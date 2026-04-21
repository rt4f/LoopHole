"""
cli.py — Command-line interface for LoopHole.

Commands:
    loophole compile -- Compile C/C++ into MLIR Affine/SCF form (Polygeist)
    loophole cgeist  -- Convert C/C++ source to MLIR with Polygeist cgeist
    loophole lift-c  -- One-step C/C++ -> MLIR -> lifted tensor dialect
  loophole lift    -- Lift a single MLIR Affine IR file to tensor dialect
  loophole verify  -- Only run formal verification, skip emit
  loophole batch   -- Lift all .mlir files in a directory
  loophole demo    -- Run the built-in demo on canonical examples

Usage examples:
  loophole lift matmul.mlir --output matmul_lifted.mlir
    loophole cgeist kernel.c --function=kernel -o kernel.mlir
    loophole lift-c kernel.c --function=kernel --target stablehlo
  loophole lift matmul.mlir --target stablehlo --verbose
  loophole verify matmul.mlir --sketch linalg.matmul
  loophole batch kernels/ --output-dir lifted/ --report
  loophole demo
"""

from __future__ import annotations

import json
import fnmatch
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table
from rich import print as rprint

from loophole.lifter import (
    POLICY_PROFILE_CHOICES,
    Lifter,
    LiftResult,
    LiftResultState,
    lift as lift_one,
    resolve_policy_profile,
)
from loophole.polygeist_frontend import (
    CgeistInvocation,
    CgeistResult,
    PolygeistFrontend,
    PolygeistFrontendError,
)
from loophole.sketch_library import SKETCH_BY_NAME, SKETCH_LIBRARY
from loophole.mlir_validator import find_mlir_verifier, validate_mlir_artifact
from loophole.z3_checker import CheckResult

console = Console()


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option("0.1.0", prog_name="loophole")
def main():
    """
    LoopHole — Automated Lifting of Scalar Loop Nests into MLIR Tensor Dialects.

    Parses MLIR Affine IR, proves formal equivalence with Z3, and emits
    high-level Linalg or StableHLO operations.
    """
    pass


def _default_frontend_for_file(source_path: Path) -> str:
    cxx_exts = {".cc", ".cpp", ".cxx", ".c++", ".cp"}
    return "cgeist++" if source_path.suffix.lower() in cxx_exts else "cgeist"


def _common_mount_root(paths: List[Path]) -> Path:
    try:
        root = Path(os.path.commonpath([str(p.resolve()) for p in paths]))
    except ValueError as exc:
        raise click.ClickException(
            "Input and output paths must be on the same drive for Docker volume mounting."
        ) from exc
    if not root.exists():
        raise click.ClickException(f"Computed mount root does not exist: {root}")
    return root


def _to_container_relpath(path: Path, mount_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(mount_root.resolve())
    except ValueError as exc:
        raise click.ClickException(
            f"Path '{path}' is not under mount root '{mount_root}'."
        ) from exc
    return rel.as_posix()


def _build_docker_compile_script(
    frontend: str,
    source_rel: str,
    output_rel: str,
    std: Optional[str],
    raise_to_affine: bool,
) -> str:
    source_in = f"/workspace/{source_rel}"
    output_out = f"/workspace/{output_rel}"

    args = [
        frontend,
        source_in,
        "-S",
        "--function=*",
        "--memref-fullrank",
    ]
    if raise_to_affine:
        args.append("-raise-scf-to-affine")
    if std:
        args.append(f"-std={std}")

    compile_cmd = " ".join(shlex.quote(arg) for arg in args)
    # Polygeist can emit legacy dynamic dims as -1 (for example memref<-1xi32>);
    # normalize to modern MLIR '?' form so mlir-opt can parse it.
    normalize_dynamic_dims_cmd = "sed -E 's/(<|x)-1([x>])/\\1?\\2/g'"
    mlir_opt_cmd = f"mlir-opt --canonicalize > {shlex.quote(output_out)}"
    return f"set -euo pipefail; {compile_cmd} | {normalize_dynamic_dims_cmd} | {mlir_opt_cmd}"


# ---------------------------------------------------------------------------
# loophole compile
# ---------------------------------------------------------------------------

@main.command(name="compile")
@click.argument("source_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Output MLIR file path (default: <source_stem>.mlir)")
@click.option("--docker-image", default="loophole-polygeist:llvm17", show_default=True,
              help="Docker image containing cgeist/cgeist++ and mlir-opt")
@click.option("--std", default=None,
              help="Language standard passed to Polygeist frontend (for example c11 or c++17)")
@click.option("--frontend-binary", default=None,
              help="Override frontend binary (default: cgeist for C, cgeist++ for C++)")
@click.option("--no-affine-raise", is_flag=True,
              help="Do not pass -raise-scf-to-affine to Polygeist")
@click.option("--print-command", is_flag=True,
              help="Print the docker command before execution")
def compile_cmd(
    source_file: Path,
    output: Optional[Path],
    docker_image: str,
    std: Optional[str],
    frontend_binary: Optional[str],
    no_affine_raise: bool,
    print_command: bool,
):
    """
    Compile C/C++ source into MLIR using Polygeist inside Docker.

    This is step 1 of the two-step flow:
      1) loophole compile kernel.c -o kernel.mlir
      2) loophole lift kernel.mlir
    """
    source_path = source_file.resolve()
    output_path = output.resolve() if output else source_path.with_suffix(".mlir")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frontend = frontend_binary or _default_frontend_for_file(source_path)
    mount_root = _common_mount_root([source_path, output_path])
    source_rel = _to_container_relpath(source_path, mount_root)
    output_rel = _to_container_relpath(output_path, mount_root)

    script = _build_docker_compile_script(
        frontend=frontend,
        source_rel=source_rel,
        output_rel=output_rel,
        std=std,
        raise_to_affine=not no_affine_raise,
    )

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{mount_root}:/workspace",
        docker_image,
        "bash",
        "-lc",
        script,
    ]

    if print_command:
        console.print("[dim]Docker command:[/dim]")
        console.print("[dim]" + " ".join(shlex.quote(part) for part in docker_cmd) + "[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Compiling {source_path.name} with Polygeist...", total=None)
        proc = subprocess.run(docker_cmd, capture_output=True, text=True)
        progress.advance(task)

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        details = stderr or stdout or "No error details reported by docker run."
        raise click.ClickException(
            "Polygeist compile failed.\n"
            f"Frontend: {frontend}\n"
            f"Image: {docker_image}\n"
            f"Command script: {script}\n\n"
            f"Details:\n{details}"
        )

    console.print(
        Panel(
            "\n".join([
                "[green bold]COMPILE SUCCEEDED[/green bold]",
                f"  Source       : [cyan]{source_path}[/cyan]",
                f"  Output MLIR  : [magenta]{output_path}[/magenta]",
                f"  Frontend     : {frontend}",
                f"  Docker image : {docker_image}",
            ]),
            title="LoopHole Compile Result",
            border_style="green",
        )
    )


def _print_cgeist_error_and_exit(exc: PolygeistFrontendError) -> None:
    lines = [f"[red]{exc}[/red]"]
    if exc.command:
        lines.append(f"\n[bold]Command:[/bold] {' '.join(exc.command)}")
    if exc.stderr:
        lines.append("\n[bold]stderr:[/bold]")
        lines.append(exc.stderr.strip()[:2000])
    stderr_lower = (exc.stderr or "").lower()
    if "llvm.call" in stderr_lower and "printf" in stderr_lower:
        lines.append(
            "\n[bold]Hint:[/bold] Detected a vararg call path (for example printf) that this "
            "Polygeist image cannot lower for full-program lifting. "
            "Try [cyan]--function <kernel_name>[/cyan] to focus a pure compute kernel."
        )
    console.print(Panel("\n".join(lines), title="cgeist failed", border_style="red"))
    sys.exit(2)


# ---------------------------------------------------------------------------
# loophole lift
# ---------------------------------------------------------------------------

@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output MLIR file (default: print to stdout)")
@click.option("--target", "-t", type=click.Choice(["linalg", "stablehlo", "both"]),
              default="linalg", show_default=True,
              help="Target dialect to lift into")
@click.option("--profile", type=click.Choice(POLICY_PROFILE_CHOICES), default="default", show_default=True,
              help="Policy profile used to set strictness/timeouts")
@click.option("--z3-timeout", type=int, default=None,
              help="Z3 solver timeout per check in milliseconds (default: profile value)")
@click.option("--top-k", type=int, default=3, show_default=True,
              help="Number of top SymPy candidates to verify with Z3")
@click.option("--verbose", "-v", is_flag=True, help="Verbose pipeline output")
@click.option("--report", is_flag=True, help="Print a summary verification report")
@click.option("--no-verify", is_flag=True,
              help="Skip Z3 verification (emit based on SymPy match only)")
@click.option("--strict", is_flag=True,
              help="Accept only formally proved results")
@click.option("--validate-emitted", is_flag=True,
              help="Validate emitted MLIR using an external verifier command")
@click.option("--mlir-verifier", default=None,
              help="Verifier command to run (default: auto-detect mlir-opt)")
def lift_cmd(
    input_file: str,
    output: Optional[str],
    target: str,
    profile: str,
    z3_timeout: Optional[int],
    top_k: int,
    verbose: bool,
    report: bool,
    no_verify: bool,
    strict: bool,
    validate_emitted: bool,
    mlir_verifier: Optional[str],
):
    """
    Lift an MLIR Affine IR file to a high-level tensor dialect.

    INPUT_FILE: Path to the .mlir file containing the scalar loop nest.
    """
    resolved_profile = _resolve_profile_or_usage_error(profile_name=profile)
    effective_strict = strict or resolved_profile.strict_mode

    if effective_strict and no_verify:
        raise click.UsageError("--strict cannot be combined with --no-verify.")

    src = Path(input_file).read_text(encoding="utf-8")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Lifting {input_file}...", total=None)

        lifter = Lifter.from_policy_profile(
            target=target,
            z3_timeout_ms=0 if no_verify else z3_timeout,
            profile_name=profile,
            top_k=top_k,
            strict_mode=True if strict else None,
            verbose=verbose,
        )
        result = lifter.lift(src)
        progress.advance(task)

    _print_lift_result(
        result,
        output,
        report,
        verbose,
        strict_mode=effective_strict,
        validate_emitted=validate_emitted,
        mlir_verifier=mlir_verifier,
    )


def _determine_exit_code(result: LiftResult, strict_mode: bool) -> int:
    if result.success:
        return 0
    if result.result_state == LiftResultState.UNPROVED_TIMEOUT:
        return 2 if strict_mode else 0
    return 1


def _result_state_color(state: LiftResultState) -> str:
    if state == LiftResultState.PROVED:
        return "green"
    if state == LiftResultState.UNPROVED_TIMEOUT:
        return "yellow"
    return "red"


def _resolve_profile_or_usage_error(profile_name: str):
    try:
        return resolve_policy_profile(profile_name=profile_name)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


def _print_lift_result(
    result: LiftResult,
    output: Optional[str],
    report: bool,
    verbose: bool,
    strict_mode: bool = False,
    validate_emitted: bool = False,
    mlir_verifier: Optional[str] = None,
):
    exit_code = _determine_exit_code(result, strict_mode)
    validation_failed = False

    if result.success:
        status = "[green bold]FORMALLY PROVED - EQUIVALENT[/green bold]"
        border = "green"
    elif result.result_state == LiftResultState.UNPROVED_TIMEOUT:
        v = result.verification
        vstr = v.result.value if v else "unknown"
        if strict_mode:
            status = f"[yellow bold]UNPROVED_TIMEOUT - STRICT REJECTED (Z3: {vstr})[/yellow bold]"
        else:
            status = f"[yellow bold]UNPROVED_TIMEOUT (Z3: {vstr})[/yellow bold]"
        border = "yellow"
    elif result.result_state == LiftResultState.REFUTED:
        v = result.verification
        vstr = v.result.value if v else "NO_VERDICT"
        status = f"[red bold]REFUTED (Z3: {vstr})[/red bold]"
        border = "red"
    else:
        status = "[red bold]LIFT FAILED[/red bold]"
        border = "red"

    header_lines = [
        status,
        f"  Source function : [cyan]{result.func_name}[/cyan]",
        f"  Lifted to       : [magenta]{result.sketch_name or 'none'}[/magenta]",
        f"  Target dialect  : {result.target_dialect}",
        f"  SymPy confidence: {result.sympy_confidence:.2f}",
        f"  Total time      : {result.total_elapsed_ms:.1f} ms",
    ]
    if result.error:
        header_lines.append(f"  Error           : {result.error}")
    if result.verification:
        v = result.verification
        header_lines.append(
            f"  Z3 result       : {v.result.value} in {v.elapsed_ms:.1f} ms"
        )
        if v.sympy_z3_disagreement:
            header_lines.append(f"  SymPy/Z3 diag   : {v.sympy_z3_disagreement}")
        if v.failed_implication:
            header_lines.append(f"  Failed side     : {v.failed_implication}")
        if v.mismatch_summary:
            header_lines.append(f"  Counterexample  : {v.mismatch_summary}")
    if verbose and result.candidates_tried:
        header_lines.append(f"  Candidates tried: {', '.join(result.candidates_tried)}")

    console.print(Panel("\n".join(header_lines), title="LoopHole Lift Result", border_style=border))

    if result.emitted_mlir:
        console.print("\n[bold]Emitted MLIR:[/bold]")
        syntax = Syntax(result.emitted_mlir, "mlir", theme="monokai", line_numbers=True)
        console.print(syntax)

        if validate_emitted:
            verifier_cmd = find_mlir_verifier(mlir_verifier)
            if not verifier_cmd:
                console.print(
                    "[red]Emitted MLIR validation requested, but no verifier command was found. "
                    "Set --mlir-verifier or LOOPHOLE_MLIR_VERIFY_CMD.[/red]"
                )
                validation_failed = True
            else:
                try:
                    validation = validate_mlir_artifact(result.emitted_mlir, verifier_cmd)
                    if validation.ok:
                        console.print(f"[green]Emitted MLIR validation passed[/green] via [dim]{verifier_cmd}[/dim]")
                    else:
                        validation_failed = True
                        console.print(f"[red]Emitted MLIR validation failed[/red] via [dim]{verifier_cmd}[/dim]")
                        if validation.stderr:
                            console.print(Panel(validation.stderr[:2000], title="Verifier stderr", border_style="red"))
                        elif validation.stdout:
                            console.print(Panel(validation.stdout[:2000], title="Verifier stdout", border_style="red"))
                except Exception as exc:
                    validation_failed = True
                    console.print(f"[red]Emitted MLIR validation error:[/red] {exc}")

        if output:
            Path(output).write_text(result.emitted_mlir, encoding="utf-8")
            console.print(f"\n[dim]Written to: {output}[/dim]")

    if report and result.verification:
        _print_verification_report(result.verification)

    if validation_failed:
        exit_code = 1

    if exit_code != 0:
        sys.exit(exit_code)


def _print_verification_report(v):
    table = Table(title="Z3 Verification Report", show_header=True)
    table.add_column("Property", style="bold")
    table.add_column("Value")
    table.add_row("Sketch", v.sketch_name)
    table.add_row("Result", v.result.value)
    table.add_row("Time (ms)", f"{v.elapsed_ms:.1f}")
    table.add_row("Notes", v.notes or "-")
    if v.sympy_confidence is not None:
        table.add_row("SymPy confidence", f"{v.sympy_confidence:.2f}")
    if v.sympy_z3_disagreement:
        table.add_row("SymPy/Z3 diagnosis", v.sympy_z3_disagreement)
    if v.failed_implication:
        table.add_row("Failed implication", v.failed_implication)
    if v.mismatch_summary:
        table.add_row("Mismatch summary", v.mismatch_summary)
    if v.z3_model:
        table.add_row("Counter-example", v.z3_model[:200])
    if v.counterexample_bindings:
        preview_items = sorted(v.counterexample_bindings.items())[:8]
        preview = ", ".join(f"{k}={val}" for k, val in preview_items)
        if len(v.counterexample_bindings) > len(preview_items):
            preview += f", ... ({len(v.counterexample_bindings)} total)"
        table.add_row("Counterexample bindings", preview)
    console.print(table)


# ---------------------------------------------------------------------------
# loophole cgeist
# ---------------------------------------------------------------------------

@main.command(name="cgeist")
@click.argument("source_files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output MLIR file (default: print to stdout)")
@click.option("--language", "-x", type=click.Choice(["c", "cpp"]), default="c",
              show_default=True, help="Source language for cgeist")
@click.option("--function", default=None,
              help="Optional entry function name to focus conversion")
@click.option("--include-dir", "-I", multiple=True, type=click.Path(exists=True, file_okay=False),
              help="Include directory (repeatable)")
@click.option("--define", "-D", multiple=True,
              help="Macro definition in KEY=VALUE form (repeatable)")
@click.option("--std", default=None,
              help="Language standard, e.g. c11, c17, c++17, c++20")
@click.option("--clang-arg", multiple=True,
              help="Extra argument forwarded to cgeist/clang (repeatable)")
@click.option("--cgeist-bin", default="cgeist", show_default=True,
              help="Path to cgeist executable")
@click.option("--docker-image", default=None,
              help="Optional Docker image for cgeist execution (auto-detect if omitted)")
@click.option("--timeout-sec", type=int, default=60, show_default=True,
              help="cgeist subprocess timeout in seconds")
@click.option("--verbose", "-v", is_flag=True, help="Print command details")
def cgeist_cmd(
    source_files: tuple[str, ...],
    output: Optional[str],
    language: str,
    function: Optional[str],
    include_dir: tuple[str, ...],
    define: tuple[str, ...],
    std: Optional[str],
    clang_arg: tuple[str, ...],
    cgeist_bin: str,
    docker_image: Optional[str],
    timeout_sec: int,
    verbose: bool,
):
    """Convert C/C++ source files to MLIR using Polygeist cgeist."""
    if not source_files:
        raise click.UsageError("At least one source file is required.")

    frontend = PolygeistFrontend(
        cgeist_bin=cgeist_bin,
        default_timeout_sec=timeout_sec,
        verbose=verbose,
        docker_image=docker_image,
        prefer_docker=True,
    )
    invocation = CgeistInvocation(
        source_files=list(source_files),
        language=language,
        function=function,
        include_dirs=list(include_dir),
        defines=list(define),
        std=std,
        extra_clang_args=list(clang_arg),
        timeout_sec=timeout_sec,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running cgeist...", total=None)
        try:
            result = frontend.generate_mlir(invocation)
        except PolygeistFrontendError as exc:
            progress.advance(task)
            _print_cgeist_error_and_exit(exc)
        progress.advance(task)

    if output:
        Path(output).write_text(result.mlir_text, encoding="utf-8")
        console.print(f"[green]Generated MLIR written to:[/green] {output}")
    else:
        syntax = Syntax(result.mlir_text, "mlir", theme="monokai", line_numbers=True)
        console.print(syntax)

    console.print(
        f"[dim]cgeist completed in {result.elapsed_ms:.1f} ms; "
        f"return code {result.returncode}[/dim]"
    )


# ---------------------------------------------------------------------------
# loophole lift-c
# ---------------------------------------------------------------------------

@main.command(name="lift-c")
@click.argument("source_files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output lifted MLIR file (default: print to stdout)")
@click.option("--mlir-output", type=click.Path(), default=None,
              help="Optional path to save intermediate MLIR from cgeist")
@click.option("--target", "-t", type=click.Choice(["linalg", "stablehlo", "both"]),
              default="linalg", show_default=True,
              help="Target dialect to lift into")
@click.option("--profile", type=click.Choice(POLICY_PROFILE_CHOICES), default="default", show_default=True,
              help="Policy profile used to set strictness/timeouts")
@click.option("--z3-timeout", type=int, default=None,
              help="Z3 solver timeout per check in milliseconds (default: profile value)")
@click.option("--top-k", type=int, default=3, show_default=True,
              help="Number of top SymPy candidates to verify with Z3")
@click.option("--no-verify", is_flag=True,
              help="Skip Z3 verification (emit based on SymPy match only)")
@click.option("--strict", is_flag=True,
              help="Accept only formally proved results")
@click.option("--report", is_flag=True, help="Print a summary verification report")
@click.option("--language", "-x", type=click.Choice(["c", "cpp"]), default="c",
              show_default=True, help="Source language for cgeist")
@click.option("--function", default=None,
              help="Optional entry function name to focus conversion")
@click.option("--include-dir", "-I", multiple=True, type=click.Path(exists=True, file_okay=False),
              help="Include directory (repeatable)")
@click.option("--define", "-D", multiple=True,
              help="Macro definition in KEY=VALUE form (repeatable)")
@click.option("--std", default=None,
              help="Language standard, e.g. c11, c17, c++17, c++20")
@click.option("--clang-arg", multiple=True,
              help="Extra argument forwarded to cgeist/clang (repeatable)")
@click.option("--cgeist-bin", default="cgeist", show_default=True,
              help="Path to cgeist executable")
@click.option("--docker-image", default=None,
              help="Optional Docker image for cgeist execution (auto-detect if omitted)")
@click.option("--timeout-sec", type=int, default=60, show_default=True,
              help="cgeist subprocess timeout in seconds")
@click.option("--verbose", "-v", is_flag=True, help="Verbose pipeline output")
@click.option("--validate-emitted", is_flag=True,
              help="Validate emitted MLIR using an external verifier command")
@click.option("--mlir-verifier", default=None,
              help="Verifier command to run (default: auto-detect mlir-opt)")
def lift_c_cmd(
    source_files: tuple[str, ...],
    output: Optional[str],
    mlir_output: Optional[str],
    target: str,
    profile: str,
    z3_timeout: Optional[int],
    top_k: int,
    no_verify: bool,
    strict: bool,
    report: bool,
    language: str,
    function: Optional[str],
    include_dir: tuple[str, ...],
    define: tuple[str, ...],
    std: Optional[str],
    clang_arg: tuple[str, ...],
    cgeist_bin: str,
    docker_image: Optional[str],
    timeout_sec: int,
    verbose: bool,
    validate_emitted: bool,
    mlir_verifier: Optional[str],
):
    """One-step flow: C/C++ source -> cgeist MLIR -> lifted tensor dialect."""
    if not source_files:
        raise click.UsageError("At least one source file is required.")
    resolved_profile = _resolve_profile_or_usage_error(profile_name=profile)
    effective_strict = strict or resolved_profile.strict_mode

    if effective_strict and no_verify:
        raise click.UsageError("--strict cannot be combined with --no-verify.")

    frontend = PolygeistFrontend(
        cgeist_bin=cgeist_bin,
        default_timeout_sec=timeout_sec,
        verbose=verbose,
        docker_image=docker_image,
        prefer_docker=True,
    )
    invocation = CgeistInvocation(
        source_files=list(source_files),
        language=language,
        function=function,
        include_dirs=list(include_dir),
        defines=list(define),
        std=std,
        extra_clang_args=list(clang_arg),
        timeout_sec=timeout_sec,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Generating MLIR with cgeist...", total=None)
        try:
            cgeist_result = frontend.generate_mlir(invocation)
        except PolygeistFrontendError as exc:
            progress.advance(task)
            _print_cgeist_error_and_exit(exc)
        progress.advance(task)

    if mlir_output:
        Path(mlir_output).write_text(cgeist_result.mlir_text, encoding="utf-8")
        if verbose:
            console.print(f"[dim]Intermediate MLIR written to: {mlir_output}[/dim]")

    lifter = Lifter.from_policy_profile(
        target=target,
        z3_timeout_ms=0 if no_verify else z3_timeout,
        profile_name=profile,
        top_k=top_k,
        strict_mode=True if strict else None,
        verbose=verbose,
    )
    result = lifter.lift(cgeist_result.mlir_text)

    if verbose:
        console.print(
            f"[dim]cgeist completed in {cgeist_result.elapsed_ms:.1f} ms with "
            f"command: {' '.join(cgeist_result.command)}[/dim]"
        )

    _print_lift_result(
        result,
        output,
        report,
        verbose,
        strict_mode=effective_strict,
        validate_emitted=validate_emitted,
        mlir_verifier=mlir_verifier,
    )


# ---------------------------------------------------------------------------
# loophole verify
# ---------------------------------------------------------------------------

@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--sketch", default=None,
              help="Sketch name to check against (e.g. linalg.matmul). "
                   "If omitted, all sketches are tried.")
@click.option("--z3-timeout", type=int, default=10_000, show_default=True)
@click.option("--verbose", "-v", is_flag=True)
def verify(input_file: str, sketch: Optional[str], z3_timeout: int, verbose: bool):
    """
    Run Z3 formal equivalence check on an MLIR Affine IR file.

    Does not emit MLIR — only proves (or disproves) equivalence.
    """
    from loophole.affine_extractor import AffineExtractor
    from loophole.z3_checker import Z3EquivalenceChecker

    src = Path(input_file).read_text(encoding="utf-8")
    extractor = AffineExtractor()
    checker = Z3EquivalenceChecker(timeout_ms=z3_timeout)

    try:
        loop = extractor.extract(src)
    except Exception as exc:
        console.print(f"[red]Parse error:[/red] {exc}")
        sys.exit(1)

    console.print(f"[cyan]Parsed:[/cyan] '{loop.func_name}' — "
                  f"{len(loop.induction_vars)} loops, "
                  f"{len(loop.reduction_vars)} reduction IVs")

    if sketch:
        target_sketches = [SKETCH_BY_NAME[sketch]] if sketch in SKETCH_BY_NAME else []
        if not target_sketches:
            console.print(f"[red]Unknown sketch:[/red] {sketch}. "
                          f"Use 'loophole sketches' to list all.")
            sys.exit(1)
    else:
        target_sketches = SKETCH_LIBRARY

    table = Table(title="Verification Results", show_header=True)
    table.add_column("Sketch")
    table.add_column("Result")
    table.add_column("Time (ms)")
    table.add_column("Notes")

    found = False
    for sk in target_sketches:
        report = checker.check(loop, sk)
        color = {
            CheckResult.EQUIVALENT: "green",
            CheckResult.NOT_EQUIVALENT: "red",
            CheckResult.STRUCTURAL_MISMATCH: "dim",
            CheckResult.TIMEOUT: "yellow",
            CheckResult.ENCODE_ERROR: "red",
            CheckResult.UNKNOWN: "yellow",
        }.get(report.result, "white")

        if report.result == CheckResult.STRUCTURAL_MISMATCH and not verbose:
            continue

        table.add_row(
            sk.name,
            f"[{color}]{report.result.value}[/{color}]",
            f"{report.elapsed_ms:.1f}",
            (report.notes or "")[:60],
        )

        if report.result == CheckResult.EQUIVALENT:
            found = True

    console.print(table)

    if found:
        console.print("\n[green bold]Equivalence formally proved.[/green bold]")
    else:
        console.print("\n[yellow]No sketch was formally proved equivalent.[/yellow]")


# ---------------------------------------------------------------------------
# loophole batch
# ---------------------------------------------------------------------------


def _collect_batch_mlir_files(
    in_path: Path,
    include_globs: tuple[str, ...],
    exclude_globs: tuple[str, ...],
    max_files: Optional[int],
) -> tuple[list[Path], int]:
    include_patterns = include_globs or ("**/*.mlir",)
    seen: set[Path] = set()
    matched: list[Path] = []

    for pattern in include_patterns:
        for candidate in in_path.glob(pattern):
            if not candidate.is_file() or candidate.suffix.lower() != ".mlir":
                continue
            # Avoid recursive re-processing of previously generated artifacts.
            if candidate.name.endswith("_lifted.mlir"):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            matched.append(candidate)

    matched.sort(key=lambda path: path.relative_to(in_path).as_posix())

    if exclude_globs:
        filtered: list[Path] = []
        for candidate in matched:
            rel_path = candidate.relative_to(in_path).as_posix()
            if any(fnmatch.fnmatch(rel_path, pattern) for pattern in exclude_globs):
                continue
            filtered.append(candidate)
        matched = filtered

    matched_before_limit = len(matched)
    if max_files is not None:
        if max_files < 1:
            raise click.UsageError("--max-files must be >= 1 when provided.")
        matched = matched[:max_files]

    return matched, matched_before_limit


@main.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", "-o", type=click.Path(), default=None,
              help="Directory for emitted .mlir files (default: same as input)")
@click.option("--target", "-t", type=click.Choice(["linalg", "stablehlo", "both"]),
              default="linalg", show_default=True)
@click.option("--profile", type=click.Choice(POLICY_PROFILE_CHOICES), default="default", show_default=True,
              help="Policy profile used to set strictness/timeouts")
@click.option("--strict", is_flag=True,
              help="Accept only formally proved outputs")
@click.option("--z3-timeout", type=int, default=None,
              help="Z3 solver timeout per check in milliseconds (default: profile value)")
@click.option("--report", is_flag=True, help="Write JSON report for batch run")
@click.option("--report-path", type=click.Path(path_type=Path), default=None,
              help="Optional JSON report path (default: <output-dir>/loophole_report.json)")
@click.option("--include-glob", "include_globs", multiple=True, default=("**/*.mlir",), show_default=True,
              help="Glob pattern (relative to input dir) to include; repeatable")
@click.option("--exclude-glob", "exclude_globs", multiple=True, default=(),
              help="Glob pattern (relative to input dir) to exclude; repeatable")
@click.option("--max-files", type=int, default=None,
              help="Process at most N files after include/exclude filtering")
@click.option("--table-limit", type=int, default=100, show_default=True,
              help="Limit displayed per-file rows in terminal output")
@click.option("--write-emitted/--no-write-emitted", default=True, show_default=True,
              help="Write emitted *_lifted.mlir artifacts")
@click.option("--top-slowest", type=int, default=10, show_default=True,
              help="Show N slowest files in terminal summary (0 to disable)")
@click.option("--validate-emitted", is_flag=True,
              help="Validate emitted MLIR using an external verifier command")
@click.option("--mlir-verifier", default=None,
              help="Verifier command to run (default: auto-detect mlir-opt)")
@click.option("--verbose", "-v", is_flag=True)
def batch(
    input_dir: str,
    output_dir: Optional[str],
    target: str,
    profile: str,
    strict: bool,
    z3_timeout: Optional[int],
    report: bool,
    report_path: Optional[Path],
    include_globs: tuple[str, ...],
    exclude_globs: tuple[str, ...],
    max_files: Optional[int],
    table_limit: int,
    write_emitted: bool,
    top_slowest: int,
    validate_emitted: bool,
    mlir_verifier: Optional[str],
    verbose: bool,
):
    """
    Lift all .mlir files in a directory.

    INPUT_DIR: Directory containing .mlir Affine IR files.
    """
    if table_limit < 1:
        raise click.UsageError("--table-limit must be >= 1.")
    if top_slowest < 0:
        raise click.UsageError("--top-slowest must be >= 0.")

    in_path = Path(input_dir)
    out_path = Path(output_dir) if output_dir else in_path
    out_path.mkdir(parents=True, exist_ok=True)

    mlir_files, matched_before_limit = _collect_batch_mlir_files(
        in_path=in_path,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        max_files=max_files,
    )
    if not mlir_files:
        console.print(
            f"[yellow]No .mlir files found in {input_dir} for the requested selection filters.[/yellow]"
        )
        return

    resolved_profile = _resolve_profile_or_usage_error(profile_name=profile)
    effective_strict = strict or resolved_profile.strict_mode

    lifter = Lifter.from_policy_profile(
        target=target,
        profile_name=profile,
        z3_timeout_ms=z3_timeout,
        strict_mode=True if strict else None,
        verbose=verbose,
    )

    results = []
    state_counts = {
        LiftResultState.PROVED: 0,
        LiftResultState.UNPROVED_TIMEOUT: 0,
        LiftResultState.REFUTED: 0,
    }
    validation_failed = 0
    displayed_rows = 0
    timings: list[dict[str, object]] = []

    verifier_cmd = None
    if validate_emitted:
        verifier_cmd = find_mlir_verifier(mlir_verifier)
        if not verifier_cmd:
            console.print(
                "[red]Batch emitted MLIR validation requested, but no verifier command was found. "
                "Set --mlir-verifier or LOOPHOLE_MLIR_VERIFY_CMD.[/red]"
            )
            sys.exit(1)

    table = Table(title=f"Batch Lift Results: {input_dir}", show_header=True)
    table.add_column("File", style="cyan")
    table.add_column("Sketch")
    table.add_column("State")
    table.add_column("Z3")
    table.add_column("Conf")
    table.add_column("ms")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing...", total=len(mlir_files))

        for mlir_file in mlir_files:
            src = mlir_file.read_text(encoding="utf-8")
            result = lifter.lift(src)

            rel_file = mlir_file.relative_to(in_path).as_posix()
            state = result.result_state
            state_counts[state] += 1
            state_label = state.value.upper()
            state_color = _result_state_color(state)

            z3_str = result.verification.result.value if result.verification else "-"
            mismatch_summary = result.verification.mismatch_summary if result.verification else None
            failed_implication = result.verification.failed_implication if result.verification else None
            disagreement = result.verification.sympy_z3_disagreement if result.verification else None
            verification_notes = result.verification.notes if result.verification else None
            sketch_str = result.sketch_name or "-"
            conf_str = f"{result.sympy_confidence:.2f}"
            ms_str = f"{result.total_elapsed_ms:.0f}"

            if displayed_rows < table_limit:
                table.add_row(
                    rel_file,
                    f"[{state_color}]{sketch_str}[/{state_color}]",
                    f"[{state_color}]{state_label}[/{state_color}]",
                    z3_str,
                    conf_str,
                    ms_str,
                )
                displayed_rows += 1

            emitted_path = None
            if write_emitted and result.emitted_mlir:
                rel_source = mlir_file.relative_to(in_path)
                out_file = out_path / rel_source.parent / f"{rel_source.stem}_lifted.mlir"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text(result.emitted_mlir, encoding="utf-8")
                emitted_path = str(out_file)

                if validate_emitted and verifier_cmd:
                    validation = validate_mlir_artifact(result.emitted_mlir, verifier_cmd)
                    if not validation.ok:
                        validation_failed += 1
                        console.print(
                            f"[red]Validation failed[/red] for {rel_file} via [dim]{verifier_cmd}[/dim]"
                        )
                        detail = validation.stderr or validation.stdout
                        if detail:
                            console.print(Panel(detail[:1200], title="Verifier output", border_style="red"))

            timings.append(
                {
                    "file_rel": rel_file,
                    "state": state_label,
                    "state_color": state_color,
                    "elapsed_ms": result.total_elapsed_ms,
                }
            )

            results.append({
                "file": str(mlir_file),
                "file_rel": rel_file,
                "file_name": mlir_file.name,
                "sketch": result.sketch_name,
                "state": state_label,
                "accepted_loose": result.is_accepted(strict_mode=False),
                "accepted_strict": result.is_accepted(strict_mode=True),
                "z3": z3_str,
                "confidence": result.sympy_confidence,
                "elapsed_ms": result.total_elapsed_ms,
                "failed_implication": failed_implication,
                "sympy_z3_disagreement": disagreement,
                "mismatch_summary": mismatch_summary,
                "verification_notes": verification_notes,
                "counterexample_bindings": result.verification.counterexample_bindings if result.verification else {},
                "success": result.success,
                "error": result.error,
                "emitted_file": emitted_path,
            })
            progress.advance(task)

    total = len(mlir_files)
    proved = state_counts[LiftResultState.PROVED]
    unproved_timeout = state_counts[LiftResultState.UNPROVED_TIMEOUT]
    refuted = state_counts[LiftResultState.REFUTED]

    console.print(table)
    if total > displayed_rows:
        console.print(f"[dim]Displayed {displayed_rows} of {total} rows. Use --table-limit to adjust.[/dim]")

    console.print(
        f"\n[bold]Summary:[/bold] "
        f"[green]{proved} PROVED[/green] / "
        f"[yellow]{unproved_timeout} UNPROVED_TIMEOUT[/yellow] / "
        f"[red]{refuted} REFUTED[/red] "
        f"out of {total} files"
    )

    if top_slowest > 0 and timings:
        slowest_rows = sorted(
            timings,
            key=lambda row: float(row["elapsed_ms"]),
            reverse=True,
        )[:top_slowest]

        slowest_table = Table(title="Slowest Files", show_header=True)
        slowest_table.add_column("File", style="cyan")
        slowest_table.add_column("State")
        slowest_table.add_column("ms")
        for row in slowest_rows:
            slowest_table.add_row(
                str(row["file_rel"]),
                f"[{row['state_color']}]{row['state']}[/{row['state_color']}]",
                f"{float(row['elapsed_ms']):.0f}",
            )
        console.print(slowest_table)

    if report:
        resolved_report_path = report_path or (out_path / "loophole_report.json")
        resolved_report_path.parent.mkdir(parents=True, exist_ok=True)

        slowest_rows = sorted(
            timings,
            key=lambda row: float(row["elapsed_ms"]),
            reverse=True,
        )[: min(25, len(timings))]

        report_payload = {
            "schema_version": "1.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_dir": str(in_path),
            "output_dir": str(out_path),
            "target": target,
            "z3_timeout_ms": lifter.z3_timeout_ms,
            "selection": {
                "include_globs": list(include_globs),
                "exclude_globs": list(exclude_globs),
                "max_files": max_files,
                "matched_before_limit": matched_before_limit,
                "processed_files": total,
            },
            "summary": {
                "total_files": total,
                "proved": proved,
                "unproved_timeout": unproved_timeout,
                "refuted": refuted,
                "profile": resolved_profile.name,
                "strict_mode": effective_strict,
                "accepted_loose": sum(1 for r in results if r["accepted_loose"]),
                "accepted_strict": sum(1 for r in results if r["accepted_strict"]),
                "write_emitted": write_emitted,
            },
            "slowest": [
                {
                    "file_rel": str(row["file_rel"]),
                    "state": str(row["state"]),
                    "elapsed_ms": float(row["elapsed_ms"]),
                }
                for row in slowest_rows
            ],
            "results": results,
        }

        resolved_report_path.write_text(
            json.dumps(report_payload, indent=2),
            encoding="utf-8",
        )
        console.print(f"\n[dim]Report written to: {resolved_report_path}[/dim]")

    if validate_emitted and validation_failed:
        sys.exit(1)

    if effective_strict and (unproved_timeout > 0 or refuted > 0):
        sys.exit(1)

# ---------------------------------------------------------------------------
# loophole sketches
# ---------------------------------------------------------------------------

@main.command()
@click.option("--dialect", "-d", type=click.Choice(["linalg", "stablehlo", "all"]),
              default="all", show_default=True)
def sketches(dialect: str):
    """List all operation sketches in the library."""
    table = Table(title="LoopHole Sketch Library", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Dialect")
    table.add_column("Loops")
    table.add_column("Parallel")
    table.add_column("Reduction")
    table.add_column("Description")

    for sk in SKETCH_LIBRARY:
        if dialect != "all" and sk.dialect != dialect:
            continue
        table.add_row(
            sk.name,
            sk.dialect,
            str(sk.num_loops),
            str(sk.num_parallel),
            str(sk.num_reduction),
            sk.description[:60],
        )

    console.print(table)
    console.print(f"\n[dim]Total: {sum(1 for s in SKETCH_LIBRARY if dialect == 'all' or s.dialect == dialect)} sketches[/dim]")


# ---------------------------------------------------------------------------
# loophole demo
# ---------------------------------------------------------------------------

@main.command()
@click.option("--target", "-t", type=click.Choice(["linalg", "stablehlo"]),
              default="linalg", show_default=True)
@click.option("--profile", type=click.Choice(POLICY_PROFILE_CHOICES), default="default", show_default=True,
              help="Policy profile used to set strictness/timeouts")
@click.option("--z3-timeout", type=int, default=None,
              help="Z3 solver timeout per check in milliseconds (default: profile value)")
@click.option("--strict", is_flag=True,
              help="Accept only formally proved results")
@click.option("--verbose", "-v", is_flag=True)
def demo(
    target: str,
    profile: str,
    z3_timeout: Optional[int],
    strict: bool,
    verbose: bool,
):
    """
    Run the built-in demo: lift canonical examples (matmul, transpose, conv1d, conv2d).

    No input files needed - uses embedded MLIR fixtures.
    """
    from loophole.tests.fixtures import DEMO_FIXTURES

    resolved_profile = _resolve_profile_or_usage_error(profile_name=profile)
    effective_strict = strict or resolved_profile.strict_mode

    lifter = Lifter.from_policy_profile(
        target=target,
        profile_name=profile,
        z3_timeout_ms=z3_timeout,
        strict_mode=True if strict else None,
        verbose=verbose,
    )

    console.print(Panel(
        "[bold cyan]LoopHole POC Demo[/bold cyan]\n"
        "Lifting canonical scalar loop nests into MLIR tensor operations\n"
        f"Target dialect: [magenta]{target}[/magenta]\n"
        f"Policy profile: [magenta]{resolved_profile.name}[/magenta] | "
        f"Strict mode: [magenta]{'on' if effective_strict else 'off'}[/magenta] | "
        f"Z3 timeout: [magenta]{lifter.z3_timeout_ms} ms[/magenta]",
        border_style="cyan",
    ))
    results_data = []

    for fixture_name, mlir_text in DEMO_FIXTURES.items():
        console.print(f"\n[bold]--- {fixture_name} ---[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=True,
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]Lifting...", total=None)
            result = lifter.lift(mlir_text)
            progress.advance(task)

        state = result.result_state
        state_label = state.value.upper()
        state_color = _result_state_color(state)
        if result.verification:
            z3_str = (
                f"[{state_color}]{result.verification.result.value}[/{state_color}] "
                f"in {result.verification.elapsed_ms:.0f}ms"
            )
        else:
            z3_str = "[dim]NO_VERDICT[/dim]"

        console.print(
            f"  [{state_color}]{state_label}[/{state_color}] {fixture_name} -> [magenta]{result.sketch_name or 'no match'}[/magenta] "
            f"| Z3: {z3_str} | conf: {result.sympy_confidence:.2f} | {result.total_elapsed_ms:.0f}ms"
        )

        if result.parser_diagnostics:
            for diag in result.parser_diagnostics:
                level_color = {"error": "red", "warning": "yellow", "debug": "cyan"}.get(diag.level, "white")
                console.print(f"    [{level_color}]{diag.level.upper()}[/{level_color}] {diag.location}: {diag.reason}")
                if diag.guidance:
                    console.print(f"      -> {diag.guidance}")

        if result.emitted_mlir:
            syntax = Syntax(result.emitted_mlir, "mlir", theme="monokai", line_numbers=False)
            console.print(syntax)

        results_data.append({
            "kernel": fixture_name,
            "sketch": result.sketch_name,
            "state": state_label,
            "confidence": result.sympy_confidence,
            "ms": result.total_elapsed_ms,
        })

    console.print("\n")
    table = Table(title="Demo Summary", show_header=True)
    table.add_column("Kernel", style="cyan")
    table.add_column("Lifted to")
    table.add_column("Status")
    table.add_column("Conf")
    table.add_column("ms")

    proved = sum(1 for r in results_data if r["state"] == "PROVED")
    unproved_timeout = sum(1 for r in results_data if r["state"] == "UNPROVED_TIMEOUT")
    refuted = sum(1 for r in results_data if r["state"] == "REFUTED")

    for r in results_data:
        if r["state"] == "PROVED":
            status = "[green]PROVED[/green]"
        elif r["state"] == "UNPROVED_TIMEOUT":
            status = "[yellow]UNPROVED_TIMEOUT[/yellow]"
        else:
            status = "[red]REFUTED[/red]"

        table.add_row(
            r["kernel"],
            r["sketch"] or "-",
            status,
            f"{r['confidence']:.2f}",
            f"{r['ms']:.0f}",
        )

    console.print(table)
    console.print(
        f"\n[bold]Result:[/bold] "
        f"[green]{proved}/{len(results_data)} formally proved[/green] | "
        f"[yellow]{unproved_timeout} unproved (timeout/unknown)[/yellow] | "
        f"[red]{refuted} refuted[/red]"
    )

    if effective_strict and (unproved_timeout > 0 or refuted > 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
