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

from loophole.lifter import Lifter, LiftResult, LiftResultState, lift as lift_one
from loophole.polygeist_frontend import (
    CgeistInvocation,
    CgeistResult,
    PolygeistFrontend,
    PolygeistFrontendError,
)
from loophole.sketch_library import SKETCH_BY_NAME, SKETCH_LIBRARY
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
@click.option("--z3-timeout", type=int, default=10_000, show_default=True,
              help="Z3 solver timeout per check in milliseconds")
@click.option("--top-k", type=int, default=3, show_default=True,
              help="Number of top SymPy candidates to verify with Z3")
@click.option("--verbose", "-v", is_flag=True, help="Verbose pipeline output")
@click.option("--report", is_flag=True, help="Print a summary verification report")
@click.option("--no-verify", is_flag=True,
              help="Skip Z3 verification (emit based on SymPy match only)")
@click.option("--strict", is_flag=True,
              help="Accept only formally proved results")
def lift_cmd(
    input_file: str,
    output: Optional[str],
    target: str,
    z3_timeout: int,
    top_k: int,
    verbose: bool,
    report: bool,
    no_verify: bool,
    strict: bool,
):
    """
    Lift an MLIR Affine IR file to a high-level tensor dialect.

    INPUT_FILE: Path to the .mlir file containing the scalar loop nest.
    """
    if strict and no_verify:
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

        lifter = Lifter(
            target=target,
            z3_timeout_ms=0 if no_verify else z3_timeout,
            top_k=top_k,
            strict_mode=strict,
            verbose=verbose,
        )
        result = lifter.lift(src)
        progress.advance(task)

    _print_lift_result(result, output, report, verbose, strict_mode=strict)


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


def _print_lift_result(
    result: LiftResult,
    output: Optional[str],
    report: bool,
    verbose: bool,
    strict_mode: bool = False,
):
    exit_code = _determine_exit_code(result, strict_mode)

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
    if verbose and result.candidates_tried:
        header_lines.append(f"  Candidates tried: {', '.join(result.candidates_tried)}")

    console.print(Panel("\n".join(header_lines), title="LoopHole Lift Result", border_style=border))

    if result.emitted_mlir:
        console.print("\n[bold]Emitted MLIR:[/bold]")
        syntax = Syntax(result.emitted_mlir, "mlir", theme="monokai", line_numbers=True)
        console.print(syntax)

        if output:
            Path(output).write_text(result.emitted_mlir, encoding="utf-8")
            console.print(f"\n[dim]Written to: {output}[/dim]")

    if report and result.verification:
        _print_verification_report(result.verification)

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
    if v.z3_model:
        table.add_row("Counter-example", v.z3_model[:200])
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
@click.option("--z3-timeout", type=int, default=10_000, show_default=True,
              help="Z3 solver timeout per check in milliseconds")
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
@click.option("--timeout-sec", type=int, default=60, show_default=True,
              help="cgeist subprocess timeout in seconds")
@click.option("--verbose", "-v", is_flag=True, help="Verbose pipeline output")
def lift_c_cmd(
    source_files: tuple[str, ...],
    output: Optional[str],
    mlir_output: Optional[str],
    target: str,
    z3_timeout: int,
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
    timeout_sec: int,
    verbose: bool,
):
    """One-step flow: C/C++ source -> cgeist MLIR -> lifted tensor dialect."""
    if not source_files:
        raise click.UsageError("At least one source file is required.")
    if strict and no_verify:
        raise click.UsageError("--strict cannot be combined with --no-verify.")

    frontend = PolygeistFrontend(
        cgeist_bin=cgeist_bin,
        default_timeout_sec=timeout_sec,
        verbose=verbose,
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

    lifter = Lifter(
        target=target,
        z3_timeout_ms=0 if no_verify else z3_timeout,
        top_k=top_k,
        strict_mode=strict,
        verbose=verbose,
    )
    result = lifter.lift(cgeist_result.mlir_text)

    if verbose:
        console.print(
            f"[dim]cgeist completed in {cgeist_result.elapsed_ms:.1f} ms with "
            f"command: {' '.join(cgeist_result.command)}[/dim]"
        )

    _print_lift_result(result, output, report, verbose, strict_mode=strict)


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

@main.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", "-o", type=click.Path(), default=None,
              help="Directory for emitted .mlir files (default: same as input)")
@click.option("--target", "-t", type=click.Choice(["linalg", "stablehlo", "both"]),
              default="linalg", show_default=True)
@click.option("--z3-timeout", type=int, default=10_000, show_default=True)
@click.option("--report", is_flag=True, help="Write JSON report for batch run")
@click.option("--report-path", type=click.Path(path_type=Path), default=None,
              help="Optional JSON report path (default: <output-dir>/loophole_report.json)")
@click.option("--verbose", "-v", is_flag=True)
def batch(
    input_dir: str,
    output_dir: Optional[str],
    target: str,
    z3_timeout: int,
    report: bool,
    report_path: Optional[Path],
    verbose: bool,
):
    """
    Lift all .mlir files in a directory.

    INPUT_DIR: Directory containing .mlir Affine IR files.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir) if output_dir else in_path
    out_path.mkdir(parents=True, exist_ok=True)

    mlir_files = list(in_path.glob("**/*.mlir"))
    if not mlir_files:
        console.print(f"[yellow]No .mlir files found in {input_dir}[/yellow]")
        return

    lifter = Lifter(target=target, z3_timeout_ms=z3_timeout, verbose=verbose)

    results = []
    state_counts = {
        LiftResultState.PROVED: 0,
        LiftResultState.UNPROVED_TIMEOUT: 0,
        LiftResultState.REFUTED: 0,
    }

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

        for mlir_file in sorted(mlir_files):
            src = mlir_file.read_text(encoding="utf-8")
            result = lifter.lift(src)

            state = result.result_state
            state_counts[state] += 1
            state_label = state.value.upper()
            state_color = _result_state_color(state)

            z3_str = result.verification.result.value if result.verification else "-"
            sketch_str = result.sketch_name or "-"
            conf_str = f"{result.sympy_confidence:.2f}"
            ms_str = f"{result.total_elapsed_ms:.0f}"

            table.add_row(
                mlir_file.name,
                f"[{state_color}]{sketch_str}[/{state_color}]",
                f"[{state_color}]{state_label}[/{state_color}]",
                z3_str,
                conf_str,
                ms_str,
            )

            emitted_path = None
            if result.emitted_mlir:
                out_file = out_path / (mlir_file.stem + "_lifted.mlir")
                out_file.write_text(result.emitted_mlir, encoding="utf-8")
                emitted_path = str(out_file)

            results.append({
                "file": str(mlir_file),
                "file_name": mlir_file.name,
                "sketch": result.sketch_name,
                "state": state_label,
                "accepted_loose": result.is_accepted(strict_mode=False),
                "accepted_strict": result.is_accepted(strict_mode=True),
                "z3": z3_str,
                "confidence": result.sympy_confidence,
                "elapsed_ms": result.total_elapsed_ms,
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
    console.print(
        f"\n[bold]Summary:[/bold] "
        f"[green]{proved} PROVED[/green] / "
        f"[yellow]{unproved_timeout} UNPROVED_TIMEOUT[/yellow] / "
        f"[red]{refuted} REFUTED[/red] "
        f"out of {total} files"
    )

    if report:
        resolved_report_path = report_path or (out_path / "loophole_report.json")
        resolved_report_path.parent.mkdir(parents=True, exist_ok=True)

        report_payload = {
            "schema_version": "1.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_dir": str(in_path),
            "output_dir": str(out_path),
            "target": target,
            "z3_timeout_ms": z3_timeout,
            "summary": {
                "total_files": total,
                "proved": proved,
                "unproved_timeout": unproved_timeout,
                "refuted": refuted,
                "accepted_loose": sum(1 for r in results if r["accepted_loose"]),
                "accepted_strict": sum(1 for r in results if r["accepted_strict"]),
            },
            "results": results,
        }

        resolved_report_path.write_text(
            json.dumps(report_payload, indent=2),
            encoding="utf-8",
        )
        console.print(f"\n[dim]Report written to: {resolved_report_path}[/dim]")


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
@click.option("--verbose", "-v", is_flag=True)
def demo(target: str, verbose: bool):
    """
    Run the built-in demo: lift canonical examples (matmul, transpose, conv1d, conv2d).

    No input files needed - uses embedded MLIR fixtures.
    """
    from loophole.tests.fixtures import DEMO_FIXTURES

    console.print(Panel(
        "[bold cyan]LoopHole POC Demo[/bold cyan]\n"
        "Lifting canonical scalar loop nests into MLIR tensor operations\n"
        f"Target dialect: [magenta]{target}[/magenta]",
        border_style="cyan",
    ))

    lifter = Lifter(target=target, z3_timeout_ms=15_000, verbose=verbose)
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


if __name__ == "__main__":
    main()
