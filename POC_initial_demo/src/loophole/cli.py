"""
cli.py — Command-line interface for LoopHole.

Commands:
  loophole lift    -- Lift a single MLIR Affine IR file to tensor dialect
  loophole verify  -- Only run formal verification, skip emit
  loophole batch   -- Lift all .mlir files in a directory
  loophole demo    -- Run the built-in demo on canonical examples

Usage examples:
  loophole lift matmul.mlir --output matmul_lifted.mlir
  loophole lift matmul.mlir --target stablehlo --verbose
  loophole verify matmul.mlir --sketch linalg.matmul
  loophole batch kernels/ --output-dir lifted/ --report
  loophole demo
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table
from rich import print as rprint

from loophole.lifter import Lifter, LiftResult, lift as lift_one
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
def lift_cmd(
    input_file: str,
    output: Optional[str],
    target: str,
    z3_timeout: int,
    top_k: int,
    verbose: bool,
    report: bool,
    no_verify: bool,
):
    """
    Lift an MLIR Affine IR file to a high-level tensor dialect.

    INPUT_FILE: Path to the .mlir file containing the scalar loop nest.
    """
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
            verbose=verbose,
        )
        result = lifter.lift(src)
        progress.advance(task)

    _print_lift_result(result, output, report, verbose)


def _print_lift_result(
    result: LiftResult,
    output: Optional[str],
    report: bool,
    verbose: bool,
):
    if result.error and not result.partial_success:
        console.print(Panel(
            f"[red bold]LIFT FAILED[/red bold]\n\n"
            f"[red]{result.error}[/red]",
            title="LoopHole Result",
            border_style="red",
        ))
        sys.exit(1)

    # Determine status color
    if result.success:
        status = "[green bold]✓ FORMALLY PROVED — EQUIVALENT[/green bold]"
        border = "green"
    elif result.partial_success:
        v = result.verification
        vstr = v.result.value if v else "unknown"
        status = f"[yellow bold]~ MATCHED (Z3: {vstr})[/yellow bold]"
        border = "yellow"
    else:
        status = "[red]✗ LIFT FAILED[/red]"
        border = "red"

    header_lines = [
        status,
        f"  Source function : [cyan]{result.func_name}[/cyan]",
        f"  Lifted to       : [magenta]{result.sketch_name or 'none'}[/magenta]",
        f"  Target dialect  : {result.target_dialect}",
        f"  SymPy confidence: {result.sympy_confidence:.2f}",
        f"  Total time      : {result.total_elapsed_ms:.1f} ms",
    ]
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
        console.print("\n[green bold]✓ Equivalence formally proved.[/green bold]")
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
@click.option("--report", is_flag=True, help="Print JSON benchmarking report")
@click.option("--verbose", "-v", is_flag=True)
def batch(
    input_dir: str,
    output_dir: Optional[str],
    target: str,
    z3_timeout: int,
    report: bool,
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
    success = 0
    partial = 0
    failed = 0

    table = Table(title=f"Batch Lift Results: {input_dir}", show_header=True)
    table.add_column("File", style="cyan")
    table.add_column("Sketch")
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

            z3_str = result.verification.result.value if result.verification else "-"
            sketch_str = result.sketch_name or "-"
            conf_str = f"{result.sympy_confidence:.2f}"
            ms_str = f"{result.total_elapsed_ms:.0f}"

            color = "green" if result.success else ("yellow" if result.partial_success else "red")
            table.add_row(
                mlir_file.name,
                f"[{color}]{sketch_str}[/{color}]",
                z3_str,
                conf_str,
                ms_str,
            )

            if result.success:
                success += 1
            elif result.partial_success:
                partial += 1
            else:
                failed += 1

            if result.emitted_mlir:
                out_file = out_path / (mlir_file.stem + "_lifted.mlir")
                out_file.write_text(result.emitted_mlir, encoding="utf-8")

            results.append({
                "file": str(mlir_file),
                "sketch": result.sketch_name,
                "z3": z3_str,
                "confidence": result.sympy_confidence,
                "elapsed_ms": result.total_elapsed_ms,
                "success": result.success,
            })
            progress.advance(task)

    console.print(table)
    console.print(
        f"\n[bold]Summary:[/bold] "
        f"[green]{success} proved[/green] / "
        f"[yellow]{partial} partial[/yellow] / "
        f"[red]{failed} failed[/red] "
        f"out of {len(mlir_files)} files"
    )

    if report:
        report_path = out_path / "loophole_report.json"
        report_path.write_text(json.dumps(results, indent=2))
        console.print(f"\n[dim]Report written to: {report_path}[/dim]")


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

    No input files needed — uses embedded MLIR fixtures.
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
        console.print(f"\n[bold]─── {fixture_name} ───[/bold]")

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

        # Print summary line
        if result.success:
            icon = "[green]✓[/green]"
            z3_str = f"[green]{result.verification.result.value}[/green] in {result.verification.elapsed_ms:.0f}ms"
        elif result.partial_success:
            icon = "[yellow]~[/yellow]"
            v = result.verification
            z3_str = f"[yellow]{v.result.value if v else 'N/A'}[/yellow]"
        else:
            icon = "[red]✗[/red]"
            z3_str = "[red]FAIL[/red]"

        console.print(
            f"  {icon} {fixture_name} → [magenta]{result.sketch_name or 'no match'}[/magenta] "
            f"| Z3: {z3_str} | conf: {result.sympy_confidence:.2f} | {result.total_elapsed_ms:.0f}ms"
        )

        if result.emitted_mlir:
            syntax = Syntax(result.emitted_mlir, "mlir", theme="monokai", line_numbers=False)
            console.print(syntax)

        results_data.append({
            "kernel": fixture_name,
            "sketch": result.sketch_name,
            "success": result.success,
            "partial": result.partial_success,
            "confidence": result.sympy_confidence,
            "ms": result.total_elapsed_ms,
        })

    # Final summary table
    console.print("\n")
    table = Table(title="Demo Summary", show_header=True)
    table.add_column("Kernel", style="cyan")
    table.add_column("Lifted to")
    table.add_column("Status")
    table.add_column("Conf")
    table.add_column("ms")

    proved = sum(1 for r in results_data if r["success"])
    partial = sum(1 for r in results_data if r["partial"] and not r["success"])

    for r in results_data:
        status = "[green]PROVED[/green]" if r["success"] else (
            "[yellow]PARTIAL[/yellow]" if r["partial"] else "[red]FAIL[/red]"
        )
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
        f"[yellow]{partial} partial (Z3 timeout)[/yellow]"
    )
