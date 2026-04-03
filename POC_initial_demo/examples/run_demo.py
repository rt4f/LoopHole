#!/usr/bin/env python3
"""
run_demo.py — Standalone demonstration of the LoopHole lifting pipeline.

Run this directly with:
    python examples/run_demo.py

Or install the package and use the CLI:
    loophole demo
"""
from __future__ import annotations

import sys
import os
import time
import re

# ---------------------------------------------------------------------------
# Ensure we can import from src/ without installing (for running directly)
# ---------------------------------------------------------------------------
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_path = os.path.join(_repo_root, "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from loophole.lifter import lift, LiftResult
from loophole.tests.fixtures import DEMO_FIXTURES


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Useful for fast smoke tests without changing default demo behavior.
DEMO_Z3_TIMEOUT_MS = max(1, _env_int("LOOPHOLE_DEMO_Z3_TIMEOUT_MS", 8000))
DEMO_FIXTURE_LIMIT = max(0, _env_int("LOOPHOLE_DEMO_FIXTURE_LIMIT", 0))


def _iter_demo_fixtures() -> list[tuple[str, str]]:
    items = list(DEMO_FIXTURES.items())
    if DEMO_FIXTURE_LIMIT > 0:
        return items[:DEMO_FIXTURE_LIMIT]
    return items


def _extract_emitted_op(emitted_mlir: str) -> str:
    for line in emitted_mlir.splitlines():
        m = re.search(r"(linalg|stablehlo)\.\w+", line)
        if m:
            return m.group(0)
    return "--"


def _terminal_safe(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")

# ---------------------------------------------------------------------------
# Plain-text fallback helpers
# ---------------------------------------------------------------------------

def plain_header(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

def plain_result(name: str, result: LiftResult) -> None:
    status = "SUCCESS" if result.success else ("PARTIAL" if result.partial_success else "FAILED")
    print(f"\n[{status}] {name}")
    if result.matched_sketch:
        print(f"  Sketch  : {result.matched_sketch.name}")
        print(f"  Confidence: {result.sympy_confidence:.2%}")
    print(f"  Summary : {_terminal_safe(result.summary())}")
    if result.emitted_mlir:
        print(f"\n  -- Emitted MLIR --\n{_terminal_safe(result.emitted_mlir)}")


# ---------------------------------------------------------------------------
# Rich-based demo
# ---------------------------------------------------------------------------

def run_demo_fancy() -> None:
    console = Console()
    fixtures = _iter_demo_fixtures()

    console.print(Panel.fit(
        "[bold cyan]LoopHole[/bold cyan] - Loop Nest -> Tensor Dialect Lifter\n"
        "[dim]Automatically raises scalar Affine IR to Linalg/StableHLO[/dim]",
        border_style="cyan",
    ))

    results_table = Table(
        title="Lift Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    results_table.add_column("Fixture", style="cyan", no_wrap=True)
    results_table.add_column("Status", justify="center")
    results_table.add_column("Sketch Matched", style="yellow")
    results_table.add_column("Confidence", justify="right")
    results_table.add_column("Emitted Op", style="green")

    lift_results: list[tuple[str, LiftResult]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Lifting kernels...", total=len(fixtures))
        for name, mlir_text in fixtures:
            progress.update(task, description=f"Lifting [bold]{name}[/bold]...")
            t0 = time.time()
            result = lift(mlir_text, target="both", z3_timeout_ms=DEMO_Z3_TIMEOUT_MS)
            elapsed = time.time() - t0
            lift_results.append((name, result))

            if result.success:
                status = "[bold green]SUCCESS[/bold green]"
            elif result.partial_success:
                status = "[bold yellow]PARTIAL[/bold yellow]"
            else:
                status = "[bold red]FAILED[/bold red]"

            sketch_name = result.matched_sketch.name if result.matched_sketch else "--"
            conf_str = f"{result.sympy_confidence:.0%}" if result.matched_sketch else "--"

            emitted_op = _extract_emitted_op(result.emitted_mlir) if result.emitted_mlir else "--"

            results_table.add_row(name, status, sketch_name, conf_str, emitted_op)
            progress.advance(task)

    console.print(results_table)
    console.print()

    # ---------------------------------------------------------------------------
    # Show detailed output for the first successful lift
    # ---------------------------------------------------------------------------
    for name, result in lift_results:
        if (result.success or result.partial_success) and result.emitted_mlir:
            dialect_label = "MLIR"
            if "linalg." in result.emitted_mlir:
                dialect_label = "Linalg MLIR"
            elif "stablehlo." in result.emitted_mlir:
                dialect_label = "StableHLO MLIR"

            console.print(Panel.fit(
                Syntax(_terminal_safe(result.emitted_mlir), "mlir", theme="monokai", line_numbers=True),
                title=f"[bold green]{dialect_label} - {name}[/bold green]",
                border_style="green",
            ))
            break

    # Summary statistics
    n_success = sum(1 for _, r in lift_results if r.success)
    n_partial = sum(1 for _, r in lift_results if r.partial_success and not r.success)
    n_total = len(lift_results)
    console.print(Panel.fit(
        f"[bold]Total: {n_total}[/bold]  "
        f"[green]Success: {n_success}[/green]  "
        f"[yellow]Partial: {n_partial}[/yellow]  "
        f"[red]Failed: {n_total - n_success - n_partial}[/red]",
        title="Summary",
        border_style="dim",
    ))


def run_demo_plain() -> None:
    plain_header("LoopHole -- Loop Nest -> Tensor Dialect Lifter")
    print("Lifting demo fixtures...\n")
    for name, mlir_text in _iter_demo_fixtures():
        result = lift(mlir_text, target="both", z3_timeout_ms=DEMO_Z3_TIMEOUT_MS)
        plain_result(name, result)

    print("\n" + "=" * 70)
    print("Demo complete.")


# ---------------------------------------------------------------------------
# Per-kernel verbose example
# ---------------------------------------------------------------------------

def run_matmul_verbose() -> None:
    """Walk through the matmul lift step by step."""
    from loophole.tests.fixtures import MATMUL_MLIR
    from loophole.affine_extractor import AffineExtractor
    from loophole.sympy_tracer import SympyTracer
    from loophole.sketch_library import SKETCH_BY_NAME  # noqa: F401 kept for reference

    if HAS_RICH:
        console = Console()
        console.rule("[bold cyan]Step-by-step: Matmul Lift[/bold cyan]")
        console.print(Panel.fit(
            Syntax(_terminal_safe(MATMUL_MLIR), "mlir", theme="monokai"),
            title="Input MLIR",
        ))
    else:
        plain_header("Step-by-step: Matmul Lift")
        print(MATMUL_MLIR)

    extractor = AffineExtractor()
    info = extractor.extract(MATMUL_MLIR)

    if HAS_RICH:
        console = Console()
        console.print(f"\n[bold]Extracted:[/bold]")
        console.print(f"  Loops     : {info.loop_order}")
        console.print(f"  Bounds    : {[info.bounds[iv][1] for iv in info.loop_order]}")
        console.print(f"  Parallel  : {info.parallel_vars}")
        console.print(f"  Reduction : {info.reduction_vars}")
        console.print(f"  Compute   : {[op.op_type for op in info.compute_ops]}")
    else:
        print(f"\nExtracted:")
        print(f"  Loops     : {info.loop_order}")
        print(f"  Bounds    : {[info.bounds[iv][1] for iv in info.loop_order]}")
        print(f"  Parallel  : {info.parallel_vars}")
        print(f"  Reduction : {info.reduction_vars}")

    tracer = SympyTracer()
    trace = tracer.trace(info)
    best = tracer.infer_sketch(trace, info)

    if HAS_RICH:
        console = Console()
        console.print(f"\n[bold]SymPy trace confidence:[/bold] {trace.confidence:.2f}")
        if best:
            console.print(f"[bold]Best sketch match:[/bold] [yellow]{best.name}[/yellow]")
    else:
        print(f"\nSymPy trace confidence: {trace.confidence:.2f}")
        if best:
            print(f"Best sketch match: {best.name}")

    result = lift(MATMUL_MLIR, target="linalg", z3_timeout_ms=DEMO_Z3_TIMEOUT_MS)
    if (result.success or result.partial_success) and result.emitted_mlir:
        if HAS_RICH:
            console = Console()
            console.print(Panel.fit(
                Syntax(_terminal_safe(result.emitted_mlir), "mlir", theme="monokai", line_numbers=True),
                title="[bold green]Emitted Linalg MLIR[/bold green]",
                border_style="green",
            ))
        else:
            print("\n-- Emitted Linalg MLIR --")
            print(result.emitted_mlir)
    else:
        print(f"\nLift result: {result.summary()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LoopHole POC demonstration")
    parser.add_argument(
        "--mode",
        choices=["demo", "matmul", "plain"],
        default="demo",
        help="Which demo to run (default: demo)",
    )
    args = parser.parse_args()

    if args.mode == "matmul":
        run_matmul_verbose()
    elif args.mode == "plain":
        run_demo_plain()
    else:
        if HAS_RICH:
            run_demo_fancy()
        else:
            run_demo_plain()
        # Always also run the matmul verbose example
        print()
        run_matmul_verbose()
