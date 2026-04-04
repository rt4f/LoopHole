#!/usr/bin/env python3
"""
Generate a weekly benchmark report for LoopHole fixture kernels.

This script runs the lifter over a fixture directory and emits:
- A machine-readable JSON report.
- A Markdown summary for quick review in PRs/status updates.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loophole.lifter import Lifter, LiftResultState  # noqa: E402


def _discover_fixtures(fixtures_dir: Path) -> List[Path]:
    return sorted(fixtures_dir.rglob("*.mlir"))


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _render_markdown(payload: Dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Weekly Benchmark Report",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        f"- Label: {payload['label']}",
        f"- Target: {payload['target']}",
        f"- Fixture directory: {payload['fixtures_dir']}",
        f"- Total kernels: {summary['total_kernels']}",
        "",
        "## Summary",
        "",
        f"- PROVED: {summary['proved']}",
        f"- UNPROVED_TIMEOUT: {summary['unproved_timeout']}",
        f"- REFUTED: {summary['refuted']}",
        f"- Accepted (loose): {summary['accepted_loose']}",
        f"- Accepted (strict): {summary['accepted_strict']}",
        f"- Average elapsed ms: {summary['avg_elapsed_ms']:.2f}",
        "",
        "## Per-Kernel Results",
        "",
        "| Kernel | Sketch | State | Z3 | Conf | ms | Diag |",
        "|---|---|---|---|---:|---:|---|",
    ]

    for row in payload["results"]:
        diag = row.get("sympy_z3_disagreement") or row.get("failed_implication") or "-"
        lines.append(
            "| {kernel} | {sketch} | {state} | {z3} | {conf:.2f} | {ms:.1f} | {diag} |".format(
                kernel=row["file_name"],
                sketch=row["sketch"] or "-",
                state=row["state"],
                z3=row["z3"],
                conf=row["confidence"],
                ms=row["elapsed_ms"],
                diag=diag,
            )
        )

    return "\n".join(lines) + "\n"


def generate_report(fixtures_dir: Path, target: str, z3_timeout_ms: int, label: str) -> Dict:
    mlir_files = _discover_fixtures(fixtures_dir)
    if not mlir_files:
        raise ValueError(f"No .mlir files found under {fixtures_dir}")

    lifter = Lifter(target=target, z3_timeout_ms=z3_timeout_ms, verbose=False)

    state_counts = {
        LiftResultState.PROVED: 0,
        LiftResultState.UNPROVED_TIMEOUT: 0,
        LiftResultState.REFUTED: 0,
    }
    rows = []

    for mlir_file in mlir_files:
        src = mlir_file.read_text(encoding="utf-8")
        result = lifter.lift(src)

        state = result.result_state
        state_counts[state] += 1

        z3_result = result.verification.result.value if result.verification else "-"
        failed_implication = result.verification.failed_implication if result.verification else None
        disagreement = result.verification.sympy_z3_disagreement if result.verification else None
        mismatch_summary = result.verification.mismatch_summary if result.verification else None
        rows.append(
            {
                "file": str(mlir_file),
                "file_name": mlir_file.name,
                "sketch": result.sketch_name,
                "state": state.value.upper(),
                "accepted_loose": result.is_accepted(strict_mode=False),
                "accepted_strict": result.is_accepted(strict_mode=True),
                "z3": z3_result,
                "confidence": result.sympy_confidence,
                "elapsed_ms": result.total_elapsed_ms,
                "failed_implication": failed_implication,
                "sympy_z3_disagreement": disagreement,
                "mismatch_summary": mismatch_summary,
                "counterexample_bindings": result.verification.counterexample_bindings if result.verification else {},
                "error": result.error,
            }
        )

    summary = {
        "total_kernels": len(rows),
        "proved": state_counts[LiftResultState.PROVED],
        "unproved_timeout": state_counts[LiftResultState.UNPROVED_TIMEOUT],
        "refuted": state_counts[LiftResultState.REFUTED],
        "accepted_loose": sum(1 for r in rows if r["accepted_loose"]),
        "accepted_strict": sum(1 for r in rows if r["accepted_strict"]),
        "avg_elapsed_ms": _safe_mean([r["elapsed_ms"] for r in rows]),
    }

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "target": target,
        "z3_timeout_ms": z3_timeout_ms,
        "fixtures_dir": str(fixtures_dir),
        "summary": summary,
        "results": rows,
    }


def write_reports(payload: Dict, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = payload["label"].replace(" ", "_") if payload["label"] else "weekly"
    stem = f"weekly_benchmark_{safe_label}_{timestamp}"

    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")

    return {"json": json_path, "markdown": md_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weekly LoopHole benchmark report")
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures",
        help="Directory containing .mlir fixture files",
    )
    parser.add_argument(
        "--target",
        choices=["linalg", "stablehlo", "both"],
        default="linalg",
        help="Target dialect for lifting",
    )
    parser.add_argument(
        "--z3-timeout-ms",
        type=int,
        default=10_000,
        help="Z3 timeout per check in milliseconds",
    )
    parser.add_argument(
        "--label",
        default="weekly",
        help="Short report label used in output filenames",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "benchmarks",
        help="Directory where report files are written",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    fixtures_dir = args.fixtures_dir.resolve()
    if not fixtures_dir.exists():
        print(f"Fixture directory does not exist: {fixtures_dir}", file=sys.stderr)
        return 1

    try:
        payload = generate_report(
            fixtures_dir=fixtures_dir,
            target=args.target,
            z3_timeout_ms=args.z3_timeout_ms,
            label=args.label,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    paths = write_reports(payload, args.output_dir.resolve())
    print(f"JSON report: {paths['json']}")
    print(f"Markdown report: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
