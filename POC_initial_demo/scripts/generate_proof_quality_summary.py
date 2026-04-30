#!/usr/bin/env python3
"""
A-16: Standalone proof-quality summary artifact generator.

Runs the lifter over a fixture directory and emits a compact proof-quality
JSON + Markdown artifact suitable for inclusion in weekly status reports.

Usage
-----
    python scripts/generate_proof_quality_summary.py [options]

Options
-------
    --fixtures-dir PATH     Directory of .mlir fixture files (default: tests/fixtures)
    --target {linalg,stablehlo,both}
    --z3-timeout-ms INT     Z3 timeout per kernel in ms (default: 10000)
    --label LABEL           Short label for the artifact filename
    --output-dir PATH       Where to write the JSON + Markdown (default: reports/benchmarks)
    --baseline-report PATH  Prior JSON report for trend comparison (optional)

Output files
------------
    proof_quality_<label>_<timestamp>.json
    proof_quality_<label>_<timestamp>.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_weekly_benchmark_report import (  # noqa: E402
    _build_proof_quality_summary,
    _render_proof_quality_section,
    _discover_fixtures,
    _sha256_text,
    _compute_fixtures_fingerprint,
)
from loophole.lifter import Lifter, LiftResultState  # noqa: E402


def generate_proof_quality_artifact(
    fixtures_dir: Path,
    target: str,
    z3_timeout_ms: int,
    label: str,
    baseline_pq: dict | None,
) -> dict:
    mlir_files = _discover_fixtures(fixtures_dir)
    if not mlir_files:
        raise ValueError(f"No .mlir files found under {fixtures_dir}")

    lifter = Lifter(target=target, z3_timeout_ms=z3_timeout_ms, verbose=False)
    rows = []
    source_hashes: dict[str, str] = {}

    for mlir_file in mlir_files:
        src = mlir_file.read_text(encoding="utf-8")
        result = lifter.lift(src)
        rel_file = mlir_file.relative_to(fixtures_dir).as_posix()
        source_hashes[rel_file] = _sha256_text(src)
        rows.append(
            {
                "file_rel": rel_file,
                "file_name": mlir_file.name,
                "sketch": result.sketch_name,
                "state": result.result_state.value.upper(),
                "accepted_strict": result.is_accepted(strict_mode=True),
                "confidence": result.sympy_confidence,
                "elapsed_ms": result.total_elapsed_ms,
            }
        )

    pq = _build_proof_quality_summary(rows)

    trend: dict = {}
    if baseline_pq:
        prev_grade = baseline_pq.get("proof_quality_grade", "?")
        prev_rate = baseline_pq.get("strict_acceptance_rate", 0.0)
        delta_rate = pq["strict_acceptance_rate"] - prev_rate
        trend = {
            "available": True,
            "baseline_label": baseline_pq.get("label", "baseline"),
            "prev_grade": prev_grade,
            "prev_strict_acceptance_rate": prev_rate,
            "delta_strict_acceptance_rate": round(delta_rate, 4),
        }
    else:
        trend = {"available": False}

    return {
        "artifact_type": "proof_quality_summary",
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "target": target,
        "z3_timeout_ms": z3_timeout_ms,
        "fixtures_dir": str(fixtures_dir),
        "fixtures_fingerprint_sha256": _compute_fixtures_fingerprint(source_hashes),
        "proof_quality_summary": pq,
        "trend": trend,
    }


def _render_markdown(artifact: dict) -> str:
    pq = artifact["proof_quality_summary"]
    trend = artifact.get("trend", {})
    lines = [
        "# Proof Quality Summary",
        "",
        f"- Generated: {artifact['generated_at_utc']}",
        f"- Label: {artifact['label']}",
        f"- Target: {artifact['target']}",
        "",
    ]
    lines.extend(_render_proof_quality_section(pq))

    if trend.get("available"):
        lines.extend([
            "## Trend vs Baseline",
            "",
            f"- Baseline label: {trend['baseline_label']}",
            f"- Previous grade: {trend['prev_grade']}",
            f"- Previous strict acceptance rate: {trend['prev_strict_acceptance_rate'] * 100:.1f}%",
            f"- Delta: {trend['delta_strict_acceptance_rate'] * 100:+.1f}%",
            "",
        ])

    return "\n".join(lines) + "\n"


def write_artifact(artifact: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = artifact["label"].replace(" ", "_") if artifact["label"] else "pq"
    stem = f"proof_quality_{safe_label}_{ts}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(artifact), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LoopHole proof-quality summary artifact")
    p.add_argument("--fixtures-dir", type=Path, default=REPO_ROOT / "tests" / "fixtures")
    p.add_argument("--target", choices=["linalg", "stablehlo", "both"], default="linalg")
    p.add_argument("--z3-timeout-ms", type=int, default=10_000)
    p.add_argument("--label", default="a16_pq")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports" / "benchmarks")
    p.add_argument("--baseline-report", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    fixtures_dir = args.fixtures_dir.resolve()
    if not fixtures_dir.exists():
        print(f"Fixture directory does not exist: {fixtures_dir}", file=sys.stderr)
        return 1

    baseline_pq: dict | None = None
    if args.baseline_report:
        bp = args.baseline_report.resolve()
        if not bp.exists():
            print(f"Baseline report does not exist: {bp}", file=sys.stderr)
            return 1
        raw = json.loads(bp.read_text(encoding="utf-8"))
        baseline_pq = raw.get("proof_quality_summary") or raw

    try:
        artifact = generate_proof_quality_artifact(
            fixtures_dir=fixtures_dir,
            target=args.target,
            z3_timeout_ms=args.z3_timeout_ms,
            label=args.label,
            baseline_pq=baseline_pq,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    paths = write_artifact(artifact, args.output_dir.resolve())
    print(f"JSON: {paths['json']}")
    print(f"Markdown: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
