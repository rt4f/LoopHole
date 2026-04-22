#!/usr/bin/env python3
"""
Generate a weekly benchmark report for LoopHole fixture kernels.

This script runs the lifter over a fixture directory and emits:
- A machine-readable JSON report.
- A Markdown summary for quick review in PRs/status updates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loophole.lifter import Lifter, LiftResultState  # noqa: E402


CANONICAL_STABLEHLO_SKETCHES = {
    "stablehlo.dot_general",
    "stablehlo.dot_general_matvec",
    "stablehlo.dot_general_vecdot",
    "stablehlo.transpose",
    "stablehlo.add",
    "stablehlo.subtract",
    "stablehlo.multiply",
    "stablehlo.reduce{add}",
    "stablehlo.reduce{add}_colsum",
    "stablehlo.reduce{max}",
    "stablehlo.convolution",
}


def _canonicalize_stablehlo_sketch_name(sketch_name: Optional[str]) -> Optional[str]:
    if not sketch_name:
        return sketch_name
    if sketch_name.startswith("stablehlo.convolution"):
        return "stablehlo.convolution"
    return sketch_name


def _discover_fixtures(fixtures_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in fixtures_dir.rglob("*.mlir")
        if path.is_file() and not path.name.endswith("_lifted.mlir")
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compute_fixtures_fingerprint(source_hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for rel_path in sorted(source_hashes):
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_hashes[rel_path].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_trend_summary(
    current_summary: Dict,
    baseline_payload: Optional[Dict],
    regression_threshold_fraction: float,
) -> Dict:
    if not baseline_payload:
        return {
            "available": False,
            "baseline_label": None,
            "baseline_generated_at_utc": None,
            "delta": {},
            "anomalies": [],
            "regression_threshold_fraction": regression_threshold_fraction,
        }

    baseline_summary = baseline_payload.get("summary", {})
    baseline_avg_elapsed = float(baseline_summary.get("avg_elapsed_ms", 0.0))
    current_avg_elapsed = float(current_summary.get("avg_elapsed_ms", 0.0))
    latency_delta_fraction = 0.0
    if baseline_avg_elapsed > 0:
        latency_delta_fraction = (current_avg_elapsed - baseline_avg_elapsed) / baseline_avg_elapsed

    delta = {
        "proved": int(current_summary.get("proved", 0)) - int(baseline_summary.get("proved", 0)),
        "unproved_timeout": int(current_summary.get("unproved_timeout", 0)) - int(baseline_summary.get("unproved_timeout", 0)),
        "refuted": int(current_summary.get("refuted", 0)) - int(baseline_summary.get("refuted", 0)),
        "avg_elapsed_ms": current_avg_elapsed - baseline_avg_elapsed,
        "avg_elapsed_fraction": latency_delta_fraction,
    }

    anomalies: List[Dict[str, str]] = []
    if delta["proved"] < 0:
        anomalies.append(
            {
                "code": "proved_drop",
                "message": f"PROVED count dropped by {abs(delta['proved'])} compared to baseline.",
            }
        )
    if delta["refuted"] > 0:
        anomalies.append(
            {
                "code": "refuted_increase",
                "message": f"REFUTED count increased by {delta['refuted']} compared to baseline.",
            }
        )
    if latency_delta_fraction > regression_threshold_fraction:
        anomalies.append(
            {
                "code": "latency_regression",
                "message": (
                    "Average latency increased by "
                    f"{latency_delta_fraction * 100:.1f}% (threshold "
                    f"{regression_threshold_fraction * 100:.1f}%)."
                ),
            }
        )

    return {
        "available": True,
        "baseline_label": baseline_payload.get("label", "baseline"),
        "baseline_generated_at_utc": baseline_payload.get("generated_at_utc"),
        "delta": delta,
        "anomalies": anomalies,
        "regression_threshold_fraction": regression_threshold_fraction,
    }


def _build_canonical_stablehlo_summary(target: str, rows: List[Dict]) -> Dict:
    if target not in ("stablehlo", "both"):
        return {
            "applicable": False,
            "required_ops": [],
            "observed_required_ops": [],
            "missing_required_ops": [],
            "proved": 0,
            "unproved_timeout": 0,
            "refuted": 0,
            "unmatched_files": [],
        }

    matched_required = {
        _canonicalize_stablehlo_sketch_name(row.get("sketch"))
        for row in rows
        if _canonicalize_stablehlo_sketch_name(row.get("sketch")) in CANONICAL_STABLEHLO_SKETCHES
    }

    return {
        "applicable": True,
        "required_ops": sorted(CANONICAL_STABLEHLO_SKETCHES),
        "observed_required_ops": sorted(matched_required),
        "missing_required_ops": sorted(CANONICAL_STABLEHLO_SKETCHES - matched_required),
        "proved": sum(
            1
            for row in rows
            if _canonicalize_stablehlo_sketch_name(row.get("sketch")) in CANONICAL_STABLEHLO_SKETCHES
            and row.get("state") == "PROVED"
        ),
        "unproved_timeout": sum(
            1
            for row in rows
            if _canonicalize_stablehlo_sketch_name(row.get("sketch")) in CANONICAL_STABLEHLO_SKETCHES
            and row.get("state") == "UNPROVED_TIMEOUT"
        ),
        "refuted": sum(
            1
            for row in rows
            if _canonicalize_stablehlo_sketch_name(row.get("sketch")) in CANONICAL_STABLEHLO_SKETCHES
            and row.get("state") == "REFUTED"
        ),
        "unmatched_files": [
            row["file_rel"]
            for row in rows
            if row.get("sketch") is None and row.get("state") == "REFUTED"
        ],
    }


def _render_markdown(payload: Dict) -> str:
    summary = payload["summary"]
    trend = payload.get("trend_summary", {})
    canonical = payload.get("canonical_stablehlo_summary", {})

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
    ]

    if trend.get("available"):
        delta = trend.get("delta", {})
        lines.extend(
            [
                "## Trend Summary",
                "",
                f"- Baseline label: {trend.get('baseline_label')}",
                f"- Baseline generated: {trend.get('baseline_generated_at_utc')}",
                f"- Delta PROVED: {delta.get('proved', 0):+d}",
                f"- Delta UNPROVED_TIMEOUT: {delta.get('unproved_timeout', 0):+d}",
                f"- Delta REFUTED: {delta.get('refuted', 0):+d}",
                f"- Delta avg elapsed ms: {delta.get('avg_elapsed_ms', 0.0):+.2f}",
                "",
            ]
        )
        anomalies = trend.get("anomalies", [])
        if anomalies:
            lines.append("### Trend Alerts")
            lines.append("")
            for anomaly in anomalies:
                lines.append(f"- [{anomaly['code']}] {anomaly['message']}")
            lines.append("")

    if canonical.get("applicable"):
        lines.extend(
            [
                "## StableHLO Canonical Suite",
                "",
                f"- Proved: {canonical['proved']}",
                f"- Unproved timeout: {canonical['unproved_timeout']}",
                f"- Refuted: {canonical['refuted']}",
                f"- Missing required ops: {', '.join(canonical['missing_required_ops']) or '-'}",
                "",
            ]
        )

        if canonical["unmatched_files"]:
            lines.append("### Unmatched Canonical Candidates")
            lines.append("")
            for rel_path in canonical["unmatched_files"]:
                lines.append(f"- {rel_path}")
            lines.append("")

    lines.extend(
        [
            "## Per-Kernel Results",
            "",
            "| Kernel | Sketch | State | Z3 | Conf | ms | Diag |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )

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


def generate_report(
    fixtures_dir: Path,
    target: str,
    z3_timeout_ms: int,
    label: str,
    baseline_payload: Optional[Dict],
    regression_threshold_fraction: float,
    docker_image: Optional[str],
) -> Dict:
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
    source_hashes: Dict[str, str] = {}

    for mlir_file in mlir_files:
        src = mlir_file.read_text(encoding="utf-8")
        result = lifter.lift(src)
        rel_file = mlir_file.relative_to(fixtures_dir).as_posix()
        source_hash = _sha256_text(src)
        source_hashes[rel_file] = source_hash

        state = result.result_state
        state_counts[state] += 1

        z3_result = result.verification.result.value if result.verification else "-"
        failed_implication = result.verification.failed_implication if result.verification else None
        disagreement = result.verification.sympy_z3_disagreement if result.verification else None
        mismatch_summary = result.verification.mismatch_summary if result.verification else None
        rows.append(
            {
                "file": str(mlir_file),
                "file_rel": rel_file,
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
                "source_sha256": source_hash,
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

    trend_summary = _build_trend_summary(
        current_summary=summary,
        baseline_payload=baseline_payload,
        regression_threshold_fraction=regression_threshold_fraction,
    )
    canonical_summary = _build_canonical_stablehlo_summary(target=target, rows=rows)
    fixtures_fingerprint = _compute_fixtures_fingerprint(source_hashes)

    return {
        "schema_version": "1.1",
        "compatible_schema_versions": ["1.0"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "target": target,
        "z3_timeout_ms": z3_timeout_ms,
        "fixtures_dir": str(fixtures_dir),
        "run_metadata": {
            "run_id": str(uuid.uuid4()),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "fixtures_fingerprint_sha256": fixtures_fingerprint,
            "fixture_count": len(source_hashes),
            "docker_image": docker_image,
            "loophole_policy_profile_env": os.environ.get("LOOPHOLE_POLICY_PROFILE"),
        },
        "summary": summary,
        "trend_summary": trend_summary,
        "canonical_stablehlo_summary": canonical_summary,
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
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=None,
        help="Optional prior JSON report for trend comparison",
    )
    parser.add_argument(
        "--regression-threshold-fraction",
        type=float,
        default=0.05,
        help="Fractional threshold for latency regression alerts (default: 0.05)",
    )
    parser.add_argument(
        "--docker-image",
        default=None,
        help="Optional docker image tag to record in report metadata",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    fixtures_dir = args.fixtures_dir.resolve()
    if not fixtures_dir.exists():
        print(f"Fixture directory does not exist: {fixtures_dir}", file=sys.stderr)
        return 1

    baseline_payload: Optional[Dict] = None
    if args.baseline_report:
        baseline_path = args.baseline_report.resolve()
        if not baseline_path.exists():
            print(f"Baseline report does not exist: {baseline_path}", file=sys.stderr)
            return 1
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))

    try:
        payload = generate_report(
            fixtures_dir=fixtures_dir,
            target=args.target,
            z3_timeout_ms=args.z3_timeout_ms,
            label=args.label,
            baseline_payload=baseline_payload,
            regression_threshold_fraction=args.regression_threshold_fraction,
            docker_image=args.docker_image,
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
