"""
A-16: Tests for proof-quality summary artifact.

Validates:
  1. _build_proof_quality_summary returns correct counts from row data.
  2. proof_rate and strict_acceptance_rate are computed correctly.
  3. confidence_distribution buckets are correct.
  4. per_sketch_breakdown groups rows by sketch name.
  5. Grading thresholds (A/B/C/D/F) are enforced.
  6. corpus_refuted_count vs unexpected_refuted_count are distinguished.
  7. Empty rows produce safe zero-valued output (no division-by-zero).
  8. _build_proof_quality_summary is included in generate_report payload.
  9. Markdown render includes the proof quality section.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.generate_weekly_benchmark_report import (
    _build_proof_quality_summary,
    _render_proof_quality_section,
    generate_report,
)
from loophole.lifter import LiftResult
from loophole.z3_checker import CheckResult, VerificationReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(
    state: str,
    sketch: str | None,
    confidence: float,
    file_rel: str = "kernels/foo.mlir",
    accepted_strict: bool | None = None,
) -> dict:
    if accepted_strict is None:
        accepted_strict = state == "PROVED"
    return {
        "file_rel": file_rel,
        "file_name": file_rel.split("/")[-1],
        "sketch": sketch,
        "state": state,
        "accepted_loose": state in ("PROVED", "UNPROVED_TIMEOUT"),
        "accepted_strict": accepted_strict,
        "confidence": confidence,
        "elapsed_ms": 5.0,
    }


# ---------------------------------------------------------------------------
# 1. Basic counts
# ---------------------------------------------------------------------------

def test_basic_counts() -> None:
    rows = [
        _row("PROVED", "linalg.matmul", 1.0),
        _row("PROVED", "linalg.matmul", 1.0),
        _row("REFUTED", None, 0.0),
    ]
    pq = _build_proof_quality_summary(rows)
    assert pq["total_kernels"] == 3
    assert pq["proof_rate"] == pytest.approx(2 / 3, rel=1e-4)
    assert pq["strict_acceptance_rate"] == pytest.approx(2 / 3, rel=1e-4)


# ---------------------------------------------------------------------------
# 2. Confidence metrics
# ---------------------------------------------------------------------------

def test_avg_confidence_proved_only() -> None:
    rows = [
        _row("PROVED", "linalg.matmul", 1.0),
        _row("PROVED", "linalg.matvec", 0.9),
        _row("REFUTED", None, 0.0),
    ]
    pq = _build_proof_quality_summary(rows)
    assert pq["avg_confidence_proved"] == pytest.approx(0.95, rel=1e-4)
    # avg over ALL rows includes 0.0 from refuted
    assert pq["avg_confidence"] == pytest.approx((1.0 + 0.9 + 0.0) / 3, rel=1e-4)


def test_fully_confident_count() -> None:
    rows = [
        _row("PROVED", "linalg.matmul", 1.0),
        _row("PROVED", "linalg.matmul", 1.0),
        _row("PROVED", "linalg.matvec", 0.9),
        _row("REFUTED", None, 0.0),
    ]
    pq = _build_proof_quality_summary(rows)
    assert pq["fully_confident_count"] == 2


# ---------------------------------------------------------------------------
# 3. Confidence distribution
# ---------------------------------------------------------------------------

def test_confidence_distribution() -> None:
    rows = [
        _row("PROVED", "linalg.matmul", 1.0),
        _row("PROVED", "linalg.matmul", 0.97),
        _row("PROVED", "linalg.matvec", 0.8),
        _row("REFUTED", None, 0.0),
    ]
    pq = _build_proof_quality_summary(rows)
    dist = pq["confidence_distribution"]
    assert dist["perfect_1_0"] == 1
    assert dist["high_0_9_to_1_0"] == 1
    assert dist["below_0_9"] == 2  # 0.8 and 0.0


# ---------------------------------------------------------------------------
# 4. Per-sketch breakdown
# ---------------------------------------------------------------------------

def test_per_sketch_breakdown() -> None:
    rows = [
        _row("PROVED", "linalg.matmul", 1.0),
        _row("PROVED", "linalg.matmul", 1.0),
        _row("REFUTED", "linalg.matmul", 0.0, accepted_strict=False),
        _row("PROVED", "linalg.matvec", 1.0),
        _row("REFUTED", None, 0.0, accepted_strict=False),
    ]
    pq = _build_proof_quality_summary(rows)
    bd = pq["per_sketch_breakdown"]
    assert bd["linalg.matmul"]["proved"] == 2
    assert bd["linalg.matmul"]["refuted"] == 1
    assert bd["linalg.matmul"]["total"] == 3
    assert bd["linalg.matvec"]["proved"] == 1
    assert bd["_no_match"]["refuted"] == 1


# ---------------------------------------------------------------------------
# 5. Grading thresholds
# ---------------------------------------------------------------------------

def _rows_with_rate(proved: int, total: int, corpus_refuted: int = 0) -> list:
    result = []
    for i in range(proved):
        result.append(_row("PROVED", "linalg.matmul", 1.0, file_rel=f"kernels/p{i}.mlir"))
    for i in range(corpus_refuted):
        result.append(_row("REFUTED", None, 0.0, file_rel=f"corpus/r{i}.mlir", accepted_strict=False))
    remaining = total - proved - corpus_refuted
    for i in range(remaining):
        result.append(_row("REFUTED", None, 0.0, file_rel=f"kernels/bad{i}.mlir", accepted_strict=False))
    return result


def test_grade_a_requires_90pct_and_no_unexpected_refuted() -> None:
    rows = _rows_with_rate(proved=10, total=10, corpus_refuted=0)
    pq = _build_proof_quality_summary(rows)
    assert pq["proof_quality_grade"] == "A"


def test_grade_a_with_corpus_refuted_still_a() -> None:
    rows = _rows_with_rate(proved=10, total=11, corpus_refuted=1)
    pq = _build_proof_quality_summary(rows)
    # 10/11 >= 0.90 and unexpected == 0 -> A
    assert pq["proof_quality_grade"] == "A"


def test_grade_b() -> None:
    rows = _rows_with_rate(proved=8, total=10)
    pq = _build_proof_quality_summary(rows)
    # 80% >= 0.75 but unexpected_refuted > 0 -> B
    assert pq["proof_quality_grade"] == "B"


def test_grade_c() -> None:
    rows = _rows_with_rate(proved=6, total=10)
    pq = _build_proof_quality_summary(rows)
    assert pq["proof_quality_grade"] == "C"


def test_grade_d() -> None:
    rows = _rows_with_rate(proved=4, total=10)
    pq = _build_proof_quality_summary(rows)
    assert pq["proof_quality_grade"] == "D"


def test_grade_f() -> None:
    rows = _rows_with_rate(proved=3, total=10)
    pq = _build_proof_quality_summary(rows)
    assert pq["proof_quality_grade"] == "F"


# ---------------------------------------------------------------------------
# 6. Corpus vs unexpected refuted
# ---------------------------------------------------------------------------

def test_corpus_vs_unexpected_refuted() -> None:
    rows = [
        _row("REFUTED", None, 0.0, file_rel="corpus/a.mlir", accepted_strict=False),
        _row("REFUTED", None, 0.0, file_rel="corpus/b.mlir", accepted_strict=False),
        _row("REFUTED", None, 0.0, file_rel="kernels/bad.mlir", accepted_strict=False),
        _row("PROVED", "linalg.matmul", 1.0),
    ]
    pq = _build_proof_quality_summary(rows)
    assert pq["corpus_refuted_count"] == 2
    assert pq["unexpected_refuted_count"] == 1


# ---------------------------------------------------------------------------
# 7. Empty rows
# ---------------------------------------------------------------------------

def test_empty_rows_no_crash() -> None:
    pq = _build_proof_quality_summary([])
    assert pq["total_kernels"] == 0
    assert pq["proof_rate"] == 0.0
    assert pq["proof_quality_grade"] == "F"


# ---------------------------------------------------------------------------
# 8. generate_report includes proof_quality_summary
# ---------------------------------------------------------------------------

def _make_lift_result(check_result: CheckResult, confidence: float = 1.0) -> LiftResult:
    sketch = SimpleNamespace(name="linalg.matmul")
    verification = VerificationReport(
        result=check_result,
        sketch_name=sketch.name,
        elapsed_ms=1.0,
        notes="",
    )
    return LiftResult(
        func_name="dummy",
        matched_sketch=sketch,
        sketch_name=sketch.name,
        emitted_mlir="module {}",
        verification=verification,
        sympy_confidence=confidence,
        total_elapsed_ms=2.0,
        target_dialect="linalg",
        error=None,
    )


def test_generate_report_includes_proof_quality_summary(monkeypatch, tmp_path) -> None:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "a.mlir").write_text("func.func @a() { return }", encoding="utf-8")
    (fixtures_dir / "b.mlir").write_text("func.func @b() { return }", encoding="utf-8")

    proved = _make_lift_result(CheckResult.EQUIVALENT, confidence=1.0)
    refuted = _make_lift_result(CheckResult.NOT_EQUIVALENT, confidence=0.0)
    seq = iter([proved, refuted])

    monkeypatch.setattr("scripts.generate_weekly_benchmark_report.Lifter.lift", lambda _s, _src: next(seq))

    payload = generate_report(
        fixtures_dir=fixtures_dir,
        target="linalg",
        z3_timeout_ms=5000,
        label="test",
        baseline_payload=None,
        regression_threshold_fraction=0.05,
        docker_image=None,
    )

    assert "proof_quality_summary" in payload
    pq = payload["proof_quality_summary"]
    assert pq["total_kernels"] == 2
    assert pq["proof_rate"] == pytest.approx(0.5)
    assert "proof_quality_grade" in pq


# ---------------------------------------------------------------------------
# 9. Markdown render includes proof quality section
# ---------------------------------------------------------------------------

def test_markdown_render_includes_proof_quality(monkeypatch, tmp_path) -> None:
    from scripts.generate_weekly_benchmark_report import _render_markdown

    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "a.mlir").write_text("func.func @a() { return }", encoding="utf-8")

    proved = _make_lift_result(CheckResult.EQUIVALENT, confidence=1.0)
    monkeypatch.setattr("scripts.generate_weekly_benchmark_report.Lifter.lift", lambda _s, _src: proved)

    payload = generate_report(
        fixtures_dir=fixtures_dir,
        target="linalg",
        z3_timeout_ms=5000,
        label="test",
        baseline_payload=None,
        regression_threshold_fraction=0.05,
        docker_image=None,
    )
    md = _render_markdown(payload)
    assert "## Proof Quality Summary" in md
    assert "Grade:" in md
    assert "Proof rate:" in md
