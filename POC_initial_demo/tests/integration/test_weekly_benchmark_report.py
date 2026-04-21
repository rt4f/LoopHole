"""
Integration tests for weekly benchmark report generation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.generate_weekly_benchmark_report import generate_report, write_reports
from loophole.lifter import LiftResult
from loophole.z3_checker import CheckResult, VerificationReport


def _make_lift_result(
    check_result: CheckResult,
    *,
    emitted_mlir: bool,
    error: str | None = None,
) -> LiftResult:
    sketch = SimpleNamespace(name="linalg.dummy")
    verification = VerificationReport(
        result=check_result,
        sketch_name=sketch.name,
        elapsed_ms=1.0,
        notes="test",
    )
    return LiftResult(
        func_name="dummy",
        matched_sketch=sketch,
        sketch_name=sketch.name,
        emitted_mlir="module {}" if emitted_mlir else None,
        verification=verification,
        sympy_confidence=0.9,
        total_elapsed_ms=2.0,
        target_dialect="linalg",
        error=error,
    )


def test_generate_report_includes_schema_and_summary(monkeypatch, tmp_path) -> None:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "a.mlir").write_text("func.func @dummy_a() { return }", encoding="utf-8")
    (fixtures_dir / "b.mlir").write_text("func.func @dummy_b() { return }", encoding="utf-8")

    proved = _make_lift_result(CheckResult.EQUIVALENT, emitted_mlir=True)
    unproved = _make_lift_result(CheckResult.TIMEOUT, emitted_mlir=True)
    sequence = iter([proved, unproved])

    def _fake_lift(_self, _src: str) -> LiftResult:
        return next(sequence)

    monkeypatch.setattr("scripts.generate_weekly_benchmark_report.Lifter.lift", _fake_lift)

    payload = generate_report(
        fixtures_dir=fixtures_dir,
        target="linalg",
        z3_timeout_ms=10_000,
        label="sprint1_week1",
    )

    assert payload["schema_version"] == "1.0"
    assert payload["label"] == "sprint1_week1"
    assert payload["target"] == "linalg"
    assert payload["summary"]["total_kernels"] == 2
    assert payload["summary"]["proved"] == 1
    assert payload["summary"]["unproved_timeout"] == 1
    assert payload["summary"]["refuted"] == 0
    assert payload["summary"]["accepted_loose"] == 2
    assert payload["summary"]["accepted_strict"] == 1
    assert len(payload["results"]) == 2
    assert payload["results"][0]["state"] == "PROVED"
    assert payload["results"][1]["state"] == "UNPROVED_TIMEOUT"


def test_write_reports_writes_json_and_markdown(monkeypatch, tmp_path) -> None:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "kernel.mlir").write_text("func.func @dummy() { return }", encoding="utf-8")

    proved = _make_lift_result(CheckResult.EQUIVALENT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return proved

    monkeypatch.setattr("scripts.generate_weekly_benchmark_report.Lifter.lift", _fake_lift)

    payload = generate_report(
        fixtures_dir=fixtures_dir,
        target="stablehlo",
        z3_timeout_ms=10_000,
        label="stablehlo_week1",
    )

    paths = write_reports(payload, tmp_path / "out")

    assert paths["json"].exists()
    assert paths["markdown"].exists()

    persisted_payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert persisted_payload["schema_version"] == "1.0"
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "# Weekly Benchmark Report" in markdown
    assert "stablehlo_week1" in markdown


def test_generate_report_raises_for_empty_fixture_dir(tmp_path) -> None:
    fixtures_dir = tmp_path / "empty"
    fixtures_dir.mkdir()

    with pytest.raises(ValueError, match="No .mlir files found"):
        generate_report(
            fixtures_dir=fixtures_dir,
            target="linalg",
            z3_timeout_ms=10_000,
            label="empty",
        )
