"""
Integration tests for batch JSON reporting semantics.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

from loophole.cli import main
from loophole.lifter import LiftResult
from loophole.z3_checker import CheckResult, VerificationReport


def _make_lift_result(
    check_result: CheckResult,
    *,
    emitted_mlir: bool,
    error: str | None = None,
    failed_implication: str | None = None,
    disagreement: str | None = None,
    mismatch_summary: str | None = None,
    counterexample_bindings: dict[str, str] | None = None,
) -> LiftResult:
    sketch = SimpleNamespace(name="linalg.dummy")
    verification = VerificationReport(
        result=check_result,
        sketch_name=sketch.name,
        elapsed_ms=1.0,
        notes="test",
        failed_implication=failed_implication,
        sympy_z3_disagreement=disagreement,
        mismatch_summary=mismatch_summary,
        counterexample_bindings=counterexample_bindings or {},
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


def test_batch_report_writes_state_summary_and_results(monkeypatch, tmp_path) -> None:
    for idx in range(3):
        (tmp_path / f"kernel_{idx}.mlir").write_text("func.func @dummy() { return }", encoding="utf-8")

    proved = _make_lift_result(CheckResult.EQUIVALENT, emitted_mlir=True)
    unproved = _make_lift_result(CheckResult.TIMEOUT, emitted_mlir=True)
    refuted = _make_lift_result(CheckResult.NOT_EQUIVALENT, emitted_mlir=False, error="refuted")
    sequence = iter([proved, unproved, refuted])

    def _fake_lift(_self, _src: str) -> LiftResult:
        return next(sequence)

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    report_path = tmp_path / "batch_report.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "batch",
            str(tmp_path),
            "--report",
            "--report-path",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["summary"]["total_files"] == 3
    assert payload["summary"]["proved"] == 1
    assert payload["summary"]["unproved_timeout"] == 1
    assert payload["summary"]["refuted"] == 1
    assert payload["summary"]["accepted_loose"] == 2
    assert payload["summary"]["accepted_strict"] == 1

    states = [row["state"] for row in payload["results"]]
    assert states == ["PROVED", "UNPROVED_TIMEOUT", "REFUTED"]


def test_batch_report_defaults_to_output_dir(monkeypatch, tmp_path) -> None:
    (tmp_path / "kernel.mlir").write_text("func.func @dummy() { return }", encoding="utf-8")

    proved = _make_lift_result(CheckResult.EQUIVALENT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return proved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    output_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "batch",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--report",
        ],
    )

    assert result.exit_code == 0
    report_path = output_dir / "loophole_report.json"
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total_files"] == 1
    assert payload["summary"]["proved"] == 1


def test_batch_report_includes_verification_diagnostics(monkeypatch, tmp_path) -> None:
    (tmp_path / "kernel.mlir").write_text("func.func @dummy() { return }", encoding="utf-8")

    refuted = _make_lift_result(
        CheckResult.NOT_EQUIVALENT,
        emitted_mlir=False,
        error="refuted",
        failed_implication="source_to_sketch",
        disagreement="HIGH_CONFIDENCE_REFUTED",
        mismatch_summary="i=0, j=1",
        counterexample_bindings={"i": "0", "j": "1"},
    )

    def _fake_lift(_self, _src: str) -> LiftResult:
        return refuted

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    report_path = tmp_path / "diag_report.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "batch",
            str(tmp_path),
            "--report",
            "--report-path",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    row = payload["results"][0]
    assert row["failed_implication"] == "source_to_sketch"
    assert row["sympy_z3_disagreement"] == "HIGH_CONFIDENCE_REFUTED"
    assert row["mismatch_summary"] == "i=0, j=1"
    assert row["counterexample_bindings"] == {"i": "0", "j": "1"}
