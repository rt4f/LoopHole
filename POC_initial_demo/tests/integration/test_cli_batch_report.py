"""
Integration tests for batch JSON reporting semantics.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

from loophole.cli import main
from loophole.lifter import LiftResult
from loophole.mlir_validator import MlirValidationResult
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
    assert payload["schema_version"] == "1.1"
    assert "1.0" in payload["compatible_schema_versions"]
    assert payload["summary"]["total_files"] == 3
    assert payload["summary"]["proved"] == 1
    assert payload["summary"]["unproved_timeout"] == 1
    assert payload["summary"]["refuted"] == 1
    assert payload["summary"]["accepted_loose"] == 2
    assert payload["summary"]["accepted_strict"] == 1
    assert payload["run_metadata"]["profile_requested"] == "default"
    assert payload["run_metadata"]["profile_resolved"] == "local-explore"
    assert len(payload["run_metadata"]["fixtures_fingerprint_sha256"]) == 64

    states = [row["state"] for row in payload["results"]]
    assert states == ["PROVED", "UNPROVED_TIMEOUT", "REFUTED"]
    assert all(len(row["source_sha256"]) == 64 for row in payload["results"])


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


def test_batch_report_includes_profile_and_strict_mode(monkeypatch, tmp_path) -> None:
    (tmp_path / "kernel.mlir").write_text("func.func @dummy() { return }", encoding="utf-8")

    proved = _make_lift_result(CheckResult.EQUIVALENT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return proved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    report_path = tmp_path / "profile_report.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "batch",
            str(tmp_path),
            "--profile",
            "ci-strict",
            "--report",
            "--report-path",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["profile"] == "ci-strict"
    assert payload["summary"]["strict_mode"] is True
    assert payload["run_metadata"]["profile_requested"] == "ci-strict"
    assert payload["run_metadata"]["profile_resolved"] == "ci-strict"


def test_batch_profile_alias_trusted_resolves_to_ci_strict(monkeypatch, tmp_path) -> None:
    (tmp_path / "kernel.mlir").write_text("func.func @dummy() { return }", encoding="utf-8")
    proved = _make_lift_result(CheckResult.EQUIVALENT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return proved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    report_path = tmp_path / "alias_report.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "batch",
            str(tmp_path),
            "--profile",
            "trusted",
            "--report",
            "--report-path",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["profile"] == "ci-strict"
    assert payload["summary"]["strict_mode"] is True
    assert payload["run_metadata"]["profile_requested"] == "trusted"
    assert payload["run_metadata"]["profile_resolved"] == "ci-strict"


def test_batch_strict_flag_overrides_profile_strictness(monkeypatch, tmp_path) -> None:
    (tmp_path / "kernel.mlir").write_text("func.func @dummy() { return }", encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeLifter:
        def lift(self, _src: str) -> LiftResult:
            return _make_lift_result(CheckResult.EQUIVALENT, emitted_mlir=True)

    def _fake_factory(_cls, **kwargs):
        captured.update(kwargs)
        return _FakeLifter()

    monkeypatch.setattr("loophole.cli.Lifter.from_policy_profile", classmethod(_fake_factory))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "batch",
            str(tmp_path),
            "--profile",
            "local-explore",
            "--strict",
        ],
    )

    assert result.exit_code == 0
    assert captured["profile_name"] == "local-explore"
    assert captured["strict_mode"] is True


def test_batch_selection_filters_and_limits_are_reported(monkeypatch, tmp_path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "one.mlir").write_text("func.func @dummy() { return }", encoding="utf-8")
    (tmp_path / "b" / "two.mlir").write_text("func.func @dummy() { return }", encoding="utf-8")
    (tmp_path / "b" / "skip.mlir").write_text("func.func @dummy() { return }", encoding="utf-8")

    proved = _make_lift_result(CheckResult.EQUIVALENT, emitted_mlir=False)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return proved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    report_path = tmp_path / "selection_report.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "batch",
            str(tmp_path),
            "--include-glob",
            "**/*.mlir",
            "--exclude-glob",
            "b/skip.mlir",
            "--max-files",
            "1",
            "--report",
            "--report-path",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["selection"]["matched_before_limit"] == 2
    assert payload["summary"]["total_files"] == 1
    assert payload["results"][0]["file_rel"] == "a/one.mlir"


def test_batch_emitted_outputs_preserve_relative_layout(monkeypatch, tmp_path) -> None:
    (tmp_path / "case_a").mkdir()
    (tmp_path / "case_b").mkdir()
    (tmp_path / "case_a" / "kernel.mlir").write_text("func.func @dummy() { return }", encoding="utf-8")
    (tmp_path / "case_b" / "kernel.mlir").write_text("func.func @dummy() { return }", encoding="utf-8")

    sequence = iter(
        [
            _make_lift_result(CheckResult.EQUIVALENT, emitted_mlir=True),
            _make_lift_result(CheckResult.EQUIVALENT, emitted_mlir=True),
        ]
    )

    def _fake_lift(_self, _src: str) -> LiftResult:
        return next(sequence)

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
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "case_a" / "kernel_lifted.mlir").exists()
    assert (output_dir / "case_b" / "kernel_lifted.mlir").exists()


def test_batch_no_write_emitted_keeps_output_clean(monkeypatch, tmp_path) -> None:
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
            "--no-write-emitted",
            "--report",
        ],
    )

    assert result.exit_code == 0
    assert list(output_dir.rglob("*_lifted.mlir")) == []


def test_trusted_batch_requires_validation_even_without_flag(monkeypatch, tmp_path) -> None:
    (tmp_path / "kernel.mlir").write_text("func.func @dummy() { return }", encoding="utf-8")
    proved = _make_lift_result(CheckResult.EQUIVALENT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return proved

    validations: list[str] = []

    def _fake_validate(mlir_text: str, verifier_cmd: str, timeout_sec: int = 15) -> MlirValidationResult:
        validations.append(verifier_cmd)
        return MlirValidationResult(ok=True, command=verifier_cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)
    monkeypatch.setattr("loophole.cli.find_mlir_verifier", lambda _cmd=None: "mock-mlir-opt")
    monkeypatch.setattr("loophole.cli.validate_mlir_artifact", _fake_validate)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "batch",
            str(tmp_path),
            "--profile",
            "ci-strict",
            "--no-write-emitted",
        ],
    )

    assert result.exit_code == 0
    assert validations == ["mock-mlir-opt"]


def test_trusted_batch_fails_when_verifier_missing(monkeypatch, tmp_path) -> None:
    (tmp_path / "kernel.mlir").write_text("func.func @dummy() { return }", encoding="utf-8")
    proved = _make_lift_result(CheckResult.EQUIVALENT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return proved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)
    monkeypatch.setattr("loophole.cli.find_mlir_verifier", lambda _cmd=None: None)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "batch",
            str(tmp_path),
            "--profile",
            "trusted",
        ],
    )

    assert result.exit_code == 1
    assert "Trusted lane requires emitted MLIR validation" in result.output


def test_trusted_batch_fails_on_invalid_artifact(monkeypatch, tmp_path) -> None:
    (tmp_path / "kernel.mlir").write_text("func.func @dummy() { return }", encoding="utf-8")
    proved = _make_lift_result(CheckResult.EQUIVALENT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return proved

    def _fake_validate(_mlir_text: str, verifier_cmd: str, timeout_sec: int = 15) -> MlirValidationResult:
        return MlirValidationResult(
            ok=False,
            command=verifier_cmd,
            returncode=1,
            stdout="",
            stderr="invalid mlir",
        )

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)
    monkeypatch.setattr("loophole.cli.find_mlir_verifier", lambda _cmd=None: "mock-mlir-opt")
    monkeypatch.setattr("loophole.cli.validate_mlir_artifact", _fake_validate)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "batch",
            str(tmp_path),
            "--profile",
            "ci-strict",
        ],
    )

    assert result.exit_code == 1
    assert "Validation failed" in result.output
