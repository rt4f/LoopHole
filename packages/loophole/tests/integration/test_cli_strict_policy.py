"""
Integration tests for strict CLI verification semantics.
"""

from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from loophole.cli import main
from loophole.lifter import LiftResult
from loophole.z3_checker import CheckResult, VerificationReport


def _set_single_demo_fixture(monkeypatch) -> None:
    monkeypatch.setattr(
        "loophole.tests.fixtures.DEMO_FIXTURES",
        {"demo_kernel": "func.func @demo_kernel() { return }"},
    )


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


def test_lift_rejects_strict_with_no_verify(tmp_path) -> None:
    input_file = tmp_path / "input.mlir"
    input_file.write_text("func.func @dummy() { return }", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, ["lift", str(input_file), "--strict", "--no-verify"])

    assert result.exit_code != 0
    assert "--strict cannot be combined with --no-verify" in result.output


def test_lift_rejects_no_verify_with_ci_strict_profile(tmp_path) -> None:
    input_file = tmp_path / "input.mlir"
    input_file.write_text("func.func @dummy() { return }", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, ["lift", str(input_file), "--profile", "ci-strict", "--no-verify"])

    assert result.exit_code != 0
    assert "--strict cannot be combined with --no-verify" in result.output


def test_lift_c_rejects_strict_with_no_verify(tmp_path) -> None:
    source_file = tmp_path / "kernel.c"
    source_file.write_text("void kernel(void) {}", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, ["lift-c", str(source_file), "--strict", "--no-verify"])

    assert result.exit_code != 0
    assert "--strict cannot be combined with --no-verify" in result.output


def test_lift_c_rejects_no_verify_with_ci_strict_profile(tmp_path) -> None:
    source_file = tmp_path / "kernel.c"
    source_file.write_text("void kernel(void) {}", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, ["lift-c", str(source_file), "--profile", "ci-strict", "--no-verify"])

    assert result.exit_code != 0
    assert "--strict cannot be combined with --no-verify" in result.output


def test_strict_exit_code_is_zero_for_proved(monkeypatch, tmp_path) -> None:
    input_file = tmp_path / "input.mlir"
    input_file.write_text("func.func @dummy() { return }", encoding="utf-8")
    proved = _make_lift_result(CheckResult.EQUIVALENT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return proved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    runner = CliRunner()
    result = runner.invoke(main, ["lift", str(input_file), "--strict"])

    assert result.exit_code == 0


def test_strict_exit_code_is_two_for_unproved_timeout(monkeypatch, tmp_path) -> None:
    input_file = tmp_path / "input.mlir"
    input_file.write_text("func.func @dummy() { return }", encoding="utf-8")
    unproved = _make_lift_result(CheckResult.TIMEOUT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return unproved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    runner = CliRunner()
    result = runner.invoke(main, ["lift", str(input_file), "--strict"])

    assert result.exit_code == 2


def test_strict_exit_code_is_one_for_refuted(monkeypatch, tmp_path) -> None:
    input_file = tmp_path / "input.mlir"
    input_file.write_text("func.func @dummy() { return }", encoding="utf-8")
    refuted = _make_lift_result(
        CheckResult.NOT_EQUIVALENT,
        emitted_mlir=False,
        error="refuted",
    )

    def _fake_lift(_self, _src: str) -> LiftResult:
        return refuted

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    runner = CliRunner()
    result = runner.invoke(main, ["lift", str(input_file), "--strict"])

    assert result.exit_code == 1


def test_ci_profile_timeout_returns_strict_exit_code_two(monkeypatch, tmp_path) -> None:
    input_file = tmp_path / "input.mlir"
    input_file.write_text("func.func @dummy() { return }", encoding="utf-8")
    unproved = _make_lift_result(CheckResult.TIMEOUT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return unproved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    runner = CliRunner()
    result = runner.invoke(main, ["lift", str(input_file), "--profile", "ci-strict"])

    assert result.exit_code == 2


def test_local_explore_profile_timeout_returns_exit_code_zero(monkeypatch, tmp_path) -> None:
    input_file = tmp_path / "input.mlir"
    input_file.write_text("func.func @dummy() { return }", encoding="utf-8")
    unproved = _make_lift_result(CheckResult.TIMEOUT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return unproved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    runner = CliRunner()
    result = runner.invoke(main, ["lift", str(input_file), "--profile", "local-explore"])

    assert result.exit_code == 0


def test_trusted_alias_timeout_returns_strict_exit_code_two(monkeypatch, tmp_path) -> None:
    input_file = tmp_path / "input.mlir"
    input_file.write_text("func.func @dummy() { return }", encoding="utf-8")
    unproved = _make_lift_result(CheckResult.TIMEOUT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return unproved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    runner = CliRunner()
    result = runner.invoke(main, ["lift", str(input_file), "--profile", "trusted"])

    assert result.exit_code == 2


def test_exploratory_alias_timeout_returns_exit_code_zero(monkeypatch, tmp_path) -> None:
    input_file = tmp_path / "input.mlir"
    input_file.write_text("func.func @dummy() { return }", encoding="utf-8")
    unproved = _make_lift_result(CheckResult.TIMEOUT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return unproved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    runner = CliRunner()
    result = runner.invoke(main, ["lift", str(input_file), "--profile", "exploratory"])

    assert result.exit_code == 0


def test_env_profile_applies_when_cli_profile_is_default(monkeypatch, tmp_path) -> None:
    input_file = tmp_path / "input.mlir"
    input_file.write_text("func.func @dummy() { return }", encoding="utf-8")
    unproved = _make_lift_result(CheckResult.TIMEOUT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return unproved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)
    monkeypatch.setenv("LOOPHOLE_POLICY_PROFILE", "ci-strict")

    runner = CliRunner()
    result = runner.invoke(main, ["lift", str(input_file)])

    assert result.exit_code == 2


def test_demo_strict_exit_code_is_zero_for_proved(monkeypatch) -> None:
    _set_single_demo_fixture(monkeypatch)
    proved = _make_lift_result(CheckResult.EQUIVALENT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return proved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    runner = CliRunner()
    result = runner.invoke(main, ["demo", "--strict"])

    assert result.exit_code == 0


def test_demo_strict_exit_code_is_one_for_unproved_timeout(monkeypatch) -> None:
    _set_single_demo_fixture(monkeypatch)
    unproved = _make_lift_result(CheckResult.TIMEOUT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return unproved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    runner = CliRunner()
    result = runner.invoke(main, ["demo", "--strict"])

    assert result.exit_code == 1


def test_demo_ci_strict_profile_rejects_unproved_timeout(monkeypatch) -> None:
    _set_single_demo_fixture(monkeypatch)
    unproved = _make_lift_result(CheckResult.TIMEOUT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return unproved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    runner = CliRunner()
    result = runner.invoke(main, ["demo", "--profile", "ci-strict"])

    assert result.exit_code == 1


def test_demo_local_explore_profile_allows_unproved_timeout(monkeypatch) -> None:
    _set_single_demo_fixture(monkeypatch)
    unproved = _make_lift_result(CheckResult.TIMEOUT, emitted_mlir=True)

    def _fake_lift(_self, _src: str) -> LiftResult:
        return unproved

    monkeypatch.setattr("loophole.cli.Lifter.lift", _fake_lift)

    runner = CliRunner()
    result = runner.invoke(main, ["demo", "--profile", "local-explore"])

    assert result.exit_code == 0
