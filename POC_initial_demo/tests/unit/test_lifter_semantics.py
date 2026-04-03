"""
Unit tests for LiftResult trust semantics and strict-mode behavior.
"""

from types import SimpleNamespace

from loophole.lifter import Lifter, LiftResult, LiftResultState
from loophole.z3_checker import CheckResult, VerificationReport


def _make_result(
    check_result: CheckResult,
    *,
    emitted_mlir: bool,
    error: str | None = None,
) -> LiftResult:
    sketch = SimpleNamespace(name="linalg.dummy")
    verification = VerificationReport(
        result=check_result,
        sketch_name="linalg.dummy",
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


def test_result_state_mapping_is_explicit() -> None:
    proved = _make_result(CheckResult.EQUIVALENT, emitted_mlir=True)
    timed_out = _make_result(CheckResult.TIMEOUT, emitted_mlir=True)
    unknown = _make_result(CheckResult.UNKNOWN, emitted_mlir=True)
    refuted = _make_result(CheckResult.NOT_EQUIVALENT, emitted_mlir=False, error="refuted")

    assert proved.result_state == LiftResultState.PROVED
    assert timed_out.result_state == LiftResultState.UNPROVED_TIMEOUT
    assert unknown.result_state == LiftResultState.UNPROVED_TIMEOUT
    assert refuted.result_state == LiftResultState.REFUTED


def test_partial_success_excludes_refuted() -> None:
    timed_out = _make_result(CheckResult.TIMEOUT, emitted_mlir=True)
    refuted = _make_result(CheckResult.NOT_EQUIVALENT, emitted_mlir=True)

    assert timed_out.partial_success
    assert not refuted.partial_success


def test_strict_acceptance_rejects_unproved_and_refuted() -> None:
    proved = _make_result(CheckResult.EQUIVALENT, emitted_mlir=True)
    timed_out = _make_result(CheckResult.TIMEOUT, emitted_mlir=True)
    refuted = _make_result(CheckResult.NOT_EQUIVALENT, emitted_mlir=False, error="refuted")

    assert proved.is_accepted(strict_mode=True)
    assert not timed_out.is_accepted(strict_mode=True)
    assert not refuted.is_accepted(strict_mode=True)


def test_lifter_strict_mode_rejects_timeout_without_emission() -> None:
    lifter = Lifter(target="linalg", strict_mode=True)

    loop = SimpleNamespace(
        func_name="dummy",
        induction_vars=[],
        reduction_vars=[],
        reads=[],
        writes=[],
        diagnostics=[],
    )
    sketch = SimpleNamespace(name="linalg.dummy", dialect="linalg")
    report = VerificationReport(
        result=CheckResult.TIMEOUT,
        sketch_name=sketch.name,
        elapsed_ms=1.0,
        notes="timeout",
    )

    lifter._extractor.extract = lambda _text: loop
    lifter._match_candidates = lambda _loop: [(sketch, 0.9)]
    lifter._verify_candidates = lambda _loop, _candidates: (sketch, report, 0.9)
    lifter._emit = lambda _sketch, _loop: (_ for _ in ()).throw(AssertionError("emit must not run"))

    result = lifter.lift("dummy")

    assert result.verification is not None
    assert result.verification.result == CheckResult.TIMEOUT
    assert result.result_state == LiftResultState.UNPROVED_TIMEOUT
    assert result.emitted_mlir is None
    assert not result.partial_success


def test_lifter_non_strict_timeout_allows_partial_emission() -> None:
    lifter = Lifter(target="linalg", strict_mode=False)

    loop = SimpleNamespace(
        func_name="dummy",
        induction_vars=[],
        reduction_vars=[],
        reads=[],
        writes=[],
        diagnostics=[],
    )
    sketch = SimpleNamespace(name="linalg.dummy", dialect="linalg")
    report = VerificationReport(
        result=CheckResult.TIMEOUT,
        sketch_name=sketch.name,
        elapsed_ms=1.0,
        notes="timeout",
    )

    lifter._extractor.extract = lambda _text: loop
    lifter._match_candidates = lambda _loop: [(sketch, 0.9)]
    lifter._verify_candidates = lambda _loop, _candidates: (sketch, report, 0.9)
    lifter._emit = lambda _sketch, _loop: "module {}"

    result = lifter.lift("dummy")

    assert result.result_state == LiftResultState.UNPROVED_TIMEOUT
    assert result.emitted_mlir is not None
    assert result.partial_success
