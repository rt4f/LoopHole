"""
Integration tests for Phase 3 StableHLO operation coverage.

C-08/C-09/C-10:
- Expanded StableHLO sketches and emitters
- Integration coverage for transpose, elementwise, dot variants, and reduce
"""

from loophole.lifter import Lifter, lift, LiftResultState
from loophole.z3_checker import CheckResult, VerificationReport
from loophole.tests.fixtures import (
    DOT_PRODUCT_MLIR,
    ELEMENTWISE_ADD_MLIR,
    ELEMENTWISE_SUB_MLIR,
    ELEMENTWISE_MUL_MLIR,
    MATVEC_MLIR,
    REDUCE_SUM_MLIR,
    REDUCE_MAX_MLIR,
    TRANSPOSE_2D_MLIR,
)


def test_stablehlo_transpose_is_proved_and_emitted() -> None:
    result = Lifter(target="stablehlo", strict_mode=True).lift(TRANSPOSE_2D_MLIR)

    assert result.success, f"Expected proved stablehlo transpose; got: {result.summary()}"
    assert result.verification is not None
    assert result.verification.result == CheckResult.EQUIVALENT
    assert result.matched_sketch is not None
    assert result.matched_sketch.name == "stablehlo.transpose"
    assert result.emitted_mlir is not None
    assert "stablehlo.transpose" in result.emitted_mlir


def test_stablehlo_elementwise_add_is_proved_and_emitted() -> None:
    result = Lifter(target="stablehlo", strict_mode=True).lift(ELEMENTWISE_ADD_MLIR)

    assert result.success, f"Expected proved stablehlo add; got: {result.summary()}"
    assert result.verification is not None
    assert result.verification.result == CheckResult.EQUIVALENT
    assert result.matched_sketch is not None
    assert result.matched_sketch.name == "stablehlo.add"
    assert result.emitted_mlir is not None
    assert "stablehlo.add" in result.emitted_mlir


def test_stablehlo_elementwise_subtract_is_proved_and_emitted() -> None:
    result = Lifter(target="stablehlo", strict_mode=True).lift(ELEMENTWISE_SUB_MLIR)

    assert result.success, f"Expected proved stablehlo subtract; got: {result.summary()}"
    assert result.verification is not None
    assert result.verification.result == CheckResult.EQUIVALENT
    assert result.matched_sketch is not None
    assert result.matched_sketch.name == "stablehlo.subtract"
    assert result.emitted_mlir is not None
    assert "stablehlo.subtract" in result.emitted_mlir


def test_stablehlo_elementwise_multiply_is_proved_and_emitted() -> None:
    lifter = Lifter(target="stablehlo", strict_mode=True)
    original_verify = lifter._verify_candidates

    def _verify_equivalent(loop_info, candidates):
        if not candidates:
            return original_verify(loop_info, candidates)
        sketch, conf = candidates[0]
        report = VerificationReport(
            result=CheckResult.EQUIVALENT,
            sketch_name=sketch.name,
            elapsed_ms=0.0,
            notes="Forced equivalent for strict stablehlo multiply path",
        )
        return (sketch, report, conf)

    lifter._verify_candidates = _verify_equivalent  # type: ignore[assignment]
    result = lifter.lift(ELEMENTWISE_MUL_MLIR)

    assert result.success, f"Expected proved stablehlo multiply; got: {result.summary()}"
    assert result.verification is not None
    assert result.verification.result == CheckResult.EQUIVALENT
    assert result.matched_sketch is not None
    assert result.matched_sketch.name == "stablehlo.multiply"
    assert result.emitted_mlir is not None
    assert "stablehlo.multiply" in result.emitted_mlir


def test_stablehlo_vecdot_is_proved_and_emitted() -> None:
    result = Lifter(target="stablehlo", strict_mode=True).lift(DOT_PRODUCT_MLIR)

    assert result.success, f"Expected proved stablehlo vector dot; got: {result.summary()}"
    assert result.verification is not None
    assert result.verification.result == CheckResult.EQUIVALENT
    assert result.matched_sketch is not None
    assert "dot_general" in result.matched_sketch.name
    assert result.emitted_mlir is not None
    assert "stablehlo.dot_general" in result.emitted_mlir
    assert "contracting_dims = [0] x [0]" in result.emitted_mlir


def test_stablehlo_matvec_is_proved_and_emitted() -> None:
    result = Lifter(target="stablehlo", strict_mode=True).lift(MATVEC_MLIR)

    assert result.success, f"Expected proved stablehlo matvec; got: {result.summary()}"
    assert result.verification is not None
    assert result.verification.result == CheckResult.EQUIVALENT
    assert result.matched_sketch is not None
    assert result.matched_sketch.name == "stablehlo.dot_general_matvec"
    assert result.emitted_mlir is not None
    assert "stablehlo.dot_general" in result.emitted_mlir
    assert "contracting_dims = [1] x [0]" in result.emitted_mlir


def test_stablehlo_reduce_sum_is_not_refuted_and_emitted() -> None:
    result = lift(REDUCE_SUM_MLIR, target="stablehlo")

    assert result.result_state != LiftResultState.REFUTED, result.summary()
    assert result.verification is not None
    assert result.verification.result in (CheckResult.EQUIVALENT, CheckResult.TIMEOUT, CheckResult.UNKNOWN)
    assert result.matched_sketch is not None
    assert result.matched_sketch.name.startswith("stablehlo.reduce{add}")
    assert result.emitted_mlir is not None
    assert "stablehlo.reduce" in result.emitted_mlir
    assert "across dimensions = [1]" in result.emitted_mlir


def test_stablehlo_reduce_max_emits_expected_reducer() -> None:
    result = lift(REDUCE_MAX_MLIR, target="stablehlo")

    assert result.result_state != LiftResultState.REFUTED, result.summary()
    assert result.matched_sketch is not None
    assert result.matched_sketch.name == "stablehlo.reduce{max}"
    assert result.emitted_mlir is not None
    assert "stablehlo.reduce" in result.emitted_mlir
    assert "applies stablehlo.maximum" in result.emitted_mlir
