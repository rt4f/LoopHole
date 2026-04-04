"""
Integration tests for Phase 3 StableHLO operation coverage.

C-08/C-09/C-10:
- Expanded StableHLO sketches and emitters
- Integration coverage for transpose, elementwise, dot variants, and reduce
"""

from loophole.lifter import Lifter, lift, LiftResultState
from loophole.z3_checker import CheckResult
from loophole.tests.fixtures import (
    DOT_PRODUCT_MLIR,
    ELEMENTWISE_ADD_MLIR,
    MATVEC_MLIR,
    REDUCE_SUM_MLIR,
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
