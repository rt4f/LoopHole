"""
Integration tests — elementwise kernels are proved as the operation they
compute, or reach no verdict. They must never be proved as a simpler op.
"""
import pytest

from loophole.lifter import Lifter, LiftResultState
from loophole.tests.fixtures import (
    ADD_CONSTANT_MLIR,
    ELEMENTWISE_ADD3_MLIR,
    ELEMENTWISE_DIV_MLIR,
    ELEMENTWISE_MUL_ADD_MLIR,
    ELEMENTWISE_MUL_MLIR,
    MAX_HEX_CONSTANT_MLIR,
    NEGATE_MLIR,
    RELU_MLIR,
    SCALE_BY_CONSTANT_MLIR,
)
from loophole.z3_checker import CheckResult


def _lift(mlir: str, target: str = "linalg"):
    return Lifter(target=target, z3_timeout_ms=5000).lift(mlir)


@pytest.mark.parametrize(
    "mlir, target, sketch_name",
    [
        (ELEMENTWISE_MUL_MLIR, "linalg", "linalg.map{arith.mulf}"),
        (ELEMENTWISE_MUL_MLIR, "stablehlo", "stablehlo.multiply"),
        (RELU_MLIR, "linalg", "linalg.map{arith.maxf_zero}"),
    ],
    ids=["mul-linalg", "mul-stablehlo", "relu-linalg"],
)
def test_kernel_is_proved_as_the_operation_it_computes(mlir, target, sketch_name):
    result = _lift(mlir, target)

    assert result.sketch_name == sketch_name
    assert result.result_state == LiftResultState.PROVED


@pytest.mark.parametrize(
    "mlir, reason",
    [
        (ELEMENTWISE_DIV_MLIR, "unknown_op"),
        (SCALE_BY_CONSTANT_MLIR, "input_count"),
        (ADD_CONSTANT_MLIR, "input_count"),
        (ELEMENTWISE_ADD3_MLIR, "input_count"),
        (ELEMENTWISE_MUL_ADD_MLIR, "input_count"),
        (MAX_HEX_CONSTANT_MLIR, "unparsed_constant"),
    ],
    ids=["div", "scale-by-constant", "add-constant", "add3", "mul-add", "max-hex-constant"],
)
def test_unmodelled_kernel_reaches_no_verdict(mlir, reason):
    result = _lift(mlir)

    assert result.verification is None
    # No verdict is still reported as REFUTED; separating the two is a separate task.
    assert result.result_state == LiftResultState.REFUTED
    assert "unsupported_form=unmodelled_compute_payload" in result.error, result.error
    assert f"reason={reason}" in result.error, result.error


def test_negation_is_refuted_rather_than_proved_as_copy():
    result = _lift(NEGATE_MLIR)

    assert result.result_state == LiftResultState.REFUTED
    assert result.verification.result == CheckResult.NOT_EQUIVALENT
