"""
Integration test — full lift pipeline for dot-product kernels.
"""

from loophole.lifter import Lifter, LiftResult, lift
from loophole.z3_checker import CheckResult
from loophole.tests.fixtures import (
    DOT_PRODUCT_MLIR,
    DOT_PRODUCT_16_MLIR,
    DOT_PRODUCT_SYMBOLIC_N_MLIR,
    DOT_PRODUCT_SYMBOLIC_ARITH_MLIR,
)


class TestDotLiftPipeline:
    def test_dot_lift_returns_result(self):
        result = lift(DOT_PRODUCT_MLIR, target="linalg")
        assert isinstance(result, LiftResult)

    def test_dot_is_formally_proved(self):
        result = Lifter(target="linalg", strict_mode=True).lift(DOT_PRODUCT_MLIR)
        assert result.success, f"Expected proved dot lift; got: {result.summary()}"
        assert result.verification is not None
        assert result.verification.result == CheckResult.EQUIVALENT

    def test_dot_16_is_formally_proved(self):
        result = Lifter(target="linalg", strict_mode=True).lift(DOT_PRODUCT_16_MLIR)
        assert result.success, f"Expected proved dot16 lift; got: {result.summary()}"
        assert result.verification is not None
        assert result.verification.result == CheckResult.EQUIVALENT

    def test_dot_proof_has_no_sympy_z3_disagreement(self):
        result = Lifter(target="linalg", strict_mode=True).lift(DOT_PRODUCT_MLIR)
        assert result.verification is not None
        assert result.verification.result == CheckResult.EQUIVALENT
        assert result.verification.sympy_z3_disagreement is None

    def test_dot_symbolic_n_is_formally_proved(self):
        result = Lifter(target="linalg", strict_mode=True).lift(DOT_PRODUCT_SYMBOLIC_N_MLIR)
        assert result.success, f"Expected proved symbolic-N dot lift; got: {result.summary()}"
        assert result.verification is not None
        assert result.verification.result == CheckResult.EQUIVALENT

    def test_dot_symbolic_arith_fixture_is_formally_proved(self):
        result = Lifter(target="linalg", strict_mode=True).lift(DOT_PRODUCT_SYMBOLIC_ARITH_MLIR)
        assert result.success, f"Expected proved symbolic-arith dot lift; got: {result.summary()}"
        assert result.verification is not None
        assert result.verification.result == CheckResult.EQUIVALENT
