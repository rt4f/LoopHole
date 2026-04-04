"""
Integration test — full lift pipeline for matvec kernels.
"""

from loophole.lifter import Lifter, LiftResult, lift
from loophole.z3_checker import CheckResult
from loophole.tests.fixtures import MATVEC_MLIR, MATVEC_TALL_MLIR, MATVEC_WIDE_MLIR


class TestMatvecLiftPipeline:
    def test_matvec_lift_returns_result(self):
        result = lift(MATVEC_MLIR, target="linalg")
        assert isinstance(result, LiftResult)

    def test_matvec_is_formally_proved(self):
        result = Lifter(target="linalg", strict_mode=True).lift(MATVEC_MLIR)
        assert result.success, f"Expected proved matvec lift; got: {result.summary()}"
        assert result.verification is not None
        assert result.verification.result == CheckResult.EQUIVALENT

    def test_matvec_tall_is_formally_proved(self):
        result = Lifter(target="linalg", strict_mode=True).lift(MATVEC_TALL_MLIR)
        assert result.success, f"Expected proved tall matvec lift; got: {result.summary()}"
        assert result.verification is not None
        assert result.verification.result == CheckResult.EQUIVALENT

    def test_matvec_wide_is_formally_proved(self):
        result = Lifter(target="linalg", strict_mode=True).lift(MATVEC_WIDE_MLIR)
        assert result.success, f"Expected proved wide matvec lift; got: {result.summary()}"
        assert result.verification is not None
        assert result.verification.result == CheckResult.EQUIVALENT
