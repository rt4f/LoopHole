"""
Integration test — full lift pipeline for 2D transpose.
"""
import pytest
from loophole.lifter import lift, LiftResult
from loophole.z3_checker import CheckResult
from loophole.tests.fixtures import TRANSPOSE_2D_MLIR, TRANSPOSE_NONSQUARE_MLIR


class TestTransposeLiftPipeline:
    def test_lift_returns_result(self):
        result = lift(TRANSPOSE_2D_MLIR)
        assert isinstance(result, LiftResult)

    def test_lift_identifies_transpose(self):
        result = lift(TRANSPOSE_2D_MLIR)
        assert result.success, f"Expected proved transpose lift; got: {result.summary()}"
        assert result.verification is not None
        assert result.verification.result == CheckResult.EQUIVALENT

    def test_sketch_is_transpose(self):
        result = lift(TRANSPOSE_2D_MLIR)
        if result.success:
            assert result.matched_sketch is not None
            assert "transpose" in result.matched_sketch.name.lower(), \
                f"Expected transpose sketch; got: {result.matched_sketch.name}"

    def test_linalg_emitted(self):
        result = lift(TRANSPOSE_2D_MLIR, target="linalg")
        if result.success:
            assert result.emitted_mlir is not None
            assert "linalg" in result.emitted_mlir.lower()

    def test_nonsquare_transpose_lifted(self):
        result = lift(TRANSPOSE_NONSQUARE_MLIR)
        assert isinstance(result, LiftResult)
        assert result.success, f"Expected proved non-square transpose lift; got: {result.summary()}"
        assert result.verification is not None
        assert result.verification.result == CheckResult.EQUIVALENT
