"""
Integration test — full lift pipeline for 2D transpose.
"""
import pytest
from loophole.lifter import lift, LiftResult
from loophole.tests.fixtures import TRANSPOSE_2D_MLIR, TRANSPOSE_NONSQUARE_MLIR


class TestTransposeLiftPipeline:
    def test_lift_returns_result(self):
        result = lift(TRANSPOSE_2D_MLIR)
        assert isinstance(result, LiftResult)

    def test_lift_identifies_transpose(self):
        result = lift(TRANSPOSE_2D_MLIR)
        assert result.success or result.partial_success, \
            f"Expected lift to succeed for transpose; got: {result.summary()}"

    def test_sketch_is_transpose(self):
        result = lift(TRANSPOSE_2D_MLIR)
        if result.success or result.partial_success:
            assert result.matched_sketch is not None
            assert "transpose" in result.matched_sketch.name.lower(), \
                f"Expected transpose sketch; got: {result.matched_sketch.name}"

    def test_linalg_emitted(self):
        result = lift(TRANSPOSE_2D_MLIR, target="linalg")
        if result.success or result.partial_success:
            assert result.emitted_mlir is not None
            assert "linalg" in result.emitted_mlir.lower()

    def test_nonsquare_transpose_lifted(self):
        result = lift(TRANSPOSE_NONSQUARE_MLIR)
        assert isinstance(result, LiftResult)
        assert result.success or result.partial_success, \
            f"Non-square transpose lift should succeed; got: {result.summary()}"
