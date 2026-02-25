"""
Integration test — full lift pipeline for 2D convolution.
"""
import pytest
from loophole.lifter import lift, LiftResult
from loophole.tests.fixtures import CONV_2D_SIMPLE_MLIR, CONV_2D_NHWC_MLIR


class TestConv2DSimpleLiftPipeline:
    def test_lift_returns_result(self):
        result = lift(CONV_2D_SIMPLE_MLIR)
        assert isinstance(result, LiftResult)

    def test_lift_identifies_conv2d(self):
        result = lift(CONV_2D_SIMPLE_MLIR)
        assert result.success or result.partial_success, \
            f"Expected lift to succeed for conv2d; got: {result.summary()}"

    def test_sketch_is_conv(self):
        result = lift(CONV_2D_SIMPLE_MLIR)
        if result.success or result.partial_success:
            assert result.matched_sketch is not None
            assert "conv" in result.matched_sketch.name.lower(), \
                f"Expected conv sketch; got: {result.matched_sketch.name}"

    def test_linalg_emitted(self):
        result = lift(CONV_2D_SIMPLE_MLIR, target="linalg")
        if result.success or result.partial_success:
            assert result.emitted_mlir
            assert "linalg" in result.emitted_mlir.lower()


class TestConv2DNHWCLiftPipeline:
    def test_lift_returns_result(self):
        result = lift(CONV_2D_NHWC_MLIR)
        assert isinstance(result, LiftResult)

    def test_lift_identifies_nhwc(self):
        result = lift(CONV_2D_NHWC_MLIR)
        # NHWC has 7 loops — may match nhwc or generic conv
        assert result.success or result.partial_success, \
            f"NHWC conv2d lift: {result.summary()}"
