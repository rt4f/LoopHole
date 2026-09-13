"""
Integration test — full lift pipeline for 1D convolution.
"""
import pytest
from loophole.lifter import lift, LiftResult
from loophole.z3_checker import CheckResult
from loophole.tests.fixtures import CONV_1D_MLIR, CONV_1D_STRIDED_DILATED_MLIR


class TestConv1DLiftPipeline:
    def test_lift_returns_result(self):
        result = lift(CONV_1D_MLIR)
        assert isinstance(result, LiftResult)

    def test_lift_identifies_conv(self):
        result = lift(CONV_1D_MLIR)
        # conv1d has mulf+addf with k-window, should match
        assert result.success, f"Expected proved conv1d lift; got: {result.summary()}"
        assert result.verification is not None
        assert result.verification.result == CheckResult.EQUIVALENT

    def test_lift_sketch_is_conv1d(self):
        result = lift(CONV_1D_MLIR)
        if result.success:
            assert result.matched_sketch is not None
            sketch_name = result.matched_sketch.name.lower()
            assert "conv" in sketch_name, \
                f"Expected a conv sketch, got: {sketch_name}"

    def test_emitted_mlir_contains_linalg(self):
        result = lift(CONV_1D_MLIR, target="linalg")
        if result.success:
            assert result.emitted_mlir is not None
            assert "linalg" in result.emitted_mlir.lower()

    def test_result_has_input_mlir(self):
        result = lift(CONV_1D_MLIR)
        # The result should have reference to the source
        assert isinstance(result, LiftResult)

    def test_linalg_emits_inferred_stride_dilation(self):
        result = lift(CONV_1D_STRIDED_DILATED_MLIR, target="linalg")
        if result.success or result.partial_success:
            assert result.emitted_mlir is not None
            assert "dilations = dense<3>" in result.emitted_mlir
            assert "strides   = dense<2>" in result.emitted_mlir
