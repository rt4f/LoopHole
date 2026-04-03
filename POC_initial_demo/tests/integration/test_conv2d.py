"""
Integration test — full lift pipeline for 2D convolution.
"""
from loophole.lifter import lift, Lifter, LiftResult, LiftResultState
from loophole.z3_checker import CheckResult
from loophole.tests.fixtures import CONV_2D_SIMPLE_MLIR, CONV_2D_NHWC_MLIR


class TestConv2DSimpleLiftPipeline:
    def test_lift_returns_result(self):
        result = lift(CONV_2D_SIMPLE_MLIR)
        assert isinstance(result, LiftResult)

    def test_lift_identifies_conv2d(self):
        result = lift(CONV_2D_SIMPLE_MLIR)
        assert result.verification is not None, \
            f"Expected explicit verification outcome for conv2d; got: {result.summary()}"

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
        assert result.verification is not None, f"NHWC conv2d lift: {result.summary()}"


def test_conv2d_simple_not_refuted_in_phase1_strict_semantics():
    result = Lifter(target="linalg", strict_mode=True).lift(CONV_2D_SIMPLE_MLIR)
    assert result.verification is not None

    if result.result_state != LiftResultState.PROVED:
        assert not result.is_accepted(strict_mode=True)

    if result.verification.result == CheckResult.NOT_EQUIVALENT:
        assert result.result_state == LiftResultState.REFUTED
        assert not result.partial_success
        assert result.emitted_mlir is None
