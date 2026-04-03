"""
Integration test — full lift pipeline for 2D convolution.
"""
import inspect

import pytest
from loophole.lifter import lift, Lifter, LiftResult
from loophole.z3_checker import CheckResult
from loophole.tests.fixtures import CONV_2D_SIMPLE_MLIR, CONV_2D_NHWC_MLIR


_LIFTER_INIT_PARAMS = inspect.signature(Lifter.__init__).parameters
_STRICT_PARAM = "strict_mode" if "strict_mode" in _LIFTER_INIT_PARAMS else (
    "strict" if "strict" in _LIFTER_INIT_PARAMS else None
)


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


@pytest.mark.xfail(
    _STRICT_PARAM is None,
    reason="Depends on A-02 strict semantics API in Lifter constructor.",
)
def test_conv2d_simple_not_refuted_in_phase1_strict_semantics():
    kwargs = {"target": "linalg"}
    if _STRICT_PARAM is not None:
        kwargs[_STRICT_PARAM] = True

    result = Lifter(**kwargs).lift(CONV_2D_SIMPLE_MLIR)
    assert result.verification is not None
    if result.verification.result == CheckResult.NOT_EQUIVALENT:
        assert not result.partial_success
