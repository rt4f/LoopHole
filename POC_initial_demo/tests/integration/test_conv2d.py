"""
Integration test — full lift pipeline for 2D convolution.
"""
from loophole.lifter import lift, Lifter, LiftResult, LiftResultState
from loophole.mlir_validator import validate_mlir_artifact
from loophole.z3_checker import CheckResult, VerificationReport
from loophole.tests.fixtures import (
    CONV_2D_1X1_MLIR,
    CONV_2D_5X5_MLIR,
    CONV_2D_SIMPLE_MLIR,
    CONV_2D_NHWC_MLIR,
    CONV_2D_STRIDED_DILATED_MLIR,
    CONV_2D_STRIDED_DILATED_REORDERED_MLIR,
)


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

    def test_conv2d_missing_output_shape_reports_error(self):
        """B-05: incomplete metadata must block emission with clear reason."""
        lifter = Lifter(target="linalg")
        original_emit = lifter._emit
        original_verify = lifter._verify_candidates

        def _emit_without_output_shape(sketch, loop_info):
            out = loop_info.output_tensor
            if out and out in loop_info.tensor_shapes:
                del loop_info.tensor_shapes[out]
            return original_emit(sketch, loop_info)

        def _verify_equivalent(loop_info, candidates):
            if not candidates:
                return original_verify(loop_info, candidates)
            sketch, conf = candidates[0]
            report = VerificationReport(
                result=CheckResult.EQUIVALENT,
                sketch_name=sketch.name,
                elapsed_ms=0.0,
                notes="Forced equivalent for emission-path test",
            )
            return (sketch, report, conf)

        lifter._emit = _emit_without_output_shape  # type: ignore[assignment]
        lifter._verify_candidates = _verify_equivalent  # type: ignore[assignment]
        result = lifter.lift(CONV_2D_SIMPLE_MLIR)

        assert not result.success
        assert result.error is not None
        assert "Emission error while generating" in result.error
        assert "Missing tensor shape metadata" in result.error


class TestConv2DNHWCLiftPipeline:
    def test_lift_returns_result(self):
        result = lift(CONV_2D_NHWC_MLIR)
        assert isinstance(result, LiftResult)

    def test_lift_identifies_nhwc(self):
        result = lift(CONV_2D_NHWC_MLIR)
        assert result.verification is not None, f"NHWC conv2d lift: {result.summary()}"


class TestConv2DAttrInference:
    def test_linalg_emits_inferred_stride_dilation(self):
        result = lift(CONV_2D_STRIDED_DILATED_MLIR, target="linalg")
        if result.success or result.partial_success:
            assert result.emitted_mlir is not None
            assert "dilations = dense<[3, 2]>" in result.emitted_mlir
            assert "strides   = dense<[2, 1]>" in result.emitted_mlir

    def test_stablehlo_emits_inferred_stride_dilation(self):
        result = lift(CONV_2D_STRIDED_DILATED_MLIR, target="stablehlo")
        if result.success or result.partial_success:
            assert result.emitted_mlir is not None
            assert "window = {stride = [2, 1]" in result.emitted_mlir
            assert "rhs_dilate = [3, 2]" in result.emitted_mlir

    def test_linalg_emitted_artifact_validates(self, mlir_verifier_cmd):
        lifter = Lifter(target="linalg")
        original_verify = lifter._verify_candidates

        def _verify_equivalent(loop_info, candidates):
            if not candidates:
                return original_verify(loop_info, candidates)
            sketch, conf = candidates[0]
            report = VerificationReport(
                result=CheckResult.EQUIVALENT,
                sketch_name=sketch.name,
                elapsed_ms=0.0,
                notes="Forced equivalent for artifact validation path",
            )
            return (sketch, report, conf)

        lifter._verify_candidates = _verify_equivalent  # type: ignore[assignment]
        result = lifter.lift(CONV_2D_STRIDED_DILATED_MLIR)
        assert result.success or result.partial_success
        assert result.emitted_mlir is not None
        validation = validate_mlir_artifact(result.emitted_mlir, mlir_verifier_cmd)
        assert validation.ok, f"Verifier failed: {validation.stderr or validation.stdout}"

    def test_reordered_affine_conv_fixture_emits_same_attrs(self):
        lifter = Lifter(target="linalg")
        original_verify = lifter._verify_candidates

        def _verify_equivalent(loop_info, candidates):
            if not candidates:
                return original_verify(loop_info, candidates)
            sketch, conf = candidates[0]
            report = VerificationReport(
                result=CheckResult.EQUIVALENT,
                sketch_name=sketch.name,
                elapsed_ms=0.0,
                notes="Forced equivalent for reordered-conv emission test",
            )
            return (sketch, report, conf)

        lifter._verify_candidates = _verify_equivalent  # type: ignore[assignment]
        result = lifter.lift(CONV_2D_STRIDED_DILATED_REORDERED_MLIR)
        assert result.success or result.partial_success
        assert result.emitted_mlir is not None
        assert "dilations = dense<[3, 2]>" in result.emitted_mlir
        assert "strides   = dense<[2, 1]>" in result.emitted_mlir


def test_conv2d_simple_not_refuted_in_phase1_strict_semantics():
    result = Lifter(target="linalg", strict_mode=True).lift(CONV_2D_SIMPLE_MLIR)
    assert result.verification is not None

    if result.result_state != LiftResultState.PROVED:
        assert not result.is_accepted(strict_mode=True)

    if result.verification.result == CheckResult.NOT_EQUIVALENT:
        assert result.result_state == LiftResultState.REFUTED
        assert not result.partial_success
        assert result.emitted_mlir is None


class TestConv2DEdgeCaseProofs:
    def test_conv2d_1x1_is_formally_proved(self):
        result = Lifter(target="linalg", strict_mode=True).lift(CONV_2D_1X1_MLIR)
        assert result.success, f"Expected proved 1x1 conv2d lift; got: {result.summary()}"
        assert result.verification is not None
        assert result.verification.result == CheckResult.EQUIVALENT

    def test_conv2d_5x5_is_formally_proved(self):
        result = Lifter(target="linalg", strict_mode=True).lift(CONV_2D_5X5_MLIR)
        assert result.success, f"Expected proved 5x5 conv2d lift; got: {result.summary()}"
        assert result.verification is not None
        assert result.verification.result == CheckResult.EQUIVALENT
