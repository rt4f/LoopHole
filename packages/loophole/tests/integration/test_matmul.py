"""
Integration test — full lift pipeline for matrix multiply.
Parse → match → verify → emit.
"""
import pytest
from loophole.lifter import lift, Lifter, LiftResult
from loophole.mlir_validator import validate_mlir_artifact
from loophole.z3_checker import CheckResult
from loophole.tests.fixtures import (
    MATMUL_MLIR,
    MATMUL_128_MLIR,
    MIXED_REALWORLD_MATMUL_MLIR,
    MATMUL_DYNAMIC_DIMS_MLIR,
    MATMUL_ITER_ARGS_MLIR,
)


class TestMatmulLiftPipeline:
    def test_lift_succeeds(self):
        result = lift(MATMUL_MLIR)
        assert isinstance(result, LiftResult)

    def test_lift_identifies_matmul(self):
        result = lift(MATMUL_MLIR)
        assert result.success, f"Expected proved lift; got: {result.summary()}"
        assert result.verification is not None
        assert result.verification.result == CheckResult.EQUIVALENT

    def test_lift_emits_linalg(self):
        result = lift(MATMUL_MLIR, target="linalg")
        if result.success:
            assert result.emitted_mlir is not None
            assert "linalg" in result.emitted_mlir.lower()

    def test_lift_emits_stablehlo(self):
        result = lift(MATMUL_MLIR, target="stablehlo")
        # stablehlo emission is best-effort
        if result.success:
            assert result.emitted_mlir is not None

    def test_lift_stablehlo_strict_matmul_path(self):
        result = Lifter(target="stablehlo", strict_mode=True, z3_timeout_ms=5000).lift(MATMUL_MLIR)
        assert result.success, f"Expected strict proved stablehlo matmul; got: {result.summary()}"
        assert result.matched_sketch is not None
        assert result.matched_sketch.name == "stablehlo.dot_general"
        assert result.emitted_mlir is not None
        assert "stablehlo.dot_general" in result.emitted_mlir

    def test_lift_both_targets(self):
        result = lift(MATMUL_MLIR, target="both")
        assert isinstance(result, LiftResult)

    def test_lift_sketch_name_contains_matmul(self):
        result = lift(MATMUL_MLIR)
        if result.success:
            assert result.matched_sketch is not None
            assert "matmul" in result.matched_sketch.name.lower(), \
                f"Expected sketch name to contain 'matmul', got: {result.matched_sketch.name}"

    def test_lift_result_has_summary(self):
        result = lift(MATMUL_MLIR)
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_lifter_class_interface(self):
        lifter = Lifter(target="linalg", z3_timeout_ms=5000)
        result = lifter.lift(MATMUL_MLIR)
        assert isinstance(result, LiftResult)

    def test_lift_confidence_positive(self):
        result = lift(MATMUL_MLIR)
        if result.success:
            assert result.sympy_confidence > 0.0

    def test_lift_larger_matmul(self):
        """Larger bounds: Z3 may timeout but lift should still return a result."""
        result = lift(MATMUL_128_MLIR, z3_timeout_ms=3000)
        assert isinstance(result, LiftResult)
        # Under strict emission policies, an unproved match can still fail to emit.
        if not (result.success or result.partial_success):
            assert result.error is not None
            assert "Emission error while generating" in result.error

    def test_lift_fails_on_missing_shape_metadata(self):
        """B-05: emission must fail when required tensor shape metadata is missing."""
        lifter = Lifter(target="linalg", z3_timeout_ms=5000)
        original_emit = lifter._emit

        def _emit_without_shapes(sketch, loop_info):
            loop_info.tensor_shapes = {}
            return original_emit(sketch, loop_info)

        lifter._emit = _emit_without_shapes  # type: ignore[assignment]
        result = lifter.lift(MATMUL_MLIR)

        assert not result.success
        assert result.error is not None
        assert "Emission error while generating" in result.error
        assert "Missing tensor_shapes metadata" in result.error

    def test_lift_emitted_artifact_validates(self, mlir_verifier_cmd):
        result = lift(MATMUL_MLIR, target="linalg")
        assert result.success, f"Expected proved matmul lift; got: {result.summary()}"
        assert result.emitted_mlir is not None
        validation = validate_mlir_artifact(result.emitted_mlir, mlir_verifier_cmd)
        assert validation.ok, f"Verifier failed: {validation.stderr or validation.stdout}"

    def test_lift_mixed_realworld_matmul_fixture(self):
        result = lift(MIXED_REALWORLD_MATMUL_MLIR, target="linalg")
        assert result.success, f"Expected proved mixed-loop matmul lift; got: {result.summary()}"
        assert result.matched_sketch is not None
        assert "matmul" in result.matched_sketch.name

    def test_lift_dynamic_dim_matmul_fixture(self):
        result = lift(MATMUL_DYNAMIC_DIMS_MLIR, target="linalg")
        assert result.success or result.partial_success
        assert result.emitted_mlir is not None
        assert "?x?xf32" in result.emitted_mlir

    def test_lift_iter_args_reduction_form_linalg(self):
        """Regression: Polygeist iter_args SSA reduction form must lift to linalg.matmul."""
        result = lift(MATMUL_ITER_ARGS_MLIR, target="linalg")
        assert result.success, f"iter_args matmul lift failed: {result.summary()}"
        assert result.matched_sketch is not None
        assert "matmul" in result.matched_sketch.name.lower()

    def test_lift_iter_args_reduction_form_stablehlo(self):
        """Regression: Polygeist iter_args SSA reduction form must lift to stablehlo.dot_general."""
        result = Lifter(target="stablehlo", strict_mode=True, z3_timeout_ms=10000).lift(
            MATMUL_ITER_ARGS_MLIR
        )
        assert result.success, f"iter_args stablehlo lift failed: {result.summary()}"
        assert result.matched_sketch is not None
        assert result.matched_sketch.name == "stablehlo.dot_general"
