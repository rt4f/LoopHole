"""
Unit tests for the MLIR emitter — verifies that emitted MLIR text is well-formed.
"""
import pytest
import re
from copy import deepcopy
from loophole.affine_extractor import AffineExtractor
from loophole.mlir_validator import validate_mlir_artifact
from loophole.sketch_library import SKETCH_BY_NAME, SKETCH_LIBRARY
from loophole.emitter import LinalgEmitter, StableHLOEmitter, EmissionError
from loophole.tests.fixtures import (
    MATMUL_MLIR,
    TRANSPOSE_2D_MLIR,
    TRANSPOSE_3D_PERM_120_MLIR,
    TRANSPOSE_3D_PERM_201_MLIR,
    CONV_2D_SIMPLE_MLIR,
    CONV_1D_STRIDED_DILATED_MLIR,
    CONV_2D_STRIDED_DILATED_MLIR,
    DOT_PRODUCT_MLIR,
    ELEMENTWISE_ADD_MLIR,
    ELEMENTWISE_SUB_MLIR,
    ELEMENTWISE_MUL_MLIR,
    REDUCE_SUM_MLIR,
    REDUCE_SUM_COLWISE_MLIR,
    REDUCE_MAX_MLIR,
    MATMUL_DYNAMIC_DIMS_MLIR,
    MATMUL_UNRANKED_MEMREF_MLIR,
    MATMUL_DYNAMIC_CONFLICTING_ANNOTATIONS_MLIR,
    MIXED_REALWORLD_MATMUL_MLIR,
    DOT_PRODUCT_SYMBOLIC_ARITH_MLIR,
    CONV_2D_STRIDED_DILATED_REORDERED_MLIR,
)


@pytest.fixture
def extractor():
    return AffineExtractor()


@pytest.fixture
def linalg_emitter():
    return LinalgEmitter()


@pytest.fixture
def stablehlo_emitter():
    return StableHLOEmitter()


class TestLinalgEmitter:
    def test_emit_matmul_contains_linalg(self, extractor, linalg_emitter):
        info = extractor.extract(MATMUL_MLIR)
        sketch = SKETCH_BY_NAME["linalg.matmul"]
        result = linalg_emitter.emit(sketch, info)
        assert "linalg" in result.lower(), "Expected 'linalg' in emitted MLIR"

    def test_emit_matmul_has_func(self, extractor, linalg_emitter):
        info = extractor.extract(MATMUL_MLIR)
        sketch = SKETCH_BY_NAME["linalg.matmul"]
        result = linalg_emitter.emit(sketch, info)
        assert "func.func" in result or "func " in result

    def test_emit_matmul_has_return(self, extractor, linalg_emitter):
        info = extractor.extract(MATMUL_MLIR)
        sketch = SKETCH_BY_NAME["linalg.matmul"]
        result = linalg_emitter.emit(sketch, info)
        assert "return" in result

    def test_emit_matmul_braces_balanced(self, extractor, linalg_emitter):
        info = extractor.extract(MATMUL_MLIR)
        sketch = SKETCH_BY_NAME["linalg.matmul"]
        result = linalg_emitter.emit(sketch, info)
        assert result.count("{") == result.count("}"), "Unbalanced braces in emitted MLIR"

    def test_emit_transpose_contains_linalg(self, extractor, linalg_emitter):
        info = extractor.extract(TRANSPOSE_2D_MLIR)
        transpose_sketches = [s for n, s in SKETCH_BY_NAME.items() if "transpose" in n]
        if not transpose_sketches:
            pytest.skip("No transpose sketch")
        result = linalg_emitter.emit(transpose_sketches[0], info)
        assert "linalg" in result.lower()

    def test_emit_nonempty_for_all_linalg_sketches(self, linalg_emitter):
        from loophole.sketch_library import LINALG_SKETCHES
        from loophole.tests.fixtures import MATMUL_MLIR
        extractor = AffineExtractor()
        info = extractor.extract(MATMUL_MLIR)
        # Incompatible sketches may fail under strict metadata/attr policies.
        for sketch in LINALG_SKETCHES[:6]:  # limit to first 6 to keep test fast
            try:
                result = linalg_emitter.emit(sketch, info)
                assert isinstance(result, str), f"emit returned non-string for {sketch.name}"
                assert len(result) > 0, f"emit returned empty string for {sketch.name}"
            except EmissionError as exc:
                msg = str(exc).lower()
                assert (
                    "policy failure" in msg
                    or "missing" in msg
                    or "invalid tensor rank" in msg
                    or "unsupported" in msg
                )

    def test_emit_matmul_named_op(self, extractor, linalg_emitter):
        info = extractor.extract(MATMUL_MLIR)
        sketch = SKETCH_BY_NAME["linalg.matmul"]
        result = linalg_emitter.emit(sketch, info)
        # should prefer linalg.matmul over linalg.generic for matmul
        assert "linalg.matmul" in result or "linalg.generic" in result

    def test_emit_dot_product(self, extractor, linalg_emitter):
        info = extractor.extract(DOT_PRODUCT_MLIR)
        dot_sketches = [s for n, s in SKETCH_BY_NAME.items() if "dot" in n]
        if not dot_sketches:
            pytest.skip("No dot sketch")
        result = linalg_emitter.emit(dot_sketches[0], info)
        assert "linalg" in result.lower()
        assert len(result) > 0

    def test_emit_matmul_missing_shapes_fails(self, extractor, linalg_emitter):
        """B-05: missing tensor metadata must fail, not guess defaults."""
        info = extractor.extract(MATMUL_MLIR)
        info = deepcopy(info)
        info.tensor_shapes = {}

        sketch = SKETCH_BY_NAME["linalg.matmul"]
        with pytest.raises(EmissionError, match="Missing tensor_shapes metadata"):
            linalg_emitter.emit(sketch, info)

    def test_emit_transpose_exact_permutation_2d(self, extractor, linalg_emitter):
        info = extractor.extract(TRANSPOSE_2D_MLIR)
        transpose_sketches = [s for n, s in SKETCH_BY_NAME.items() if "transpose" in n]
        if not transpose_sketches:
            pytest.skip("No transpose sketch")
        result = linalg_emitter.emit(transpose_sketches[0], info)
        assert "permutation = [1, 0]" in result

    def test_emit_transpose_exact_permutation_3d_120(self, extractor, linalg_emitter):
        info = extractor.extract(TRANSPOSE_3D_PERM_120_MLIR)
        transpose_sketches = [s for n, s in SKETCH_BY_NAME.items() if "transpose" in n]
        if not transpose_sketches:
            pytest.skip("No transpose sketch")
        result = linalg_emitter.emit(transpose_sketches[0], info)
        assert "permutation = [1, 2, 0]" in result

    def test_emit_transpose_exact_permutation_3d_201(self, extractor, linalg_emitter):
        info = extractor.extract(TRANSPOSE_3D_PERM_201_MLIR)
        transpose_sketches = [s for n, s in SKETCH_BY_NAME.items() if "transpose" in n]
        if not transpose_sketches:
            pytest.skip("No transpose sketch")
        result = linalg_emitter.emit(transpose_sketches[0], info)
        assert "permutation = [2, 0, 1]" in result

    def test_emit_transpose_ambiguous_inference_fails(self, extractor, linalg_emitter):
        """B-06: ambiguous/non-direct output indexing must fail, no reverse fallback."""
        info = extractor.extract(TRANSPOSE_2D_MLIR)
        info = deepcopy(info)

        # Force rank-3 input but keep 2-D write indexing, so inference is incomplete.
        info.tensor_shapes["%A"] = [2, 3, 4]
        info.tensor_shapes["%B"] = [4, 3, 2]

        transpose_sketches = [s for n, s in SKETCH_BY_NAME.items() if "transpose" in n]
        if not transpose_sketches:
            pytest.skip("No transpose sketch")
        with pytest.raises(EmissionError, match="Cannot infer transpose permutation|Invalid output rank"):
            linalg_emitter.emit(transpose_sketches[0], info)

    def test_emit_conv2d_missing_output_shape_fails(self, extractor, linalg_emitter):
        """B-05: conv emission must fail when output shape metadata is missing."""
        from loophole.tests.fixtures import CONV_2D_SIMPLE_MLIR

        info = extractor.extract(CONV_2D_SIMPLE_MLIR)
        info = deepcopy(info)
        out = info.output_tensor
        if out and out in info.tensor_shapes:
            del info.tensor_shapes[out]

        sketch = SKETCH_BY_NAME["linalg.conv_2d"]
        with pytest.raises(EmissionError, match="Missing tensor shape metadata"):
            linalg_emitter.emit(sketch, info)

    def test_emit_conv1d_infers_non_unit_attrs(self, extractor, linalg_emitter):
        """B-07: infer explicit non-unit stride/dilation for 1-D conv."""
        info = extractor.extract(CONV_1D_STRIDED_DILATED_MLIR)
        sketch = SKETCH_BY_NAME["linalg.conv_1d_ncw_fcw"]
        result = linalg_emitter.emit(sketch, info)
        assert "dilations = dense<3>" in result
        assert "strides   = dense<2>" in result

    def test_emit_conv2d_infers_non_unit_attrs(self, extractor, linalg_emitter):
        """B-07: infer per-axis non-unit stride/dilation for 2-D conv."""
        info = extractor.extract(CONV_2D_STRIDED_DILATED_MLIR)
        sketch = SKETCH_BY_NAME["linalg.conv_2d"]
        result = linalg_emitter.emit(sketch, info)
        assert "dilations = dense<[3, 2]>" in result
        assert "strides   = dense<[2, 1]>" in result

    def test_emit_conv2d_attr_inference_failure_is_explicit(self, extractor, linalg_emitter):
        """B-07: unsupported indexing must fail with policy error, never guess attrs."""
        from loophole.tests.fixtures import CONV_2D_SIMPLE_MLIR

        info = extractor.extract(CONV_2D_SIMPLE_MLIR)
        info = deepcopy(info)
        i_name = info.input_tensors[0]

        for read in info.reads:
            if read.tensor_name == i_name and len(read.index_exprs) >= 2:
                read.index_exprs[0] = "%oh + %ow"
                break

        sketch = SKETCH_BY_NAME["linalg.conv_2d"]
        with pytest.raises(EmissionError, match="Convolution attr policy failure"):
            linalg_emitter.emit(sketch, info)

    def test_emit_matmul_artifact_validates(self, extractor, linalg_emitter, mlir_verifier_cmd):
        info = extractor.extract(MATMUL_MLIR)
        sketch = SKETCH_BY_NAME["linalg.matmul"]
        result = linalg_emitter.emit(sketch, info)
        validation = validate_mlir_artifact(result, mlir_verifier_cmd)
        assert validation.ok, f"Verifier failed: {validation.stderr or validation.stdout}"

    def test_emit_transpose_artifact_validates(self, extractor, linalg_emitter, mlir_verifier_cmd):
        info = extractor.extract(TRANSPOSE_2D_MLIR)
        transpose_sketches = [s for n, s in SKETCH_BY_NAME.items() if "transpose" in n]
        if not transpose_sketches:
            pytest.skip("No transpose sketch")
        result = linalg_emitter.emit(transpose_sketches[0], info)
        validation = validate_mlir_artifact(result, mlir_verifier_cmd)
        assert validation.ok, f"Verifier failed: {validation.stderr or validation.stdout}"

    def test_emit_conv2d_artifact_validates(self, extractor, linalg_emitter, mlir_verifier_cmd):
        info = extractor.extract(CONV_2D_STRIDED_DILATED_MLIR)
        sketch = SKETCH_BY_NAME["linalg.conv_2d"]
        result = linalg_emitter.emit(sketch, info)
        validation = validate_mlir_artifact(result, mlir_verifier_cmd)
        assert validation.ok, f"Verifier failed: {validation.stderr or validation.stdout}"

    def test_emit_dynamic_dim_matmul_uses_symbolic_dims(self, extractor, linalg_emitter):
        info = extractor.extract(MATMUL_DYNAMIC_DIMS_MLIR)
        sketch = SKETCH_BY_NAME["linalg.matmul"]
        result = linalg_emitter.emit(sketch, info)
        assert "memref<?x?xf32>" in result

    def test_emit_unranked_memref_matmul_normalizes_to_ranked_dynamic(self, extractor, linalg_emitter):
        info = extractor.extract(MATMUL_UNRANKED_MEMREF_MLIR)
        sketch = SKETCH_BY_NAME["linalg.matmul"]
        result = linalg_emitter.emit(sketch, info)
        assert "memref<?x?xf32>" in result

    def test_emit_conflicting_dynamic_dim_annotations_remain_dynamic(self, extractor, linalg_emitter):
        info = extractor.extract(MATMUL_DYNAMIC_CONFLICTING_ANNOTATIONS_MLIR)
        sketch = SKETCH_BY_NAME["linalg.matmul"]
        result = linalg_emitter.emit(sketch, info)
        assert "memref<?x?xf32>" in result

    def test_emit_mixed_realworld_matmul_path(self, extractor, linalg_emitter):
        info = extractor.extract(MIXED_REALWORLD_MATMUL_MLIR)
        sketch = SKETCH_BY_NAME["linalg.matmul"]
        result = linalg_emitter.emit(sketch, info)
        assert "linalg.matmul" in result

    def test_emit_symbolic_arith_dot_path(self, extractor, linalg_emitter):
        info = extractor.extract(DOT_PRODUCT_SYMBOLIC_ARITH_MLIR)
        sketch = SKETCH_BY_NAME["linalg.dot"]
        result = linalg_emitter.emit(sketch, info)
        assert "linalg.dot" in result

    def test_emit_reordered_conv_attr_inference(self, extractor, linalg_emitter):
        info = extractor.extract(CONV_2D_STRIDED_DILATED_REORDERED_MLIR)
        sketch = SKETCH_BY_NAME["linalg.conv_2d"]
        result = linalg_emitter.emit(sketch, info)
        assert "dilations = dense<[3, 2]>" in result
        assert "strides   = dense<[2, 1]>" in result


class TestStableHLOEmitter:
    def test_stablehlo_handler_map_covers_all_sketches(self):
        from loophole.sketch_library import STABLEHLO_SKETCHES

        mapped = set(StableHLOEmitter._NAMED_OP_HANDLERS.keys())
        library = {s.name for s in STABLEHLO_SKETCHES}
        assert mapped == library, f"StableHLO handler map mismatch: mapped={sorted(mapped)}, library={sorted(library)}"

    def test_emit_stablehlo_matmul(self, extractor, stablehlo_emitter):
        info = extractor.extract(MATMUL_MLIR)
        # find a stablehlo sketch
        from loophole.sketch_library import STABLEHLO_SKETCHES
        if not STABLEHLO_SKETCHES:
            pytest.skip("No StableHLO sketches")
        sketch = STABLEHLO_SKETCHES[0]
        result = stablehlo_emitter.emit(sketch, info)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_emit_stablehlo_dot_general(self, extractor, stablehlo_emitter):
        info = extractor.extract(MATMUL_MLIR)
        from loophole.sketch_library import STABLEHLO_SKETCHES
        dot_sketches = [s for s in STABLEHLO_SKETCHES if "dot" in s.name.lower()]
        if not dot_sketches:
            pytest.skip("No StableHLO dot sketch")
        result = stablehlo_emitter.emit(dot_sketches[0], info)
        assert "stablehlo" in result.lower() or "func" in result.lower()

    def test_emit_stablehlo_convolution_infers_non_unit_attrs(self, extractor, stablehlo_emitter):
        info = extractor.extract(CONV_2D_STRIDED_DILATED_MLIR)
        from loophole.sketch_library import STABLEHLO_SKETCHES

        conv_sketches = [s for s in STABLEHLO_SKETCHES if "convolution" in s.name.lower()]
        if not conv_sketches:
            pytest.skip("No StableHLO convolution sketch")

        result = stablehlo_emitter.emit(conv_sketches[0], info)
        assert "window = {stride = [2, 1]" in result
        assert "rhs_dilate = [3, 2]" in result

    def test_emit_stablehlo_subtract(self, extractor, stablehlo_emitter):
        info = extractor.extract(ELEMENTWISE_SUB_MLIR)
        sketch = SKETCH_BY_NAME["stablehlo.subtract"]
        result = stablehlo_emitter.emit(sketch, info)
        assert "stablehlo.subtract" in result

    def test_emit_stablehlo_multiply(self, extractor, stablehlo_emitter):
        info = extractor.extract(ELEMENTWISE_MUL_MLIR)
        sketch = SKETCH_BY_NAME["stablehlo.multiply"]
        result = stablehlo_emitter.emit(sketch, info)
        assert "stablehlo.multiply" in result

    def test_emit_stablehlo_reduce_sum_rowwise(self, extractor, stablehlo_emitter):
        info = extractor.extract(REDUCE_SUM_MLIR)
        sketch = SKETCH_BY_NAME["stablehlo.reduce{add}"]
        result = stablehlo_emitter.emit(sketch, info)
        assert "stablehlo.reduce" in result
        assert "across dimensions = [1]" in result

    def test_emit_stablehlo_reduce_sum_colwise(self, extractor, stablehlo_emitter):
        info = extractor.extract(REDUCE_SUM_COLWISE_MLIR)
        sketch = SKETCH_BY_NAME["stablehlo.reduce{add}_colsum"]
        result = stablehlo_emitter.emit(sketch, info)
        assert "stablehlo.reduce" in result
        assert "across dimensions = [0]" in result

    def test_emit_stablehlo_reduce_max(self, extractor, stablehlo_emitter):
        info = extractor.extract(REDUCE_MAX_MLIR)
        sketch = SKETCH_BY_NAME["stablehlo.reduce{max}"]
        result = stablehlo_emitter.emit(sketch, info)
        assert "stablehlo.reduce" in result
        assert "applies stablehlo.maximum" in result


class TestEmitterValidationConsistencyB13:
    def test_missing_tensor_shapes_gate_is_consistent_across_dialects(self, extractor, linalg_emitter, stablehlo_emitter):
        info = deepcopy(extractor.extract(MATMUL_MLIR))
        info.tensor_shapes = {}

        with pytest.raises(EmissionError, match="Missing tensor_shapes metadata"):
            linalg_emitter.emit(SKETCH_BY_NAME["linalg.matmul"], info)

        with pytest.raises(EmissionError, match="Missing tensor_shapes metadata"):
            stablehlo_emitter.emit(SKETCH_BY_NAME["stablehlo.dot_general"], info)

    def test_missing_element_type_gate_is_consistent_across_dialects(self, extractor, linalg_emitter, stablehlo_emitter):
        info = deepcopy(extractor.extract(MATMUL_MLIR))
        info.element_type = ""

        with pytest.raises(EmissionError, match="Missing element type metadata"):
            linalg_emitter.emit(SKETCH_BY_NAME["linalg.matmul"], info)

        with pytest.raises(EmissionError, match="Missing element type metadata"):
            stablehlo_emitter.emit(SKETCH_BY_NAME["stablehlo.dot_general"], info)

    def test_rank_gate_is_actionable_for_linalg_matmul(self, extractor, linalg_emitter):
        info = deepcopy(extractor.extract(MATMUL_MLIR))
        info.tensor_shapes["%A"] = [4]

        with pytest.raises(EmissionError, match="Invalid tensor rank"):
            linalg_emitter.emit(SKETCH_BY_NAME["linalg.matmul"], info)

    def test_rank_gate_is_actionable_for_stablehlo_convolution(self, extractor, stablehlo_emitter):
        info = deepcopy(extractor.extract(CONV_2D_STRIDED_DILATED_MLIR))
        info.tensor_shapes["%I"] = [8, 8, 1]

        with pytest.raises(EmissionError, match="Invalid tensor rank"):
            stablehlo_emitter.emit(SKETCH_BY_NAME["stablehlo.convolution"], info)

    def test_convolution_attr_policy_failure_is_consistent_across_dialects(self, extractor, linalg_emitter, stablehlo_emitter):
        info = deepcopy(extractor.extract(CONV_2D_SIMPLE_MLIR))

        for read in info.reads:
            if read.tensor_name == "%I" and len(read.index_exprs) >= 2:
                read.index_exprs[0] = "%oh + %ow"
                break

        with pytest.raises(EmissionError, match="Convolution attr policy failure"):
            linalg_emitter.emit(SKETCH_BY_NAME["linalg.conv_2d"], info)

        with pytest.raises(EmissionError, match="Convolution attr policy failure"):
            stablehlo_emitter.emit(SKETCH_BY_NAME["stablehlo.convolution"], info)
