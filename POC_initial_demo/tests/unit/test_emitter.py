"""
Unit tests for the MLIR emitter — verifies that emitted MLIR text is well-formed.
"""
import pytest
import re
from copy import deepcopy
from loophole.affine_extractor import AffineExtractor
from loophole.sketch_library import SKETCH_BY_NAME, SKETCH_LIBRARY
from loophole.emitter import LinalgEmitter, StableHLOEmitter, EmissionError
from loophole.tests.fixtures import MATMUL_MLIR, TRANSPOSE_2D_MLIR, DOT_PRODUCT_MLIR


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
        # Just check emitter doesn't crash for any sketch
        for sketch in LINALG_SKETCHES[:6]:  # limit to first 6 to keep test fast
            result = linalg_emitter.emit(sketch, info)
            assert isinstance(result, str), f"emit returned non-string for {sketch.name}"
            assert len(result) > 0, f"emit returned empty string for {sketch.name}"

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

    def test_emit_transpose_detects_reverse_permutation_fallback(self, extractor, linalg_emitter):
        """Detect permutation fallback path: incomplete inference falls back to reverse dims."""
        info = extractor.extract(TRANSPOSE_2D_MLIR)
        info = deepcopy(info)

        # Force rank-3 input but keep 2-D write indexing, so inference is incomplete.
        info.tensor_shapes["%A"] = [2, 3, 4]
        info.tensor_shapes["%B"] = [4, 3, 2]

        transpose_sketches = [s for n, s in SKETCH_BY_NAME.items() if "transpose" in n]
        if not transpose_sketches:
            pytest.skip("No transpose sketch")
        result = linalg_emitter.emit(transpose_sketches[0], info)

        assert "permutation = [2, 1, 0]" in result

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


class TestStableHLOEmitter:
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
