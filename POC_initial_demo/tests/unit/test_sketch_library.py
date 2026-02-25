"""
Unit tests for the sketch library — verifies each sketch has valid structure.
"""
import pytest
from loophole.sketch_library import (
    SKETCH_LIBRARY,
    SKETCH_BY_NAME,
    LINALG_SKETCHES,
    STABLEHLO_SKETCHES,
    OperationSketch,
)


class TestSketchLibraryStructure:
    def test_library_nonempty(self):
        assert len(SKETCH_LIBRARY) > 0

    def test_all_are_operation_sketches(self):
        for sketch in SKETCH_LIBRARY:
            assert isinstance(sketch, OperationSketch)

    def test_by_name_matches_library(self):
        for sketch in SKETCH_LIBRARY:
            assert sketch.name in SKETCH_BY_NAME
            assert SKETCH_BY_NAME[sketch.name] is sketch

    def test_names_unique(self):
        names = [s.name for s in SKETCH_LIBRARY]
        assert len(names) == len(set(names)), "Duplicate sketch names found"

    def test_linalg_and_stablehlo_subsets(self):
        all_names = {s.name for s in SKETCH_LIBRARY}
        for s in LINALG_SKETCHES:
            assert s.name in all_names
        for s in STABLEHLO_SKETCHES:
            assert s.name in all_names

    def test_each_sketch_has_required_fields(self):
        for sketch in SKETCH_LIBRARY:
            assert sketch.name, f"Empty name in sketch"
            assert len(sketch.dim_names) > 0, f"{sketch.name}: empty dim_names"
            assert len(sketch.iterator_types) > 0, f"{sketch.name}: empty iterator_types"
            assert sketch.compute_payload, f"{sketch.name}: empty compute_payload"
            assert sketch.num_inputs >= 1, f"{sketch.name}: num_inputs < 1"
            assert sketch.num_outputs >= 1, f"{sketch.name}: num_outputs < 1"

    def test_iterator_types_valid(self):
        valid = {"parallel", "reduction"}
        for sketch in SKETCH_LIBRARY:
            for it in sketch.iterator_types:
                assert it in valid, f"{sketch.name}: invalid iterator type '{it}'"


class TestKeySketchPresence:
    def test_matmul_present(self):
        assert any("matmul" in n for n in SKETCH_BY_NAME.keys())

    def test_transpose_present(self):
        # may be 'transpose_2d' or similar
        names = list(SKETCH_BY_NAME.keys())
        assert any("transpose" in n for n in names)

    def test_dot_product_present(self):
        names = list(SKETCH_BY_NAME.keys())
        assert any("dot" in n for n in names)

    def test_conv1d_present(self):
        names = list(SKETCH_BY_NAME.keys())
        assert any("conv_1d" in n or "conv1d" in n for n in names)

    def test_conv2d_present(self):
        names = list(SKETCH_BY_NAME.keys())
        assert any("conv_2d" in n or "conv2d" in n for n in names)

    def test_elementwise_add_present(self):
        names = list(SKETCH_BY_NAME.keys())
        assert any("elementwise_add" in n or "add" in n for n in names)

    def test_reduce_sum_present(self):
        names = list(SKETCH_BY_NAME.keys())
        assert any("reduce_sum" in n or "reduce" in n for n in names)

    def test_matvec_present(self):
        names = list(SKETCH_BY_NAME.keys())
        assert any("matvec" in n for n in names)

    def test_stablehlo_sketches_nonempty(self):
        assert len(STABLEHLO_SKETCHES) > 0

    def test_linalg_sketches_nonempty(self):
        assert len(LINALG_SKETCHES) > 0


class TestMatmulSketch:
    def setup_method(self):
        self.sketch = SKETCH_BY_NAME["linalg.matmul"]

    def test_three_dims(self):
        assert len(self.sketch.dim_names) == 3

    def test_two_inputs_one_output(self):
        assert self.sketch.num_inputs == 2
        assert self.sketch.num_outputs == 1

    def test_one_reduction(self):
        assert self.sketch.iterator_types.count("reduction") == 1

    def test_two_parallel(self):
        assert self.sketch.iterator_types.count("parallel") == 2

    def test_mul_and_add_in_payload(self):
        payload = str(self.sketch.compute_payload).lower()
        # compute_payload is an enum like multiply_accumulate; check for multiply
        assert "mul" in payload or "accumulate" in payload
