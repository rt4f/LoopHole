"""
Unit tests for AffineExtractor — verifies parsing of MLIR Affine IR text.
"""
import pytest
from loophole.affine_extractor import AffineExtractor
from loophole.tests.fixtures import (
    MATMUL_MLIR,
    TRANSPOSE_2D_MLIR,
    CONV_1D_MLIR,
    CONV_2D_SIMPLE_MLIR,
    DOT_PRODUCT_MLIR,
    MATVEC_MLIR,
    ELEMENTWISE_ADD_MLIR,
    REDUCE_SUM_MLIR,
    MIXED_AFFINE_SCF_MATMUL_MLIR,
    MIXED_SCF_AFFINE_REDUCTION_MLIR,
    INDEX_VARIATION_EQ_A_MLIR,
    INDEX_VARIATION_EQ_B_MLIR,
)


@pytest.fixture
def extractor():
    return AffineExtractor()


class TestMatmulParsing:
    def test_loop_count(self, extractor):
        info = extractor.extract(MATMUL_MLIR)
        assert len(info.loop_order) == 3

    def test_loop_bounds(self, extractor):
        info = extractor.extract(MATMUL_MLIR)
        bounds = [info.bounds[iv][1] for iv in info.loop_order]
        assert bounds == [4, 4, 4]

    def test_loop_iv_names(self, extractor):
        info = extractor.extract(MATMUL_MLIR)
        names = info.loop_order
        assert names == ["i", "j", "k"]

    def test_parallel_iv_count(self, extractor):
        info = extractor.extract(MATMUL_MLIR)
        assert len(info.parallel_vars) == 2

    def test_reduction_iv_count(self, extractor):
        info = extractor.extract(MATMUL_MLIR)
        assert len(info.reduction_vars) == 1

    def test_has_mulf_addf(self, extractor):
        info = extractor.extract(MATMUL_MLIR)
        ops = [op.op_type for op in info.compute_ops]
        assert "mulf" in ops or any("mulf" in o for o in ops)
        assert "addf" in ops or any("addf" in o for o in ops)

    def test_num_reads(self, extractor):
        info = extractor.extract(MATMUL_MLIR)
        assert len(info.reads) == 3  # A, B, C (load before update)

    def test_num_writes(self, extractor):
        info = extractor.extract(MATMUL_MLIR)
        assert len(info.writes) == 1  # C store

    def test_func_name(self, extractor):
        info = extractor.extract(MATMUL_MLIR)
        assert info.func_name == "matmul"


class TestTransposeParsing:
    def test_loop_count(self, extractor):
        info = extractor.extract(TRANSPOSE_2D_MLIR)
        assert len(info.loop_order) == 2

    def test_no_reduction_ivs(self, extractor):
        info = extractor.extract(TRANSPOSE_2D_MLIR)
        assert len(info.reduction_vars) == 0

    def test_all_parallel_ivs(self, extractor):
        info = extractor.extract(TRANSPOSE_2D_MLIR)
        assert len(info.parallel_vars) == 2

    def test_write_index_swapped(self, extractor):
        """B[j,i] means the store subscripts are reversed relative to loops."""
        info = extractor.extract(TRANSPOSE_2D_MLIR)
        assert len(info.writes) == 1
        w = info.writes[0]
        # subscripts should be [j, i] — the second IV first
        assert w.index_exprs[0] != w.index_exprs[1]


class TestConv1DParsing:
    def test_loop_count(self, extractor):
        info = extractor.extract(CONV_1D_MLIR)
        assert len(info.loop_order) == 3

    def test_has_reduction_iv(self, extractor):
        info = extractor.extract(CONV_1D_MLIR)
        assert len(info.reduction_vars) >= 1

    def test_compute_ops_mul_add(self, extractor):
        info = extractor.extract(CONV_1D_MLIR)
        opcodes = [op.op_type for op in info.compute_ops]
        assert any("mulf" in o for o in opcodes)
        assert any("addf" in o for o in opcodes)


class TestConv2DSimpleParsing:
    def test_loop_count(self, extractor):
        info = extractor.extract(CONV_2D_SIMPLE_MLIR)
        assert len(info.loop_order) == 4

    def test_reduction_iv_count(self, extractor):
        info = extractor.extract(CONV_2D_SIMPLE_MLIR)
        assert len(info.reduction_vars) == 2  # kh, kw

    def test_parallel_iv_count(self, extractor):
        info = extractor.extract(CONV_2D_SIMPLE_MLIR)
        assert len(info.parallel_vars) == 2  # oh, ow


class TestDotProductParsing:
    def test_loop_count(self, extractor):
        info = extractor.extract(DOT_PRODUCT_MLIR)
        assert len(info.loop_order) == 1

    def test_single_reduction_iv(self, extractor):
        info = extractor.extract(DOT_PRODUCT_MLIR)
        assert len(info.reduction_vars) == 1
        assert len(info.parallel_vars) == 0


class TestMatvecParsing:
    def test_loop_count(self, extractor):
        info = extractor.extract(MATVEC_MLIR)
        assert len(info.loop_order) == 2

    def test_one_parallel_one_reduction(self, extractor):
        info = extractor.extract(MATVEC_MLIR)
        assert len(info.parallel_vars) == 1
        assert len(info.reduction_vars) == 1


class TestElementwiseAddParsing:
    def test_loop_count(self, extractor):
        info = extractor.extract(ELEMENTWISE_ADD_MLIR)
        assert len(info.loop_order) == 2

    def test_no_reductions(self, extractor):
        info = extractor.extract(ELEMENTWISE_ADD_MLIR)
        assert len(info.reduction_vars) == 0

    def test_addf_present(self, extractor):
        info = extractor.extract(ELEMENTWISE_ADD_MLIR)
        opcodes = [op.op_type for op in info.compute_ops]
        assert any("addf" in o for o in opcodes)


class TestReduceSumParsing:
    def test_loop_count(self, extractor):
        info = extractor.extract(REDUCE_SUM_MLIR)
        assert len(info.loop_order) == 2

    def test_one_reduction_iv(self, extractor):
        info = extractor.extract(REDUCE_SUM_MLIR)
        assert len(info.reduction_vars) == 1

    def test_one_parallel_iv(self, extractor):
        info = extractor.extract(REDUCE_SUM_MLIR)
        assert len(info.parallel_vars) == 1


class TestMixedAffineScfParsing:
    def test_mixed_matmul_loop_count(self, extractor):
        info = extractor.extract(MIXED_AFFINE_SCF_MATMUL_MLIR)
        assert len(info.loop_order) == 3

    def test_mixed_matmul_loop_order(self, extractor):
        info = extractor.extract(MIXED_AFFINE_SCF_MATMUL_MLIR)
        assert info.loop_order == ["i", "j", "k"]

    def test_mixed_matmul_bounds(self, extractor):
        info = extractor.extract(MIXED_AFFINE_SCF_MATMUL_MLIR)
        bounds = [info.bounds[iv][1] for iv in info.loop_order]
        assert bounds == [4, 4, 4]

    def test_mixed_matmul_reduction_vars(self, extractor):
        info = extractor.extract(MIXED_AFFINE_SCF_MATMUL_MLIR)
        assert len(info.reduction_vars) == 1
        assert info.reduction_vars[0] == "k"

    def test_mixed_matmul_access_extraction_stable(self, extractor):
        info = extractor.extract(MIXED_AFFINE_SCF_MATMUL_MLIR)
        assert len(info.reads) == 3
        assert len(info.writes) == 1

    def test_mixed_reduction_loop_count(self, extractor):
        info = extractor.extract(MIXED_SCF_AFFINE_REDUCTION_MLIR)
        assert len(info.loop_order) == 2

    def test_mixed_reduction_loop_order(self, extractor):
        info = extractor.extract(MIXED_SCF_AFFINE_REDUCTION_MLIR)
        assert info.loop_order == ["m", "k"]

    def test_mixed_reduction_bounds(self, extractor):
        info = extractor.extract(MIXED_SCF_AFFINE_REDUCTION_MLIR)
        bounds = [info.bounds[iv][1] for iv in info.loop_order]
        assert bounds == [4, 4]

    def test_mixed_reduction_role_classification(self, extractor):
        info = extractor.extract(MIXED_SCF_AFFINE_REDUCTION_MLIR)
        assert info.parallel_vars == ["m"]
        assert info.reduction_vars == ["k"]


class TestIndexNormalization:
    def test_equivalent_index_variants_have_same_read_index(self, extractor):
        info_a = extractor.extract(INDEX_VARIATION_EQ_A_MLIR)
        info_b = extractor.extract(INDEX_VARIATION_EQ_B_MLIR)

        assert len(info_a.reads) == 1
        assert len(info_b.reads) == 1
        assert info_a.reads[0].index_exprs == info_b.reads[0].index_exprs

    def test_equivalent_index_variants_have_same_write_index(self, extractor):
        info_a = extractor.extract(INDEX_VARIATION_EQ_A_MLIR)
        info_b = extractor.extract(INDEX_VARIATION_EQ_B_MLIR)

        assert len(info_a.writes) == 1
        assert len(info_b.writes) == 1
        assert info_a.writes[0].index_exprs == info_b.writes[0].index_exprs

    def test_normalized_index_collapses_symbolic_offsets(self, extractor):
        info = extractor.extract(INDEX_VARIATION_EQ_A_MLIR)
        idx = info.reads[0].index_exprs[0]
        # c1 + c2 should canonicalize to a literal offset.
        assert "c1" not in idx and "c2" not in idx and "%" not in idx

    def test_normalized_index_is_canonical_order(self, extractor):
        info = extractor.extract(INDEX_VARIATION_EQ_B_MLIR)
        read_idx = info.reads[0].index_exprs[0]
        write_idx = info.writes[0].index_exprs[0]
        assert read_idx == write_idx
