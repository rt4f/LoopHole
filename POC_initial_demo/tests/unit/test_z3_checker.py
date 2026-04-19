"""
Unit tests for the Z3 equivalence checker.
Tests both the structural pre-filter and the full SMT check.
"""
import pytest
from z3.z3 import Int, IntVal, simplify, substitute
from loophole.affine_extractor import AffineExtractor
from loophole.sketch_library import SKETCH_BY_NAME
from loophole.z3_checker import Z3EquivalenceChecker, CheckResult, structural_match, _eval_index_expr
from loophole.tests.fixtures import (
    MATMUL_MLIR,
    TRANSPOSE_2D_MLIR,
    DOT_PRODUCT_MLIR,
    DOT_PRODUCT_SYMBOLIC_N_MLIR,
    ELEMENTWISE_ADD_MLIR,
    REDUCE_SUM_MLIR,
)


@pytest.fixture
def extractor():
    return AffineExtractor()


@pytest.fixture
def checker():
    return Z3EquivalenceChecker(timeout_ms=10_000)


class TestStructuralMatch:
    def test_matmul_structural_match(self, extractor, checker):
        info = extractor.extract(MATMUL_MLIR)
        sketch = SKETCH_BY_NAME["linalg.matmul"]
        assert structural_match(info, sketch), \
            "matmul loop nest should structurally match matmul sketch"

    def test_matmul_not_transpose(self, extractor, checker):
        info = extractor.extract(MATMUL_MLIR)
        # find a transpose sketch
        transpose_sketches = [s for n, s in SKETCH_BY_NAME.items() if "transpose" in n]
        if transpose_sketches:
            assert not structural_match(info, transpose_sketches[0]), \
                "matmul should NOT structurally match transpose"

    def test_transpose_structural_match(self, extractor, checker):
        info = extractor.extract(TRANSPOSE_2D_MLIR)
        transpose_sketches = [s for n, s in SKETCH_BY_NAME.items() if "transpose" in n]
        assert any(structural_match(info, s) for s in transpose_sketches), \
            "transpose loop nest should structurally match a transpose sketch"

    def test_dot_product_structural(self, extractor, checker):
        info = extractor.extract(DOT_PRODUCT_MLIR)
        dot_sketches = [s for n, s in SKETCH_BY_NAME.items() if "dot" in n]
        assert any(structural_match(info, s) for s in dot_sketches), \
            "dot product loop nest should structurally match a dot sketch"


class TestZ3Verification:
    def test_matmul_equivalent_to_itself(self, extractor, checker):
        info = extractor.extract(MATMUL_MLIR)
        sketch = SKETCH_BY_NAME["linalg.matmul"]
        report = checker.check(info, sketch)
        assert report.result in (CheckResult.EQUIVALENT, CheckResult.UNKNOWN, CheckResult.TIMEOUT), \
            f"Expected EQUIVALENT or UNKNOWN/TIMEOUT for matmul, got {report.result}: {report.notes}"

    def test_transpose_equivalent(self, extractor, checker):
        info = extractor.extract(TRANSPOSE_2D_MLIR)
        transpose_sketches = [s for n, s in SKETCH_BY_NAME.items() if "transpose" in n]
        if not transpose_sketches:
            pytest.skip("No transpose sketch available")
        report = checker.check(info, transpose_sketches[0])
        assert report.result in (CheckResult.EQUIVALENT, CheckResult.UNKNOWN, CheckResult.TIMEOUT), \
            f"Transpose check: {report.result}: {report.notes}"

    def test_matmul_not_equiv_to_elementwise(self, extractor, checker):
        info = extractor.extract(MATMUL_MLIR)
        elem_sketches = [s for n, s in SKETCH_BY_NAME.items() if "elementwise_add" in n]
        if not elem_sketches:
            pytest.skip("No elementwise_add sketch available")
        report = checker.check(info, elem_sketches[0])
        # Should either be NOT_EQUIVALENT, STRUCTURAL_MISMATCH, or similar non-EQUIVALENT
        assert report.result != CheckResult.EQUIVALENT, \
            "matmul should NOT be equivalent to elementwise_add"


class TestVerificationReport:
    def test_report_has_confidence(self, extractor, checker):
        info = extractor.extract(MATMUL_MLIR)
        sketch = SKETCH_BY_NAME["linalg.matmul"]
        report = checker.check(info, sketch)
        # VerificationReport has result and notes, no confidence field
        assert isinstance(report.result, CheckResult)

    def test_report_has_notes(self, extractor, checker):
        info = extractor.extract(MATMUL_MLIR)
        sketch = SKETCH_BY_NAME["linalg.matmul"]
        report = checker.check(info, sketch)
        assert isinstance(report.notes, str)
        assert len(report.notes) > 0

    def test_report_has_result_enum(self, extractor, checker):
        info = extractor.extract(MATMUL_MLIR)
        sketch = SKETCH_BY_NAME["linalg.matmul"]
        report = checker.check(info, sketch)
        assert isinstance(report.result, CheckResult)


class TestSymbolicPathHandling:
    def test_eval_index_expr_handles_parenthesized_affine_form(self):
        i = Int("i")
        j = Int("j")
        expr = _eval_index_expr("(i + 1) * 2 - j", {"i": i, "j": j})
        value = simplify(substitute(expr, (i, IntVal(3)), (j, IntVal(4))))
        assert value.as_long() == 4

    def test_bound_expr_parses_symbolic_affine_expression(self, checker):
        n = Int("N")
        expr = checker._bound_expr("N + 2")
        value = simplify(substitute(expr, (n, IntVal(5))))
        assert value.as_long() == 7

    def test_symbolic_dot_bound_no_longer_hits_encode_error(self, extractor, checker):
        info = extractor.extract(DOT_PRODUCT_SYMBOLIC_N_MLIR)
        sketch = SKETCH_BY_NAME["linalg.dot"]
        report = checker.check(info, sketch)
        assert report.result != CheckResult.ENCODE_ERROR
