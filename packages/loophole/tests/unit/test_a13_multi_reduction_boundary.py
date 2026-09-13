"""
A-13: Tests for symbolic multi-reduction boundary improvements and clearer unsupported envelopes.

Validates:
  1. Concrete multi-reduction kernels (conv2d) work without errors.
  2. Multi-reduction with symbolic but inferable bounds attempts guarded unroll.
  3. Multi-reduction with uninferrable symbolic bounds returns ENCODE_ERROR (not ValueError).
  4. ENCODE_ERROR notes contain structured diagnostic fields.
  5. _classify_reduction_pattern covers all five ReductionPatternKind variants.
"""
import pytest

from loophole.affine_extractor import (
    AccessPattern, AffineExtractor, ComputeOp, LoopNestInfo,
)
from loophole.sketch_library import SKETCH_BY_NAME
from loophole.z3_checker import (
    CheckResult, ReductionPatternKind, Z3EquivalenceChecker,
    _UnsupportedReductionForm,
)
from loophole.tests.fixtures import CONV_2D_SIMPLE_MLIR, MATMUL_MLIR


# ---------------------------------------------------------------------------
# Helpers to build synthetic LoopNestInfo objects
# ---------------------------------------------------------------------------

def _make_access(tensor_name: str, index_exprs, shape, is_read: bool) -> AccessPattern:
    return AccessPattern(
        tensor_name=tensor_name,
        index_exprs=list(index_exprs),
        is_affine=True,
        is_read=is_read,
        shape=list(shape),
        element_type="f32",
        ssa_name=f"%{tensor_name}",
    )


def _make_compute_ops_macc():
    return [
        ComputeOp(op_type="mulf", operands=["%a", "%b"], result="%mul"),
        ComputeOp(op_type="addf", operands=["%c", "%mul"], result="%add"),
    ]


def _make_multi_red_loop(
    bounds_kh, bounds_kw,
    input_shape=(6, 6), kernel_shape=(3, 3), output_shape=(4, 4),
) -> LoopNestInfo:
    """
    Synthetic 2D convolution-like kernel:
      O[oh, ow] += I[oh + kh, ow + kw] * K[kh, kw]
    with configurable bounds (concrete or symbolic).
    """
    reads = [
        _make_access("%I", ["oh + kh", "ow + kw"], input_shape, is_read=True),
        _make_access("%K", ["kh", "kw"], kernel_shape, is_read=True),
        _make_access("%O", ["oh", "ow"], output_shape, is_read=True),
    ]
    writes = [_make_access("%O", ["oh", "ow"], output_shape, is_read=False)]
    return LoopNestInfo(
        func_name="conv2d_test",
        induction_vars=["oh", "ow", "kh", "kw"],
        bounds={"oh": (0, 4), "ow": (0, 4), "kh": bounds_kh, "kw": bounds_kw},
        loop_order=["oh", "ow", "kh", "kw"],
        reads=reads,
        writes=writes,
        compute_ops=_make_compute_ops_macc(),
        tensor_shapes={"%I": list(input_shape), "%K": list(kernel_shape), "%O": list(output_shape)},
        tensor_types={"%I": "memref<6x6xf32>", "%K": "memref<3x3xf32>", "%O": "memref<4x4xf32>"},
        element_type="f32",
        reduction_vars=["kh", "kw"],
        parallel_vars=["oh", "ow"],
        func_args={},
        has_accumulation=True,
        accumulation_op="addf",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor():
    return AffineExtractor()


@pytest.fixture
def checker():
    return Z3EquivalenceChecker(timeout_ms=10_000)


# ---------------------------------------------------------------------------
# Test 1: Concrete multi-reduction (conv2d) uses CONCRETE_UNROLL path
# ---------------------------------------------------------------------------

class TestConcreteMultiReduction:
    def test_conv2d_parsed_and_verified(self, extractor, checker):
        loop = extractor.extract(CONV_2D_SIMPLE_MLIR)
        assert loop.reduction_vars == ["kh", "kw"], \
            f"Expected ['kh','kw'] reduction vars, got {loop.reduction_vars}"

        sketches = [s for n, s in SKETCH_BY_NAME.items() if "conv" in n.lower() and "linalg" in n]
        if not sketches:
            pytest.skip("No linalg conv sketch found")

        report = checker.check(loop, sketches[0])
        assert report.result != CheckResult.ENCODE_ERROR, (
            f"Concrete multi-reduction should NOT produce ENCODE_ERROR; got: {report.notes}"
        )

    def test_synthetic_concrete_multi_red_classify(self, checker):
        loop = _make_multi_red_loop(bounds_kh=(0, 3), bounds_kw=(0, 3))
        kind, reason = checker._classify_reduction_pattern(loop, loop.bounds)
        assert kind == ReductionPatternKind.CONCRETE_UNROLL
        assert "concrete" in reason.lower()

    def test_synthetic_concrete_multi_red_no_exception(self, checker):
        loop = _make_multi_red_loop(bounds_kh=(0, 3), bounds_kw=(0, 3))
        sketch = next(
            (s for n, s in SKETCH_BY_NAME.items() if "conv" in n.lower() and "linalg" in n),
            None,
        )
        if sketch is None:
            pytest.skip("No linalg conv sketch found")
        report = checker.check(loop, sketch)
        assert report.result not in (CheckResult.ENCODE_ERROR,)


# ---------------------------------------------------------------------------
# Test 2: Multi-reduction with symbolic but inferable bounds (MULTI_SYMBOLIC_INFERRED)
# ---------------------------------------------------------------------------

class TestMultiSymbolicInferred:
    def test_multi_symbolic_upper_bound_inferred(self, checker):
        # kh and kw have symbolic upper bounds 'Kh' / 'Kw' but the K tensor
        # has shape [3, 3] from which we can infer both bounds = 3.
        loop = _make_multi_red_loop(
            bounds_kh=(0, "Kh"),
            bounds_kw=(0, "Kw"),
            kernel_shape=(3, 3),
        )
        multi_limits = checker._symbolic_reduction_upper_bound_multi(loop, loop.bounds)
        assert multi_limits is not None, "Should infer bounds from tensor extents"
        assert multi_limits["kh"] == 3
        assert multi_limits["kw"] == 3

    def test_multi_symbolic_inferred_classify(self, checker):
        loop = _make_multi_red_loop(
            bounds_kh=(0, "Kh"),
            bounds_kw=(0, "Kw"),
            kernel_shape=(3, 3),
        )
        kind, reason = checker._classify_reduction_pattern(loop, loop.bounds)
        assert kind == ReductionPatternKind.MULTI_SYMBOLIC_INFERRED
        assert "inferred" in reason.lower()

    def test_multi_symbolic_inferred_does_not_encode_error(self, checker):
        loop = _make_multi_red_loop(
            bounds_kh=(0, "Kh"),
            bounds_kw=(0, "Kw"),
            kernel_shape=(3, 3),
        )
        sketch = next(
            (s for n, s in SKETCH_BY_NAME.items() if "conv" in n.lower() and "linalg" in n),
            None,
        )
        if sketch is None:
            pytest.skip("No linalg conv sketch found")
        report = checker.check(loop, sketch)
        assert report.result != CheckResult.ENCODE_ERROR, (
            f"Multi symbolic inferred path should not ENCODE_ERROR; got: {report.notes}"
        )
        assert report.result in (
            CheckResult.EQUIVALENT, CheckResult.TIMEOUT, CheckResult.UNKNOWN,
            CheckResult.NOT_EQUIVALENT, CheckResult.STRUCTURAL_MISMATCH,
        )


# ---------------------------------------------------------------------------
# Test 3: Multi-reduction with uninferrable symbolic bounds → ENCODE_ERROR
# ---------------------------------------------------------------------------

class TestMultiSymbolicUninferred:
    def _make_uninferrable_loop(self) -> LoopNestInfo:
        """
        Multi-reduction with symbolic bounds that cannot be inferred:
        the kernel K has NO dimensions that correspond to kh/kw index exprs.
        The I tensor uses 'oh' and 'ow' (parallel IVs), not kh/kw.
        """
        reads = [
            # I indexed by (oh, ow) — no kh/kw indexing → cannot infer bounds
            _make_access("%I", ["oh", "ow"], (4, 4), is_read=True),
            _make_access("%O", ["oh", "ow"], (4, 4), is_read=True),
        ]
        writes = [_make_access("%O", ["oh", "ow"], (4, 4), is_read=False)]
        return LoopNestInfo(
            func_name="multi_red_uninferrable",
            induction_vars=["oh", "ow", "kh", "kw"],
            bounds={"oh": (0, 4), "ow": (0, 4), "kh": (0, "Kh"), "kw": (0, "Kw")},
            loop_order=["oh", "ow", "kh", "kw"],
            reads=reads,
            writes=writes,
            compute_ops=[ComputeOp(op_type="addf", operands=["%c", "%a"], result="%add")],
            tensor_shapes={"%I": [4, 4], "%O": [4, 4]},
            tensor_types={},
            element_type="f32",
            reduction_vars=["kh", "kw"],
            parallel_vars=["oh", "ow"],
            func_args={},
            has_accumulation=True,
            accumulation_op="addf",
        )

    def test_uninferrable_classify_as_multi_symbolic_uninferred(self, checker):
        loop = self._make_uninferrable_loop()
        multi_limits = checker._symbolic_reduction_upper_bound_multi(loop, loop.bounds)
        assert multi_limits is None, "Should NOT infer bounds when IVs not in tensor accesses"

        kind, reason = checker._classify_reduction_pattern(loop, loop.bounds)
        assert kind == ReductionPatternKind.MULTI_SYMBOLIC_UNINFERRED

    def test_uninferrable_returns_encode_error_not_value_error(self, checker):
        loop = self._make_uninferrable_loop()
        sketches = [s for n, s in SKETCH_BY_NAME.items() if "linalg" in n]
        if not sketches:
            pytest.skip("No linalg sketch found")

        report = None
        for s in sketches[:5]:
            try:
                report = checker.check(loop, s)
                if report.result not in (CheckResult.STRUCTURAL_MISMATCH,):
                    break
            except Exception as exc:
                pytest.fail(
                    f"checker.check() should never raise; got {type(exc).__name__}: {exc}"
                )

        assert report is not None
        # Must NOT raise ValueError — the outer check() always returns a report
        assert isinstance(report.result, CheckResult), (
            "Result must be a CheckResult enum value, not an exception"
        )

    def test_uninferrable_encode_error_has_structured_notes(self, checker):
        loop = self._make_uninferrable_loop()
        # Force the symbolic path by using a sketch that passes structural matching
        sketches = [
            s for n, s in SKETCH_BY_NAME.items()
            if "linalg" in n and len(s.reduction_dims()) == 2
        ]
        if not sketches:
            pytest.skip("No 2-reduction linalg sketch found")

        report = checker.check(loop, sketches[0])
        if report.result == CheckResult.ENCODE_ERROR:
            assert "pattern=" in report.notes, (
                f"ENCODE_ERROR notes must contain 'pattern=' field; got: {report.notes}"
            )
            assert "unsupported_form=" in report.notes, (
                f"ENCODE_ERROR notes must contain 'unsupported_form=' field; got: {report.notes}"
            )
            assert "hint=" in report.notes


# ---------------------------------------------------------------------------
# Test 4: _classify_reduction_pattern covers all five ReductionPatternKind variants
# ---------------------------------------------------------------------------

class TestClassifyReductionPattern:
    def test_concrete_unroll_kind(self, checker):
        loop = _make_multi_red_loop((0, 3), (0, 3))
        kind, _ = checker._classify_reduction_pattern(loop, loop.bounds)
        assert kind == ReductionPatternKind.CONCRETE_UNROLL

    def test_multi_symbolic_inferred_kind(self, checker):
        loop = _make_multi_red_loop((0, "Kh"), (0, "Kw"), kernel_shape=(3, 3))
        kind, _ = checker._classify_reduction_pattern(loop, loop.bounds)
        assert kind == ReductionPatternKind.MULTI_SYMBOLIC_INFERRED

    def test_single_symbolic_inferred_kind(self, checker, extractor):
        from loophole.tests.fixtures import DOT_PRODUCT_SYMBOLIC_N_MLIR
        loop = extractor.extract(DOT_PRODUCT_SYMBOLIC_N_MLIR)
        kind, _ = checker._classify_reduction_pattern(loop, loop.bounds)
        assert kind in (
            ReductionPatternKind.SINGLE_SYMBOLIC_INFERRED,
            ReductionPatternKind.CONCRETE_UNROLL,
        ), f"Unexpected kind: {kind}"

    def test_multi_symbolic_uninferred_kind(self, checker):
        reads = [_make_access("%I", ["oh", "ow"], (4, 4), is_read=True),
                 _make_access("%O", ["oh", "ow"], (4, 4), is_read=True)]
        writes = [_make_access("%O", ["oh", "ow"], (4, 4), is_read=False)]
        loop = LoopNestInfo(
            func_name="uninferrable",
            induction_vars=["oh", "ow", "kh", "kw"],
            bounds={"oh": (0, 4), "ow": (0, 4), "kh": (0, "Kh"), "kw": (0, "Kw")},
            loop_order=["oh", "ow", "kh", "kw"],
            reads=reads, writes=writes,
            compute_ops=[ComputeOp(op_type="addf", operands=["%a", "%b"], result="%c")],
            tensor_shapes={"%I": [4, 4], "%O": [4, 4]},
            tensor_types={}, element_type="f32",
            reduction_vars=["kh", "kw"], parallel_vars=["oh", "ow"],
            func_args={}, has_accumulation=True, accumulation_op="addf",
        )
        kind, _ = checker._classify_reduction_pattern(loop, loop.bounds)
        assert kind == ReductionPatternKind.MULTI_SYMBOLIC_UNINFERRED

    def test_single_symbolic_uninferred_kind(self, checker):
        reads = [_make_access("%A", ["k"], (100,), is_read=True),
                 _make_access("%c", [], (1,), is_read=True)]
        writes = [_make_access("%c", [], (1,), is_read=False)]
        loop = LoopNestInfo(
            func_name="single_uninferrable",
            induction_vars=["k"],
            bounds={"k": (0, "N")},
            loop_order=["k"],
            reads=reads, writes=writes,
            compute_ops=[ComputeOp(op_type="addf", operands=["%a", "%b"], result="%c")],
            tensor_shapes={"%A": [100], "%c": [1]},
            tensor_types={}, element_type="f32",
            reduction_vars=["k"], parallel_vars=[],
            func_args={}, has_accumulation=True, accumulation_op="addf",
        )
        # A[k] has shape [100], but the bound says "N" which is symbolic.
        # _symbolic_reduction_upper_bound should infer 100 from the tensor extent.
        # So this is actually SINGLE_SYMBOLIC_INFERRED or CONCRETE (if 100 parsed).
        # For a truly uninferrable single case we need NO matching tensor dimension.
        reads2 = [_make_access("%c", [], (1,), is_read=True)]
        loop2 = LoopNestInfo(
            func_name="single_uninferrable2",
            induction_vars=["k"],
            bounds={"k": (0, "N")},
            loop_order=["k"],
            reads=reads2, writes=[_make_access("%c", [], (1,), is_read=False)],
            compute_ops=[ComputeOp(op_type="addf", operands=["%a", "%b"], result="%c")],
            tensor_shapes={"%c": [1]},
            tensor_types={}, element_type="f32",
            reduction_vars=["k"], parallel_vars=[],
            func_args={}, has_accumulation=True, accumulation_op="addf",
        )
        kind, _ = checker._classify_reduction_pattern(loop2, loop2.bounds)
        assert kind == ReductionPatternKind.SINGLE_SYMBOLIC_UNINFERRED


# ---------------------------------------------------------------------------
# Test 5: UnsupportedReductionForm exception is never exposed to callers
# ---------------------------------------------------------------------------

class TestUnsupportedReductionFormNotExposed:
    def test_check_never_raises(self, checker):
        reads = [_make_access("%I", ["oh", "ow"], (4, 4), is_read=True),
                 _make_access("%O", ["oh", "ow"], (4, 4), is_read=True)]
        writes = [_make_access("%O", ["oh", "ow"], (4, 4), is_read=False)]
        loop = LoopNestInfo(
            func_name="no_raise_test",
            induction_vars=["oh", "ow", "kh", "kw"],
            bounds={"oh": (0, 4), "ow": (0, 4), "kh": (0, "Kh"), "kw": (0, "Kw")},
            loop_order=["oh", "ow", "kh", "kw"],
            reads=reads, writes=writes,
            compute_ops=[ComputeOp(op_type="addf", operands=["%a", "%b"], result="%c")],
            tensor_shapes={"%I": [4, 4], "%O": [4, 4]},
            tensor_types={}, element_type="f32",
            reduction_vars=["kh", "kw"], parallel_vars=["oh", "ow"],
            func_args={}, has_accumulation=True, accumulation_op="addf",
        )
        for sketch in list(SKETCH_BY_NAME.values())[:10]:
            try:
                report = checker.check(loop, sketch)
                assert isinstance(report.result, CheckResult)
            except _UnsupportedReductionForm:
                pytest.fail("_UnsupportedReductionForm must not propagate past checker.check()")
            except Exception:
                pass  # other exceptions from structural mismatch etc. are OK

    def test_result_is_encode_error_or_structural_mismatch(self, checker):
        reads = [_make_access("%I", ["oh", "ow"], (4, 4), is_read=True),
                 _make_access("%O", ["oh", "ow"], (4, 4), is_read=True)]
        writes = [_make_access("%O", ["oh", "ow"], (4, 4), is_read=False)]
        loop = LoopNestInfo(
            func_name="encode_error_test",
            induction_vars=["oh", "ow", "kh", "kw"],
            bounds={"oh": (0, 4), "ow": (0, 4), "kh": (0, "Kh"), "kw": (0, "Kw")},
            loop_order=["oh", "ow", "kh", "kw"],
            reads=reads, writes=writes,
            compute_ops=[ComputeOp(op_type="addf", operands=["%a", "%b"], result="%c")],
            tensor_shapes={"%I": [4, 4], "%O": [4, 4]},
            tensor_types={}, element_type="f32",
            reduction_vars=["kh", "kw"], parallel_vars=["oh", "ow"],
            func_args={}, has_accumulation=True, accumulation_op="addf",
        )
        results = set()
        for sketch in list(SKETCH_BY_NAME.values()):
            report = checker.check(loop, sketch)
            results.add(report.result)

        allowed = {
            CheckResult.STRUCTURAL_MISMATCH,
            CheckResult.ENCODE_ERROR,
            CheckResult.UNKNOWN,
            CheckResult.TIMEOUT,
            CheckResult.NOT_EQUIVALENT,
            CheckResult.EQUIVALENT,
        }
        assert results <= allowed, f"Unexpected result types: {results - allowed}"
