"""
A-14: Tests for counterexample replay utility and triage workflow.

Validates:
  1. parse_z3_binding handles integers, rationals, floats, rejects unparseable values.
  2. replay_counterexample returns None for empty bindings.
  3. replay_counterexample returns a CounterexampleReport for non-empty bindings.
  4. IV values are correctly parsed from counterexample_bindings.
  5. Source access patterns are populated from loop info.
  6. Sketch access patterns are populated from indexing maps.
"""
import pytest

from loophole.affine_extractor import (
    AccessPattern, AffineExtractor, ComputeOp, LoopNestInfo,
)
from loophole.replay_checker import (
    CounterexampleReport,
    _eval_float_expr,
    parse_z3_binding,
    replay_counterexample,
    simulate_sketch,
    simulate_source,
)
from loophole.sketch_library import SKETCH_BY_NAME
from loophole.tests.fixtures import MATMUL_MLIR
from loophole.z3_checker import CheckResult, VerificationReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor():
    return AffineExtractor()


@pytest.fixture
def matmul_loop(extractor):
    return extractor.extract(MATMUL_MLIR)


@pytest.fixture
def matmul_sketch():
    return SKETCH_BY_NAME["linalg.matmul"]


def _make_empty_report(sketch_name: str = "linalg.matmul") -> VerificationReport:
    return VerificationReport(
        result=CheckResult.NOT_EQUIVALENT,
        sketch_name=sketch_name,
        elapsed_ms=0.0,
        counterexample_bindings={},
    )


def _make_report_with_bindings(
    bindings: dict,
    failed_implication: str = "source_to_sketch",
    mismatch_summary: str = None,
) -> VerificationReport:
    return VerificationReport(
        result=CheckResult.NOT_EQUIVALENT,
        sketch_name="linalg.matmul",
        elapsed_ms=10.0,
        counterexample_bindings=bindings,
        failed_implication=failed_implication,
        mismatch_summary=mismatch_summary,
    )


# ---------------------------------------------------------------------------
# Test 1: parse_z3_binding
# ---------------------------------------------------------------------------

class TestParseZ3Binding:
    def test_integer(self):
        assert parse_z3_binding("42") == 42.0

    def test_negative_integer(self):
        assert parse_z3_binding("-3") == -3.0

    def test_zero(self):
        assert parse_z3_binding("0") == 0.0

    def test_rational_one_third(self):
        result = parse_z3_binding("1/3")
        assert abs(result - 1 / 3) < 1e-9

    def test_rational_negative(self):
        result = parse_z3_binding("-7/4")
        assert abs(result - (-7 / 4)) < 1e-9

    def test_float_string(self):
        assert parse_z3_binding("5.0") == 5.0

    def test_float_decimal(self):
        assert abs(parse_z3_binding("2.5") - 2.5) < 1e-9

    def test_unparseable_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_z3_binding("<unavailable>")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_z3_binding("")

    def test_symbol_raises(self):
        with pytest.raises(ValueError):
            parse_z3_binding("x + y")


# ---------------------------------------------------------------------------
# Test 2: _eval_float_expr
# ---------------------------------------------------------------------------

class TestEvalFloatExpr:
    def test_integer_literal(self):
        assert _eval_float_expr("3", {}) == 3.0

    def test_variable_lookup(self):
        assert _eval_float_expr("i", {"i": 2.0}) == 2.0

    def test_addition(self):
        assert _eval_float_expr("i + 1", {"i": 3.0}) == 4.0

    def test_subtraction(self):
        assert _eval_float_expr("i - 2", {"i": 5.0}) == 3.0

    def test_missing_variable_returns_none(self):
        assert _eval_float_expr("unknown", {}) is None

    def test_complex_expr(self):
        assert _eval_float_expr("i + kh", {"i": 2.0, "kh": 1.0}) == 3.0


# ---------------------------------------------------------------------------
# Test 3: replay_counterexample returns None for empty bindings
# ---------------------------------------------------------------------------

class TestReplayNoneOnEmptyBindings:
    def test_none_when_no_bindings(self, matmul_loop, matmul_sketch):
        report = _make_empty_report()
        result = replay_counterexample(matmul_loop, matmul_sketch, report)
        assert result is None

    def test_none_for_equivalent_result_no_bindings(self, matmul_loop, matmul_sketch):
        report = VerificationReport(
            result=CheckResult.EQUIVALENT,
            sketch_name="linalg.matmul",
            elapsed_ms=5.0,
            counterexample_bindings={},
        )
        assert replay_counterexample(matmul_loop, matmul_sketch, report) is None


# ---------------------------------------------------------------------------
# Test 4: replay_counterexample returns CounterexampleReport for non-empty bindings
# ---------------------------------------------------------------------------

class TestReplayWithBindings:
    def test_returns_counterexample_report(self, matmul_loop, matmul_sketch):
        bindings = {"i": "2", "j": "1", "k": "3"}
        report = _make_report_with_bindings(bindings)
        ce = replay_counterexample(matmul_loop, matmul_sketch, report)
        assert ce is not None
        assert isinstance(ce, CounterexampleReport)

    def test_iv_values_correctly_parsed(self, matmul_loop, matmul_sketch):
        bindings = {"i": "2", "j": "1", "k": "3"}
        report = _make_report_with_bindings(bindings)
        ce = replay_counterexample(matmul_loop, matmul_sketch, report)
        assert ce.iv_values.get("i") == 2.0
        assert ce.iv_values.get("j") == 1.0
        assert ce.iv_values.get("k") == 3.0

    def test_rational_iv_values_parsed(self, matmul_loop, matmul_sketch):
        bindings = {"i": "1/2", "j": "0", "k": "1"}
        report = _make_report_with_bindings(bindings)
        ce = replay_counterexample(matmul_loop, matmul_sketch, report)
        assert abs(ce.iv_values.get("i", -1) - 0.5) < 1e-9

    def test_verdict_summary_is_non_empty(self, matmul_loop, matmul_sketch):
        bindings = {"i": "0", "j": "0", "k": "0"}
        report = _make_report_with_bindings(bindings, failed_implication="source_to_sketch")
        ce = replay_counterexample(matmul_loop, matmul_sketch, report)
        assert ce.verdict_summary
        assert len(ce.verdict_summary) > 10

    def test_verdict_summary_mentions_failed_implication(self, matmul_loop, matmul_sketch):
        bindings = {"i": "2", "j": "1", "k": "0"}
        report = _make_report_with_bindings(bindings, failed_implication="sketch_to_source")
        ce = replay_counterexample(matmul_loop, matmul_sketch, report)
        assert "sketch_to_source" in ce.verdict_summary

    def test_max_abs_diff_is_none_without_tensor_values(self, matmul_loop, matmul_sketch):
        bindings = {"i": "0", "j": "0", "k": "0"}
        report = _make_report_with_bindings(bindings)
        ce = replay_counterexample(matmul_loop, matmul_sketch, report)
        assert ce.max_abs_diff is None

    def test_parse_errors_recorded_for_tensor_functions(self, matmul_loop, matmul_sketch):
        # Z3 model entries like "A -> [...]" are not scalar IVs and should be skipped
        bindings = {"i": "1", "j": "2", "A": "[(const 0.5)]"}
        report = _make_report_with_bindings(bindings)
        ce = replay_counterexample(matmul_loop, matmul_sketch, report)
        assert "i" in ce.iv_values
        assert "A" not in ce.iv_values  # tensor function skipped, not an IV


# ---------------------------------------------------------------------------
# Test 5: simulate_source — access patterns populated
# ---------------------------------------------------------------------------

class TestSimulateSource:
    def test_source_access_pattern_has_tensors(self, matmul_loop):
        iv_vals = {"i": 2.0, "j": 1.0, "k": 3.0}
        pat = simulate_source(matmul_loop, iv_vals)
        assert len(pat) > 0, "Expected at least one tensor in source access pattern"
        for tensor, idxs in pat.items():
            assert isinstance(idxs, list)
            assert len(idxs) > 0

    def test_source_output_tensor_present(self, matmul_loop):
        iv_vals = {"i": 1.0, "j": 1.0, "k": 0.0}
        pat = simulate_source(matmul_loop, iv_vals)
        out = matmul_loop.output_tensor
        assert out in pat, f"Output tensor '{out}' should appear in source access pattern"

    def test_source_index_tuples_are_evaluated(self, matmul_loop):
        iv_vals = {"i": 2.0, "j": 3.0, "k": 1.0}
        pat = simulate_source(matmul_loop, iv_vals)
        # For matmul: %C[i, j] → index (2, 3)
        out = matmul_loop.output_tensor
        if out in pat:
            assert (2, 3) in pat[out] or any(t[0] == 2 for t in pat[out] if t)


# ---------------------------------------------------------------------------
# Test 6: simulate_sketch — access patterns populated
# ---------------------------------------------------------------------------

class TestSimulateSketch:
    def test_sketch_access_pattern_has_entries(self, matmul_loop, matmul_sketch):
        iv_vals = {"i": 1.0, "j": 2.0, "k": 0.0}
        pat = simulate_sketch(matmul_sketch, matmul_loop, iv_vals)
        assert len(pat) > 0, "Expected at least one tensor in sketch access pattern"

    def test_sketch_access_indices_computed(self, matmul_loop, matmul_sketch):
        iv_vals = {"i": 1.0, "j": 2.0, "k": 3.0}
        pat = simulate_sketch(matmul_sketch, matmul_loop, iv_vals)
        for tensor, idxs in pat.items():
            assert isinstance(idxs, list)


# ---------------------------------------------------------------------------
# Test 7: CounterexampleReport fields
# ---------------------------------------------------------------------------

class TestCounterexampleReportFields:
    def test_report_has_all_fields(self, matmul_loop, matmul_sketch):
        bindings = {"i": "0", "j": "1", "k": "2"}
        report = _make_report_with_bindings(bindings)
        ce = replay_counterexample(matmul_loop, matmul_sketch, report)

        assert hasattr(ce, "iv_values")
        assert hasattr(ce, "source_access_pattern")
        assert hasattr(ce, "sketch_access_pattern")
        assert hasattr(ce, "parse_errors")
        assert hasattr(ce, "diverging_cells")
        assert hasattr(ce, "max_abs_diff")
        assert hasattr(ce, "verdict_summary")

    def test_parse_errors_is_list(self, matmul_loop, matmul_sketch):
        bindings = {"i": "0"}
        report = _make_report_with_bindings(bindings)
        ce = replay_counterexample(matmul_loop, matmul_sketch, report)
        assert isinstance(ce.parse_errors, list)

    def test_diverging_cells_is_list(self, matmul_loop, matmul_sketch):
        bindings = {"i": "0"}
        report = _make_report_with_bindings(bindings)
        ce = replay_counterexample(matmul_loop, matmul_sketch, report)
        assert isinstance(ce.diverging_cells, list)
