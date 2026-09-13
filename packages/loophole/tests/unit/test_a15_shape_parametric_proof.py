"""
A-15: Tests for shape-parametric proof pilot.

Validates:
  1. VerificationReport.parametric_result and parametric_notes default to None/"".
  2. _verify_parametric returns a valid CheckResult for canonical kernels.
  3. The parametric notes mention the parametrised dimension names.
  4. Lifter(parametric_mode=True) populates parametric_result after a successful concrete proof.
  5. Parametric mode does not change the concrete proof result.
  6. Pilot kernels (matmul, matvec, reduce_sum) do not produce ENCODE_ERROR.
"""
import pytest

from loophole.affine_extractor import AffineExtractor
from loophole.lifter import Lifter
from loophole.sketch_library import SKETCH_BY_NAME
from loophole.tests.fixtures import MATMUL_MLIR, MATVEC_MLIR, REDUCE_SUM_MLIR
from loophole.z3_checker import CheckResult, VerificationReport, Z3EquivalenceChecker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor():
    return AffineExtractor()


@pytest.fixture
def checker():
    return Z3EquivalenceChecker(timeout_ms=10_000)


@pytest.fixture
def matmul_loop(extractor):
    return extractor.extract(MATMUL_MLIR)


@pytest.fixture
def matvec_loop(extractor):
    return extractor.extract(MATVEC_MLIR)


@pytest.fixture
def reduce_sum_loop(extractor):
    return extractor.extract(REDUCE_SUM_MLIR)


@pytest.fixture
def matmul_sketch():
    return SKETCH_BY_NAME["linalg.matmul"]


# ---------------------------------------------------------------------------
# Test 1: VerificationReport parametric fields default
# ---------------------------------------------------------------------------

class TestVerificationReportParametricFields:
    def test_parametric_result_default_none(self):
        report = VerificationReport(
            result=CheckResult.EQUIVALENT,
            sketch_name="linalg.matmul",
            elapsed_ms=5.0,
        )
        assert report.parametric_result is None

    def test_parametric_notes_default_empty(self):
        report = VerificationReport(
            result=CheckResult.EQUIVALENT,
            sketch_name="linalg.matmul",
            elapsed_ms=5.0,
        )
        assert report.parametric_notes == ""

    def test_checker_check_does_not_populate_parametric_fields(self, checker, matmul_loop, matmul_sketch):
        report = checker.check(matmul_loop, matmul_sketch)
        assert report.parametric_result is None, (
            "checker.check() should NOT set parametric_result — that's only via _verify_parametric()"
        )
        assert report.parametric_notes == ""

    def test_parametric_fields_are_settable(self):
        report = VerificationReport(
            result=CheckResult.EQUIVALENT,
            sketch_name="test",
            elapsed_ms=1.0,
        )
        report.parametric_result = CheckResult.TIMEOUT
        report.parametric_notes = "parametric over dims=['M', 'N']"
        assert report.parametric_result == CheckResult.TIMEOUT
        assert "parametric" in report.parametric_notes


# ---------------------------------------------------------------------------
# Test 2: _verify_parametric returns a valid result
# ---------------------------------------------------------------------------

_VALID_RESULTS = {
    CheckResult.EQUIVALENT,
    CheckResult.TIMEOUT,
    CheckResult.UNKNOWN,
    CheckResult.NOT_EQUIVALENT,
    CheckResult.ENCODE_ERROR,
}


class TestVerifyParametric:
    def test_matmul_parametric_returns_valid_result(self, checker, matmul_loop, matmul_sketch):
        report = checker._verify_parametric(matmul_loop, matmul_sketch)
        assert report.result in _VALID_RESULTS, (
            f"Unexpected parametric result for matmul: {report.result}"
        )

    def test_parametric_returns_verification_report(self, checker, matmul_loop, matmul_sketch):
        report = checker._verify_parametric(matmul_loop, matmul_sketch)
        assert isinstance(report, VerificationReport)

    def test_matmul_parametric_not_structural_mismatch(self, checker, matmul_loop, matmul_sketch):
        report = checker._verify_parametric(matmul_loop, matmul_sketch)
        assert report.result != CheckResult.STRUCTURAL_MISMATCH, (
            "Structural mismatch should be caught before parametric attempt"
        )

    def test_matvec_parametric_returns_valid_result(self, checker, matvec_loop):
        matvec_sketches = [s for n, s in SKETCH_BY_NAME.items() if "matvec" in n]
        if not matvec_sketches:
            pytest.skip("No matvec sketch found")
        report = checker._verify_parametric(matvec_loop, matvec_sketches[0])
        assert report.result in _VALID_RESULTS

    def test_reduce_sum_parametric_returns_valid_result(self, checker, reduce_sum_loop):
        reduce_sketches = [s for n, s in SKETCH_BY_NAME.items() if "reduce" in n and "linalg" in n]
        if not reduce_sketches:
            pytest.skip("No linalg reduce sketch found")
        report = checker._verify_parametric(reduce_sum_loop, reduce_sketches[0])
        assert report.result in _VALID_RESULTS


# ---------------------------------------------------------------------------
# Test 3: Parametric notes mention dimension names
# ---------------------------------------------------------------------------

class TestParametricNotes:
    def test_parametric_note_mentions_parametric(self, checker, matmul_loop, matmul_sketch):
        report = checker._verify_parametric(matmul_loop, matmul_sketch)
        assert "parametric" in report.notes.lower(), (
            f"Notes should mention 'parametric'; got: {report.notes}"
        )

    def test_parametric_note_mentions_dims_when_successful(self, checker, matmul_loop, matmul_sketch):
        report = checker._verify_parametric(matmul_loop, matmul_sketch)
        if report.result != CheckResult.UNKNOWN:
            assert "dims=" in report.notes or "dims" in report.notes.lower(), (
                f"Notes should mention dims; got: {report.notes}"
            )

    def test_collect_shape_dims_returns_symbol_names(self, checker, matmul_loop):
        dims = checker._collect_shape_dims(matmul_loop)
        assert isinstance(dims, dict)
        assert all(isinstance(k, int) for k in dims.keys())
        assert all(isinstance(v, str) for v in dims.values())

    def test_collect_shape_dims_uses_mnk_names(self, checker, matmul_loop):
        dims = checker._collect_shape_dims(matmul_loop)
        for sym in dims.values():
            assert sym in "MNKPQRS", f"Unexpected symbol name: {sym}"

    def test_collect_shape_dims_non_empty_for_matmul(self, checker, matmul_loop):
        dims = checker._collect_shape_dims(matmul_loop)
        assert len(dims) >= 1, "Should find at least one parametrisable shape dim for matmul"


# ---------------------------------------------------------------------------
# Test 4: Lifter(parametric_mode=True) populates field after successful concrete proof
# ---------------------------------------------------------------------------

class TestLifterParametricMode:
    def test_lifter_parametric_mode_flag_accepted(self):
        lifter = Lifter(parametric_mode=True)
        assert lifter.parametric_mode is True

    def test_lifter_parametric_mode_default_false(self):
        lifter = Lifter()
        assert lifter.parametric_mode is False

    def test_from_policy_profile_parametric_mode(self):
        lifter = Lifter.from_policy_profile(parametric_mode=True)
        assert lifter.parametric_mode is True

    def test_lifter_parametric_populates_field_on_success(self):
        lifter = Lifter(parametric_mode=True, z3_timeout_ms=10_000)
        result = lifter.lift(MATMUL_MLIR)
        if result.success and result.verification:
            assert result.verification.parametric_result is not None, (
                "parametric_result should be populated after a successful concrete proof with parametric_mode=True"
            )

    def test_lifter_parametric_populates_notes_on_success(self):
        lifter = Lifter(parametric_mode=True, z3_timeout_ms=10_000)
        result = lifter.lift(MATMUL_MLIR)
        if result.success and result.verification and result.verification.parametric_result is not None:
            assert isinstance(result.verification.parametric_notes, str)


# ---------------------------------------------------------------------------
# Test 5: Parametric mode does not change the concrete proof result
# ---------------------------------------------------------------------------

class TestParametricModeDoesNotChangeConcreteResult:
    def test_matmul_concrete_result_unchanged(self):
        lifter_normal = Lifter(parametric_mode=False, z3_timeout_ms=10_000)
        lifter_param = Lifter(parametric_mode=True, z3_timeout_ms=10_000)

        r_normal = lifter_normal.lift(MATMUL_MLIR)
        r_param = lifter_param.lift(MATMUL_MLIR)

        assert r_normal.success == r_param.success, (
            "Parametric mode must not change whether lift succeeds"
        )
        assert r_normal.sketch_name == r_param.sketch_name, (
            "Parametric mode must not change matched sketch"
        )
        if r_normal.verification and r_param.verification:
            assert r_normal.verification.result == r_param.verification.result, (
                "Parametric mode must not change the concrete Z3 result"
            )

    def test_matvec_concrete_result_unchanged(self):
        lifter_normal = Lifter(parametric_mode=False, z3_timeout_ms=10_000)
        lifter_param = Lifter(parametric_mode=True, z3_timeout_ms=10_000)

        r_normal = lifter_normal.lift(MATVEC_MLIR)
        r_param = lifter_param.lift(MATVEC_MLIR)

        assert r_normal.success == r_param.success

    def test_parametric_result_none_if_concrete_not_proved(self):
        # If concrete proof doesn't succeed, parametric should NOT be attempted
        lifter = Lifter(parametric_mode=True, z3_timeout_ms=1)  # tiny timeout
        result = lifter.lift(MATMUL_MLIR)
        if not result.success and result.verification:
            # Only EQUIVALENT result triggers parametric
            if result.verification.result != CheckResult.EQUIVALENT:
                assert result.verification.parametric_result is None, (
                    "Parametric should not be attempted if concrete proof did not return EQUIVALENT"
                )


# ---------------------------------------------------------------------------
# Test 6: Pilot kernels don't produce ENCODE_ERROR from parametric attempt
# ---------------------------------------------------------------------------

class TestPilotKernelsParametricRate:
    def _run_parametric(self, checker, extractor, mlir_text, sketch_name=None):
        loop = extractor.extract(mlir_text)
        if sketch_name:
            sketch = SKETCH_BY_NAME.get(sketch_name)
        else:
            from loophole.z3_checker import structural_match
            sketch = next(
                (s for n, s in SKETCH_BY_NAME.items() if structural_match(loop, s)),
                None,
            )
        if sketch is None:
            return None
        return checker._verify_parametric(loop, sketch)

    def test_matmul_no_encode_error(self, checker, extractor):
        report = self._run_parametric(checker, extractor, MATMUL_MLIR, "linalg.matmul")
        if report is not None:
            assert report.result != CheckResult.ENCODE_ERROR, (
                f"matmul parametric should not ENCODE_ERROR; got: {report.notes}"
            )

    def test_matvec_no_encode_error(self, checker, extractor):
        report = self._run_parametric(checker, extractor, MATVEC_MLIR)
        if report is not None:
            assert report.result != CheckResult.ENCODE_ERROR, (
                f"matvec parametric should not ENCODE_ERROR; got: {report.notes}"
            )

    def test_reduce_sum_no_encode_error(self, checker, extractor):
        report = self._run_parametric(checker, extractor, REDUCE_SUM_MLIR)
        if report is not None:
            assert report.result != CheckResult.ENCODE_ERROR, (
                f"reduce_sum parametric should not ENCODE_ERROR; got: {report.notes}"
            )

    def test_at_least_one_pilot_kernel_attempted(self, checker, extractor):
        results = []
        for mlir, sketch_name in [
            (MATMUL_MLIR, "linalg.matmul"),
            (MATVEC_MLIR, None),
            (REDUCE_SUM_MLIR, None),
        ]:
            r = self._run_parametric(checker, extractor, mlir, sketch_name)
            if r is not None:
                results.append(r.result)

        assert len(results) >= 1, "At least one pilot kernel should be parametrically attempted"
        for r in results:
            assert r in _VALID_RESULTS
