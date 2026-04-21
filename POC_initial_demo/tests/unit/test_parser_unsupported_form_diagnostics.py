"""B-12 diagnostics closure tests for tracked unsupported corpus forms."""

from pathlib import Path

from loophole.affine_extractor import AffineExtractor


def _read_corpus_fixture(name: str) -> str:
    root = Path(__file__).resolve().parents[1] / "fixtures" / "corpus"
    return (root / name).read_text(encoding="utf-8")


def _diagnostic_texts(result) -> tuple[list[str], list[str]]:
    reasons = [d.reason for d in result.diagnostics]
    guidances = [d.guidance or "" for d in result.diagnostics]
    return reasons, guidances


def test_iter_args_form_has_explicit_diagnostic_and_guidance() -> None:
    ext = AffineExtractor(validate_with_mlir=False)
    result = ext.extract(_read_corpus_fixture("polygeist_dot_iter_args.mlir"))
    reasons, guidances = _diagnostic_texts(result)

    assert any("iter_args" in reason for reason in reasons)
    assert any("explicit output memref accumulation" in g for g in guidances)


def test_conditional_region_has_explicit_diagnostic_and_guidance() -> None:
    ext = AffineExtractor(validate_with_mlir=False)
    result = ext.extract(_read_corpus_fixture("polygeist_if_guarded_store.mlir"))
    reasons, guidances = _diagnostic_texts(result)

    assert any("Conditional region form" in reason for reason in reasons)
    assert any("predicatize" in g or "Specialize" in g for g in guidances)


def test_affine_apply_form_has_explicit_diagnostic_and_guidance() -> None:
    ext = AffineExtractor(validate_with_mlir=False)
    result = ext.extract(_read_corpus_fixture("polygeist_affine_apply_indices.mlir"))
    reasons, guidances = _diagnostic_texts(result)

    assert any("affine.apply" in reason for reason in reasons)
    assert any("Canonicalize affine.apply" in g for g in guidances)


def test_no_loop_and_no_store_have_explicit_diagnostics_and_guidance() -> None:
    ext = AffineExtractor(validate_with_mlir=False)
    result = ext.extract(_read_corpus_fixture("frontend_no_loop_vectorized.mlir"))
    reasons, guidances = _diagnostic_texts(result)

    assert any("No affine.for/scf.for loop nest detected" in reason for reason in reasons)
    assert any("No explicit output store detected" in reason for reason in reasons)
    assert any("Lower vectorized/region-only frontend IR" in g for g in guidances)
    assert any("writes output via affine.store/memref.store" in g for g in guidances)
