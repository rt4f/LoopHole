"""Regression tests for B-11 parser corpus compatibility analysis."""

from pathlib import Path

from loophole.parser_corpus_compat import analyze_parser_compatibility


def _corpus_root() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "corpus"


def test_corpus_summary_counts_are_stable() -> None:
    summary = analyze_parser_compatibility(_corpus_root())

    assert summary.total_files == 7
    assert summary.compatible_files == 3
    assert summary.incompatible_files == 4

    assert summary.class_frequency["no_store_detected"] == 3
    assert summary.class_frequency["scf_iter_args_not_modeled"] == 1
    assert summary.class_frequency["conditional_region_not_modeled"] == 1
    assert summary.class_frequency["no_loop_construct_detected"] == 3
    assert "affine_apply_index_materialization" not in summary.class_frequency


def test_top_failure_classes_are_reproducible_per_file() -> None:
    summary = analyze_parser_compatibility(_corpus_root())
    by_file = {row.file_path: row for row in summary.files}

    assert by_file["polygeist_matmul_affine.mlir"].compatible is True
    assert by_file["polygeist_conv1d_scf.mlir"].compatible is True
    assert by_file["polygeist_affine_apply_indices.mlir"].compatible is True  # affine.apply now resolved

    assert "scf_iter_args_not_modeled" in by_file["polygeist_dot_iter_args.mlir"].classes
    assert "no_store_detected" in by_file["polygeist_dot_iter_args.mlir"].classes

    assert "conditional_region_not_modeled" in by_file["polygeist_if_guarded_store.mlir"].classes

    vectorized = by_file["frontend_no_loop_vectorized.mlir"]
    assert "no_loop_construct_detected" in vectorized.classes
    assert "no_store_detected" in vectorized.classes


def test_tracked_unsupported_corpus_misses_have_explicit_diagnostics() -> None:
    summary = analyze_parser_compatibility(_corpus_root())

    incompatible = [row for row in summary.files if not row.compatible]
    assert incompatible
    assert all(row.diagnostics_count > 0 for row in incompatible)
