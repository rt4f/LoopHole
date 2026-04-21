# B-11: Real-World MLIR Corpus Parser Compatibility Pass

## Goal

Run the parser across a realistic frontend-style MLIR corpus, classify incompatibility classes with frequencies, and lock top failure classes into reproducible regression tests.

## Scope Implemented

### 1. Added corpus compatibility analyzer

New module:
- `POC_initial_demo/src/loophole/parser_corpus_compat.py`

What it does:
- runs `AffineExtractor` across every `.mlir` file under a corpus root
- computes file-level compatibility status
- classifies incompatibility classes
- produces deterministic class-frequency summary

Compatibility classes currently tracked:
- `no_loop_construct_detected`
- `no_store_detected`
- `parser_error_diagnostics`
- `scf_iter_args_not_modeled`
- `conditional_region_not_modeled`
- `non_for_loop_control_not_modeled`
- `affine_apply_index_materialization`

### 2. Added realistic corpus fixtures

New corpus fixtures:
- `POC_initial_demo/tests/fixtures/corpus/polygeist_matmul_affine.mlir`
- `POC_initial_demo/tests/fixtures/corpus/polygeist_conv1d_scf.mlir`
- `POC_initial_demo/tests/fixtures/corpus/polygeist_dot_iter_args.mlir`
- `POC_initial_demo/tests/fixtures/corpus/polygeist_if_guarded_store.mlir`
- `POC_initial_demo/tests/fixtures/corpus/polygeist_affine_apply_indices.mlir`
- `POC_initial_demo/tests/fixtures/corpus/frontend_no_loop_vectorized.mlir`

These represent a mixed compatibility set:
- compatible affine/scf loop kernels
- unsupported/missed forms (iter_args reductions, conditional regions, affine.apply index materialization, loop-less vectorized region)

### 3. Added reproducible corpus analysis script and artifacts

New script:
- `POC_initial_demo/scripts/analyze_parser_corpus_compat.py`

Generated artifacts:
- `POC_initial_demo/tests/fixtures/corpus/parser_compat_summary.json`
- `POC_initial_demo/tests/fixtures/corpus/parser_compat_summary.md`

Measured summary for current corpus:
- total files: 6
- compatible: 2
- incompatible: 4

Top incompatibility frequencies:
1. `no_loop_construct_detected`: 2
2. `no_store_detected`: 2
3. `affine_apply_index_materialization`: 1
4. `conditional_region_not_modeled`: 1
5. `scf_iter_args_not_modeled`: 1

## Test Additions (B-11)

New regression test file:
- `POC_initial_demo/tests/unit/test_parser_corpus_compat.py`

What is locked by tests:
- corpus summary totals (files, compatible/incompatible)
- class-frequency counts for top misses
- per-file reproducibility for each top incompatibility class

## Validation

Executed in `POC_initial_demo`:

- B-11 focused regression:
  - `PYTHONPATH=src python -m pytest tests/unit/test_parser_corpus_compat.py -q`
- Parser-focused regression:
  - `PYTHONPATH=src python -m pytest tests/unit/test_affine_extractor.py tests/unit/test_parser_diagnostics.py tests/unit/test_parser_corpus_compat.py -q`
- Full regression:
  - `PYTHONPATH=src python -m pytest tests/ -q`

Latest result:
- `239 passed, 1 skipped`

## Acceptance Mapping (B-11)

- Action: run parser across realistic frontend-generated MLIR corpus and capture incompatibility classes.
  - Done: analyzer + corpus run + JSON/Markdown summaries.
- Action: produce compatibility summary with categorized misses and frequency in docs/B lane reports.
  - Done: summary artifacts generated and this B-11 report records frequency results.
- Test addition: add corpus-derived parser regression fixtures for top failure classes.
  - Done: new corpus fixtures and dedicated regression tests.
- Done when: known corpus incompatibilities are tracked and reproducible in regression tests.
  - Done: class frequencies and per-file class membership are test-locked.
