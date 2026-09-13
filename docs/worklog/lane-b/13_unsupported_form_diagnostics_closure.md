# B-12: Unsupported-Form Diagnostics Closure for Corpus Misses

## Goal

Close top parser miss categories from B-11 by adding explicit unsupported-form diagnostics with remediation guidance, and lock these diagnostics with targeted tests.

## B-11 Closure Checklist Used

Tracked incompatibility classes carried forward from B-11:

1. `scf_iter_args_not_modeled`
2. `conditional_region_not_modeled`
3. `affine_apply_index_materialization`
4. `no_loop_construct_detected`
5. `no_store_detected`

## Implementation Summary

### 1. Added explicit unsupported-form diagnostics in parser extraction flow

Updated:
- `packages/loophole/src/loophole/affine_extractor.py`

Changes:
- Added `_add_unsupported_form_diagnostics(...)` and invoked it in `extract(...)`.
- Added explicit warnings plus remediation guidance for:
  - unsupported `scf.for ... iter_args(...)` form
  - unsupported conditional regions (`affine.if`, `scf.if`)
  - unsupported `affine.apply` index materialization
  - missing loop nest (`no affine.for/scf.for`)
  - missing explicit output stores

This closes silent-miss behavior by emitting diagnostics for all tracked unsupported corpus forms.

### 2. Added targeted diagnostics tests per tracked unsupported class

New test file:
- `packages/loophole/tests/unit/test_parser_unsupported_form_diagnostics.py`

Assertions added:
- each tracked unsupported corpus form produces an explicit diagnostic reason
- each unsupported form includes remediation guidance text

### 3. Strengthened corpus compatibility test for explicit diagnostics

Updated:
- `packages/loophole/tests/unit/test_parser_corpus_compat.py`

Added assertion:
- all incompatible corpus files must have `diagnostics_count > 0`

## Corpus Summary After B-12

Regenerated:
- `packages/loophole/tests/fixtures/corpus/parser_compat_summary.json`
- `packages/loophole/tests/fixtures/corpus/parser_compat_summary.md`

Compatibility counts remain intentionally unchanged (B-12 is diagnostics closure, not parser-support expansion):
- total files: 6
- compatible: 2
- incompatible: 4

Diagnostics are now explicit on all incompatible files:
- `frontend_no_loop_vectorized.mlir`: diagnostics_count = 3
- `polygeist_affine_apply_indices.mlir`: diagnostics_count = 1
- `polygeist_dot_iter_args.mlir`: diagnostics_count = 4
- `polygeist_if_guarded_store.mlir`: diagnostics_count = 1

## Acceptance Mapping (B-12)

- Action: Close top parser miss categories from B-11 by adding explicit unsupported-form diagnostics.
  - Done: explicit diagnostics added for all tracked classes.
- Action: Ensure each unsupported class reports clear remediation guidance.
  - Done: guidance text added per unsupported class.
- Test addition: Add targeted diagnostics assertions for each tracked unsupported corpus form.
  - Done: new targeted unit tests plus corpus-level diagnostic count assertion.
- Done when: Tracked corpus misses no longer fail silently and all have explicit diagnostics.
  - Done: all incompatible tracked corpus files now report diagnostics (`diagnostics_count > 0`).

## Validation

Executed in `POC_initial_demo`:

- Focused diagnostics closure tests:
  - `PYTHONPATH=src python -m pytest tests/unit/test_parser_unsupported_form_diagnostics.py tests/unit/test_parser_corpus_compat.py tests/unit/test_parser_diagnostics.py -q`
- Full regression:
  - `PYTHONPATH=src python -m pytest tests/ -q`

Latest result:
- `247 passed, 1 skipped`
