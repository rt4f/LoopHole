# 16 - B-06 Transpose Permutation Inference and Validation

Date: 2026-04-03
Objective: Strengthen transpose permutation inference to remove reverse-dims fallback, add explicit permutation validation, and enforce correctness in tests.

---

## A) Scope

B-06 required:

1. Improve permutation inference in `emitter.py` to avoid reverse fallback in ambiguous cases.
2. Add explicit permutation validation before emission.
3. Add transpose variants to fixtures (including non-reverse permutations).
4. Add targeted emitter assertions for exact permutation values.
5. Extend transpose integration checks to validate permutation correctness.

---

## B) Implementation Changes

### 1) Strict permutation inference and validation

File: `packages/loophole/src/loophole/emitter.py`

Updated transpose path:

- `_emit_transpose(...)`
  - now requires explicit input/output tensor metadata and shapes
  - infers permutation strictly
  - validates inferred permutation against shape/rank before emission

Reworked `_infer_transpose_perm(...)`:

- removed reverse-order fallback behavior
- now fails with `EmissionError` if:
  - no write pattern exists
  - write rank does not match input rank
  - output index expressions are ambiguous (not direct IV references)

Added `_validate_transpose_perm(...)`:

- checks permutation length/range/uniqueness
- checks output shape matches input shape permuted by inferred permutation
- raises `EmissionError` with actionable reason on mismatch

### 2) New transpose fixtures

File: `packages/loophole/src/loophole/tests/fixtures.py`

Added 3-D transpose fixtures with non-reverse permutations:

- `TRANSPOSE_3D_PERM_120_MLIR` -> permutation `[1, 2, 0]`
- `TRANSPOSE_3D_PERM_201_MLIR` -> permutation `[2, 0, 1]`

Both added to `ALL_FIXTURES`.

---

## C) Tests Added/Updated

### Unit tests

File: `packages/loophole/tests/unit/test_emitter.py`

Added/updated tests:

- `test_emit_transpose_exact_permutation_2d`
- `test_emit_transpose_exact_permutation_3d_120`
- `test_emit_transpose_exact_permutation_3d_201`
- `test_emit_transpose_ambiguous_inference_fails`

These assert exact emitted permutations and strict failure for ambiguous cases.

### Integration tests

File: `packages/loophole/tests/integration/test_transpose.py`

Added/updated checks:

- `test_transpose_2d_permutation_correct`
- `test_nonsquare_transpose_permutation_correct`

These ensure permutation correctness is validated in full lift pipeline tests for supported transpose sketches.

---

## D) Validation

Commands executed:

1. `python -m pytest tests/unit/test_emitter.py tests/integration/test_transpose.py tests/integration/test_conv2d.py -q`
   - Result: 31 passed

2. `python -m pytest tests/ -q`
   - Result: 161 passed, 1 skipped

No regressions introduced.

---

## E) Done Criteria Check

Requirement: Transpose emits only validated, correct permutations.

Status: Done

Evidence:

- Reverse fallback removed for ambiguous transpose inference.
- Strict inference and validation logic raises explicit errors on ambiguity/mismatch.
- Unit tests assert exact permutations for 2-D and non-reverse 3-D variants.
- Integration tests verify permutation correctness on supported transpose lift paths.