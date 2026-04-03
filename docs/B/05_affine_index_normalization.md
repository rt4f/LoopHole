# 14 - B-04 Affine Index Normalization

Date: 2026-04-03
Objective: Harden index normalization so equivalent indexing expressions map to deterministic canonical strings for matcher/verifier stability.

---

## A) Scope

B-04 required:

1. Improve index normalization in `affine_extractor.py` for:
   - `iv + offset`
   - nested symbolic constants
   - reordered equivalent forms
2. Ensure normalized output is stable for downstream matching and verification.
3. Add expression-variation fixtures and normalization equality tests.

---

## B) Implementation Changes

### 1) Normalization hardening in extractor

File: `POC_initial_demo/src/loophole/affine_extractor.py`

Added:

- `_apply_known_constants(expr, const_map)`
  - Replaces symbolic SSA constants (for example `%c1`) with literal values.

- `_canonicalize_index_expr(expr)`
  - Uses SymPy (when available) to canonicalize arithmetic forms.
  - Normalizes expressions such as `1 + i`, `i + (1 + 2)`, `2 + (1 + i)` into a consistent form.

Updated:

- `_normalize_index(...)`
  - Now applies IV substitution, constant substitution, SSA cleanup, and canonicalization.

- `_parse_load(...)` and `_parse_store(...)`
  - Now accept `const_map` and normalize indices with known constants.

- `_collect_accesses(...)`
  - Now accepts `const_map` and passes it through to load/store parsing.

- `extract(...)`
  - Now passes `const_map` into `_collect_accesses`.

Result:

- Equivalent index expressions normalize to deterministic canonical strings.
- Nested/reordered forms no longer create semantically duplicate but textually different index expressions.

### 2) Expression-variation fixtures

File: `POC_initial_demo/src/loophole/tests/fixtures.py`

Added fixtures:

- `INDEX_VARIATION_EQ_A_MLIR`
  - Uses nested symbolic constants: `%i + (%c1 + %c2)` and `(%i + 3)`.

- `INDEX_VARIATION_EQ_B_MLIR`
  - Uses reordered equivalent forms: `%c2 + (1 + %i)` and `1 + (2 + %i)`.

Added to `ALL_FIXTURES`:

- `index_variation_eq_a`
- `index_variation_eq_b`

### 3) Normalization assertions

File: `POC_initial_demo/tests/unit/test_affine_extractor.py`

Added class `TestIndexNormalization` with assertions:

- Equivalent fixtures produce equal normalized read indices.
- Equivalent fixtures produce equal normalized write indices.
- Symbolic constants are collapsed to literals in normalized strings.
- Canonicalized read/write forms match across syntactic variants.

---

## C) Validation

Commands executed:

1. `python -m pytest tests/unit/test_affine_extractor.py -q`
   - Result: 42 passed

2. `python -m pytest tests/ -q`
   - Result: 137 passed, 1 skipped, 1 xfailed

No regressions detected.

---

## D) Done Criteria Check

Requirement: Equivalent index expressions normalize to consistent canonical strings.

Status: Done

Evidence:

- Canonical normalization pipeline is implemented and exercised on equivalent expression variants.
- New fixtures cover nested symbols and reordered arithmetic forms.
- Equality assertions pass for normalized read/write indices across variants.