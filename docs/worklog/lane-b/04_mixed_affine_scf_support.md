# 13 - B-03 Mixed Affine/SCF Parser Support

Date: 2026-04-03
Objective: Extend loop extraction to consistently parse mixed `affine.for` and `scf.for` nests while preserving deterministic loop semantics.

---

## A) Scope

Task B-03 required:

1. Extend loop structure extraction in `affine_extractor.py` for mixed nests.
2. Ensure read/write extraction remains stable for non-trivial nesting order.
3. Add mixed-loop fixtures in `fixtures.py`.
4. Add parser expectations in `test_affine_extractor.py` for loop count, order, reduction vars, and bounds.

---

## B) Code Changes

### 1) Loop extraction logic

File: `packages/loophole/src/loophole/affine_extractor.py`

- Updated `_parse_loop_structure` to attempt both parsers on every loop line in source order:
  - First `affine.for`
  - If not matched, then `scf.for`
- Removed previous gating behavior that only tried `scf.for` before first discovered loop.
- Mixed-loop detection remains logged (debug diagnostic), but parsing is now supported deterministically.

Behavioral result:

- Mixed nests like `affine -> scf -> affine` and `scf -> affine` now produce stable `loop_order`, `bounds`, and IV mapping.

### 2) Mixed-loop fixtures

File: `packages/loophole/src/loophole/tests/fixtures.py`

Added fixtures:

- `MIXED_AFFINE_SCF_MATMUL_MLIR`
  - Structure: `affine.for %i` -> `scf.for %j` -> `affine.for %k`
  - Semantics: matmul-like update with `%k` reduction

- `MIXED_SCF_AFFINE_REDUCTION_MLIR`
  - Structure: `scf.for %m` -> `affine.for %k`
  - Semantics: row-wise reduction with `%k` reduction variable

Also added both fixtures to `ALL_FIXTURES`:

- `mixed_affine_scf_matmul`
- `mixed_scf_affine_reduction`

### 3) Parser expectation tests

File: `packages/loophole/tests/unit/test_affine_extractor.py`

Added class `TestMixedAffineScfParsing` with assertions for:

- loop count
- loop order
- bounds
- reduction and parallel role classification
- read/write extraction stability

---

## C) Validation

Executed tests:

1. `python -m pytest tests/unit/test_affine_extractor.py -q`
   - Result: 38 passed

2. `python -m pytest tests/ -q`
   - Result: 133 passed, 1 skipped, 1 xfailed

No regressions were introduced.

---

## D) Done Criteria Check

Requirement: Mixed affine/scf fixtures parse deterministically and keep stable loop semantics.

Status: Done

Evidence:

- Deterministic mixed-loop parsing is implemented in `_parse_loop_structure`.
- Dedicated mixed-loop fixtures added to shared fixture source.
- Parser expectations validate loop count/order/reduction/bounds and access stability.
- Full suite remains green.