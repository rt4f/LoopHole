# 15 - B-05 Remove Silent Shape/Type Defaults in Emitter

Date: 2026-04-03
Objective: Replace guessed shape defaults in `emitter.py` with explicit validation and hard failure when required metadata is missing, with clear failure propagation via `lifter.py`.

---

## A) Scope

B-05 required:

1. Remove silent shape/type fallback behavior in emitter paths.
2. Ensure failures are explicit and preserved in `LiftResult.error`.
3. Add unit tests that assert hard failure on incomplete metadata.
4. Add integration checks for strict metadata behavior in matmul/conv2d pipelines.

---

## B) Implementation Changes

### 1) Strict metadata validation in emitter

File: `POC_initial_demo/src/loophole/emitter.py`

Added strict validators:

- `_require_output_tensor(...)`
- `_require_input_tensor(...)`
- `_require_shape(...)`

Added pre-validation hooks:

- `LinalgEmitter._validate_required_metadata(...)`
- `StableHLOEmitter._validate_required_metadata(...)`

Both `emit(...)` methods now validate required input/output tensors and shape metadata before emission. Missing metadata now raises `EmissionError` with actionable text instead of falling back to guessed defaults.

### 2) Failure propagation through lifter

File: `POC_initial_demo/src/loophole/lifter.py`

`Lifter.lift(...)` Stage 4 emission path already catches emitter exceptions and returns structured `LiftResult.error` text:

- Prefix: `Emission error while generating ...`
- Includes original failure reason from emitter

This ensures user-facing diagnostics describe why emission was blocked.

---

## C) Tests Added/Updated

### Unit tests

File: `POC_initial_demo/tests/unit/test_emitter.py`

Updates:

- Replaced fallback assertion test with strict-failure assertion:
  - `test_emit_matmul_missing_shapes_fails`
  - now expects `EmissionError` when `tensor_shapes` is missing

- Added strict conv metadata test:
  - `test_emit_conv2d_missing_output_shape_fails`
  - expects `EmissionError` when output shape metadata is missing

### Integration tests

Files:

- `POC_initial_demo/tests/integration/test_matmul.py`
- `POC_initial_demo/tests/integration/test_conv2d.py`

Added checks that intentionally remove metadata at emission stage and assert:

- lift does not succeed
- `result.error` exists
- `result.error` contains actionable emission failure text

---

## D) Validation

Commands executed:

1. `python -m pytest tests/unit/test_emitter.py -q`
   - Result: 13 passed

2. `python -m pytest tests/integration/test_matmul.py tests/integration/test_conv2d.py -q`
   - Result: 18 passed, 1 xfailed

3. `python -m pytest tests/ -q`
   - Result: 140 passed, 1 skipped, 1 xfailed

No regressions introduced.

---

## E) Done Criteria Check

Requirement: Missing metadata blocks emission and reports why.

Status: Done

Evidence:

- Emitter now hard-fails on missing required metadata via explicit validation.
- Lifter preserves and reports failure reason in `LiftResult.error`.
- Unit and integration tests assert failure behavior and actionable error text.