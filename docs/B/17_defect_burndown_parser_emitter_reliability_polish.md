# B-16: Defect Burn-Down and Parser/Emitter Reliability Polish

## Goal

Triage and close prioritized parser/emitter reliability defects, add regression guards for each fix, include guards in trusted CI lanes, and update risk/status artifacts.

## Burn-Down Scope

Prioritized high-severity reliability defects addressed in this task:

1. Static zero-dimension shape corruption
- Symptom: parser and emitter normalized dimension `0` to dynamic (`?`), which degraded shape fidelity and could mask correctness issues.
- Severity: High (metadata correctness).

2. No-SymPy convolution coefficient inference gaps
- Symptom: fallback coefficient parsing only handled `N*var` form, missing `var*N` and mixed signed-term variants; could cause avoidable convolution attr inference failures.
- Severity: High (emitter reliability under constrained environments).

## Implementation Summary

Updated:
- `POC_initial_demo/src/loophole/affine_extractor.py`
- `POC_initial_demo/src/loophole/emitter.py`

### Fix 1: Preserve static zero dimensions end-to-end

Changes:
- Parser memref dimension normalization now preserves `0` as a valid static dimension.
- Metadata reconciliation now treats `0` as a static candidate dimension (not dynamic).
- Emitter memref/tensor type formatting now preserves `0` dimensions in emitted IR.

Reliability effect:
- Shape metadata is no longer silently degraded for zero-sized tensors.

### Fix 2: Harden no-SymPy linear coefficient fallback

Changes:
- Extended regex fallback in `_infer_linear_coeff(...)` to handle:
  - `var`
  - `N*var`
  - `var*N`
  - linear combinations of signed terms where applicable

Reliability effect:
- Convolution stride/dilation inference remains stable when `sympy` is unavailable.

## Regression Test Additions

Updated:
- `POC_initial_demo/tests/unit/test_affine_extractor.py`
- `POC_initial_demo/tests/unit/test_emitter.py`
- `POC_initial_demo/src/loophole/tests/fixtures.py`

Added fixtures:
1. `MATMUL_ZERO_DIM_STATIC_MLIR`
2. `CONV_2D_VAR_TIMES_CONST_INDEX_MLIR`

Added tests:
1. `test_static_zero_dimension_is_preserved` (parser)
2. `test_emit_matmul_preserves_static_zero_dim` (emitter)
3. `test_conv_attr_inference_var_times_const_without_sympy` (emitter fallback)

## Trusted-Lane Integration

Updated:
- `.github/workflows/ci.yml`

Added CI guard step:
- `B-16 parser/emitter reliability guard checks`
- Executes the three B-16 regression tests in CI with trusted verifier policy environment enabled.

## Risk/Status Update

Status after burn-down:
- Closed in agreed scope: 2/2 prioritized high-severity parser/emitter reliability defects.
- Residual risk level (B-lane parser/emitter): Reduced from Medium to Low-Medium for currently tracked defects.

Residual watch items:
1. Additional symbolic bound edge cases beyond current fixture corpus.
2. Environment variability around optional symbolic tooling beyond SymPy.

## Validation

Executed in `POC_initial_demo`:

- Focused parser/emitter:
  - `PYTHONPATH=src python -m pytest tests/unit/test_affine_extractor.py tests/unit/test_emitter.py -q`
- Full regression:
  - `PYTHONPATH=src python -m pytest tests -q`

Latest result:
- Focused: `91 passed`
- Full: `273 passed`
