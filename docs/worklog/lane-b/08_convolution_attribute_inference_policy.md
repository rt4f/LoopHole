# B-07: Convolution Attribute Inference and Policy Enforcement

## Goal

Infer convolution `strides` and `dilations` from extracted affine access expressions, and fail explicitly when inference is ambiguous or unsupported.

## Scope

- Linalg conv emitters:
  - `linalg.conv_1d_ncw_fcw`
  - `linalg.conv_1d_nwc_wcf`
  - `linalg.conv_2d`
  - `linalg.conv_2d_nhwc_hwcf`
  - `linalg.conv_2d_nchw_fchw`
- StableHLO convolution emitter:
  - `stablehlo.convolution`
- Unit/integration coverage for:
  - non-unit stride/dilation propagation
  - explicit failure mode behavior

## Implementation Summary

### 1. Added explicit 2D conv window-attribute inference

A shared helper was added in the emitter layer to infer 2D convolution window attributes from input tensor read index expressions:

- infers `(stride_h, dilation_h, stride_w, dilation_w)` from linear coefficients
- supports multiple input layouts by expression analysis rather than fixed IV positions
- accepts a spatial axis only when an expression depends on:
  - exactly one parallel IV with positive coefficient
  - exactly one reduction IV with positive coefficient
- fails with `EmissionError` if two spatial axes cannot be inferred

This prevents silent guessing while still supporting plain 2D, NHWC, and NCHW-style index layouts.

### 2. Wired inference into Linalg and StableHLO conv paths

The conv emitters now consume inferred attributes instead of hardcoded unit windows:

- Linalg emits explicit
  - `strides = dense<[...]>`
  - `dilations = dense<[...]>`
- StableHLO emits explicit
  - `window = {stride = [...], rhs_dilate = [...]}`

### 3. Preserved strict policy behavior for unsupported patterns

If inference fails, emitters now raise clear policy errors instead of falling back to defaults.

## Test Updates

### Unit tests

- Added/validated non-unit attr emission checks for conv1d and conv2d emitters.
- Added/validated explicit failure checks when index expressions are unsupported.
- Updated broad "emit first N sketches" smoke test to allow `EmissionError` for incompatible sketch/input combinations under strict policies.

### Integration tests

- Added/validated strided+dilated conv1d and conv2d fixtures for pipeline-level checks.
- StableHLO convolution integration verifies inferred non-unit window attrs are present.
- Adjusted large-matmul timeout expectation to allow explicit emission-policy failure messaging when an unproved candidate cannot be emitted.

## Acceptance Mapping (B-07)

- Infer convolution attrs from affine accesses: Done.
- Propagate attrs into emitted Linalg/StableHLO ops: Done.
- No silent defaulting on unsupported patterns: Done (explicit policy errors).
- Regression suite stability: Done.

## Validation

From `POC_initial_demo`:

- Focused checks:
  - `pytest tests/unit/test_emitter.py tests/integration/test_conv1d.py tests/integration/test_conv2d.py tests/integration/test_matmul.py -q`
- Full regression:
  - `pytest tests/ -q`

Latest result:

- `168 passed, 1 skipped`

## Notes

- This task intentionally tightens correctness semantics for convolution emission.
- Some previous permissive assumptions in generic smoke tests were updated to align with explicit-failure policy behavior.
