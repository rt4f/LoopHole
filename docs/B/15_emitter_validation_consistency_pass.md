# B-13: Emitter Validation Consistency Pass

## Goal

Normalize emitter validation behavior so Linalg and StableHLO paths enforce metadata gates consistently and fail deterministically with actionable diagnostics for missing shape/type/rank and convolution-attribute policy failures.

## Implementation Summary

### 1. Unified metadata gates across emitters

Updated:
- `POC_initial_demo/src/loophole/emitter.py`

Changes:
- Added explicit element-type gate helper:
  - `_require_element_type(loop, op_name)`
- Enforced element-type checks in both emitters before op-specific emission:
  - `LinalgEmitter._validate_required_metadata(...)`
  - `StableHLOEmitter._validate_required_metadata(...)`
- Removed silent fallbacks for tensor names and shapes in emitter op handlers.
  - Replaced default guessing with strict helpers:
    - `_require_input_tensor(...)`
    - `_require_output_tensor(...)`
    - `_require_shape(...)`

Result:
- Missing metadata now fails early with explicit, deterministic `EmissionError` messages.

### 2. Rank validation consistency and deterministic behavior

Updated:
- `POC_initial_demo/src/loophole/emitter.py`

Changes:
- Added/expanded explicit rank requirements for key ops in Linalg paths (matmul family, convolution families, pool).
- Added stricter shape compatibility gate for elementwise Linalg emission.
- StableHLO convolution now supports deterministic normalization for both:
  - rank-2 metadata tuples `(I,K,O)` for 2D-style inputs
  - rank-4 metadata tuples `(I,K,O)` for NHWC/HWCF-style inputs
- All other rank combinations fail with actionable rank diagnostics.

### 3. Cross-op consistency tests for B-13 closure criteria

Updated:
- `POC_initial_demo/tests/unit/test_emitter.py`

Added test class:
- `TestEmitterValidationConsistencyB13`

New tests:
1. `test_missing_tensor_shapes_gate_is_consistent_across_dialects`
2. `test_missing_element_type_gate_is_consistent_across_dialects`
3. `test_rank_gate_is_actionable_for_linalg_matmul`
4. `test_rank_gate_is_actionable_for_stablehlo_convolution`
5. `test_convolution_attr_policy_failure_is_consistent_across_dialects`

Also updated an existing broad emitter smoke assertion to accept strict rank/unsupported validation failures as expected policy outcomes.

## Acceptance Mapping (B-13)

- Action: Ensure uniform metadata validation gates across emitter paths.
  - Done: helper-based strict input/output/shape/type requirements are enforced before emission.
- Action: Standardize failure behavior for missing shape/type/rank metadata.
  - Done: deterministic `EmissionError` paths for both Linalg and StableHLO emitters; cross-dialect tests added.
- Action: Ensure convolution attribute policy failures are consistent and actionable.
  - Done: shared policy failure behavior validated in both dialect paths by tests.
- Done when: Emitter validation behavior is deterministic, policy-aligned, and test-covered for closure classes.
  - Done: targeted B-13 tests and full regression suite pass.

## Validation

Executed in `POC_initial_demo`:

- Emitter suite:
  - `PYTHONPATH=src python -m pytest tests/unit/test_emitter.py -q`
- Full regression:
  - `PYTHONPATH=src python -m pytest tests -q`

Latest result:
- `252 passed, 1 skipped`
