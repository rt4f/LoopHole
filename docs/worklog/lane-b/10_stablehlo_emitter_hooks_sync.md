# B-09: StableHLO Emitter Hooks and Mapping Synchronization

## Goal

Ensure StableHLO emitter hooks stay synchronized with `sketch_library.py`, and every newly added StableHLO sketch has tested emission coverage.

## Implementation Summary

### 1. Added explicit StableHLO sketch-to-handler map

Updated `packages/loophole/src/loophole/emitter.py`:

- Added `StableHLOEmitter._NAMED_OP_HANDLERS` mapping all StableHLO sketch names to concrete emitter methods.
- Switched `StableHLOEmitter.emit()` from family-based dispatch to explicit name-based dispatch.
- Added explicit synchronization failure message when a StableHLO sketch is missing in mapping.

This removes implicit routing ambiguity and makes mapping drift obvious.

### 2. Added fixtures for newly covered StableHLO paths

Updated `packages/loophole/src/loophole/tests/fixtures.py` with additional kernels:

- `ELEMENTWISE_SUB_MLIR`
- `ELEMENTWISE_MUL_MLIR`
- `REDUCE_SUM_COLWISE_MLIR`
- `REDUCE_MAX_MLIR`

Also added these fixtures to demo/all fixture dictionaries for broader test access.

### 3. Upgraded StableHLO unit emission assertions

Updated `packages/loophole/tests/unit/test_emitter.py`:

- Added mapping synchronization test:
  - verifies `StableHLOEmitter._NAMED_OP_HANDLERS` exactly matches StableHLO sketches in library.
- Added StableHLO output assertions for:
  - subtract
  - multiply
  - reduce sum (rowwise)
  - reduce sum (colwise)
  - reduce max

### 4. Expanded strict integration coverage

Updated integration tests:

- `packages/loophole/tests/integration/test_matmul.py`
  - added strict StableHLO matmul path assertion (`stablehlo.dot_general`).
- `packages/loophole/tests/integration/test_stablehlo_phase3_ops.py`
  - added strict StableHLO subtract path assertion.
  - added strict StableHLO multiply emission-path assertion (forced equivalent verification path to stabilize strict route coverage).
  - added reduce-max StableHLO emission assertions.

## Acceptance Mapping (B-09)

- Expand StableHLO handlers in emitter as sketches are added: Done.
- Keep sketch-to-emitter mapping synchronized with sketch library: Done (explicit map + synchronization unit test).
- Add StableHLO-specific output assertions in unit emitter tests: Done.
- Add strict integration tests in matmul and expanded integration coverage: Done.
- Every newly added StableHLO sketch has a tested emission path: Done.

## Validation

Executed in `POC_initial_demo`:

- Focused: `pytest tests/unit/test_emitter.py tests/integration/test_matmul.py tests/integration/test_stablehlo_phase3_ops.py -q`
- Full: `pytest tests/ -q`

Latest result:

- `220 passed, 1 skipped`
