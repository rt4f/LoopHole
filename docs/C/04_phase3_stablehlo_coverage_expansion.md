# Phase 3 StableHLO Coverage Expansion

## Scope

This update starts Person C Phase 3 tasks C-08, C-09, and C-10:

- C-08: Expand StableHLO sketch-library coverage.
- C-09: Add emitter support for newly covered StableHLO operations.
- C-10: Add integration tests for each new StableHLO operation path.

## Implemented

### C-08: Sketch Coverage

The StableHLO sketch catalog now includes:

- dot variants:
  - `stablehlo.dot_general` (matmul)
  - `stablehlo.dot_general_matvec`
  - `stablehlo.dot_general_vecdot`
- transpose:
  - `stablehlo.transpose`
- elementwise:
  - `stablehlo.add`
  - `stablehlo.subtract`
  - `stablehlo.multiply`
- reduce variants:
  - `stablehlo.reduce{add}`
  - `stablehlo.reduce{add}_colsum`
  - `stablehlo.reduce{max}`
- existing convolution sketch remains supported:
  - `stablehlo.convolution`

### C-09: Emitter Support

StableHLO emitter now supports:

- dot_general rank variants (matmul, matvec, vecdot)
- transpose emission with inferred permutation
- elementwise add/subtract/multiply
- reduce emission with inferred reduction axes and add/max reducer selection

Additional robustness improvements:

- scalar tensor/memref type rendering now emits `tensor<f32>` / `memref<f32>` instead of malformed scalar forms.

### C-10: Integration Coverage

Added:

- `tests/integration/test_stablehlo_phase3_ops.py`

This suite validates:

- StableHLO transpose path
- StableHLO elementwise add path
- StableHLO vector-dot path
- StableHLO matvec path
- StableHLO reduce-sum path

## Validation Result

Full repository POC tests after this update:

- `201 passed, 7 skipped`
