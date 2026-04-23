# B-15: Dynamic Memref Normalization Hardening Toward IR-Aware Handling

## Goal

Improve dynamic memref normalization in parser metadata extraction to be more IR-aware, reduce ambiguous dynamic-shape handling that degrades matcher/emitter quality, and lock behavior with regression tests.

## Implementation Summary

### 1. Parser dynamic memref normalization hardening

Updated:
- `POC_initial_demo/src/loophole/affine_extractor.py`

Changes:
- Hardened memref parsing:
  - legacy dynamic dimensions (`-1`) and wildcard dims (`*`) now normalize consistently to dynamic (`-1`).
  - unranked memref form (`memref<*xT>`) is parsed as unknown-rank metadata instead of a misleading pseudo-rank.
- Reworked tensor metadata resolution from first-hit behavior to aggregated normalization:
  - collect shape/type observations across function args and all read/write accesses.
  - infer a stable rank using majority observations (including IR index-arity evidence from access expressions).
  - normalize per-dimension static values only when consistent.
  - preserve dynamic dimension (`-1`) when conflicting static annotations are observed.
- Added explicit diagnostics for metadata ambiguity:
  - conflicting rank observations
  - conflicting static dimension candidates
  - conflicting element type candidates (resolved by dominant type)

### 2. Dynamic memref edge fixtures

Updated:
- `POC_initial_demo/src/loophole/tests/fixtures.py`

Added fixtures:
1. `MATMUL_UNRANKED_MEMREF_MLIR`
   - uses unranked memrefs (`memref<*xf32>`) with ranked index usage
   - validates rank normalization from IR access arity
2. `MATMUL_DYNAMIC_CONFLICTING_ANNOTATIONS_MLIR`
   - includes intentionally conflicting static dimension annotations for a dynamic tensor
   - validates that parser keeps ambiguous dimension dynamic with explicit diagnostics

### 3. Parser and emitter regression coverage

Updated:
- `POC_initial_demo/tests/unit/test_affine_extractor.py`
- `POC_initial_demo/tests/unit/test_emitter.py`

Added assertions:
- unranked memref rank normalization yields stable dynamic-ranked shapes (`[-1, -1]` for matmul tensors)
- conflicting static dim observations are normalized to dynamic with explicit diagnostics
- emitter output remains stable and dynamic-safe (`memref<?x?xf32>`) for normalized edge fixtures

## Acceptance Mapping (B-15)

- Action: Improve dynamic memref normalization logic toward more IR-aware parser metadata extraction.
  - Done: metadata normalization now aggregates cross-IR observations, uses index-arity rank evidence, and resolves shape/type ambiguities deterministically.
- Action: Reduce ambiguous dynamic-shape handling that can degrade matcher/verifier quality.
  - Done: conflicting static dimensions are now explicitly diagnosed and normalized to dynamic instead of unstable shape selection.
- Test addition: Add dynamic memref edge fixtures and parser/emitter assertions for normalized handling.
  - Done: two new edge fixtures plus parser and emitter regression tests.
- Done when: Dynamic memref behavior is stable, documented, and covered by regression tests.
  - Done: implementation + tests + B-15 documentation delivered.

## Validation

Run in `POC_initial_demo`:

- Focused:
  - `PYTHONPATH=src python -m pytest tests/unit/test_affine_extractor.py tests/unit/test_emitter.py -q`
- Full:
  - `PYTHONPATH=src python -m pytest tests -q`

Latest result:
- Focused parser/emitter regression: `88 passed`
- Full regression: `270 passed`
