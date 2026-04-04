# Person B Execution Documentation

This folder contains all documentation for Person B's parallel execution lane in the three-person compiler hardening plan.

## Task Index

### Phase 1: Baseline & Inventory

| Doc | Task | Status |
|---|---|---|
| [01_baseline_report.md](01_baseline_report.md) | Lock baseline test behavior + demo output | Done |
| [02_fallback_audit.md](02_fallback_audit.md) | B-01: Audit all parser/emitter fallbacks | Done |

### Phase 2: Parser Hardening

| Doc | Task | Status |
|---|---|---|
| [03_diagnostics_plan.md](03_diagnostics_plan.md) | B-02: Add diagnostics for unsupported forms | Done |
| [04_mixed_affine_scf_support.md](04_mixed_affine_scf_support.md) | B-03: Expand parser support for mixed affine/scf patterns | Done |
| [05_affine_index_normalization.md](05_affine_index_normalization.md) | B-04: Improve affine index normalization | Done |
| [06_remove_silent_shape_type_defaults.md](06_remove_silent_shape_type_defaults.md) | B-05: Remove silent shape/type defaults in emitter | Done |
| [07_transpose_permutation_inference_validation.md](07_transpose_permutation_inference_validation.md) | B-06: Improve transpose permutation inference and validation | Done |

### Phase 3: Emitter & Advanced Hardening

| Doc | Task | Status |
|---|---|---|
| [08_convolution_attribute_inference_policy.md](08_convolution_attribute_inference_policy.md) | B-07: Infer/propagate convolution attributes | Done |
| [09_emitted_mlir_validation_pipeline.md](09_emitted_mlir_validation_pipeline.md) | B-08: Add artifact-level emitted MLIR validation pipeline | Done |
| [10_stablehlo_emitter_hooks_sync.md](10_stablehlo_emitter_hooks_sync.md) | B-09: Add emitter hooks for new StableHLO operations | Done |
| [11_fixture_realism_stress_coverage.md](11_fixture_realism_stress_coverage.md) | B-10: Expand fixture realism for parser and emitter stress testing | Done |

Future docs will be added to this folder as B-11 through B-12 tasks are executed.

## Quick Reference

**Completed tasks:** B-01 (fallback audit), B-02 (diagnostics), B-03 (mixed affine/scf parsing), B-04 (index normalization), B-05 (strict emitter metadata), B-06 (transpose permutation validation), B-07 (conv attribute inference/propagation policy), B-08 (artifact-level emitted MLIR validation pipeline), B-09 (StableHLO emitter hook/mapping sync and coverage), B-10 (fixture realism stress coverage for parser/emitter/integration)  
**Current focus:** Ready for B-11  
**Test status:** 232 passing, 1 skipped  
**Demo status:** CLI 7/10 proved, 3 partial; standalone script has stale field reference  

See individual docs for detailed audit tables, disposition mappings, and implementation details.
