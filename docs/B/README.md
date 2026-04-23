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

### Phase 4: Corpus Compatibility and Diagnostics Closure

| Doc | Task | Status |
|---|---|---|
| [12_realworld_mlir_corpus_parser_compatibility.md](12_realworld_mlir_corpus_parser_compatibility.md) | B-11: Real-world MLIR corpus parser compatibility pass | Done |
| [13_unsupported_form_diagnostics_closure.md](13_unsupported_form_diagnostics_closure.md) | B-12: Unsupported-form diagnostics closure for corpus misses | Done |

### Phase 5: Emitter Consistency Pass

| Doc | Task | Status |
|---|---|---|
| [15_emitter_validation_consistency_pass.md](15_emitter_validation_consistency_pass.md) | B-13: Emitter validation consistency pass | Done |

### Phase 6: Trusted Lane Enforcement

| Doc | Task | Status |
|---|---|---|
| [14_trusted_lane_artifact_verification_enforcement_ci.md](14_trusted_lane_artifact_verification_enforcement_ci.md) | B-14: Trusted-lane artifact verification enforcement in CI | Done |

### Phase 7: Dynamic Memref Hardening

| Doc | Task | Status |
|---|---|---|
| [16_dynamic_memref_normalization_hardening.md](16_dynamic_memref_normalization_hardening.md) | B-15: Dynamic memref normalization hardening toward IR-aware handling | Done |

### Phase 8: Reliability Burn-Down

| Doc | Task | Status |
|---|---|---|
| [17_defect_burndown_parser_emitter_reliability_polish.md](17_defect_burndown_parser_emitter_reliability_polish.md) | B-16: Defect burn-down and parser/emitter reliability polish | Done |

Future docs will be added to this folder as new B-lane tasks are defined.

## Quick Reference

**Completed tasks:** B-01 (fallback audit), B-02 (diagnostics), B-03 (mixed affine/scf parsing), B-04 (index normalization), B-05 (strict emitter metadata), B-06 (transpose permutation validation), B-07 (conv attribute inference/propagation policy), B-08 (artifact-level emitted MLIR validation pipeline), B-09 (StableHLO emitter hook/mapping sync and coverage), B-10 (fixture realism stress coverage for parser/emitter/integration), B-11 (real-world MLIR corpus parser compatibility pass), B-12 (unsupported-form diagnostics closure for corpus misses), B-13 (emitter validation consistency pass), B-14 (trusted-lane artifact verification enforcement in CI), B-15 (dynamic memref normalization hardening), B-16 (defect burn-down and parser/emitter reliability polish)  
**Current focus:** B-lane 12-week scope complete; monitor reliability trend and open next-cycle tasks  
**Test status:** 273 passing, 0 skipped  
**Demo status:** CLI 7/10 proved, 3 partial; standalone script has stale field reference  

See individual docs for detailed audit tables, disposition mappings, and implementation details.
