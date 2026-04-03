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

### Phase 3: Emitter & Advanced Hardening (Pending)

Future docs will be added to this folder as B-06 through B-12 tasks are executed:
- B-06: Improve transpose permutation inference
- B-07: Infer/propagate convolution attributes
- B-08 through B-12: (details TBD)

## Quick Reference

**Completed tasks:** B-01 (fallback audit), B-02 (diagnostics), B-03 (mixed affine/scf parsing), B-04 (index normalization), B-05 (strict emitter metadata)  
**Current focus:** Ready for B-06 (transpose permutation inference/validation)  
**Test status:** 140 passing, 1 skipped, 1 xfailed  
**Demo status:** CLI 7/10 proved, 3 partial; standalone script has stale field reference  

See individual docs for detailed audit tables, disposition mappings, and implementation details.
