# A-16: Proof-Quality Summary Artifact — Closure

**Date:** 2026-04-29  
**Task:** A-16 — Proof-quality summary artifact for weekly reports

---

## Scope Delivered

A-16 adds a `proof_quality_summary` block to every weekly benchmark report and provides
a standalone generator for emitting the artifact independently.

### Changes

#### `scripts/generate_weekly_benchmark_report.py`

- Added `_build_proof_quality_summary(rows)` — computes all proof-quality metrics from
  the per-kernel result rows already produced by `generate_report`.
- Added `_render_proof_quality_section(pq)` — renders the proof-quality block in Markdown.
- Wired both into `generate_report` (payload) and `_render_markdown` (output).
- Bumped `schema_version` to `"1.2"`, added `"1.1"` to `compatible_schema_versions`.

#### `scripts/generate_proof_quality_summary.py` (new)

Standalone script that runs the lifter and emits only the proof-quality artifact:

```
python scripts/generate_proof_quality_summary.py \
    --fixtures-dir tests/fixtures \
    --target linalg \
    --z3-timeout-ms 10000 \
    --label weekly_YYYY_MM_DD \
    --output-dir reports/benchmarks
```

Outputs `proof_quality_<label>_<timestamp>.json` and `.md`. Accepts
`--baseline-report` to emit a trend delta vs a prior artifact.

### Proof-Quality Metrics

| Field | Description |
|---|---|
| `proof_rate` | PROVED / total kernels |
| `strict_acceptance_rate` | accepted_strict / total kernels |
| `avg_confidence` | mean SymPy confidence across all kernels |
| `avg_confidence_proved` | mean SymPy confidence for proved-only kernels |
| `fully_confident_count` | kernels with confidence = 1.0 and state = PROVED |
| `confidence_distribution` | buckets: perfect (1.0), high (0.9–1.0), below 0.9 |
| `per_sketch_breakdown` | per-sketch counts of proved / refuted / timeout |
| `refuted_kernels` | list of file_rel paths for all refuted results |
| `corpus_refuted_count` | refuted kernels in `corpus/` (expected unsupported forms) |
| `unexpected_refuted_count` | refuted kernels outside `corpus/` (actionable regressions) |
| `proof_quality_grade` | A / B / C / D / F (see grading below) |

### Grading Rules

| Grade | Condition |
|---|---|
| A | strict_acceptance_rate >= 0.90 AND unexpected_refuted == 0 |
| B | strict_acceptance_rate >= 0.75 |
| C | strict_acceptance_rate >= 0.60 |
| D | strict_acceptance_rate >= 0.40 |
| F | strict_acceptance_rate < 0.40 |

The grade distinguishes corpus refusals (known unsupported forms, not regressions) from
unexpected refusals that require triage.

---

## Baseline Snapshot (2026-04-29, linalg, 23 kernels)

| Metric | Value |
|---|---|
| Grade | **B** |
| Proof rate | 78.3% (18 / 23) |
| Strict acceptance rate | 78.3% |
| Avg confidence (all) | 0.777 |
| Avg confidence (proved) | 0.993 |
| Fully confident (conf=1.0) | 14 |
| Corpus refuted (expected) | 5 |
| Unexpected refuted | **0** |

All 5 refuted kernels are in `corpus/` and represent known-unsupported frontend forms
(vectorized IR, scf.for, affine_apply indices, if-guarded stores). No unexpected
regressions.

Baseline artifact:
- `docs/A/baselines/proof_quality_a16_pq_20260429_20260429T110512Z.json`
- `docs/A/baselines/proof_quality_a16_pq_20260429_20260429T110512Z.md`

---

## Tests

New file: `tests/unit/test_a16_proof_quality_summary.py`

15 tests covering:

1. Basic proved/refuted counts
2. Avg confidence across all kernels and proved-only subset
3. Fully-confident count
4. Confidence distribution buckets
5. Per-sketch breakdown grouping
6. Grade thresholds A / B / C / D / F
7. Corpus vs unexpected refuted distinction
8. Empty rows (no division-by-zero)
9. `generate_report` payload includes `proof_quality_summary`
10. Markdown render includes `## Proof Quality Summary` section

---

## Validation

```
365 passed in 5.77s
```

---

## Path to Grade A

The current B grade reflects 5 corpus refusals that are intentional (known-unsupported
frontend patterns). To reach grade A without touching the corpus fixtures, the
`unexpected_refuted_count` must stay 0 **and** `strict_acceptance_rate` must reach 90%.
That requires lifting 3 more of the 23 kernels (currently 18/23). Candidates are the
corpus files once their unsupported forms are added to the parser (B-lane work).
