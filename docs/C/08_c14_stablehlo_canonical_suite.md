# C-14 StableHLO Canonical Suite and Report Integration

Last updated: 2026-04-20

This document defines the canonical StableHLO suite used for C-14 confidence tracking and report integration.

## Canonical Required Operations

1. stablehlo.dot_general
2. stablehlo.dot_general_matvec
3. stablehlo.dot_general_vecdot
4. stablehlo.transpose
5. stablehlo.add
6. stablehlo.subtract
7. stablehlo.multiply
8. stablehlo.reduce{add}
9. stablehlo.reduce{add}_colsum
10. stablehlo.reduce{max}
11. stablehlo.convolution

## Integration Behavior

Canonical-suite metrics are now emitted in report payloads:
- `canonical_stablehlo_summary.required_ops`
- `canonical_stablehlo_summary.observed_required_ops`
- `canonical_stablehlo_summary.missing_required_ops`
- `canonical_stablehlo_summary.proved`
- `canonical_stablehlo_summary.unproved_timeout`
- `canonical_stablehlo_summary.refuted`
- `canonical_stablehlo_summary.unmatched_files`

## Interpretation Rules

1. Missing required ops indicates coverage gap in the current fixture run.
2. Refuted canonical entries are explicit trust-envelope misses and must remain visible.
3. Unmatched canonical candidates are tracked explicitly and are never silently accepted.
4. Canonical summary is only applicable for `stablehlo` or `both` targets.

## No-Silent-Fallback Guardrail

A canonical op is considered healthy only if one of the following is true:
1. Result state is `PROVED`.
2. Result state is `UNPROVED_TIMEOUT` and run is exploratory.

A canonical op is considered unhealthy if:
1. Result state is `REFUTED`.
2. No sketch is matched and the file appears under `unmatched_files`.

This keeps trust and coverage signals explicit, matching the plan requirement to avoid silent fallback behavior.
