# C-12 to C-17 Completion and Verification

Last updated: 2026-04-20
Status: Completed and verified.

## Scope

This document closes Person C Phase 4 tasks:
1. C-12 quickstart and runbook consolidation.
2. C-13 weekly benchmark dashboard hardening.
3. C-14 canonical StableHLO suite expansion and reporting.
4. C-15 profile preset UX and reproducibility metadata.
5. C-16 release-readiness troubleshooting matrix.
6. C-17 research-track decision package.

Per request, emitted-IR verifier triage was not performed in this closeout pass.

## Implementation Summary

Code and test implementation highlights:
1. Profile alias resolution and canonical policy mapping added for `trusted` and `exploratory` profiles.
2. Batch/report payloads advanced to schema `1.1` with compatibility marker for `1.0` readers.
3. Reproducibility metadata added (run id, fixture fingerprint, source hashes, profile-resolved metadata).
4. Weekly benchmark reporting hardened with trend and anomaly support.
5. Canonical StableHLO suite summary integrated into batch and weekly reports.
6. Docker helper scripts added for batch and weekly report workflows.
7. Integration and unit tests updated to cover new schema and strict/profile behavior.

## Verification Executed

### Local verification

Command:
`c:/Users/Yasho/LoopHole/.venv/Scripts/python.exe -m pytest POC_initial_demo/tests/integration/test_cli_batch_report.py POC_initial_demo/tests/integration/test_cli_strict_policy.py POC_initial_demo/tests/unit/test_weekly_benchmark_report.py -q`

Result:
`23 passed`

### Docker verification (required image)

Image:
`ghcr.io/schizoid-man/loophole-polygeist:llvm17`

Executed:
1. Targeted pytest suite in-container (using container-local venv and editable install).
2. Batch helper run in strict mode without emitted-IR validation triage.
3. Weekly helper run for completion label.

Result:
1. Docker targeted tests passed.
2. Batch report generated successfully.
3. Weekly JSON/Markdown benchmark reports generated successfully.

## Generated Artifacts

1. `packages/loophole/reports/loophole_report.json`
2. `packages/loophole/reports/benchmarks/weekly_benchmark_c_lane_completion_20260420T070036Z.json`
3. `packages/loophole/reports/benchmarks/weekly_benchmark_c_lane_completion_20260420T070036Z.md`

## Notes

1. The Docker helper scripts bootstrap a container-local virtual environment, install project requirements, and install the local package in editable mode before execution.
2. This avoids dependence on host MLIR installs and avoids system Python package-management constraints inside the image.