# C-16 Release Readiness and Troubleshooting Matrix

Last updated: 2026-04-20

This document captures release-readiness checks and a troubleshooting matrix for trusted and exploratory workflows.

## Release Readiness Checklist

1. Trusted profile runs are reproducible on a clean machine.
2. Batch and weekly reports are generated with schema `1.1` and compatibility markers.
3. Canonical StableHLO summary is present for stablehlo targets.
4. No silent fallback is introduced for canonical suite misses.
5. Docker image tag used for MLIR workflows is documented in `run_metadata.docker_image`.
6. Trusted run rejects unproved/refuted outputs as expected.
7. Known risks are reflected in project status docs.

## Troubleshooting Matrix

| Symptom | Likely Cause | Resolution |
|---|---|---|
| `--strict cannot be combined with --no-verify` | Strict policy requested with verification disabled | Remove `--no-verify` for trusted/ci-strict runs |
| Trusted run exits non-zero on timeout | Expected strict behavior | Re-run with `--profile exploratory` for diagnostic triage |
| Batch report lacks canonical section | Target is not stablehlo/both | Run with `--target stablehlo` or `--target both` |
| Docker run fails with mount path errors on Windows | Input/output paths across different drives | Keep paths on same drive and rerun |
| `mlir-opt` verifier not found in batch validation | Verifier not available in runtime path | Use Docker image workflow or set `--mlir-verifier` explicitly |
| Trend report shows false regression noise | Changed fixture set or timeout baseline | Keep fixture set, timeout, and target fixed for trend comparisons |
| Canonical ops missing unexpectedly | Fixture run does not include representative kernels | Expand fixture corpus and rerun baseline |

## Operational Notes

1. For MLIR-related commands, prefer Docker helper scripts under `POC_initial_demo/scripts`.
2. Store weekly baseline artifacts in versioned benchmark directories.
3. Keep release candidate evidence linked to C-17 decision package.

## Related References

- `docs/09_docker_polygeist_delivery_handoff.md`
- `docs/10_project_status_achievements_limitations.md`
- `docs/11_future_plan_and_improvements.md`
- `docs/12_three_person_parallel_plan_v2_12_weeks.md`
