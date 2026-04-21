# Sprint 1 Person A Closure (A-11, A-12 Baseline)

Date: 2026-04-17
Owner: Person A lane
Reference: [12_three_person_parallel_plan_v2_12_weeks.md](../12_three_person_parallel_plan_v2_12_weeks.md)

## Objective

Complete Person A Sprint 1 scope in the 12-week plan:
1. A-11: strict-profile contract tests across lift, batch, and demo.
2. A-12 (Sprint 1 portion): capture week-1 weak-kernel baseline metrics.

## A-11 Completion Summary

1. Demo command now follows the same profile and strictness contract used by lift and batch:
   - `--profile` support (`default`, `local-explore`, `ci-strict`).
   - `--z3-timeout` support with profile default fallback.
   - `--strict` support with strict exit behavior.

2. Demo strict behavior now enforces trusted semantics:
   - strict mode exits non-zero when any fixture result is `UNPROVED_TIMEOUT` or `REFUTED`.
   - non-strict mode remains exploratory and does not fail on unproved-only outcomes.

3. Contract tests were expanded:
   - `tests/integration/test_cli_strict_policy.py`
     - `test_demo_strict_exit_code_is_zero_for_proved`
     - `test_demo_strict_exit_code_is_one_for_unproved_timeout`
     - `test_demo_ci_strict_profile_rejects_unproved_timeout`
     - `test_demo_local_explore_profile_allows_unproved_timeout`
   - `tests/integration/test_cli_batch_report.py`
     - `test_batch_strict_exits_nonzero_for_unproved_timeout`

## A-12 Week-1 Baseline Capture Summary

Dual-target baseline reports were generated with fixed timeout `10000` ms:

1. Linalg baseline artifacts (committed snapshot + generated report bundle):
   - `docs/A/baselines/sprint1_week1_linalg_20260417.json`
   - `POC_initial_demo/reports/benchmarks/weekly_benchmark_sprint1_week1_linalg_20260417T145918Z.json` (generated local artifact)
   - `POC_initial_demo/reports/benchmarks/weekly_benchmark_sprint1_week1_linalg_20260417T145918Z.md` (generated local artifact)

2. StableHLO baseline artifacts (committed snapshot + generated report bundle):
   - `docs/A/baselines/sprint1_week1_stablehlo_20260417.json`
   - `POC_initial_demo/reports/benchmarks/weekly_benchmark_sprint1_week1_stablehlo_20260417T145919Z.json` (generated local artifact)
   - `POC_initial_demo/reports/benchmarks/weekly_benchmark_sprint1_week1_stablehlo_20260417T145919Z.md` (generated local artifact)

3. Baseline summary counts:
   - Linalg: total 7, proved 7, unproved_timeout 0, refuted 0.
   - StableHLO: total 7, proved 5, unproved_timeout 0, refuted 2.

4. StableHLO baseline refuted fixtures for follow-up hardening:
   - `conv1d.mlir`
   - `conv2d.mlir`

## Report Pipeline Test Hardening

Added direct integration coverage for weekly report generation script:

1. New file:
   - `tests/integration/test_weekly_benchmark_report.py`

2. Covered behaviors:
   - schema and summary correctness (`schema_version`, counts, accepted strict/loose).
   - persisted JSON and Markdown output writing.
   - explicit error on empty fixture directories.

## Validation Evidence

Focused tests for changed surfaces:
1. `.\.venv\Scripts\python.exe -m pytest tests/integration/test_cli_strict_policy.py tests/integration/test_cli_batch_report.py tests/integration/test_weekly_benchmark_report.py -q -rs`
   - Result: `26 passed`.

Broader Sprint 1 slice:
1. `.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifter_semantics.py tests/integration/test_cli_strict_policy.py tests/integration/test_cli_batch_report.py tests/integration/test_conv2d.py tests/integration/test_dot.py tests/integration/test_matvec.py tests/integration/test_stablehlo_phase3_ops.py tests/integration/test_weekly_benchmark_report.py -q -rs`
   - Result: `66 passed, 1 skipped`.

## Sprint 1 Gate Mapping (Person A Lane)

1. G1 item: "Trusted profile rejects refuted outputs" -> Completed for lift, lift-c, batch, and demo surfaces.
2. G1 item: "Baseline week-1 metrics report committed" -> Person A baseline artifacts generated and committed for both targets.

Note:
1. Overall G1 sign-off still depends on Person B B-14 CI enforcement and Person C dashboard/runbook lane deliverables.

## Handoff Notes

1. Person A Sprint 1 implementation scope is complete for A-11 and A-12 baseline capture.
2. StableHLO weak-kernel hardening for refuted conv fixtures remains in later sprint scope (A-12 completion phase).
3. This document serves as the lane summary artifact required by sprint handoff rules.
