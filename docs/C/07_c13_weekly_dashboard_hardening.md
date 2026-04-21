# C-13 Weekly Dashboard Hardening and Trend Summaries

Last updated: 2026-04-20

This document defines the hardened weekly reporting workflow and trend interpretation rules.

## Scope Delivered

1. Weekly benchmark report schema advanced to `1.1` with compatibility marker for `1.0` readers.
2. Added trend summary support using an optional baseline report.
3. Added anomaly flags for:
- proved count drops
- refuted count increases
- latency regression beyond threshold
4. Added StableHLO canonical-suite summary section to weekly artifacts.
5. Added Docker helper script for weekly report generation.

## Weekly Execution (Docker-first)

Week 1 baseline run:

```powershell
powershell -ExecutionPolicy Bypass -File POC_initial_demo/scripts/run_weekly_report_in_docker.ps1 -Target stablehlo -Label week_01
```

Week N trend run against a baseline JSON report:

```powershell
powershell -ExecutionPolicy Bypass -File POC_initial_demo/scripts/run_weekly_report_in_docker.ps1 -Target stablehlo -Label week_02 -BaselineReport reports/benchmarks/weekly_benchmark_week_01_20260420T000000Z.json -RegressionThresholdFraction 0.05
```

## Report Sections

Top-level additions in schema `1.1`:
- `run_metadata`
- `trend_summary`
- `canonical_stablehlo_summary`
- `compatible_schema_versions`

Trend summary fields:
- `available`
- `baseline_label`
- `baseline_generated_at_utc`
- `delta.proved`
- `delta.unproved_timeout`
- `delta.refuted`
- `delta.avg_elapsed_ms`
- `anomalies[]`

## Quality Gate Usage

Use trend outputs directly in Friday gate decisions:
1. If `proved_drop` appears two weeks in a row, trigger stabilization sprint.
2. If `refuted_increase` appears, block non-critical feature expansion.
3. If `latency_regression` exceeds threshold, inspect slowest-kernel list before merge.

## Baseline Discipline

1. Keep fixture corpus fixed while comparing trend deltas.
2. Keep target dialect and timeout fixed per trend stream.
3. Record Docker image tag in each report (`run_metadata.docker_image`).
4. Keep a clean week-1 baseline artifact under version control or release artifact storage.
