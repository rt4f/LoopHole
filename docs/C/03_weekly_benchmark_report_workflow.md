# Weekly Benchmark Report Workflow

## C-06 Baseline Script

Script path:

- `POC_initial_demo/scripts/generate_weekly_benchmark_report.py`

The script runs all `.mlir` fixtures in a directory and emits:

- JSON report for automation/dashboard ingestion,
- Markdown report for quick human review.

Current schema level:

- `1.1` with compatibility marker for `1.0` readers.

## Default Behavior

- fixtures directory: `POC_initial_demo/tests/fixtures`
- target dialect: `linalg`
- timeout: `10000` ms
- output directory: `POC_initial_demo/reports/benchmarks`

## Example Commands

```bash
python POC_initial_demo/scripts/generate_weekly_benchmark_report.py
python POC_initial_demo/scripts/generate_weekly_benchmark_report.py --target stablehlo --label weekly_stablehlo
python POC_initial_demo/scripts/generate_weekly_benchmark_report.py --fixtures-dir POC_initial_demo/tests/fixtures --output-dir POC_initial_demo/reports/benchmarks
```

Docker-first (recommended):

```powershell
powershell -ExecutionPolicy Bypass -File POC_initial_demo/scripts/run_weekly_report_in_docker.ps1 -Target stablehlo -Label week_01
```

Trend run against a prior baseline:

```powershell
powershell -ExecutionPolicy Bypass -File POC_initial_demo/scripts/run_weekly_report_in_docker.ps1 -Target stablehlo -Label week_02 -BaselineReport reports/benchmarks/<week_01_report>.json -RegressionThresholdFraction 0.05
```

## Weekly Tracking Guidance

- Run once per week on the same fixture set.
- Keep the target and timeout stable to avoid noise.
- Compare summary counts and average latency across weeks.
- Use `trend_summary` deltas for gate decisions.
- Investigate regressions where PROVED count drops, REFUTED count rises, or latency exceeds threshold.
