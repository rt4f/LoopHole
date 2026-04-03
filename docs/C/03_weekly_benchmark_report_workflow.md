# Weekly Benchmark Report Workflow

## C-06 Baseline Script

Script path:

- `POC_initial_demo/scripts/generate_weekly_benchmark_report.py`

The script runs all `.mlir` fixtures in a directory and emits:

- JSON report for automation/dashboard ingestion,
- Markdown report for quick human review.

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

## Weekly Tracking Guidance

- Run once per week on the same fixture set.
- Keep the target and timeout stable to avoid noise.
- Compare summary counts and average latency across weeks.
- Investigate regressions where PROVED count drops or REFUTED count rises.
