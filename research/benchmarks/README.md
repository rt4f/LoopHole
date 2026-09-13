# Benchmarks

```
benchmarks/
├── baselines/   dated snapshots of batch/benchmark/proof-quality runs (JSON + MD)
└── suites/      (future) benchmark kernel sets, e.g. PolyBench-derived C kernels
```

## Baselines

Snapshot files are named `<topic>_<target-or-scope>_<YYYYMMDD>.json`, with an
optional `.md` rendering. Generate new ones from `packages/loophole`:

```bash
python scripts/generate_weekly_benchmark_report.py --help
python scripts/generate_proof_quality_summary.py --help
loophole batch tests/fixtures --report --output-dir <dir>
```

Then copy the result here and reference it from the relevant worklog record or experiment.

Older baselines contain absolute paths from the machine that produced them
(`...\POC_initial_demo\tests\fixtures\...`). They are historical records; don't rewrite them.
