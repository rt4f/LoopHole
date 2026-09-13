# research/

Home for the research phase: questions we are testing, the evidence, and the
benchmarks that evidence is measured on. Production code stays in `packages/`.
Polished write-ups go to `papers/`.

```
research/
├── experiments/     one dated folder per experiment (copy _template/)
├── benchmarks/      baselines (JSON/MD snapshots) and, later, kernel suites
└── notebooks/       exploratory analysis
```

## From idea to paper

1. **Idea:** start from [novel research directions](../docs/background/novel-research-directions.md), the [future plan](../docs/planning/future-plan.md), or the [research-track decision package](../docs/worklog/lane-c/11_c17_research_track_decision_package.md).
2. **Decision:** if it changes the system's design, write an [ADR](../docs/decisions/README.md).
3. **Experiment:** `research/experiments/YYYY-MM-DD-slug/` from `_template/`. Record the hypothesis *before* running.
4. **Promote:**
   - Code that proves useful moves into `packages/loophole` (or a new package) with tests.
   - Stable measurements become baselines in `benchmarks/`.
5. **Write up:** cite the experiment folder from the paper in `papers/`.

## Reproducibility

Every experiment README must record the commit SHA, the Docker image tag and
`docker/polygeist/toolchain.lock` refs, the exact commands, and where raw outputs are.
Large raw outputs go in an `outputs/` subfolder, which is git-ignored. Commit a summary instead.
