# Experiments

One folder per experiment, named `YYYY-MM-DD-short-slug` (date the experiment started).

```
2026-09-20-z3-timeout-sweep/
├── README.md        # from _template/README.md
├── run.ps1 / run.sh # exact commands (or a script under packages/*/scripts)
├── config/          # inputs, parameter files (optional)
├── results/         # small committed summaries: tables, plots, JSON
└── outputs/         # large raw outputs (git-ignored)
```

Start one:

```bash
cp -r research/experiments/_template research/experiments/2026-09-20-my-experiment
```

| Experiment | Question | Status |
|---|---|---|
| _(none yet)_ | | |
