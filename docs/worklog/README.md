# Worklog

Historical execution records. These are kept as written (only links were
updated during the 2026-09 repo restructure). Put current, living documentation in
`docs/guides`, `docs/architecture`, or `docs/status` instead.

| Folder | Lane | Focus |
|---|---|---|
| [lane-a/](lane-a/README.md) | Person A | Trust boundary and core engine: trust states, strict mode, Z3 proof quality, counterexample replay, shape-parametric proofs |
| [lane-b/](lane-b/README.md) | Person B | Parser and emitter reliability: fallbacks, diagnostics, index normalization, emitted-MLIR validation, corpus compatibility |
| [lane-c/](lane-c/README.md) | Person C | CLI and workflow: batch JSON reports, weekly benchmarks, StableHLO coverage, profiles, release readiness, research-track decision |
| [development-log.md](development-log.md) | — | Original POC bring-up log (first install through first green test run) |

Lanes follow the plans in [docs/planning](../planning/). Benchmark baselines referenced by
lane records live in [research/benchmarks/baselines](../../research/benchmarks/baselines/).

## Adding a record

Continue the lane's numbering: `lane-b/18_<topic>.md`, then add a row to that lane's README.
