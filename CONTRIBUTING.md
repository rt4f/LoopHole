# Contributing to LoopHole

## Where does it go?

| You are adding… | Put it in | Notes |
|---|---|---|
| Library / CLI code | `packages/loophole/src/loophole/` | |
| Tests | `packages/loophole/tests/{unit,integration}/` | MLIR inputs go in `tests/fixtures/` |
| A C/C++ example kernel | `packages/loophole/examples/` | |
| A report generator or helper script for the package | `packages/loophole/scripts/` | |
| A new, separately installable component | `packages/<name>/` | See [packages/README.md](packages/README.md) |
| Docker toolchain changes | `docker/polygeist/` | Bump `toolchain.lock` with any ref change |
| How-to / user guide | `docs/guides/` | |
| Design / how the system works | `docs/architecture/` | |
| Literature, problem statement, references | `docs/background/` | |
| Roadmaps, sprint or team plans | `docs/planning/` | |
| Status snapshots, audits, handoffs | `docs/status/` | |
| Sprint / task closure records | `docs/worklog/lane-<x>/` | Keep the per-lane numbering |
| A significant technical decision | `docs/decisions/NNNN-title.md` | Copy `0000-template.md` |
| An experiment (research phase) | `research/experiments/YYYY-MM-DD-slug/` | Copy `_template/` |
| Benchmark baselines / suites | `research/benchmarks/` | |
| Exploratory notebooks | `research/notebooks/` | |
| Thesis report / papers | `papers/<name>/` | Only sources + final PDF are committed |
| Anything unsupported or superseded | `tools/legacy/` | Say why in its README |

Nothing else belongs at the repository root.

## Naming

- Files and folders: `kebab-case` (docs, experiments, packages). Python modules: `snake_case`.
- No numeric prefixes on new docs. Order is given by the index in [docs/README.md](docs/README.md). Exceptions: worklog records (`NN_topic.md`) and ADRs (`NNNN-title.md`).
- Experiments are dated: `2026-09-20-z3-timeout-sweep`.

## Workflow

1. Branch from `main`.
2. Set up once with [SETUP.md](SETUP.md).
3. Run the tests from `packages/loophole`: `python -m pytest -q`.
4. If you moved or renamed a file, update links to it and the index in `docs/README.md`.
5. Open a PR. CI (`.github/workflows/ci.yml`) runs the test suite with MLIR verification required.

## Reproducibility rules for results

Any number that ends up in a doc or paper must be traceable to:

- a git commit SHA,
- a Docker image tag (`loophole-polygeist:llvm17` + `docker/polygeist/toolchain.lock`),
- the exact command, and
- the raw output file (in the experiment folder or `research/benchmarks/`).
