# 0001 — Monorepo layout with packages/, research/, papers/

- **Status:** Accepted
- **Date:** 2026-09-13
- **Deciders:** LoopHole team

## Context

The repository had grown organically during the POC:

- The code lived in `POC_initial_demo/`, a name that no longer described it.
- There were two `docs/` trees, two Dockerfiles, two `scripts/` folders, and doc files numbered `01_`–`12_` (two files numbered `08_`).
- A 74 MB failed MinGW LLVM build directory, an empty and broken `Polygeist` submodule entry, stray `hi.c` files, and LaTeX build artifacts were committed.
- Superseded WSL native-build scripts sat at the root next to the supported Docker flow.

The project is moving into a research phase that will add experiments, benchmarks,
papers, and probably more components. Without a clear home for each, the mess would grow.

## Options considered

1. **Flat src layout:** the package at the repo root. Simplest paths, but a second component would mean restructuring again.
2. **Monorepo with `packages/`:** each installable component in its own folder, with research and papers as siblings. Slightly deeper paths, room to grow.

## Decision

Option 2:

```
packages/loophole/    code, tests, examples, package scripts
docker/polygeist/     supported toolchain image
docs/                 guides, architecture, background, planning, status, decisions, worklog
research/             experiments, benchmarks, notebooks
papers/               thesis report, IEEE paper
tools/legacy/         superseded tooling, kept for reference
```

- Committed junk was deleted (still available in git history).
- Superseded tooling was moved to `tools/legacy/`.
- Lane A/B/C records were kept intact under `docs/worklog/`.
- `setup.py` was replaced by `pyproject.toml`.

## Consequences

- Paths changed. Any open branch touching `POC_initial_demo/`, `docs/NN_*.md`, `report/`, or `rpaper/` must be rebased; git's rename detection handles most of it.
- CI and the image-publish workflow now point at `packages/loophole` and `docker/polygeist`.
- Old absolute paths inside historical baseline JSON files (`POC_initial_demo\tests\fixtures\…`) were left untouched on purpose. They are records.
- The scripts in `tools/legacy/` reference old paths and won't run as-is.
