# LoopHole Documentation

Start with [SETUP.md](../SETUP.md) to get running, then use this index.

## Guides — how to do things

| Doc | Description |
|---|---|
| [Setup (Docker + Polygeist)](../SETUP.md) | Build the image, install the CLI, compile and lift, troubleshooting |
| [Docker Polygeist toolchain](guides/docker-polygeist.md) | Image details, two-step compile/lift workflow, publishing to GHCR |

## Background — why this project exists

| Doc | Description |
|---|---|
| [Problem statement](background/problem-statement.md) | Feasibility and novelty analysis of lifting legacy code to tensor IR |
| [Project overview](background/project-overview.md) | Goals, motivation, scope, success criteria |
| [State of the art](background/state-of-the-art.md) | mlirSynth, Tenspiler, Tensorize, STAGG and others |
| [Novel research directions](background/novel-research-directions.md) | Sparse synthesis, Transform-dialect scheduling, dynamic shapes, … |
| [References](background/references.md) | Papers, repositories, tools, benchmarks |

## Architecture — how it works

| Doc | Description |
|---|---|
| [POC documentation](architecture/poc-documentation.md) | Module-by-module walkthrough of the lifter |
| [Architecture diagram](architecture/architecture.mmd) | Mermaid diagram of the pipeline |
| [MLIR architecture deep dive](architecture/mlir-architecture.md) | Dialects (Affine, SCF, Linalg, StableHLO, …) and the pipeline |

## Planning — where it is going

| Doc | Description |
|---|---|
| [Implementation roadmap](planning/implementation-roadmap.md) | POC, extension, and long-term phases |
| [Future plan and improvements](planning/future-plan.md) | Prioritized improvement strategy |
| [Three-person plan v2 (12 weeks)](planning/three-person-execution-plan-v2-12-weeks.md) | Current parallel execution plan |
| [Three-person plan v1](planning/three-person-execution-plan-v1.md) | Original parallel execution plan |
| [Docker/Polygeist migration plan](planning/docker-polygeist-migration-plan.md) | Plan for moving the toolchain into Docker |

## Status — where it is now

| Doc | Description |
|---|---|
| [Project status](status/project-status.md) | Achievements, limitations, failure analysis |
| [POC implementation audit](status/poc-implementation-audit.md) | Ground-truth implementation status and fix plan |
| [Docker/Polygeist delivery handoff](status/docker-polygeist-delivery-handoff.md) | Change log and validation of the Docker toolchain |

## Decisions

[Architecture Decision Records](decisions/README.md): one short file per significant decision.

## Worklog — what was done, by lane

[Worklog index](worklog/README.md): per-lane sprint and task closure records, and the original development log.

## Elsewhere

- Research experiments and benchmarks: [research/](../research/README.md)
- Thesis report and papers: [papers/](../papers/README.md)
