# LoopHole — Project Documentation Index

**Project:** Automated Lifting and Synthesis of Legacy Code into Tensor IR  
**Status:** Active implementation (documentation, planning, and working POC)  
**Date:** February 2026

---

## What Is This Project?

**LoopHole** is a compiler research and engineering project aimed at automatically *lifting* legacy scalar loop-based code (C, C++, Fortran) written in element-wise, pointer-arithmetic style into high-level tensor dialects within the [MLIR](https://mlir.llvm.org/) (Multi-Level Intermediate Representation) ecosystem — specifically targeting the **Linalg** and **StableHLO** dialects.

Once lifted, these programs can be passed to domain-specific ML compilers (XLA, IREE) and executed on modern hardware accelerators (TPUs, GPUs, NPUs) with order-of-magnitude performance improvements, without any manual rewriting.

---

## Documentation Structure

| Document | Description |
|---|---|
| [01 — Project Overview](./docs/01_project_overview.md) | Goals, motivation, problem scope, and success criteria |
| [02 — State of the Art](./docs/02_state_of_the_art.md) | Comprehensive survey of all relevant SOTA frameworks, benchmarks, and limitations (mlirSynth, Tenspiler, Tensorize, STAGG, QiMeng-Xpiler, and more) |
| [03 — MLIR Architecture Deep Dive](./docs/03_mlir_architecture.md) | Technical reference for the MLIR infrastructure, dialects used (Affine, SCF, Linalg, StableHLO, SparseTensor, Transform), and the compilation pipeline |
| [04 — Implementation Roadmap](./docs/04_implementation_roadmap.md) | Step-by-step phased build plan: 3-month POC, 6-month extension, and long-term research phases with concrete deliverables |
| [05 — Novel Research Directions](./docs/05_novel_research_directions.md) | Unexplored frontiers: sparse tensor synthesis, Transform dialect scheduling, dynamic shape inference, and auto-documentation |
| [06 — References & Resources](./docs/06_references.md) | All papers, repositories, tools, benchmarks, and links cited throughout this documentation |
<<<<<<< HEAD
| [07 — POC Implementation Audit](./docs/07_poc_implementation_audit.md) | Ground-truth implementation status, limitations, hardcoding/stubs inventory, and prioritized fix plan |
| [08 — Three-Person Parallel Execution Plan](./docs/08_three_person_parallel_execution_plan.md) | Execution-ready task graph for three contributors with dependencies, checkpoints, and metrics |
| [09 — Docker/Polygeist Delivery Handoff](./docs/09_docker_polygeist_delivery_handoff.md) | Full change log, validation evidence, drawbacks, limitations, and distribution plan for prebuilt images |
| [POC Docker Polygeist Quickstart](./POC_initial_demo/docker/README.md) | Reproducible Polygeist+MLIR container with two-step C/C++ compile-then-lift workflow |
=======
| [07 — POC Implementation Audit](./docs/07_poc_implementation_audit.md) | Deep audit of implemented POC behavior, limitations, hardcoding/stubs, and prioritized fixes |
| [08 — MLIR + Polygeist Docker Migration Plan](./docs/08_mlir_polygeist_docker_migration_plan.md) | Sequential implementation plan for Dockerized Polygeist integration and regex-to-MLIR parser migration |
>>>>>>> e4273a6 (C-01, C-02, C-03: Fix demo script, add smoke test, wire strict-mode integration test)

---

## Quick-Start: Core Concepts

1. **Program Lifting / Raising** — The reverse of compilation lowering. Takes low-level scalar loops and reconstructs their high-level mathematical intent as tensor operations.
2. **MLIR Dialect** — A modular, composable IR abstraction in the MLIR framework. Key dialects for this project: `affine`, `linalg`, `stablehlo`, `scf`, `sparse_tensor`, `transform`.
3. **Program Synthesis** — Automated construction of programs that satisfy a given specification, verified either by I/O testing, formal SMT proving, or algebraic equivalence.
4. **Neuro-Symbolic Synthesis** — Combining LLM-based pattern recognition with rigorous formal verification to guide and prune the synthesis search space.

---

## Key Performance Benchmarks (SOTA as of Feb 2026)

| Framework | Venue | Best Speedup | Mechanism |
|---|---|---|---|
| mlirSynth | PACT 2023 | 21.6× (TPU) | Bottom-up enumerative synthesis |
| Tenspiler | ECOOP 2024 | 105× (kernel avg) | SMT-verified lifting via Rosette |
| Tensorize | CGO 2025 | **4,102× (GPU)** | Symbolic tracing + algebraic solving |
| STAGG | PLDI 2025 | 99% accuracy, 3.19s avg | LLM-guided probabilistic grammar + A\* |

---

## How To Use This Documentation

- If you are **starting from scratch**, read documents in order: `01 → 02 → 03 → 04`.
- If you want to **understand what has already been done** and avoid duplicating existing work, focus on `02` (State of the Art) and `05` (Novel Research Directions).
- If you want to **start implementing** the 3-month POC, jump directly to `04` (Implementation Roadmap).
- For **dialect-level MLIR API reference**, see `03` (Architecture Deep Dive).
