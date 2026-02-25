# LoopHole — Project Documentation Index

**Project:** Automated Lifting and Synthesis of Legacy Code into Tensor IR  
**Status:** Pre-implementation (documentation & planning phase)  
**Date:** February 2026

---

## What Is This Project?

**LoopHole** is a compiler research and engineering project aimed at automatically *lifting* legacy scalar loop-based code (C, C++, Fortran) written in element-wise, pointer-arithmetic style into high-level tensor dialects within the [MLIR](https://mlir.llvm.org/) (Multi-Level Intermediate Representation) ecosystem — specifically targeting the **Linalg** and **StableHLO** dialects.

Once lifted, these programs can be passed to domain-specific ML compilers (XLA, IREE) and executed on modern hardware accelerators (TPUs, GPUs, NPUs) with order-of-magnitude performance improvements, without any manual rewriting.

---

## Documentation Structure

| Document | Description |
|---|---|
| [01 — Project Overview](./01_project_overview.md) | Goals, motivation, problem scope, and success criteria |
| [02 — State of the Art](./02_state_of_the_art.md) | Comprehensive survey of all relevant SOTA frameworks, benchmarks, and limitations (mlirSynth, Tenspiler, Tensorize, STAGG, QiMeng-Xpiler, and more) |
| [03 — MLIR Architecture Deep Dive](./03_mlir_architecture.md) | Technical reference for the MLIR infrastructure, dialects used (Affine, SCF, Linalg, StableHLO, SparseTensor, Transform), and the compilation pipeline |
| [04 — Implementation Roadmap](./04_implementation_roadmap.md) | Step-by-step phased build plan: 3-month POC, 6-month extension, and long-term research phases with concrete deliverables |
| [05 — Novel Research Directions](./05_novel_research_directions.md) | Unexplored frontiers: sparse tensor synthesis, Transform dialect scheduling, dynamic shape inference, and auto-documentation |
| [06 — References & Resources](./06_references.md) | All papers, repositories, tools, benchmarks, and links cited throughout this documentation |

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
