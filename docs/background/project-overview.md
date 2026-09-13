# 01 — Project Overview

## 1.1 Motivation: The Hardware-Software Scaling Crisis

The cessation of **Dennard scaling** (transistor power density kept constant as transistors shrank) and the deceleration of **Moore's Law** (transistor density doubling every ~18 months) have ended the era of free, automatic CPU performance improvements. To sustain computational scaling, the semiconductor industry has pivoted aggressively toward **domain-specific accelerators**:

- **Google TPUs (Tensor Processing Units)** — systolic array-based matrix engines
- **NVIDIA GPUs** — massively parallel vector compute units with dedicated Tensor Cores
- **Apple Neural Engines / AMD NPUs** — low-latency, energy-efficient inference chips
- **Intel Gaudi / Habana processors** — HPC-class AI training accelerators

To program these efficiently, modern software architecture relies on **high-level Tensor DSLs** (Domain-Specific Languages):

| DSL / Framework | Primary Use |
|---|---|
| **JAX** | Functional NumPy-like, XLA-compiled |
| **PyTorch 2.x (torch.compile)** | Dynamic graph capture, TorchInductor backend |
| **StableHLO** | Stable portability layer between ML frameworks and compilers |
| **TACO** (Tensor Algebra Compiler) | Sparse/dense tensor algebra |
| **NumPy** | Scientific array programming (CPU-focused) |

These abstractions allow specialized compilers to map computations to low-level hardware intrinsics, exploit optimal memory layouts, and apply target-specific scheduling optimizations — achieving speedups of 10×–4000× over naive scalar C on accelerators.

---

## 1.2 The Legacy Code Problem

Despite the emergence of these modern frameworks, a critical bottleneck persists:

> **Billions of lines of scientifically validated, mission-critical legacy code** remain locked in scalar, loop-based languages — C, C++, and Fortran — where computations are expressed at the individual element level.

Examples include:

- **Climate and weather simulation codes** (e.g., NCAR/CESM, WRF) in Fortran
- **Financial quantitative analysis** libraries in C/C++
- **Medical imaging algorithms** (CT reconstruction, MRI signal processing) in C
- **Computational fluid dynamics solvers** (OpenFOAM, legacy LAPACK routines) in Fortran/C
- **Physics and particle simulation** codebases (CERN geant4 dependencies)

In these codebases, a matrix-matrix multiplication is not expressed as `C = A @ B` — it is expressed as:

```c
for (int i = 0; i < M; i++)
  for (int j = 0; j < N; j++)
    for (int k = 0; k < K; k++)
      C[i][j] += A[i][k] * B[k][j];
```

This scalar representation **completely obscures the mathematical intent** from the compiler and from automated tooling. A domain compiler cannot infer that this is a `linalg.matmul` without explicit semantic analysis.

**The economic and engineering cost** of manually rewriting these codebases is:
- Exceptionally time-consuming (~$100M+ for a typical HPC codebase)
- Highly error-prone (semantic bugs in numerical code are catastrophic)
- Practically infeasible given the volume of existing code

---

## 1.3 The Solution: Automated Program Lifting

**Program lifting** (also called *program raising* or *decompilation to semantics*) is the automated reverse of compilation. Instead of lowering high-level abstractions to machine code, a lifter extracts the underlying mathematical semantics from low-level scalar code and reconstructs an equivalent, higher-level representation.

```
Legacy C Code → [Frontend (Polygeist)] → MLIR Affine IR
MLIR Affine IR → [LIFTER] → MLIR Linalg / StableHLO
MLIR Linalg / StableHLO → [XLA / IREE Compiler] → Hardware-Optimized Binary
```

The key innovation is that the lifter operates **inside the MLIR ecosystem**, which provides a multi-level, extensible IR framework where different dialects (levels of abstraction) can coexist and be transformed progressively.

---

## 1.4 Project Goals

### Primary Goal
Build a **verification-driven program synthesis tool** that ingests nested scalar loops expressed in **MLIR Affine IR** and automatically raises them to high-level tensor operations in the **Linalg** or **StableHLO** dialect.

### Three-Month Proof of Concept (POC) Goal
Demonstrate feasibility of a Python-based tool that:
1. Parses a C `for`-loop implementing a basic linear algebra kernel (convolution, matrix multiply, transpose)
2. Translates the loop into an intermediate **symbolic mathematical representation**
3. Uses the **Z3 SMT solver** to prove equivalence against a pre-defined library of tensor operation sketches
4. Emits valid **`linalg.generic`** or **`stablehlo` operations** as output

### Long-Term Research Goal (12–24 months)
Extend the POC with a **neuro-symbolic synthesis engine** that:
1. Uses a fine-tuned or prompted **LLM** to predict probable tensor operation structures from the source loop
2. Converts these predictions into a **Probabilistic Context-Free Grammar (pCFG)**
3. Uses the pCFG to guide a **heuristically pruned enumerative synthesizer**
4. Verifies candidates via **bounded model checking** (CBMC) for formal correctness guarantees

---

## 1.5 Success Criteria

### POC Success Criteria (3 months)
| Criterion | Target |
|---|---|
| Successfully lifts 1D convolution | ✓ |
| Successfully lifts 2D convolution | ✓ |
| Successfully lifts matrix transposition | ✓ |
| Z3 formally proves equivalence (not just I/O testing) | ✓ |
| Output is valid and compilable MLIR | ✓ |
| Handles constrained (static-shaped) inputs | ✓ |

### Long-Term Research Success Criteria
| Criterion | Target |
|---|---|
| Lifting accuracy on Polybench benchmark suite | ≥ 90% |
| Average synthesis time per kernel | < 10 seconds |
| Speedup over LLVM-O3 on CPU (geomean) | ≥ 3× |
| GPU speedup on at least 5 benchmarks | ≥ 100× |
| Novel contribution beyond STAGG (e.g., sparse, dynamic shapes) | Required for publication |

---

## 1.6 Out of Scope (for POC Phase)

- Irregular / non-affine loop bounds (e.g., while loops, pointer-chased memory)
- Sparse tensor representations (deferred to research phase)
- Dynamic (runtime-determined) tensor shapes
- Multi-function codebases (POC handles single kernels only)
- Auto-tuning or hardware-specific scheduling (deferred to research phase)

---

## 1.7 Target Hardware Platforms

The lifted programs should be compilable and deployable on:

| Platform | Compiler Backend |
|---|---|
| Intel CPUs (AVX2/AVX-512) | LLVM via Linalg lowering |
| AMD CPUs (AVX2/Zen) | LLVM via Linalg lowering |
| NVIDIA GPUs (CUDA) | IREE / MLX backend |
| Google TPU v3/v4 | XLA from StableHLO |
| ARM CPUs (SVE/SVE2) | MLIR ARM-SVE lowering |

---

## 1.8 Positioning Statement

LoopHole is positioned at the intersection of:

- **Compiler Infrastructure** — MLIR ecosystem, dialect engineering, pass management
- **Formal Methods** — SMT solving (Z3), bounded model checking (CBMC), algebraic equivalence proofs
- **Machine Learning for Code** — LLMs as probabilistic heuristic guides for program synthesis
- **High-Performance Computing** — Targeting hardware accelerators, vectorization, auto-tuning

It builds directly on the work of mlirSynth, Tenspiler, Tensorize, and STAGG while targeting novel frontiers that these systems have left open (see [05 — Novel Research Directions](./novel-research-directions.md)).
