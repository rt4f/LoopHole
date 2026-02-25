# 02 — State of the Art in Program Lifting and Tensor Synthesis

> **Last Updated:** February 2026  
> This document provides an exhaustive, technical survey of all state-of-the-art tools, frameworks, and paradigms directly relevant to the LoopHole project. It is organized chronologically and by architectural paradigm.

---

## 2.1 Overview: Three Generations of Program Lifting

The field of automated program lifting within the MLIR ecosystem has evolved through three distinct architectural generations:

| Generation | Paradigm | Representative Tool | Core Limitation |
|---|---|---|---|
| **1st Gen (2021–2023)** | Rule-based pattern matching | MultiLevel Tactics, Polygeist + LIFT | Requires manually-written rewrite rules; brittle and not retargetable |
| **2nd Gen (2023–2024)** | Enumerative synthesis + formal verification | mlirSynth, Tenspiler | Exponential state-space explosion (enumerative) or SMT timeouts (formal) |
| **3rd Gen (2025–present)** | Neuro-symbolic synthesis (LLM-guided) | Tensorize, STAGG, QiMeng-Xpiler | Sparse/dynamic shapes remain unsolved; hardware-specific scheduling untackled |

---

## 2.2 Pre-Synthesis Foundations

### 2.2.1 Polygeist (MIT/ETH, 2021–2023)
**Role in the pipeline:** Frontend that converts C, C++, and Fortran into MLIR Affine IR.

- Converts `for` loops with affine bounds into `affine.for` operations with polyhedral loop bounds
- Converts pointer-based memory accesses into `memref` (typed, strided memory reference) operations
- Preserves the loop nest structure (critical for downstream synthesis)
- Supports a broad subset of C including multi-dimensional arrays, structs, and basic control flow
- **Limitation:** Irregular loops (while, pointer-chased), complex conditional control flow, and aliasing reduce analysis accuracy

**Key repository:** [llvm/Polygeist](https://github.com/llvm/Polygeist)

### 2.2.2 Lifting Loops in MLIR: MultiLevel Tactics (Grosser et al., EuroLLVM 2020)
- The first systematic exploration of "loop lifting" within the MLIR framework
- Introduced the concept of progressive *raising* (the reverse of lowering) for perfectly nested loops
- Used manually-crafted rewrite patterns (e.g., detecting a 3-deep loop nest with specific index arithmetic and mapping it to `linalg.matmul`)
- **Critical limitation:** Every new target operation requires a new hand-written rule — completely non-scalable for the full breadth of the Linalg or StableHLO dialect

### 2.2.3 C2TACO (2022–2023)
- A lifting tool targeting the **TACO** (Tensor Algebra Compiler) expression language
- Operates by pattern-matching loop nests against a catalogue of TACO templates
- Reports 67/77 correctly lifted on a large benchmark suite
- Serves as the direct predecessor and baseline against which STAGG is evaluated
- **Critical limitation:** Template-based; cannot generalize to operations outside the pre-defined catalogue

---

## 2.3 Second Generation: mlirSynth (PACT 2023)

**Paper:** *mlirSynth: Automatic, Retargetable Program Raising in Multi-Level IR using Program Synthesis*  
**Authors:** Alexander Brauckmann, Elizabeth Polgreen, Tobias Grosser, Michael F. P. O'Boyle  
**Venue:** PACT 2023 (International Conference on Parallel Architectures and Compilation Techniques)  
**Repository:** [alexanderb14/mlirSynth](https://github.com/alexanderb14/mlirSynth)

### Architecture

```
Affine IR → Preprocessing → [Bottom-Up Enumerative Synthesizer] → Candidate Programs
                                        ↕
                             [Observational Equivalence Check (JIT)]
Candidate Programs → Postprocessing → Target Dialect Output (Linalg / StableHLO)
```

### Core Mechanisms

**1. Dialect-Informed Search Space Construction**  
mlirSynth ingests MLIR's **TableGen dialect definitions** to automatically extract the full set of available operations, their type signatures, and their structural constraints. This eliminates the need for manually-written rules and makes the tool retargetable to any dialect simply by loading new TableGen definitions.

**2. Smallest-Program-First, Bottom-Up Enumeration**  
The synthesizer constructs candidate programs in the target dialect using a smallest-first strategy — it enumerates all programs of length 1, then length 2, etc. Type constraints from the dialect definitions aggressively prune invalid candidates. For example, `linalg.matmul` requires `memref<?x?xf32>` inputs, eliminating candidates that feed scalar values.

**3. Observational Equivalence via JIT Execution**  
To verify a candidate program, mlirSynth uses MLIR's built-in JIT compilation infrastructure. Both the candidate and the original Affine IR program are compiled and executed on a set of input-output test vectors. If all outputs match, the programs are declared observationally equivalent.

### Performance Results

| Target | Benchmark | Geomean Speedup vs LLVM-O3 |
|---|---|---|
| Intel CPU | Polybench | **2.5×** |
| AMD CPU | Polybench | **3.4×** |
| Google TPU v3 | Polybench (via XLA) | **21.6×** |

### Critical Limitations

| Limitation | Technical Root Cause |
|---|---|
| **Exponential explosion** | Search space grows as O(|ops|^n) for programs of n operations |
| **Practical ceiling: ~4 ops** | Programs requiring >4 DSL operations fail to synthesize within time limits |
| **No formal correctness guarantee** | Observational equivalence on finite test vectors cannot prove universal correctness — edge cases (overflow, denormals, corner-case array sizes) may pass testing but fail on production inputs |
| **Scalability wall** | The bottom-up approach has no mechanism for leveraging semantic knowledge of the source program to guide the search |

---

## 2.4 Second Generation: Tenspiler (ECOOP 2024)

**Paper:** *Tenspiler: A Verified Lifting-Based Compiler for Tensor Operations*  
**Authors:** Jie Qiu, Colin Cai, Hersh Kumar, Ruiqi Gao, Guannan Wei, Shannon Huang, et al.  
**Venue:** ECOOP 2024 (European Conference on Object-Oriented Programming)  
**Repository:** [tenspiler/tenspiler](https://github.com/tenspiler/tenspiler)

### Architecture

```
C++ / Python Source → Frontend Analysis
                     ↓
              TensIR (Intermediate Representation)
                     ↓
     [Rosette Symbolic Synthesis Engine]
           → Constructs verification conditions
           → Encodes loop invariants
           → Translates to SMT-LIB format
                     ↓
         [Z3 / CVC5 SMT Solver]
           → Proves functional equivalence
                     ↓
      Target Backend (NumPy / TensorFlow / PyTorch / MLX / Gemmini / TPC-C)
```

### Core Mechanisms

**1. TensIR: A Verified Lifting Intermediate Language**  
TensIR is a carefully designed intermediate language that expresses tensor operations in a way that (a) captures the semantic intent of the original program, and (b) can be efficiently translated to multiple target backends. TensIR operations correspond to high-level functional primitives like `map`, `reduce`, `zip`, and `take`.

**2. Symbolic Synthesis via Rosette**  
[Rosette](https://emina.github.io/rosette/) is a solver-aided programming language built on top of Racket that provides facilities for symbolic execution, verification, and synthesis. Tenspiler encodes both the source program and candidate target programs symbolically in Rosette, then queries SMT solvers for a proof of equivalence.

**3. Loop Invariant Generation**  
To prove that a looping source program is equivalent to a non-looping tensor expression, Tenspiler must automatically generate **loop invariants** — mathematical predicates that hold at the start and end of every loop iteration. These invariants are the formal bridge between the scalar and tensor representations.

**4. SMT-LIB Backend (Z3 / CVC5)**  
The verification conditions are serialized to SMT-LIB 2.0 format and dispatched to Z3 or CVC5 for proof search.

### Supported Backends

| Backend | Platform |
|---|---|
| NumPy | CPU (Python) |
| TensorFlow | CPU / GPU |
| PyTorch | CPU / GPU |
| Apple MLX | Apple Silicon (M-series) |
| TPC-C | Intel Gaudi AI accelerator |
| Gemmini | RISC-V Deep Learning accelerator |

### Performance Results

Evaluated on 10 real-world benchmark suites including:
- Image blending modes (alpha compositing, screen blend, overlay, etc.)
- Llama-2 inference kernels (dot products, softmax, RMS normalization)

| Metric | Result |
|---|---|
| Average kernel execution speedup | **105×** over optimized sequential baseline |
| Average end-to-end speedup | **9.65×** over optimized sequential baseline |

### Critical Limitations

| Limitation | Technical Root Cause |
|---|---|
| **SMT solver timeouts on complex invariants** | Multi-dimensional loop nests require invariants over multi-dimensional index spaces — the number of quantified variables grows exponentially with loop depth |
| **Requires user-provided templates for hard cases** | Complex irregular control flow cannot produce invariants automatically; the user must supply partial specifications |
| **Restricted to relatively simple kernels** | Full automation is practically limited to 2D loops with straightforward index arithmetic |
| **Language scope** | Designed primarily for C++ and Python — converting Fortran requires additional preprocessing |

---

## 2.5 Third Generation: Tensorize (CGO 2025)

**Paper:** *Fast Synthesis of Tensor Programs from Legacy Code using Symbolic Tracing, Sketching and Solving*  
**Authors:** Alexander Brauckmann, Ludovic Jaulmes, José Wesley O. Magalhães, Elizabeth Polgreen, Michael F. P. O'Boyle  
**Venue:** CGO 2025 (International Symposium on Code Generation and Optimization)  
**Repository:** [alexanderb14/tensorize](https://github.com/alexanderb14/tensorize)

### Architecture

```
Legacy C (or Python) Code
          ↓
  [Polygeist / Numba-MLIR Frontend]
          ↓
     MLIR Affine IR
          ↓
  [Symbolic Tracer]
    → Builds a complete symbolic expression tree
    → Represents the program as a mathematical formula
          ↓
  [Sketch Generator]
    → Derives symbolic sketches from target DSL op definitions
    → Each sketch is a partial program with symbolic holes
          ↓
  [Algebraic Solver (SymPy)]
    → Recursive "sketch and solve" algorithm
    → Uses algebraic simplification to fill holes
    → Decomposes complex specs into subproblems
          ↓
  Lifted Target Program (Linalg / StableHLO / NumPy / PyTorch / JAX)
```

### Core Innovation: Symbolic Tracing + Algebraic Solving

The critical departure from prior work: instead of testing programs on sample inputs (mlirSynth) or generating formal SMT proofs (Tenspiler), Tensorize **represents both source and target programs as symbolic algebraic expressions** and uses **algebraic simplification** to determine equivalence.

**Example — Matrix Transposition:**
The symbolic trace of:
```c
for (i=0; i<M; i++)
  for (j=0; j<N; j++)
    B[j][i] = A[i][j];
```
produces a symbolic expression:
```
B[j][i] = A[i][j]  ∀ i ∈ [0,M), j ∈ [0,N)
```
The sketch for `linalg.transpose` produces:
```
output[?₁][?₂] = input[?₂][?₁]  ∀ ...
```
The algebraic solver instantiates `?₁ = j`, `?₂ = i` and confirms equivalence symbolically — **no runtime needed**.

### Three Key Properties

**1. Correctness by Construction**  
Because synthesis proceeds via algebraic identity (not I/O sampling), any discovered program is mathematically guaranteed to be equivalent to the source. No post-hoc verification step is needed.

**2. Linear Scalability**  
The recursive "sketch and solve" decomposes the synthesis problem into subproblems proportional to the number of operations in the target program. Runtime scales **linearly** with program length — a fundamental improvement over the exponential growth of bottom-up enumeration.

**3. Massive Performance Unlocked**

| Target | Benchmark | Geomean Speedup vs LLVM-O3 |
|---|---|---|
| Intel CPU | Polybench | **4.1×** |
| AMD CPU (Ryzen) | Polybench | **11.7×** |
| NVIDIA GPU | Polybench | **4,102×** |

Specific kernel highlights:
- Covariance computation: **29× to 148×** speedup (varies by dimensions)
- Benchmark suite coverage: lifted **96 of 99** Polybench kernels (97% success rate)

### Benchmarks Summary

| Polybench Kernel | CPU Speedup | GPU Speedup |
|---|---|---|
| `2mm` (matrix mult chain) | 3.8× | 890× |
| `covariance` | 29× – 148× | >1000× |
| `gemm` | 4.2× | 3210× |
| `syrk` | 5.1× | 2840× |
| `correlation` | 7.3× | 4102× |

### Remaining Limitations

| Limitation | Status |
|---|---|
| Dense tensor algebra only | Sparse patterns not handled |
| Statically known shapes | Dynamic shapes not supported |
| Does not synthesize scheduling/tiling strategies | Leaves hardware optimization to downstream compilers |
| Irregular control flow (branches inside loops) | Not handled |

---

## 2.6 Third Generation: STAGG — Guided Tensor Lifting (PLDI 2025)

**Paper:** *Guided Tensor Lifting*  
**Authors:** Yixuan Li, José Wesley O. Magalhães, Alexander Brauckmann, Michael F. P. O'Boyle, Elizabeth Polgreen  
**Venue:** PLDI 2025 (Proceedings of the ACM on Programming Languages)  
**Repository:** [BugBugSurvival/Guided-Tensor-Lifting](https://github.com/BugBugSurvival/Guided-Tensor-Lifting)

### Architecture: LLM-Guided Probabilistic Grammar + Enumeration

```
Legacy C Tensor Program
          ↓
  [Prompt Construction]
    → Formats source code as structured LLM prompt
          ↓
  [LLM Query] (GPT-4, Claude, etc.)
    → Returns k=10 candidate TACO expressions
          ↓
  [Probabilistic Context-Free Grammar (pCFG) Construction]
    → Parses LLM candidates
    → Counts production rule frequencies
    → Normalizes to probabilities:
      P[α → β] = weight(α → β) / Σ weight(α → γ)
          ↓
  [Weighted A* Enumeration]
    → Top-down or bottom-up search over the pCFG
    → Most probable structures explored first
          ↓
  [I/O Validation + CBMC Bounded Model Checking]
    → I/O testing on sample inputs
    → CBMC formal proof for correct candidates
          ↓
  Verified TACO Expression
```

### LLM Integration Details

STAGG uses an LLM not as a **direct code translator** (which has a >92% failure rate on complex tensors) but as a **statistical estimator of the target program's structure**. The LLM is queried once, its output is parsed to extract structural patterns (which tensors appear, which operations are used, what the index structure looks like), and these patterns are formalized into a pCFG that guides a rigorous synthesis search.

**Mathematical Formalization of pCFG:**

Let $G = (N, T, P, S)$ be a probabilistic context-free grammar where:
- $N$ = non-terminal symbols (tensor variables, operators)
- $T$ = terminal symbols (specific indices, constants)
- $P$ = production rules with associated probabilities
- $S$ = start symbol

For each production rule $\alpha \to \beta_1, \ldots, \beta_k$ where $\alpha \in N$:

$$P[\alpha \to \beta_i] = \frac{\text{count}(\alpha \to \beta_i \text{ in LLM outputs})}{\sum_j \text{count}(\alpha \to \beta_j \text{ in LLM outputs})}$$

The A\* enumerator then explores candidate programs in order of decreasing probability:

$$\text{priority}(p) = \prod_{(r_1, r_2, \ldots, r_n) \in p} P[r_i]$$

### Performance Results

| Benchmark Suite | Solved (STAGG) | Solved (C2TACO) | Solved (Tenspiler) |
|---|---|---|---|
| Combined 77-benchmark suite | **76/77 (99%)** | 67/77 (87%) | 52/77 (67%) |
| Average synthesis time | **3.19s** | 12.4s | 21.15s |

### Evaluation on Specific Benchmark Categories

| Category | STAGG Accuracy |
|---|---|
| Dense linear algebra (Polybench-style) | 100% (30/30) |
| Image processing kernels | 97% (29/30) |
| Deep learning inference kernels | 94% (17/18) |

### Critical Insight for LoopHole

STAGG **directly implements** the "LLM-predicted probabilistic grammar for guided enumerative synthesis" architecture described in the LoopHole long-term horizon. This means:

1. The core neuro-symbolic architecture is **already published and validated**
2. LoopHole cannot claim novelty by simply reimplementing STAGG's approach for TACO
3. **Novel contributions must extend STAGG**, not replicate it — specifically to sparse tensors, dynamic shapes, hardware-specific scheduling, or other open problems

---

## 2.7 Adjacent SOTA: QiMeng-Xpiler (2025)

**Paper:** *QiMeng-Xpiler: Transcompiling Tensor Programs for Deep Learning Systems with a Neural-Symbolic Approach*  
**Source:** arXiv 2505.02146

### Architecture
QiMeng-Xpiler is a neural-symbolic transcompiler specifically designed for **cross-framework DL code migration** (e.g., TensorFlow → PyTorch → JAX). It combines:
- Symbolic analysis of the source computation graph
- LLM-guided mapping of source ops to target framework ops
- Verification via execution on test inputs

### Relevance to LoopHole
- Demonstrates neural-symbolic synthesis beyond just C/Fortran → MLIR
- Shows that the neuro-symbolic paradigm generalizes across framework boundaries
- Result: ~87% automated translation accuracy on real-world DL codebases

---

## 2.8 Adjacent SOTA: LLMLift and Verified Deep Learning Lifting (2024–2025)

**Paper:** *Verified Lifting of Deep Learning Operators* (arXiv 2412.20992)

### Architecture
- Targets lifting of CUDA/HIP kernel implementations of DL operators (e.g., cuBLAS GEMM implementations) into high-level dialect Op representations
- Uses LLMs to generate initial high-level sketches, then applies formal verification to confirm correctness
- Demonstrates that LLM-assisted lifting can automate operator library porting

### Relevance
- Validates LLM use in the lifting pipeline
- Shows that LLMs can handle CUDA-level code, not just simple C loops
- Error rates ~8% for well-structured BLAS kernels (vs 92% for arbitrary tensor programs)

---

## 2.9 Adjacent SOTA: STENSO — Superoptimization via Symbolic Synthesis (2024)

**Paper:** *Tensor Program Superoptimization through Cost-Guided Symbolic Program Synthesis*

### Architecture
Extends the tensor synthesis paradigm to **superoptimization** — synthesizing not just semantically equivalent programs but programs that are both semantically equivalent and **computationally cheaper** based on a hardware cost model.

- Builds a cost-guided symbolic search: the synthesizer targets programs with lower operation count / memory traffic
- Demonstrates that synthesis-based approaches can automatically discover algorithmic optimizations (e.g., exploiting symmetry in matrix operations)

### Relevance
- Suggests that LoopHole's long-term vision could incorporate cost-guided synthesis for joint correctness + performance optimization
- A potential extension vector: synthesize Linalg ops + Transform dialect schedules jointly under a hardware cost model

---

## 2.10 Adjacent SOTA: Konrul — Guess, Measure & Edit (2024)

**Paper:** *Guess, Measure & Edit: Using Lowering to Lift Tensor Code*

### Architecture
A novel "lowering to lift" approach: instead of synthesizing the high-level program directly, Konrul:
1. Proposes a candidate high-level program (the "guess")
2. Lowers it back to scalar loops (the "measure")
3. Compares the lowered version to the original source, computes a diff, and uses the diff to guide edits (the "edit")

This bidirectional approximate-then-refine loop avoids the exponential search of bottom-up enumeration.

### Relevance
- Demonstrates that *lowering-for-verification* (the reverse direction of lifting) is a viable refinement mechanism
- Could be combined with LoopHole's symbolic approach to handle the post-synthesis refinement step

---

## 2.11 MLIR Sparse Tensor Dialect: The Open Frontier (2021–2026)

**Reference:** MLIR SparseTensor Dialect documentation; MPACT Research Group (Google)

### Current Status of Sparse Lifting
The MLIR `sparse_tensor` dialect provides:
- A first-class representation of sparse tensors as MLIR types
- Level types: `Dense`, `Compressed` (CSR/CSC), `Singleton`, `LooseCompressed`
- Iteration graphs and topological lattice-based code generation
- Integration with TACO-style iteration space co-iteration

**However, automated lifting of legacy sparse code into SparseTensor dialect is completely unsolved** as of February 2026:
- Existing tools (Tensorize, STAGG, Tenspiler) only handle dense tensors
- Sparse code features indirect indexing (`A[rowptr[i]]`), conditional skipping, and irregular memory access — none of which are handled by symbolic tracing or algebraic simplification designed for dense loops
- The SparseTensor dialect's iteration lattice semantics require inferring the **sparse storage format** (CSR, CSC, COO, BCSR...) from the loop pattern — a significantly harder problem than recognizing dense matrix multiply

This represents the **highest-priority novel research opportunity** for LoopHole.

---

## 2.12 MLIR Transform Dialect: Scheduling Synthesis Frontier (CGO 2025)

**Paper:** *The MLIR Transform Dialect: Your Compiler Is More Powerful Than You Think*  
**Authors:** Michel Steuwer et al.  
**Venue:** CGO 2025

### What is the Transform Dialect?
The Transform dialect provides a **composable, IR-level transformation scheduling language** that allows performance engineers to:
- Dictate precise tiling strategies (tile sizes, tile dimensions)
- Control loop fusion and fission
- Specify vectorization patterns (which dimensions become vector dimensions)
- Reuse and compose transformation sequences without modifying compiler source code

Example Transform dialect program to tile a matrix multiply:
```mlir
transform.sequence failures(propagate) {
^bb0(%module: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %module
    : (!transform.any_op) -> !transform.any_op
  %tiled, %loops = transform.structured.tile_using_for %matmul [8, 16, 32]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
}
```

### Why This Is Relevant
Current lifting tools only raise to Linalg/StableHLO and then delegate all optimization to downstream compilers (XLA, IREE). But these downstream compilers use generic heuristics that may not be optimal for the specific kernel being lifted.

**Synthesizing the Transform dialect schedule alongside the Linalg operation** would allow LoopHole to output not just "what the computation is" (the lifted op) but "how to execute it optimally on hardware X" (the schedule).

This joint lifting + scheduling synthesis is **completely unexplored** in the current literature.

---

## 2.13 Comparative SOTA Summary Table

| Feature | mlirSynth (PACT '23) | Tenspiler (ECOOP '24) | Tensorize (CGO '25) | STAGG (PLDI '25) | LoopHole (Target) |
|---|---|---|---|---|---|
| **Synthesis Mechanism** | Bottom-up enumeration | Symbolic (Rosette) | Symbolic tracing + algebraic solving | LLM-guided pCFG + A* | TBD (build on STAGG/Tensorize) |
| **Verification** | I/O observational equiv. | SMT (Z3/CVC5) formal proof | Correct by construction | CBMC + I/O | Formal (SMT or CBMC) |
| **Scalability** | O(|ops|^n) — poor | Poor on deep nests | Linear w.r.t. program length | 3.19s avg | Target: < 5s avg |
| **Dense tensors** | ✓ | ✓ | ✓ (97%) | ✓ (99%) | ✓ |
| **Sparse tensors** | ✗ | ✗ | ✗ | ✗ | **Target novel** |
| **Dynamic shapes** | ✗ | ✗ | ✗ | ✗ | **Target novel** |
| **Scheduling synthesis** | ✗ | ✗ | ✗ | ✗ | **Target novel** |
| **Target dialect** | Linalg / StableHLO | NumPy / TF / PT / MLX | Linalg / NumPy / JAX / PT | TACO | Linalg / StableHLO |
| **GPU speedup** | — | — | 4,102× | — | — |
| **CPU speedup** | 2.5–3.4× | — | 4.1–11.7× | — | — |
| **TPU speedup** | 21.6× | — | — | — | — |
| **LLM integration** | ✗ | ✗ | ✗ | ✓ (pCFG) | ✓ (novel extension) |

---

## 2.14 Key Takeaways for LoopHole Development

1. **The 3-month POC** (SMT-based verification against predefined sketches for 1D/2D convolutions) is technically feasible but academically limited — it replicates a restricted subset of both Tenspiler and Tensorize.

2. **The enumerative + LLM-guided architecture** (original long-term vision) has been preempted by STAGG for dense TACO tensors. LoopHole must differentiate.

3. **The highest-value novel contributions** are:
   - Sparse tensor synthesis (sparse_tensor dialect) — completely open
   - Dynamic shape lifting — completely open
   - Transform dialect schedule synthesis — completely open
   - Auto-documentation of lifted abstractions — completely open

4. **Best implementation path for a novel long-term system:** Build on top of STAGG's pCFG + A* core but extend it to handle SparseTensor iteration lattice synthesis and dynamic shape inference.
