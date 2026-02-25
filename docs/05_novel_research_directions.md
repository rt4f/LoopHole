# 05 — Novel Research Directions

> This document catalogues the **genuinely novel, unexplored research opportunities** in the MLIR program lifting ecosystem, validated against the February 2026 state of the art. Each direction is evaluated for novelty, feasibility, expected impact, and approximate research effort.

---

## 5.1 Priority Matrix

| Direction | Novelty | Feasibility | Impact | Effort (months) |
|---|---|---|---|---|
| **Sparse Tensor Synthesis** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 12–18 |
| **Transform Dialect Schedule Synthesis** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 9–12 |
| **Dynamic Shape Lifting** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 9–12 |
| **Auto-Documentation of Lifted Abstractions** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 6–9 |
| **Cross-Architecture Lifting** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 12–18 |
| **Lifting from CUDA to Linalg** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 12–15 |

---

## 5.2 Direction 1: Reverse Lifting of Sparse Algorithm Implementations

### Clarifying the Research Gap

> ⚠️ **Important framing note:** Searching "sparse tensor synthesis" returns hundreds of papers. Almost all of them address the *forward* direction — compiling tensor algebra specifications into efficient sparse code. The novelty described here is the *reverse* direction — recognizing that a piece of C code is implementing a sparse algorithm and lifting it back to the abstract tensor representation. These are entirely different problems.

The sparse tensor literature has two distinct research streams:

| Stream | Direction | Representative Work | What it takes as input |
|---|---|---|---|
| **Sparse compiler / code generation** | Tensor algebra → sparse C code | TACO (Kjolstad et al., OOPSLA 2017), MLIR SparseTensor sparsifier (Bik et al., ACM TACO 2022), Looplets (CGO 2023) | A *tensor algebra expression* you wrote (`y(i) += A(i,j)*x(j)`) + a format spec you specified |
| **Sparse format conversion synthesis** | Sparse format → sparse format C code | "Code Synthesis for Sparse Tensor Format Conversion" (CGO 2023) | Two explicitly named sparse formats |
| **❌ Sparse lifting (the gap)** | Handwritten C sparse implementation → tensor dialect | **Nothing exists** | A C function with `rowptr[]`, `col[]`, `val[]` arrays written by a human 20 years ago |

The key distinction: **TACO and the MLIR sparsifier both require that you already know and can express the computation in tensor algebra form**. They automate the *code generation* step. LoopHole's problem is the prior step: automatically recognizing, from existing C source code, that the code *is* a sparse computation and what its abstract tensor algebra semantics are.

### Problem Statement

The majority of legacy scientific simulation code is dominated by **sparse computations** — matrices where the vast majority of elements are zero. Decades of HPC code was written in C/Fortran by manually implementing sparse storage formats:

```c
// Compressed Sparse Row (CSR) Sparse Matrix-Vector Multiply
// Written by hand in the 1990s — no developer today wants to maintain this
for (int i = 0; i < nrows; i++) {
  for (int j = rowptr[i]; j < rowptr[i+1]; j++) {
    y[i] += val[j] * x[col[j]];  // indirect: x[col[j]], not x[j]
  }
}
```

This code cannot be handled by any existing lifting tool because:
- The inner loop bound (`rowptr[i]` to `rowptr[i+1]`) is **data-dependent** — not affine, so Affine dialect analysis fails entirely
- The memory access `x[col[j]]` involves **double indirection** — not expressible in AffineMap
- The sparse format (CSR, COO, BCSR...) is **implicit** in the array structure: no tool automatically recovers it

The existing lifting tools (Tensorize, STAGG, Tenspiler, mlirSynth) all hard-fail at Step 1 of their analysis pipelines on this code because their loop analysis assumes affine bounds and affine subscripts.

### Target Output

LoopHole would analyze the C source and emit the **input** to the MLIR sparsifier — the annotated `linalg.generic` that a human would have had to write manually to use TACO/MLIR today. The sparsifier then handles all downstream code generation automatically.

```mlir
// This is what LoopHole synthesizes from the C source above.
// A human today would have to write this by hand to use the MLIR sparsifier.
#CSR = #sparse_tensor.encoding<{lvlTypes = ["dense", "compressed"]}>
#COO = #sparse_tensor.encoding<{lvlTypes = ["compressed-nu", "singleton"]}>

func.func @spmv(%A: tensor<?x?xf64, #CSR>,
                %x: tensor<?xf64>) -> tensor<?xf64> {
  %c0 = arith.constant 0.0 : f64
  %y = tensor.empty(??) : tensor<?xf64>
  %y_zero = linalg.fill ins(%c0 : f64) outs(%y : tensor<?xf64>) -> tensor<?xf64>
  %result = linalg.generic {
    indexing_maps = [#map_sparse, #map_dense_in, #map_dense_out],
    iterator_types = ["parallel", "reduction"]
  } ins(%A, %x : tensor<?x?xf64, #CSR>, tensor<?xf64>)
    outs(%y_zero : tensor<?xf64>) {
  ^bb0(%a_val: f64, %x_val: f64, %y_val: f64):
    %prod = arith.mulf %a_val, %x_val : f64
    %sum  = arith.addf %y_val, %prod  : f64
    linalg.yield %sum : f64
  } -> tensor<?xf64>
  return %result : tensor<?xf64>
}
```

The SparseTensor compiler then handles:
- Traversal only over non-zero elements (skipping explicit zeros)
- Optimal memory access ordering
- Efficient co-iteration for SpGEMM (sparse × sparse)

### Research Challenges

**Challenge 1: Non-Affine Access Recognition**

The source loop contains `x[col[j]]` — a two-level indirect access. No existing symbolic tracer handles this. The key insight is that indirect accesses in sparse code encode the **sparse format structure**:

| Access Pattern | Sparse Format |
|---|---|
| `val[rowptr[i]..rowptr[i+1]-1]` | CSR (row-major) |
| `val[colptr[j]..colptr[j+1]-1]` | CSC (column-major) |
| `val[k]` with `row[k]` and `col[k]` unordered | COO |
| `val[bsr_ptr[i]*bs2 + r*bs + s]` | BSR (block sparse row) |

The research challenge is to **infer the storage format** from pattern analysis of the array access structure.

**Challenge 2: Storage Format Inference**

Proposed approach:
1. Identify all pointer/index arrays (arrays whose values are used as indices into other arrays)
2. Classify the access structure:
   - Level 1 (outer): dense loop → this is the "dense" level
   - Level 2 (inner): bounded by `ptr[i]` and `ptr[i+1]` → this is the "compressed" level
3. Verify classification by checking that `ptr[0] == 0` and `ptr[n] == nnz` are asserted somewhere
4. Map to `#sparse_tensor.encoding`

**Challenge 3: Iteration Lattice Synthesis**

For multi-operand sparse operations (e.g., sparse + sparse), the SparseTensor compiler constructs an **iteration lattice** that determines the co-iteration pattern (union, intersection, or asymmetric join). Inferring the correct lattice from the source requires understanding the mathematical operation being performed.

**Challenge 4: LLM Guidance for Format Classification**

Just as STAGG uses LLMs to predict tensor operation structure for dense programs, we can query an LLM with the source C code to predict:
- What sparse format is being used?
- What mathematical operation does this represent (SpMV, SpGEMM, SpMM)?

The LLM prediction can then be validated and formalized through pattern analysis.

### Why the Existing Sparse Literature Doesn't Close This Gap

- **TACO (Kjolstad et al.)**: Goes *forward* — you write `y(i) += A(i,j)*x(j)`, specify that `A` is CSR, and TACO generates the C loops. It does not ingest existing C loops.
- **MLIR SparseTensor sparsifier (Bik et al. 2022)**: You annotate a `linalg.generic` op with `#sparse_tensor.encoding`. The compiler lowers to efficient sparse traversal. Input is already tensor IR, not C source.
- **Looplets (CGO 2023)**: A new *language* for authoring sparse co-iteration patterns. Does not perform source code analysis.
- **"Code Synthesis for Sparse Tensor Format Conversion" (CGO 2023)**: Synthesizes code to *convert data between formats* (e.g., CSR→COO). Does not analyze existing algorithms.
- **Autoschedulers (Halide, TVM)**: Optimize schedules for dense computations. Explicitly exclude irregular (sparse) access patterns.

None of these tools answer the question: *"I have this C function — what is it doing, and can I replace it with a SparseTensor dialect operation?"*

### Expected Impact

- **First tool to automate reverse-lifting of handwritten sparse C code to the SparseTensor dialect** — a completely unexplored problem direction
- All existing TACO/MLIR-sparsifier speedups become available to legacy codebases *without rewriting* the algorithm in tensor notation
- Directly enables HPC codes (climate models, FEM solvers, graph analytics) to transparently run on GPU/TPU accelerators
- Once lifted, the MLIR sparsifier handles all downstream code generation: format-specific loop emission, co-iteration, vectorization
- Potential speedups: 10×–100× on GPU for typical sparse scientific kernels (SpMV, SpGEMM, SpMM)

### Key References to Study First

- **Bik et al. 2022** (arXiv:2202.04305) — MLIR SparseTensor sparsifier; understand the *target* representation and what the dialect can express
- **Kjolstad et al. 2017** — Original TACO paper; understand the iteration lattice semantics that the SparseTensor dialect inherits
- **Chou et al. 2018** — "Format abstraction for sparse tensor algebra compilers" (OOPSLA 2018); the `#sparse_tensor.encoding` level types originate here
- **CGO 2023 Looplets** — Structured coiteration; shows what a principled sparse traversal abstraction looks like
- **Su et al., "SPARSITY: An Optimizing Framework for Sparse Matrix Kernels"** — Early work on classifying sparse patterns from C code

---

## 5.3 Direction 2: Hardware-Specific Schedule Synthesis via Transform Dialect

### Problem Statement

Current lifting tools (Tensorize, STAGG) produce high-level Linalg or TACO operations and then delegate **all** performance optimization to downstream compilers (XLA, IREE, TACO runtime). These downstream compilers use generic heuristics that may not be optimal for the specific kernel shape + target hardware combination.

For example, the optimal tile sizes for a 128×128 matrix multiply differ between:
- Intel Skylake (L1 cache 32KB, AVX-512, 16 FP32 SIMD lanes) → tiles of [16, 16, 16]
- NVIDIA RTX 4090 (shared memory 100KB, Tensor Cores WMMA 16×16×16) → tiles of [64, 64, 32]
- Apple M3 Ultra (AMX matrix unit, NEON SIMD) → tiles of [8, 8, 32]

The Transform dialect allows expressing these hardware-specific schedules as MLIR IR. The research challenge is **automatically synthesizing the optimal Transform dialect sequence** alongside the lifted Linalg operation.

### Proposed Architecture

```
Lifted linalg.matmul + Hardware Profile (cache sizes, SIMD width, memory bandwidth)
             ↓
  [LLM Schedule Predictor]
     → Prompt: "Given a {M}×{N}×{K} matmul on {hardware}, suggest optimal Transform dialect tiling, packing, and vectorization schedule"
     → Returns: k=5 candidate Transform dialect sequences
             ↓
  [Schedule Verifier]
     → Compile each candidate schedule + op
     → Run on target hardware
     → Measure peak FLOPS / bandwidth utilization
             ↓
  [Schedule Ranker + Tuner]
     → Rank candidates by performance
     → Apply Bayesian optimization to refine tile size candidates
     → Output: optimal Transform dialect sequence
```

### Target Output Example

```mlir
// Synthesized jointly: the Linalg op + its optimal schedule for AVX-512 Intel CPU
func.func @matmul_128x128x128(
    %A: tensor<128x128xf32>,
    %B: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  %init = tensor.empty() : tensor<128x128xf32>
  %result = linalg.matmul 
    ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %result : tensor<128x128xf32>
}

// Synthesized Transform dialect schedule
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %module
    // LLM-predicted + auto-tuned: tile 64x64 outer, 16x16x16 inner (AMX)
    %tiled_l1, %loops:3 = transform.structured.tile_using_for %mm [64, 64, 64]
    %tiled_l2, %inner:3 = transform.structured.tile_using_for %tiled_l1 [8, 16, 4]
    // Pack for SIMD-friendly access patterns
    %packed = transform.structured.pack %tiled_l2 packed_sizes = [0, 0, 16]
    // Vectorize
    transform.structured.vectorize %packed 
      vector_sizes = [8, 16]
      vectorize_nd_extract
    // Hoist scalar computations out of vectorized loops (LICM)
    %func = transform.structured.match ops{["func.func"]} in %module
    transform.structured.hoist_redundant_vector_transfers %func
    transform.yield
  }
}
```

### Research Novelty

This is **completely unexplored** in the literature. All existing lifting tools separate "what" (the Linalg op) from "how" (the schedule). This direction proposes synthesizing them jointly, with the LLM providing structural priors about both the operation and its optimal execution strategy.

### Implementation Feasibility

- **High feasibility** relative to sparse synthesis
- Transform dialect is well-documented and stable
- LLM prompting for code generation is mature
- The main research challenge is the feedback loop: measuring actual hardware performance and using it to refine the synthesis

---

## 5.4 Direction 3: Dynamic Shape Inference and Lifting

### Problem Statement

Production ML systems (PyTorch 2.x, JAX) operate on **dynamically shaped tensors** where dimensions are not known at compile time. Legacy scientific code with dynamic allocation patterns (e.g., `int N = atoi(argv[1]); double *A = malloc(N*N*sizeof(double))`) similarly has runtime-determined shapes.

All existing lifting tools (Tensorize, STAGG, Tenspiler, mlirSynth) require **statically known dimensions** to construct the SMT formulas, algebraic specifications, or I/O test vectors used for verification.

### Approach

**Step 1: Symbolic Dimension Lifting**

Instead of encoding concrete dimensions (M=128, N=128, K=128), extend the symbolic tracer to propagate symbolic dimension variables:

```python
# Instead of:
C_formula = Sum(A[i,k] * B[k,j], k=0..K)  # K=128 concrete

# Use:
M, N, K = symbols('M N K', positive=True, integer=True)
C_formula = Sum(A[i,k] * B[k,j], (k, 0, K))  # K symbolic
```

**Step 2: Shape-Parametric Equivalence Verification**

Prove equivalence `∀ M, N, K ∈ ℤ⁺` using Z3 with universally quantified shape variables:

```python
M, N, K = Ints('M N K')
solver.add(M > 0, N > 0, K > 0)
solver.add(ForAll([i, j, k],
    Implies(
        And(0 <= i, i < M, 0 <= j, j < N, 0 <= k, k < K),
        C[i,j] == Sum(A[i,kk] * B[kk,j], kk=0..K)
    )
))
```

**Step 3: Dynamic Shape MLIR Output**

Emit Linalg operations with `?` dimensions and shape inference:

```mlir
func.func @lifted_matmul_dynamic(
    %A: tensor<?x?xf32>,
    %B: tensor<?x?xf32>
) -> tensor<?x?xf32> {
  %d0 = tensor.dim %A, %c0 : tensor<?x?xf32>  // M
  %d1 = tensor.dim %B, %c1 : tensor<?x?xf32>  // N
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %result = linalg.matmul
    ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
```

### Research Challenges

1. **Z3 quantifier alternation**: Adding universal quantification over M, N, K creates alternating quantifiers that exponentially increase SMT solving complexity
2. **Shape-dependent loop invariants**: Loop invariants in Tenspiler-style synthesis become parametric — harder to generate and check
3. **Runtime shape inference integration**: The lifted program must correctly propagate shape information through all intermediate operations

---

## 5.5 Direction 4: Auto-Documentation and Semantic Annotation of Lifted Abstractions

### Problem Statement

When legacy C code is lifted to high-level tensor operations, the resulting output may be semantically opaque to human maintainers. For example:

**Input (familiar to the engineer):**
```c
// 400 lines of nested loops computing covariance matrix
for (int i = 0; i < N; i++)
  for (int j = 0; j < N; j++) {
    cov[i][j] = 0.0;
    for (int k = 0; k < M; k++)
      cov[i][j] += data[k][i] * data[k][j];
    cov[i][j] /= (float_n - 1.0);
  }
```

**Output (semantically rich but unfamiliar):**
```mlir
linalg.generic { ... iterator_types = ["parallel", "parallel", "reduction"] }
```

A human maintainer may not immediately recognize that this is computing a **normalized covariance matrix** and may struggle to validate the lifted code or understand the lifted abstraction's mathematical semantics.

### Proposed System: Simultaneous Lifting + Documentation

Inspired by **LILO** (Learning Interpretable Libraries by Compressing and Documenting Code), this direction proposes building a system that:

1. Lifts the source loops to a high-level Linalg / StableHLO operation
2. Simultaneously generates:
   - A **LaTeX mathematical description** of the operation (e.g., $C_{ij} = \frac{1}{M-1} \sum_k D_{ki} D_{kj}$)
   - A **natural language docstring** (e.g., "Computes the sample covariance matrix of the data matrix D")
   - A **semantic operation tag** (e.g., "statistical/covariance", "linear-algebra/matrix-product")
   - **Shape annotations** noting pre/post conditions (e.g., "requires N×N symmetric output")

3. Embeds all documentation as MLIR operation attributes:

```mlir
linalg.generic {
  indexing_maps = [...],
  iterator_types = ["parallel", "parallel", "reduction"],
  // Auto-generated documentation attributes
  doc = "Covariance matrix: C[i,j] = (1/(M-1)) * sum_k(data[k,i] * data[k,j])",
  math_notation = "C_{ij} = \\frac{1}{M-1} \\sum_{k=0}^{M} D_{ki} D_{kj}",
  semantic_tags = ["statistics", "covariance", "symmetric-output"]
} ins(%data : ...) outs(%cov : ...) { ... }
```

### LLM Integration

The documentation generation step uses an LLM to:
1. Accept the lifted Linalg operation + its affine maps + iterator types as input
2. Generate natural language description, LaTeX formula, and semantic tags
3. Validate the description by checking it against known operation catalogues (e.g., if the op matches `linalg.matmul`, the description should be "matrix-matrix multiplication")

### Research Novelty

- **No existing lifting tool generates human-readable documentation**
- Directly addresses enterprise adoption barrier (engineers cannot validate black-box lifted code)
- Enables lifted code to serve as training data for future LLMs (documented, semantically rich ML-HPC code pairs)
- Novel application of LILO-style library compression to compiler IR

---

## 5.6 Direction 5: Cross-Architecture Lifting via Intermediate Semantic DSL

### Problem Statement

Existing lifting tools target a specific dialect (Linalg, StableHLO, TACO). There is no pipeline that can automatically select the **optimal target dialect** based on the deployment target:
- If deploying to Google TPU → lift to StableHLO
- If deploying to CPU with AVX-512 → lift to Linalg + Transform dialect schedule
- If deploying to AMD GPU → lift to Linalg + ROCm backend
- If deploying to ARM SVE → lift to Linalg + SVE vectorization

### Proposed Architecture

A **meta-lifter** that:
1. Lifts to a hardware-neutral semantic IR (a new TensIR-like intermediate language)
2. Has backend lowerers that translate the semantic IR to each target dialect
3. Uses a hardware profiling model to select the optimal target

This is architecturally novel and would unify the fragmented ecosystem of per-dialect lifting tools.

---

## 5.7 Choosing Your Research Track

**Recommendation for LoopHole project:**

Given the current state of the art (February 2026), **Track 1 (Sparse Tensor Synthesis)** offers the highest combination of academic novelty and real-world impact. The SparseTensor dialect is mature and the problem is well-defined, but no lifting tool addresses it. This is the clearest path to a top-venue publication (PLDI, ASPLOS, or SC).

**If sparse synthesis proves too difficult** (storage format inference ambiguity, non-affine access analysis), **Track 2 (Transform Dialect Schedule Synthesis)** is the most feasible novel contribution with strong publication potential at CGO or OOPSLA.

**For maximum near-term publishability with lower risk**, **Track 4 (Auto-Documentation)** can be combined with any lifting result to produce a novel systems contribution with a clear user-facing value proposition.
