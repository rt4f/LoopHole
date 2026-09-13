# 03 — MLIR Architecture Deep Dive

> This document provides the technical reference for every MLIR infrastructure component used by the LoopHole project. It covers the relevant dialects, the compilation pipeline from legacy C to accelerator-ready IR, and the MLIR APIs required for implementation.

---

## 3.1 MLIR Overview

**MLIR** (Multi-Level Intermediate Representation) was developed at Google and contributed to the LLVM project (2019–present). It solves the core problem of **IR fragmentation** in modern compiler toolchains — where each framework maintained its own incompatible IR (TensorFlow Graph, XLA HLO, Halide, LLVM IR), making cross-framework optimization impossible.

MLIR's core design principles:

| Principle | Meaning |
|---|---|
| **Parsimony** | Minimal IR constructs; expressiveness comes from dialects |
| **Progressivity** | Compilation proceeds through a sequence of progressive IR lowering steps |
| **Traceability** | Source location and semantic metadata preserved throughout the pipeline |
| **Extensibility** | New operations, types, and attributes can be defined without modifying core infrastructure |

### The Dialect System

A **dialect** in MLIR is a modular namespace containing:
- **Operations** (`op`): units of computation (e.g., `affine.for`, `linalg.matmul`)
- **Types**: custom type definitions (e.g., `memref<?xf32>`, `tensor<4x4xf64>`)
- **Attributes**: compile-time constants and metadata (e.g., affine maps, iterator types)

Dialects can be mixed freely in the same IR module. This enables the "mixed abstraction" compilation model where a high-level `linalg.matmul` decomposes progressively to `scf.for` loops which decomposes to `llvm.call @cblas_sgemm` or vector intrinsics.

---

## 3.2 The Compilation Pipeline for LoopHole

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LEGACY C / C++ / FORTRAN                         │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │  [Polygeist Frontend]
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│   MLIR Affine Dialect + SCF Dialect + Arith Dialect + MemRef Dialect    │
│   (Scalar loop representation — element-level semantics)                 │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │  [LoopHole Lifter]
                                   │  (Symbolic Tracing → Sketch → Solve)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│           MLIR Linalg Dialect  OR  StableHLO Dialect                     │
│           (Tensor-level semantics — mathematical intent preserved)        │
└──────────────────────────────────┬──────────────────────────────────────┘
                  ┌────────────────┴────────────────┐
                  ▼                                  ▼
    [XLA Compiler (via StableHLO)]     [IREE Compiler (via Linalg)]
                  │                                  │
          ┌───────┴──────┐                  ┌────────┴────────┐
          ▼              ▼                  ▼                 ▼
      TPU Binary    GPU Binary          GPU Binary        CPU Binary
     (Google TPU)  (NVIDIA/AMD)        (Vulkan/CUDA)    (x86/ARM)
```

---

## 3.3 Dialect Reference: Affine Dialect

**Purpose:** Represents structured loop nests and memory accesses using polyhedral constraints. This is the **input representation** for the LoopHole lifter.

### Key Concepts

**`affine.for`** — A structured loop with affine bounds:
```mlir
// for (i = 0; i < 128; i++) { ... }
affine.for %i = 0 to 128 {
  // loop body
}
```

**`affine.load` / `affine.store`** — Memory access with an affine map describing the index computation:
```mlir
// A[i][j] read
%val = affine.load %A[%i, %j] : memref<128x128xf32>
// B[j][i] write (transposition access pattern)
affine.store %val, %B[%j, %i] : memref<128x128xf32>
```

**`AffineMap`** — A first-class MLIR attribute encoding a mapping from loop induction variables to memory indices:
```
affine_map<(i, j) -> (j, i)>   // transpose access
affine_map<(i, j, k) -> (i, k)>  // first matrix of matmul
```

### What Polygeist Produces

Given this C code:
```c
for (int i = 0; i < M; i++)
  for (int j = 0; j < N; j++)
    for (int k = 0; k < K; k++)
      C[i][j] += A[i][k] * B[k][j];
```

Polygeist produces:
```mlir
affine.for %i = 0 to %M {
  affine.for %j = 0 to %N {
    affine.for %k = 0 to %K {
      %a = affine.load %A[%i, %k] : memref<?x?xf32>
      %b = affine.load %B[%k, %j] : memref<?x?xf32>
      %c = affine.load %C[%i, %j] : memref<?x?xf32>
      %mul = arith.mulf %a, %b : f32
      %add = arith.addf %c, %mul : f32
      affine.store %add, %C[%i, %j] : memref<?x?xf32>
    }
  }
}
```

### Analysis Primitives Available in the Affine Dialect

| Analysis | API | Purpose |
|---|---|---|
| `AffineLoopAnalysis` | `mlir::affine::AffineLoopAnalysis` | Analyze loop bounds, step, induction variable |
| `DependenceAnalysis` | `mlir::affine::DependenceAnalysis` | Detect data dependencies between iterations |
| Polyhedral extraction | `mlir::affine::extractForInductionVars` | Extract all loop induction variables |
| Alias analysis | `mlir::AliasAnalysis` | Determine if two memrefs alias |

---

## 3.4 Dialect Reference: SCF (Structured Control Flow) Dialect

**Purpose:** A lower-level loop representation that serves as the bridge between Affine and LLVM IR. Used when Affine bounds are not fully affine.

### Key Operations

```mlir
// scf.for with explicit bounds
%result = scf.for %iv = %lb to %ub step %step iter_args(%arg = %init) -> f32 {
  %new = arith.addf %arg, %val : f32
  scf.yield %new : f32
}

// scf.if for conditional execution
%result = scf.if %cond -> f32 {
  scf.yield %then_val : f32
} else {
  scf.yield %else_val : f32
}

// scf.parallel for parallel loops
scf.parallel (%i, %j) = (%c0, %c0) to (%M, %N) step (%c1, %c1) {
  // body
}
```

---

## 3.5 Dialect Reference: Linalg Dialect

**Purpose:** High-level structured tensor/buffer computation dialect. This is the **primary lifting target** for the LoopHole POC.

### Design Principles

Linalg is designed around **structured operations** — operations where:
1. The iteration space is fully determined by the operand types
2. The access patterns are described by affine maps
3. The computation payload is generic (any region body)
4. The output is always a perfectly nested write to the entire output operand

### The `linalg.generic` Operation

The foundation of the entire Linalg dialect. Every named op (`linalg.matmul`, `linalg.conv_2d`, etc.) is a specialization of `linalg.generic`.

**Anatomy:**
```mlir
#map_A = affine_map<(m, n, k) -> (m, k)>
#map_B = affine_map<(m, n, k) -> (k, n)>
#map_C = affine_map<(m, n, k) -> (m, n)>

linalg.generic {
  indexing_maps = [#map_A, #map_B, #map_C],
  iterator_types = ["parallel", "parallel", "reduction"]
}
ins(%A : memref<?x?xf32>, %B : memref<?x?xf32>)
outs(%C : memref<?x?xf32>) {
^bb0(%a : f32, %b : f32, %c : f32):
  %mul = arith.mulf %a, %b : f32
  %add = arith.addf %c, %mul : f32
  linalg.yield %add : f32
}
```

This represents: `C[m,n] += A[m,k] * B[k,n]` (matrix multiplication)

**`iterator_types`:**
- `"parallel"` — loop dimension is independent; can be parallelized / vectorized
- `"reduction"` — loop dimension reduces into the output; cannot be reordered without special handling
- `"window"` — for convolution window dimensions

### Key Named Operations in Linalg

| Operation | C Equivalent | Indexing Pattern |
|---|---|---|
| `linalg.matmul` | `C[i][j] += A[i][k] * B[k][j]` | `(m,n,k) → (m,k), (k,n), (m,n)` |
| `linalg.transpose` | `B[j][i] = A[i][j]` | `(i,j) → (j,i), (i,j)` |
| `linalg.conv_1d` | 1D sliding window convolution | `(n,w,kw) → (n,w+kw), (kw), (n,w)` |
| `linalg.conv_2d_nhwc_hwcf` | 2D image convolution (NHWC layout) | Complex 4D + kernel dims |
| `linalg.dot` | `c += sum(A[i] * B[i])` | `(k) → (k), (k), ()` |
| `linalg.fill` | `memset` | `() → (), ()` |
| `linalg.reduce` | Reduction over one dimension | `(i,j) → (i,j), (i)` with reduction over `j` |
| `linalg.map` | Elementwise f(A) → B | `(i,j) → (i,j), (i,j)` |

### Key Transformations Available on Linalg

| Transformation | API | Hardware Impact |
|---|---|---|
| **Tiling** | `transform.structured.tile_using_for` | Cache reuse, parallelism |
| **Vectorization** | `transform.structured.vectorize` | SIMD utilization |
| **Fusion** | `transform.structured.fuse_into_containing_op` | Reduced memory traffic |
| **Packing** | `linalg.pack` / `linalg.unpack` | SIMD-friendly memory layouts |
| **Lowering to loops** | `-convert-linalg-to-loops` pass | Debug / fallback |
| **Bufferization** | `-one-shot-bufferize` | Tensor → memref materialization |

---

## 3.6 Dialect Reference: StableHLO Dialect

**Purpose:** A stable, versioned, fully-specified dialect serving as the portability layer between ML frameworks (JAX, PyTorch, TensorFlow) and ML compilers (XLA, IREE). Used as the **secondary lifting target** for operations that target TPUs or need XLA compatibility.

**Specification:** ~100 operations with complete formal specifications, verifiers, and type inference rules. Backward compatibility guaranteed for 5 years; forward compatibility for 2 years.

### Key Operations for LoopHole

| StableHLO Op | Mathematical Operation | When to Target |
|---|---|---|
| `stablehlo.dot_general` | General tensor contraction (covers matmul, dot, batch matmul) | Matrix operations targeting TPU/XLA |
| `stablehlo.convolution` | N-dimensional convolution with full dimension numbering | DNN convolutions targeting TPU |
| `stablehlo.transpose` | Tensor dimension permutation | Reordering operations |
| `stablehlo.reduce` | Reduction over specified dimensions | Sum/max/min reductions |
| `stablehlo.map` | Elementwise function (requires a computation region) | Custom elementwise ops |
| `stablehlo.pad` | Tensor padding with edge/interior values | Pre-convolution padding |
| `stablehlo.reshape` | Rank-preserving shape change | Layout changes |
| `stablehlo.slice` | Tensor slicing | Subarray extraction |
| `stablehlo.broadcast_in_dim` | Broadcast to a larger shape | Compatible with BLAS broadcasting rules |

### Linalg vs StableHLO: When to Choose Which

| Criterion | Use Linalg | Use StableHLO |
|---|---|---|
| **Primary target hardware** | CPU, AMD GPU, ARM | Google TPU, NVIDIA GPU via XLA |
| **Optimization focus** | Cache-aware tiling, SIMD | JIT-compiled accelerator dispatch |
| **Control over scheduling** | Full (Transform dialect) | Limited (delegated to XLA) |
| **Backwards compatibility** | Evolves with MLIR | Guaranteed 5-year stability |
| **Ecosystem maturity** | Research/production hybrid | Production-ready |

---

## 3.7 Dialect Reference: SparseTensor Dialect

**Purpose:** First-class representation of sparse tensors as MLIR types with fully automated code generation for compressed storage formats.

### Sparse Type System

Sparse tensors are annotated with a `#sparse_tensor.encoding` attribute:

```mlir
// CSR (Compressed Sparse Row) matrix
#CSR = #sparse_tensor.encoding<{
  lvlTypes = ["dense", "compressed"]
}>
func.func @spmv(%A: tensor<M x N x f64, #CSR>,
                %x: tensor<N x f64>,
                %y: tensor<M x f64>) -> tensor<M x f64>
```

### Level Types

| Level Type | Description | Use Case |
|---|---|---|
| `dense` | Standard dense dimension | Row dimension in CSR |
| `compressed` | Compressed (pointer + index arrays) | Column dimension in CSR |
| `singleton` | Exactly one stored element per dimension | Diagonal matrices |
| `loose_compressed` | Non-monotone compressed | COO-like formats |
| `block_compressed` | Block-structured compression | BCSR, BCSC |

### Iteration Lattice

The SparseTensor dialect generates iteration code using **co-iteration lattices** — when two sparse tensors are combined, the lattice determines which positions must be visited (union for add, intersection for multiply).

### Why Lifting to SparseTensor Is Hard

1. The source C code uses indirect indexing (`A[rowptr[i]]`, `A[colind[j]]`) — not affine
2. The storage format (CSR, COO, BCSR) must be *inferred* from the indexing pattern
3. Algebraic solvers (SymPy, Z3) cannot reason about non-affine indirect array accesses
4. Iteration lattice synthesis requires recognizing the *merge semantics* of two sparse iteration schemes

---

## 3.8 Dialect Reference: Transform Dialect

**Purpose:** A structured, IR-level language for precisely composing and sequencing compiler optimizations without modifying the compiler source code.

### Core Concept

Transform operations take **handles** to IR operations and apply transformations. The transformation sequence itself is represented as MLIR IR:

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    // Match the matmul operation
    %matmul = transform.structured.match ops{["linalg.matmul"]} 
              in %module : (!transform.any_op) -> !transform.any_op
    
    // Tile it with [8, 4, 8] tile sizes
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul [8, 4, 8]
              : (!transform.any_op) -> (!transform.any_op, 
                                        !transform.any_op,
                                        !transform.any_op,
                                        !transform.any_op)
    
    // Vectorize the tiled operation
    transform.structured.vectorize %tiled : !transform.any_op
    
    transform.yield
  }
}
```

### Key Transform Operations

| Transform Op | What It Does |
|---|---|
| `transform.structured.match` | Find operations in the IR matching a predicate |
| `transform.structured.tile_using_for` | Tile a structured op with given tile sizes |
| `transform.structured.vectorize` | Convert structured op to vector dialect |
| `transform.structured.fuse_into_containing_op` | Fuse producer into consumer |
| `transform.structured.pad` | Pad operands to multiples of tile size |
| `transform.structured.pack` | Repack data into SIMD-friendly layouts |
| `transform.structured.lower_pack` | Lower pack operations to loops |
| `transform.loop.get_parent_for` | Navigate to enclosing loop |
| `transform.loop.pipeline` | Apply software pipelining to a loop |

---

## 3.9 MemRef Type System

**`memref`** is MLIR's typed, strided memory reference — the primary type for buffer-based computation in Linalg.

### `memref` Type Syntax
```mlir
memref<DxDx...xElemType>           // fully static
memref<?x?xf32>                     // fully dynamic
memref<4x?xf32, strided<[?, 1], offset: ?>>  // with explicit strided layout
```

### Strided Layout
A strided layout encodes the memory layout: `element at [i,j] is at offset + i*stride[0] + j*stride[1]`.

- Identity layout (row-major): `strided<[N, 1], offset: 0>`
- Column-major (Fortran): `strided<[1, M], offset: 0>`
- Transposed view: `strided<[1, M], offset: 0>` (same memory, different strides)

---

## 3.10 Tensor Type System

**`tensor`** is MLIR's immutable, value-semantic tensor type — used in the tensor-based (as opposed to buffer-based) programming model.

```mlir
tensor<4x4xf32>     // static shape
tensor<?x?xf32>     // dynamic shape
tensor<*xf32>       // unranked
```

In Linalg with tensors (preferred for optimization):
- Inputs are `tensor` (value semantics, no aliasing concerns)
- Outputs are also `tensor`, and the operation returns a new tensor
- **Bufferization** converts `tensor` to `memref` at a later lowering stage

---

## 3.11 Key MLIR Passes Relevant to LoopHole

| Pass | Flag | Purpose |
|---|---|---|
| Affine → SCF lowering | `-lower-affine` | Convert `affine.for` to `scf.for` |
| Linalg → Loops | `-convert-linalg-to-loops` | Lower Linalg to SCF for debugging |
| Linalg → Vector | `-convert-linalg-to-vector` | Vectorize Linalg before lowering |
| Vector → LLVM | `-convert-vector-to-llvm` | Emit AVX/SVE vector intrinsics |
| Bufferization | `-one-shot-bufferize` | Convert tensor-based to memref-based |
| StableHLO → Linalg | via `stablehlo-legalize-to-linalg` | Convert StableHLO to Linalg for CPU |
| Linalg → LLVM | Full pipeline | End-to-end CPU code generation |

### Minimal compilation pipeline to executable (CPU):
```bash
mlir-opt input.mlir \
  -one-shot-bufferize \
  -convert-linalg-to-loops \
  -lower-affine \
  -convert-scf-to-cf \
  -convert-cf-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  | mlir-translate --mlir-to-llvmir \
  | llc -filetype=obj \
  | clang -o output
```

---

## 3.12 MLIR Python Bindings

The LoopHole POC is built in Python using MLIR's Python bindings (`mlir-core` package).

### Installation
```bash
pip install mlir-python-bindings
# or from source: CMAKE -DMLIR_ENABLE_PYTHON_BINDINGS=ON
```

### Key Python APIs for LoopHole

```python
from mlir.ir import Context, Module, InsertionPoint
from mlir.dialects import linalg, tensor, arith, affine, func
from mlir.passmanager import PassManager

# Create an MLIR context and load required dialects
ctx = Context()
ctx.allow_unregistered_dialects = False

# Parse MLIR from text
with ctx:
    module = Module.parse("""
        func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, 
                          %C: memref<?x?xf32>) {
          linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
                        outs(%C : memref<?x?xf32>)
          func.return
        }
    """)

# Apply a pass pipeline
pm = PassManager.parse("builtin.module(one-shot-bufferize,convert-linalg-to-loops)")
pm.run(module.operation)
```

### Z3 Python Integration

```python
from z3 import *

# Create symbolic variables for loop indices
i, j, k = Ints('i j k')
M, N, K = Ints('M N K')

# Symbolic access patterns
A_access = Function('A', IntSort(), IntSort(), RealSort())  # A[i][k]
B_access = Function('B', IntSort(), IntSort(), RealSort())  # B[k][j]
C_access = Function('C', IntSort(), IntSort(), RealSort())  # C[i][j]

# Equivalence verification
solver = Solver()
solver.add(ForAll([i, j, k], 
    Implies(
        And(0 <= i, i < M, 0 <= j, j < N, 0 <= k, k < K),
        C_access(i, j) == C_access(i, j) + A_access(i, k) * B_access(k, j)
    )
))
result = solver.check()
```

---

## 3.13 Integration Diagram: LoopHole Within MLIR Ecosystem

```
Input: C source file
       ↓
  [Polygeist CLI]     clang -O1 -emit-mlir source.c -o source.mlir
       ↓
  MLIR Affine IR      (affine.for + affine.load/store + arith ops)
       ↓
  [LoopHole Lifter]
  ┌─────────────────────────────────────────────┐
  │  1. Parse Affine IR via Python bindings      │
  │  2. Extract symbolic trace (loop bounds,     │
  │     access maps, compute payload)            │
  │  3. Construct operation sketch library       │
  │     from Linalg/StableHLO TableGen defs      │
  │  4. Call Z3 to check sketch equivalence      │
  │  5. Instantiate matched sketch               │
  │  6. Emit valid MLIR Linalg / StableHLO       │
  └─────────────────────────────────────────────┘
       ↓
  Lifted MLIR Output  (linalg.matmul / stablehlo.dot_general / ...)
       ↓
  [Standard MLIR Pass Pipeline]
       ↓
  Target Binary (CPU / GPU / TPU)
```
