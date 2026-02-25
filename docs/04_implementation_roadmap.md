# 04 — Implementation Roadmap

> This document provides a concrete, phased build plan for the LoopHole project. It spans from the 3-month Proof of Concept through a 6-month extension and into the long-term research phase.

---

## 4.1 Phase 0: Environment Setup (Week 1–2)

### 4.1.1 Prerequisites
- Python 3.10+ (recommended: 3.11 or 3.12)
- CMake 3.20+
- LLVM/MLIR built from source (or pre-built wheels)
- Z3 theorem prover (Python bindings)
- SymPy (for algebraic simplification in later phases)

### 4.1.2 MLIR / LLVM Setup

**Option A: Pre-built Python wheels (fast start)**
```bash
pip install mlir-python-bindings  # MLIR core Python bindings
pip install z3-solver             # Z3 SMT solver
pip install sympy                 # SymPy algebraic solver (Phase 2+)
```

**Option B: Build from source (required for custom dialect work)**
```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU;AArch64" \
  -DMLIR_ENABLE_PYTHON_BINDINGS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON
ninja -j$(nproc)
echo 'export PYTHONPATH=$(pwd)/tools/mlir/python_packages/mlir_core' >> ~/.bashrc
```

**Polygeist Setup (for C frontend)**
```bash
git clone --recursive https://github.com/llvm/Polygeist.git
cd Polygeist && mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
ninja -j$(nproc)
# Test: polygeist-opt --help
```

### 4.1.3 Verify Environment
```bash
# Test MLIR Python bindings
python3 -c "import mlir; print(mlir.__version__)"

# Test Z3
python3 -c "from z3 import *; s = Solver(); s.add(Int('x') > 1); print(s.check())"

# Test Polygeist on a hello-world C file
echo 'void add(int* A, int* B, int* C, int n) {
  for (int i=0; i<n; i++) C[i] = A[i] + B[i];
}' > test.c
polygeist-opt --convert-polygeist-to-mlir test.c
```

---

## 4.2 Phase 1: 3-Month Proof of Concept

**Goal:** Build a working Python tool that lifts C scalar loops for 1D/2D convolution and matrix operations into valid MLIR Linalg dialect using Z3-verified sketch matching.

### 4.2.1 Week 1–3: Affine IR Parser and Symbolic Extractor

**Deliverable:** Python class `AffineLoopExtractor` that ingests MLIR Affine IR (as text or programmatic representation) and extracts:

```python
@dataclass
class LoopNestInfo:
    """Extracted semantic information from an affine loop nest."""
    induction_vars: List[str]     # ['i', 'j', 'k']
    bounds: Dict[str, Tuple]      # {'i': (0, 'M'), 'j': (0, 'N'), ...}
    access_patterns: List[AccessPattern]  # read/write memrefs + AffineMap
    compute_ops: List[ComputeOp]  # arithmetic operations in innermost body
    output_var: str               # which memref is written (output)
    input_vars: List[str]         # which memrefs are read (inputs)
    reduction_vars: List[str]     # which iv's are reductions (present in output index)
```

**Implementation approach:**
1. Parse MLIR textual format using `mlir.ir.Module.parse(text)`
2. Walk the module with `module.operation.walk()` to find `affine.for` ops
3. For each loop nest, extract iteration bounds from `affine.for` attributes
4. Extract access patterns from `affine.load` / `affine.store` affine maps
5. Identify the compute payload from arith operations in the innermost body

**Key MLIR Python APIs:**
```python
from mlir.ir import Context, Module, Operation, Block
from mlir.dialects import affine

def extract_affine_maps(store_op: Operation) -> AffineMap:
    """Extract the affine map from an affine.store operation."""
    return AffineMap(store_op.attributes['map'])

def get_loop_bounds(for_op: Operation) -> Tuple[int, int, int]:
    """Extract (lower_bound, upper_bound, step) from affine.for."""
    lb = for_op.attributes['lower_bound']
    ub = for_op.attributes['upper_bound']
    step = for_op.attributes['step']
    return (lb, ub, step)
```

### 4.2.2 Week 4–6: Sketch Library Construction

**Deliverable:** A `SketchLibrary` containing formal symbolic sketches for target Linalg operations.

Each sketch encodes:
- The expected access pattern for each operand (as an AffineMap over symbolic dims m, n, k, ...)
- The compute payload (inner arithmetic)
- The iterator types (parallel / reduction)

```python
@dataclass
class OperationSketch:
    name: str                           # e.g., "linalg.matmul"
    num_inputs: int
    num_outputs: int
    dim_names: List[str]               # e.g., ["m", "n", "k"]
    indexing_maps: List[AffineMap]     # one per input + output
    iterator_types: List[str]          # "parallel" or "reduction"
    compute: ComputePayload            # inner arithmetic spec

SKETCH_LIBRARY = [
    OperationSketch(
        name="linalg.matmul",
        dim_names=["m", "n", "k"],
        indexing_maps=[
            AffineMap.parse("(m, n, k) -> (m, k)"),   # A
            AffineMap.parse("(m, n, k) -> (k, n)"),   # B
            AffineMap.parse("(m, n, k) -> (m, n)"),   # C
        ],
        iterator_types=["parallel", "parallel", "reduction"],
        compute=ComputePayload(ops=["mulf", "addf"], is_reduction=True)
    ),
    OperationSketch(
        name="linalg.transpose",
        dim_names=["i", "j"],
        indexing_maps=[
            AffineMap.parse("(i, j) -> (i, j)"),   # input
            AffineMap.parse("(i, j) -> (j, i)"),   # output
        ],
        iterator_types=["parallel", "parallel"],
        compute=ComputePayload(ops=["identity"], is_reduction=False)
    ),
    # ... 1D conv, 2D conv, dot product, etc.
]
```

**Target sketches for POC:**

| Sketch Name | Mathematical Operation | Linalg Op |
|---|---|---|
| `matmul` | C[m,n] += A[m,k] * B[k,n] | `linalg.matmul` |
| `transpose_2d` | B[j,i] = A[i,j] | `linalg.transpose` |
| `conv_1d` | C[n,w] += I[n,w+kw] * K[kw] | `linalg.conv_1d` |
| `conv_2d_nhwc` | C[n,h,w,f] += I[n,h+p,w+q,c] * K[p,q,c,f] | `linalg.conv_2d_nhwc_hwcf` |
| `dot_product` | c += sum(A[k] * B[k]) | `linalg.dot` |
| `elementwise_add` | C[i,j] = A[i,j] + B[i,j] | `linalg.map{arith.addf}` |
| `reduce_sum` | C[i] = sum_j A[i,j] | `linalg.reduce{arith.addf}` |

### 4.2.3 Week 7–9: Z3 Equivalence Verification Engine

**Deliverable:** `Z3EquivalenceChecker` that takes a `LoopNestInfo` and a `OperationSketch` and returns `EQUIVALENT`, `NOT_EQUIVALENT`, or `TIMEOUT`.

**Core approach:**

1. Translate the extracted loop nest into a Z3 formula representing the semantics:
   - For each write `C[f(i,j,k)] = expr(A[g(i,j,k)], B[h(i,j,k)])`, create a Z3 `ForAll` assertion
   - Handle reduction loops with a Z3 `Sum` construct or manual accumulation encoding

2. Translate the sketch into a Z3 formula representing the target operation's semantics

3. Check that the two formulas are logically equivalent (one implies the other and vice versa)

```python
from z3 import *
from typing import Optional

class Z3EquivalenceChecker:
    def __init__(self, timeout_ms: int = 5000):
        self.timeout_ms = timeout_ms
    
    def check(
        self, 
        loop_info: LoopNestInfo, 
        sketch: OperationSketch
    ) -> CheckResult:
        solver = Solver()
        solver.set("timeout", self.timeout_ms)
        
        # Symbolic index variables
        idx_vars = {name: Int(name) for name in loop_info.induction_vars}
        
        # Symbolic tensor access functions
        tensors = {
            var: Function(var, *([IntSort()] * rank + [RealSort()]))
            for var, rank in loop_info.tensor_ranks.items()
        }
        
        # Encode source program semantics
        source_formula = self._encode_loop_nest(loop_info, idx_vars, tensors)
        
        # Encode sketch semantics
        sketch_formula = self._encode_sketch(sketch, idx_vars, tensors)
        
        # Check equivalence: source ↔ sketch
        solver.add(Not(Implies(source_formula, sketch_formula)))
        
        if solver.check() == unsat:
            # No counter-example found — equivalent
            solver.reset()
            solver.add(Not(Implies(sketch_formula, source_formula)))
            if solver.check() == unsat:
                return CheckResult.EQUIVALENT
        
        if solver.check() == unknown:
            return CheckResult.TIMEOUT
        
        return CheckResult.NOT_EQUIVALENT
    
    def _encode_loop_nest(
        self, 
        loop_info: LoopNestInfo, 
        idx_vars: dict, 
        tensors: dict
    ) -> ExprRef:
        """Build Z3 formula from extracted loop nest info."""
        # Build bounds constraints
        bounds_constraints = []
        for var_name, (lb, ub) in loop_info.bounds.items():
            v = idx_vars[var_name]
            bounds_constraints.append(And(lb <= v, v < ub))
        bounds = And(*bounds_constraints)
        
        # Build compute expression
        compute_expr = self._build_compute_expr(
            loop_info.compute_ops, idx_vars, tensors
        )
        
        # Build output assignment assertion
        output_tensor = tensors[loop_info.output_var]
        output_idx = [idx_vars[v] for v in loop_info.access_patterns['output'].result_dims]
        
        return ForAll(
            list(idx_vars.values()),
            Implies(bounds, output_tensor(*output_idx) == compute_expr)
        )
```

**Limitation handling — why Z3 works for the POC:**
- POC is intentionally restricted to 1D and 2D shapes (static bounds)
- All memory accesses are affine — directly translatable to linear Z3 arithmetic
- No pointer arithmetic, no aliasing
- Reduction loops encoded as sum over a bounded integer range (solvable by Z3)

### 4.2.4 Week 10–11: MLIR Emitter

**Deliverable:** `LinalgEmitter` that takes a matched sketch + the original tensor shapes and emits valid MLIR Linalg IR.

```python
def emit_linalg_matmul(M: int, N: int, K: int) -> str:
    return f"""
    func.func @lifted_matmul(
        %A: memref<{M}x{K}xf32>,
        %B: memref<{K}x{N}xf32>,
        %C: memref<{M}x{N}xf32>
    ) {{
        linalg.matmul 
            ins(%A, %B : memref<{M}x{K}xf32>, memref<{K}x{N}xf32>)
            outs(%C : memref<{M}x{N}xf32>)
        func.return
    }}
    """
```

For `linalg.generic` (when named op doesn't exist):
```python
def emit_linalg_generic(sketch: OperationSketch, shapes: Dict[str, List[int]]) -> str:
    maps_str = "\n  ".join(
        f"#map_{i} = affine_map<{m}>" 
        for i, m in enumerate(sketch.indexing_maps)
    )
    return f"""
    {maps_str}
    linalg.generic {{
        indexing_maps = [{', '.join(f'#map_{i}' for i in range(len(sketch.indexing_maps)))}],
        iterator_types = [{', '.join(f'"{t}"' for t in sketch.iterator_types)}]
    }}
    ins(...) outs(...) {{
    ^bb0({sketch.block_args_str}):
        {sketch.compute_body_str}
        linalg.yield {sketch.yield_str}
    }}
    """
```

### 4.2.5 Week 12: Integration, Testing, and Benchmarking

**Deliverable:** End-to-end tool `loophole-lift` that accepts a C file and produces MLIR.

**CLI interface:**
```bash
# Full pipeline
loophole-lift --input matmul.c --output matmul_lifted.mlir --target linalg

# With verification output
loophole-lift --input conv2d.c --output conv2d_lifted.mlir --verify --verbose

# Batch mode on Polybench subset
loophole-lift --batch polybench/kernels/ --output-dir lifted/ --report
```

**Test suite:**
```
tests/
  unit/
    test_affine_extractor.py    # Test loop extraction from MLIR
    test_sketch_library.py      # Test sketch encoding
    test_z3_checker.py          # Unit tests for Z3 equivalence
    test_emitter.py             # Test MLIR output validity
  integration/
    test_matmul.py              # Full pipeline for matmul
    test_conv1d.py              # Full pipeline for 1D conv
    test_conv2d.py              # Full pipeline for 2D conv
    test_transpose.py           # Full pipeline for transpose
  benchmarks/
    polybench_subset/           # 10 selected Polybench kernels
    run_benchmarks.sh           # Compile and time lifted vs baseline
```

**POC Acceptance Criteria:**

| Test Case | Must Pass |
|---|---|
| 1D convolution (static shapes) | ✓ |
| 2D convolution NHWC (static shapes) | ✓ |
| Matrix multiply (M=N=K=128) | ✓ |
| Matrix transpose | ✓ |
| Z3 proves all at EQUIVALENT (not timeout) | ✓ |
| Output compiles with `mlir-opt` | ✓ |
| Lifted code runs correctly on CPU | ✓ |

---

## 4.3 Phase 2: 6-Month Extension (Months 4–6)

**Goal:** Scale the POC to handle the full Polybench suite, extend to StableHLO targeting, and begin the symbolic tracing approach (replacing Z3 with algebraic simplification for better scalability).

### 4.3.1 Symbolic Tracer (replaces/supplements Z3)

Inspired by **Tensorize**, implement a SymPy-based symbolic tracer:

1. Walk the Affine loop nest and build a **SymPy expression** representing the output:
   - `C[i,j]` becomes a SymPy Symbol indexed by `(i,j)`
   - `C[i,j] += A[i,k] * B[k,j]` becomes `C_ij = Sum(A[i,k] * B[k,j], (k, 0, K))`

2. For each sketch, build the corresponding SymPy expression

3. Use `sympy.simplify(source_expr - sketch_expr) == 0` to test equivalence algebraically

**Why this improves over Z3:**
- SymPy can handle multi-dimensional sums symbolically (Z3 struggles)
- Linear runtime with program length (vs Z3's potential exponential)
- Algebraically correct — no timeouts, no false negatives

### 4.3.2 StableHLO Target Support

Extend the sketch library and emitter with StableHLO operations:

```python
STABLEHLO_SKETCH_LIBRARY = [
    OperationSketch(
        name="stablehlo.dot_general",
        # ... batch dimensions, contracting dimensions
    ),
    OperationSketch(
        name="stablehlo.convolution",
        # ... feature dimensions, spatial dimensions, kernel dims
    ),
    OperationSketch(
        name="stablehlo.reduce",
        # ... computation region, dimensions to reduce
    ),
]
```

### 4.3.3 Polybench Full Suite Coverage

Target 90%+ coverage of Polybench linear algebra kernels:

```
Polybench/Linear-Algebra:
  blas/      → gemm, symm, syrk, syr2k, trmm
  kernels/   → 2mm, 3mm, atax, bicg, cholesky, doitgen,
               gesummv, mvt
  solvers/   → durbin, gramschmidt, lu, ludcmp, trisolv
Polybench/Stencils:
  adi, fdtd-2d, heat-3d, jacobi-1d, jacobi-2d
```

### 4.3.4 Dynamic Shape Exploration

Begin preliminary exploration of dynamic shapes:
- Identify which Z3 formulas remain decidable with symbolic dimension variables
- Explore MLIR's shape dialect (`shape.rank`, `shape.dim`) for shape inference integration
- Prototype a simple extension: if bounds are symbolic variables `M`, `N`, `K` (not concrete integers), can the sketch matching still work?

---

## 4.4 Phase 3: Long-Term Research (Months 7–24)

**Goal:** Build genuinely novel contributions beyond the current SOTA. Three candidate research tracks.

### Track A: Sparse Tensor Synthesis (Highest Impact, Highest Difficulty)

**Problem:** All existing lifting tools handle dense tensors only. The SparseTensor dialect is fully implemented but no automated lifter targets it.

**Approach:**
1. Build a **sparse indexing pattern recognizer** that identifies indirect indexing patterns in the source C code:
   ```c
   // CSR SpMV pattern
   for (i=0; i<nrows; i++)
     for (j=rowptr[i]; j<rowptr[i+1]; j++)
       y[i] += val[j] * x[col[j]];
   ```
   
2. Use LLM prompting to classify the storage format (CSR, COO, BCSR, etc.) from the array naming conventions and access structure

3. Map the recognized format to a `#sparse_tensor.encoding` attribute

4. Synthesize the target `linalg.generic` with the corresponding sparse encoding

**Target Output:**
```mlir
#CSR = #sparse_tensor.encoding<{lvlTypes = ["dense", "compressed"]}>
func.func @spmv(%A: tensor<?x?xf64, #CSR>, %x: tensor<?xf64>) -> tensor<?xf64> {
  %y = tensor.empty() : tensor<?xf64>
  %result = linalg.generic {
    indexing_maps = [affine_map<(i,j) -> (i,j)>,
                     affine_map<(i,j) -> (j)>,
                     affine_map<(i,j) -> (i)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%A, %x : tensor<?x?xf64, #CSR>, tensor<?xf64>)
    outs(%y : tensor<?xf64>) {
  ^bb0(%a: f64, %xi: f64, %yi: f64):
    %mul = arith.mulf %a, %xi : f64
    %add = arith.addf %yi, %mul : f64
    linalg.yield %add : f64
  } -> tensor<?xf64>
  return %result : tensor<?xf64>
}
```

### Track B: Transform Dialect Schedule Synthesis

**Problem:** Lifting tools produce high-level ops but leave performance optimization entirely to downstream compilers. For specific hardware targets, the downstream compiler may not choose optimal tile sizes or vectorization strategies.

**Approach:**
1. Profile the lifted Linalg op on target hardware to obtain performance data
2. Use an LLM to predict a Transform dialect schedule based on:
   - The lifted operation type (matmul, conv, etc.)
   - The target hardware (GPU compute capability, CPU SIMD width, cache sizes)
   - The tensor shapes
3. Verify the schedule by comparing output with expected results (schedules are correctness-preserving by construction)
4. Fine-tune the schedule with auto-tuning (random search or Bayesian optimization)

**Target Output:**
```mlir
// Lifted operation + synthesized schedule
module attributes {transform.with_named_sequence} {
  // ... the linalg.matmul function ...
  
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %module
    // LLM-predicted tile sizes for NVIDIA RTX 4090
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul [64, 64, 32]
    %packed = transform.structured.pack %tiled
      packed_sizes = [8, 16, 4]    // WMMA tile sizes
    transform.structured.vectorize %packed
    transform.yield
  }
}
```

### Track C: Dynamic Shape Lifting

**Problem:** Static-shape assumption is valid for batch processing but fails for production DL inference where batch sizes vary at runtime. All existing lifting tools require statically-known tensor dimensions.

**Approach:**
1. Extend SymPy-based symbolic tracer to handle symbolic dimension variables
2. Emit Linalg ops with `?` (dynamic) dimension markers instead of concrete sizes
3. Use MLIR's shape inference framework to propagate shape information through the lifted program
4. Verify correctness for parameterized families of inputs (∀ M, N, K: lifted(M,N,K) ≡ source(M,N,K))

---

## 4.5 Milestone and Timeline Summary

```
Month 1-2:   Environment setup + Affine IR parser + Sketch library (basic)
Month 3:     Z3 equivalence checker + MLIR emitter + POC integration testing
Month 4:     SymPy symbolic tracer + extend sketch library to 20+ ops
Month 5:     StableHLO target support + Polybench full suite
Month 6:     Dynamic shape prototype + benchmarking + Phase 1 write-up
Month 7-9:   Track A or B or C selection + deep research implementation
Month 10-12: Evaluation against STAGG/Tensorize baselines
Month 13-18: Paper writing + submission (PLDI, CGO, OOPSLA, or similar)
Month 19-24: Further extensions, open-source release, community adoption
```

---

## 4.6 Risk Assessment and Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Z3 timeouts on 2D convolution invariants | Medium | High | Restrict POC to statically bounded small sizes; fallback to SymPy |
| Polygeist fails to parse complex C | Medium | Medium | Pre-process C with clang-format; manually inline complex macros |
| MLIR API changes breaking Python bindings | Low | Medium | Pin mlir-python-bindings to a specific version; maintain venv |
| STAGG / Tensorize publish extensions preempting novel work | Medium | High | Choose Track A (sparse) — least risk of preemption given difficulty |
| Sparse storage format inference is ambiguous | High | High | Use LLM + user annotation for disambiguation; don't attempt fully automated format inference in first iteration |
| SMT solver cannot handle reduction invariants | High | High | Scope Z3 strictly to non-reduction equivalences; use SymPy's Sum for reductions |
