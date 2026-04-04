"""
sketch_library.py — Formal operation sketch library for target Linalg/StableHLO ops.

Each OperationSketch encodes the complete semantic signature of a target tensor
operation:
  - How many induction variables it has (and their names/roles)
  - The affine maps describing how each operand is indexed
  - Iterator types (parallel / reduction)
  - The compute payload (what arithmetic is done in the innermost body)
  - A structural fingerprint used for fast pre-filtering

The sketch matching pipeline:
  1. Structural pre-filter: num_loops, num_reads, num_writes match
  2. Access pattern match: loop IV roles and index expressions align
  3. Compute payload match: arithmetic op sequence is consistent
  4. Formal verification: Z3 proves equivalence (or SymPy algebraic check)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


class IteratorType(str, Enum):
    PARALLEL = "parallel"
    REDUCTION = "reduction"
    WINDOW = "window"


class ComputePayloadType(str, Enum):
    """High-level characterisation of the inner computation."""
    MULTIPLY_ACCUMULATE = "multiply_accumulate"  # C += A * B
    COPY = "copy"                                 # B = A  (transpose/copy)
    ADD = "add"                                   # C = A + B
    SUBTRACT = "subtract"                         # C = A - B
    MULTIPLY = "multiply"                         # C = A * B
    MAX = "max"                                   # C = max(A, B)
    MIN = "min"                                   # C = min(A, B)
    RELU = "relu"                                 # C = max(A, 0)
    SCALE = "scale"                               # C = alpha * A
    NEGATE = "negate"                             # C = -A
    ACCUMULATE_ADD = "accumulate_add"             # C += A  (reduce sum body)
    ACCUMULATE_MAX = "accumulate_max"             # C = max(C, A)


@dataclass
class OperationSketch:
    """
    Formal semantic sketch for a single tensor dialect operation.

    indexing_maps: list of strings like "(m, n, k) -> (m, k)" — one per
                   operand in the order [inputs..., outputs...]
    """
    name: str                              # e.g. "linalg.matmul"
    dialect: str                           # "linalg" | "stablehlo"
    dim_names: List[str]                   # ordered loop IV names in the sketch
    indexing_maps: List[str]               # affine_map strings
    iterator_types: List[IteratorType]
    compute_payload: ComputePayloadType
    num_inputs: int
    num_outputs: int
    description: str                       # human-readable summary

    # Structural hints for fast pre-filtering
    @property
    def num_loops(self) -> int:
        return len(self.dim_names)

    @property
    def num_parallel(self) -> int:
        return sum(1 for t in self.iterator_types if t == IteratorType.PARALLEL)

    @property
    def num_reduction(self) -> int:
        return sum(1 for t in self.iterator_types if t == IteratorType.REDUCTION)

    @property
    def total_operands(self) -> int:
        return self.num_inputs + self.num_outputs

    def parallel_dims(self) -> List[str]:
        return [
            self.dim_names[i]
            for i, t in enumerate(self.iterator_types)
            if t == IteratorType.PARALLEL
        ]

    def reduction_dims(self) -> List[str]:
        return [
            self.dim_names[i]
            for i, t in enumerate(self.iterator_types)
            if t == IteratorType.REDUCTION
        ]


# ---------------------------------------------------------------------------
# Sketch definitions — all target operations
# ---------------------------------------------------------------------------

# ----- Dense Linear Algebra ------------------------------------------------

SKETCH_MATMUL = OperationSketch(
    name="linalg.matmul",
    dialect="linalg",
    dim_names=["m", "n", "k"],
    indexing_maps=[
        "(m, n, k) -> (m, k)",   # A (input)
        "(m, n, k) -> (k, n)",   # B (input)
        "(m, n, k) -> (m, n)",   # C (output, read+write for accumulation)
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.REDUCTION],
    compute_payload=ComputePayloadType.MULTIPLY_ACCUMULATE,
    num_inputs=2,
    num_outputs=1,
    description="C[m,n] += A[m,k] * B[k,n]  — dense matrix multiplication",
)

SKETCH_TRANSPOSE_2D = OperationSketch(
    name="linalg.transpose",
    dialect="linalg",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",   # input A
        "(i, j) -> (j, i)",   # output B  (permuted)
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.PARALLEL],
    compute_payload=ComputePayloadType.COPY,
    num_inputs=1,
    num_outputs=1,
    description="B[j,i] = A[i,j]  — 2-D matrix transposition",
)

SKETCH_DOT_PRODUCT = OperationSketch(
    name="linalg.dot",
    dialect="linalg",
    dim_names=["k"],
    indexing_maps=[
        "(k) -> (k)",   # A (input vector)
        "(k) -> (k)",   # B (input vector)
        "() -> ()",     # c (scalar output)
    ],
    iterator_types=[IteratorType.REDUCTION],
    compute_payload=ComputePayloadType.MULTIPLY_ACCUMULATE,
    num_inputs=2,
    num_outputs=1,
    description="c += sum_k A[k] * B[k]  — vector dot product",
)

SKETCH_MATVEC = OperationSketch(
    name="linalg.matvec",
    dialect="linalg",
    dim_names=["m", "k"],
    indexing_maps=[
        "(m, k) -> (m, k)",   # A (matrix)
        "(m, k) -> (k)",      # x (vector)
        "(m, k) -> (m)",      # y (output vector)
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.REDUCTION],
    compute_payload=ComputePayloadType.MULTIPLY_ACCUMULATE,
    num_inputs=2,
    num_outputs=1,
    description="y[m] += sum_k A[m,k] * x[k]  — matrix-vector multiply",
)

SKETCH_VECMAT = OperationSketch(
    name="linalg.vecmat",
    dialect="linalg",
    dim_names=["k", "n"],
    indexing_maps=[
        "(k, n) -> (k)",      # x (row vector)
        "(k, n) -> (k, n)",   # A (matrix)
        "(k, n) -> (n)",      # y (output row vector)
    ],
    iterator_types=[IteratorType.REDUCTION, IteratorType.PARALLEL],
    compute_payload=ComputePayloadType.MULTIPLY_ACCUMULATE,
    num_inputs=2,
    num_outputs=1,
    description="y[n] += sum_k x[k] * A[k,n]  — vector-matrix multiply",
)

# ----- Elementwise Operations ----------------------------------------------

SKETCH_ELEMENTWISE_ADD = OperationSketch(
    name="linalg.map{arith.addf}",
    dialect="linalg",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",   # A
        "(i, j) -> (i, j)",   # B
        "(i, j) -> (i, j)",   # C
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.PARALLEL],
    compute_payload=ComputePayloadType.ADD,
    num_inputs=2,
    num_outputs=1,
    description="C[i,j] = A[i,j] + B[i,j]  — elementwise addition",
)

SKETCH_ELEMENTWISE_MUL = OperationSketch(
    name="linalg.map{arith.mulf}",
    dialect="linalg",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",   # A
        "(i, j) -> (i, j)",   # B
        "(i, j) -> (i, j)",   # C
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.PARALLEL],
    compute_payload=ComputePayloadType.MULTIPLY,
    num_inputs=2,
    num_outputs=1,
    description="C[i,j] = A[i,j] * B[i,j]  — elementwise multiplication",
)

SKETCH_ELEMENTWISE_SUB = OperationSketch(
    name="linalg.map{arith.subf}",
    dialect="linalg",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",   # A
        "(i, j) -> (i, j)",   # B
        "(i, j) -> (i, j)",   # C
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.PARALLEL],
    compute_payload=ComputePayloadType.SUBTRACT,
    num_inputs=2,
    num_outputs=1,
    description="C[i,j] = A[i,j] - B[i,j]  — elementwise subtraction",
)

SKETCH_ELEMENTWISE_ADD_1D = OperationSketch(
    name="linalg.map{arith.addf}_1d",
    dialect="linalg",
    dim_names=["i"],
    indexing_maps=[
        "(i) -> (i)",
        "(i) -> (i)",
        "(i) -> (i)",
    ],
    iterator_types=[IteratorType.PARALLEL],
    compute_payload=ComputePayloadType.ADD,
    num_inputs=2,
    num_outputs=1,
    description="C[i] = A[i] + B[i]  — 1-D elementwise addition",
)

SKETCH_RELU = OperationSketch(
    name="linalg.map{arith.maxf_zero}",
    dialect="linalg",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",   # A (input)
        "(i, j) -> (i, j)",   # B (output)
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.PARALLEL],
    compute_payload=ComputePayloadType.RELU,
    num_inputs=1,
    num_outputs=1,
    description="B[i,j] = max(A[i,j], 0)  — ReLU activation",
)

SKETCH_SCALE_2D = OperationSketch(
    name="linalg.map{arith.mulf_scalar}",
    dialect="linalg",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",   # A (input)
        "(i, j) -> (i, j)",   # B (output)
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.PARALLEL],
    compute_payload=ComputePayloadType.SCALE,
    num_inputs=1,
    num_outputs=1,
    description="B[i,j] = alpha * A[i,j]  — scalar multiplication",
)

SKETCH_COPY_2D = OperationSketch(
    name="linalg.copy",
    dialect="linalg",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",   # A (input)
        "(i, j) -> (i, j)",   # B (output)
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.PARALLEL],
    compute_payload=ComputePayloadType.COPY,
    num_inputs=1,
    num_outputs=1,
    description="B[i,j] = A[i,j]  — 2-D memcpy",
)

# ----- Reductions ----------------------------------------------------------

SKETCH_REDUCE_SUM_2D_TO_1D = OperationSketch(
    name="linalg.reduce{arith.addf}_rowsum",
    dialect="linalg",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",   # A (input matrix)
        "(i, j) -> (i)",      # B (output row-sum vector)
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.REDUCTION],
    compute_payload=ComputePayloadType.ACCUMULATE_ADD,
    num_inputs=1,
    num_outputs=1,
    description="B[i] += sum_j A[i,j]  — row-wise sum reduction",
)

SKETCH_REDUCE_SUM_COLWISE = OperationSketch(
    name="linalg.reduce{arith.addf}_colsum",
    dialect="linalg",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",   # A (input matrix)
        "(i, j) -> (j)",      # B (output col-sum vector)
    ],
    iterator_types=[IteratorType.REDUCTION, IteratorType.PARALLEL],
    compute_payload=ComputePayloadType.ACCUMULATE_ADD,
    num_inputs=1,
    num_outputs=1,
    description="B[j] += sum_i A[i,j]  — column-wise sum reduction",
)

SKETCH_REDUCE_MAX_2D_TO_1D = OperationSketch(
    name="linalg.reduce{arith.maxf}",
    dialect="linalg",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",
        "(i, j) -> (i)",
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.REDUCTION],
    compute_payload=ComputePayloadType.ACCUMULATE_MAX,
    num_inputs=1,
    num_outputs=1,
    description="B[i] = max_j A[i,j]  — row-wise max reduction",
)

# ----- 1-D Convolution -----------------------------------------------------

SKETCH_CONV_1D = OperationSketch(
    name="linalg.conv_1d_ncw_fcw",
    dialect="linalg",
    dim_names=["n", "w", "kw"],
    indexing_maps=[
        "(n, w, kw) -> (n, w + kw)",   # input I
        "(n, w, kw) -> (kw,)",          # kernel K
        "(n, w, kw) -> (n, w)",         # output O
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.REDUCTION],
    compute_payload=ComputePayloadType.MULTIPLY_ACCUMULATE,
    num_inputs=2,
    num_outputs=1,
    description="O[n,w] += sum_kw I[n,w+kw] * K[kw]  — 1-D NCW convolution (no channels)",
)

SKETCH_CONV_1D_NWC_WCF = OperationSketch(
    name="linalg.conv_1d_nwc_wcf",
    dialect="linalg",
    dim_names=["n", "w", "kw", "c", "f"],
    indexing_maps=[
        "(n, w, kw, c, f) -> (n, w + kw, c)",   # input I[n, w+kw, c]
        "(n, w, kw, c, f) -> (kw, c, f)",         # kernel K[kw, c, f]
        "(n, w, kw, c, f) -> (n, w, f)",           # output O[n, w, f]
    ],
    iterator_types=[
        IteratorType.PARALLEL, IteratorType.PARALLEL,
        IteratorType.REDUCTION, IteratorType.REDUCTION, IteratorType.PARALLEL
    ],
    compute_payload=ComputePayloadType.MULTIPLY_ACCUMULATE,
    num_inputs=2,
    num_outputs=1,
    description="O[n,w,f] += sum_{kw,c} I[n,w+kw,c] * K[kw,c,f]  — 1-D NWC/WCF conv",
)

# ----- 2-D Convolution -----------------------------------------------------

SKETCH_CONV_2D_NHWC_HWCF = OperationSketch(
    name="linalg.conv_2d_nhwc_hwcf",
    dialect="linalg",
    dim_names=["n", "oh", "ow", "kh", "kw", "ic", "oc"],
    indexing_maps=[
        "(n, oh, ow, kh, kw, ic, oc) -> (n, oh + kh, ow + kw, ic)",   # I
        "(n, oh, ow, kh, kw, ic, oc) -> (kh, kw, ic, oc)",              # K
        "(n, oh, ow, kh, kw, ic, oc) -> (n, oh, ow, oc)",               # O
    ],
    iterator_types=[
        IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.PARALLEL,   # n, oh, ow
        IteratorType.REDUCTION, IteratorType.REDUCTION, IteratorType.REDUCTION, # kh, kw, ic
        IteratorType.PARALLEL,                                                   # oc
    ],
    compute_payload=ComputePayloadType.MULTIPLY_ACCUMULATE,
    num_inputs=2,
    num_outputs=1,
    description="O[n,oh,ow,oc] += sum_{kh,kw,ic} I[n,oh+kh,ow+kw,ic] * K[kh,kw,ic,oc] — 2-D NHWC conv",
)

SKETCH_CONV_2D_NCHW_FCHW = OperationSketch(
    name="linalg.conv_2d_nchw_fchw",
    dialect="linalg",
    dim_names=["n", "oc", "oh", "ow", "ic", "kh", "kw"],
    indexing_maps=[
        "(n, oc, oh, ow, ic, kh, kw) -> (n, ic, oh + kh, ow + kw)",   # I
        "(n, oc, oh, ow, ic, kh, kw) -> (oc, ic, kh, kw)",              # K
        "(n, oc, oh, ow, ic, kh, kw) -> (n, oc, oh, ow)",               # O
    ],
    iterator_types=[
        IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.PARALLEL,  # n,oc,oh,ow
        IteratorType.REDUCTION, IteratorType.REDUCTION, IteratorType.REDUCTION,                        # ic,kh,kw
    ],
    compute_payload=ComputePayloadType.MULTIPLY_ACCUMULATE,
    num_inputs=2,
    num_outputs=1,
    description="O[n,oc,oh,ow] += sum_{ic,kh,kw} I[n,ic,oh+kh,ow+kw] * K[oc,ic,kh,kw] — 2-D NCHW conv",
)

# Simplified 2-D conv (no batch, no channels) — common in legacy image processing code
SKETCH_CONV_2D_SIMPLE = OperationSketch(
    name="linalg.conv_2d",
    dialect="linalg",
    dim_names=["oh", "ow", "kh", "kw"],
    indexing_maps=[
        "(oh, ow, kh, kw) -> (oh + kh, ow + kw)",   # I
        "(oh, ow, kh, kw) -> (kh, kw)",              # K
        "(oh, ow, kh, kw) -> (oh, ow)",              # O
    ],
    iterator_types=[
        IteratorType.PARALLEL, IteratorType.PARALLEL,
        IteratorType.REDUCTION, IteratorType.REDUCTION,
    ],
    compute_payload=ComputePayloadType.MULTIPLY_ACCUMULATE,
    num_inputs=2,
    num_outputs=1,
    description="O[oh,ow] += sum_{kh,kw} I[oh+kh,ow+kw] * K[kh,kw]  — simple 2-D convolution",
)

# ----- Pooling -------------------------------------------------------------

SKETCH_MAX_POOL_2D = OperationSketch(
    name="linalg.pooling_nhwc_max",
    dialect="linalg",
    dim_names=["n", "oh", "ow", "kh", "kw", "c"],
    indexing_maps=[
        "(n, oh, ow, kh, kw, c) -> (n, oh + kh, ow + kw, c)",   # I
        "(n, oh, ow, kh, kw, c) -> (kh, kw)",                    # window (shape only)
        "(n, oh, ow, kh, kw, c) -> (n, oh, ow, c)",              # O
    ],
    iterator_types=[
        IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.PARALLEL,
        IteratorType.REDUCTION, IteratorType.REDUCTION,
        IteratorType.PARALLEL,
    ],
    compute_payload=ComputePayloadType.ACCUMULATE_MAX,
    num_inputs=2,
    num_outputs=1,
    description="O[n,oh,ow,c] = max_{kh,kw} I[n,oh+kh,ow+kw,c]  — 2-D max pooling",
)

# ----- Batch Matrix Multiply -----------------------------------------------

SKETCH_BATCH_MATMUL = OperationSketch(
    name="linalg.batch_matmul",
    dialect="linalg",
    dim_names=["b", "m", "n", "k"],
    indexing_maps=[
        "(b, m, n, k) -> (b, m, k)",   # A
        "(b, m, n, k) -> (b, k, n)",   # B
        "(b, m, n, k) -> (b, m, n)",   # C
    ],
    iterator_types=[
        IteratorType.PARALLEL, IteratorType.PARALLEL,
        IteratorType.PARALLEL, IteratorType.REDUCTION,
    ],
    compute_payload=ComputePayloadType.MULTIPLY_ACCUMULATE,
    num_inputs=2,
    num_outputs=1,
    description="C[b,m,n] += sum_k A[b,m,k] * B[b,k,n]  — batched matrix multiplication",
)

# ----- StableHLO equivalents -----------------------------------------------

SKETCH_STABLEHLO_DOT = OperationSketch(
    name="stablehlo.dot_general",
    dialect="stablehlo",
    dim_names=["m", "n", "k"],
    indexing_maps=[
        "(m, n, k) -> (m, k)",
        "(m, n, k) -> (k, n)",
        "(m, n, k) -> (m, n)",
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.REDUCTION],
    compute_payload=ComputePayloadType.MULTIPLY_ACCUMULATE,
    num_inputs=2,
    num_outputs=1,
    description="C[m,n] = sum_k A[m,k] * B[k,n]  — StableHLO dot_general (matmul)",
)

SKETCH_STABLEHLO_DOT_MATVEC = OperationSketch(
    name="stablehlo.dot_general_matvec",
    dialect="stablehlo",
    dim_names=["m", "k"],
    indexing_maps=[
        "(m, k) -> (m, k)",
        "(m, k) -> (k)",
        "(m, k) -> (m)",
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.REDUCTION],
    compute_payload=ComputePayloadType.MULTIPLY_ACCUMULATE,
    num_inputs=2,
    num_outputs=1,
    description="y[m] = sum_k A[m,k] * x[k]  — StableHLO dot_general (matvec)",
)

SKETCH_STABLEHLO_DOT_VECDOT = OperationSketch(
    name="stablehlo.dot_general_vecdot",
    dialect="stablehlo",
    dim_names=["k"],
    indexing_maps=[
        "(k) -> (k)",
        "(k) -> (k)",
        "() -> ()",
    ],
    iterator_types=[IteratorType.REDUCTION],
    compute_payload=ComputePayloadType.MULTIPLY_ACCUMULATE,
    num_inputs=2,
    num_outputs=1,
    description="c = sum_k A[k] * B[k]  — StableHLO dot_general (vector dot)",
)

SKETCH_STABLEHLO_TRANSPOSE_2D = OperationSketch(
    name="stablehlo.transpose",
    dialect="stablehlo",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",
        "(i, j) -> (j, i)",
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.PARALLEL],
    compute_payload=ComputePayloadType.COPY,
    num_inputs=1,
    num_outputs=1,
    description="B[j,i] = A[i,j]  — StableHLO transpose",
)

SKETCH_STABLEHLO_ELEMENTWISE_ADD = OperationSketch(
    name="stablehlo.add",
    dialect="stablehlo",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",
        "(i, j) -> (i, j)",
        "(i, j) -> (i, j)",
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.PARALLEL],
    compute_payload=ComputePayloadType.ADD,
    num_inputs=2,
    num_outputs=1,
    description="C[i,j] = A[i,j] + B[i,j]  — StableHLO elementwise add",
)

SKETCH_STABLEHLO_ELEMENTWISE_SUB = OperationSketch(
    name="stablehlo.subtract",
    dialect="stablehlo",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",
        "(i, j) -> (i, j)",
        "(i, j) -> (i, j)",
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.PARALLEL],
    compute_payload=ComputePayloadType.SUBTRACT,
    num_inputs=2,
    num_outputs=1,
    description="C[i,j] = A[i,j] - B[i,j]  — StableHLO elementwise subtract",
)

SKETCH_STABLEHLO_ELEMENTWISE_MUL = OperationSketch(
    name="stablehlo.multiply",
    dialect="stablehlo",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",
        "(i, j) -> (i, j)",
        "(i, j) -> (i, j)",
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.PARALLEL],
    compute_payload=ComputePayloadType.MULTIPLY,
    num_inputs=2,
    num_outputs=1,
    description="C[i,j] = A[i,j] * B[i,j]  — StableHLO elementwise multiply",
)

SKETCH_STABLEHLO_REDUCE_SUM = OperationSketch(
    name="stablehlo.reduce{add}",
    dialect="stablehlo",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",
        "(i, j) -> (i)",
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.REDUCTION],
    compute_payload=ComputePayloadType.ACCUMULATE_ADD,
    num_inputs=1,
    num_outputs=1,
    description="B[i] = sum_j A[i,j]  — StableHLO row-wise reduce sum",
)

SKETCH_STABLEHLO_REDUCE_SUM_COLWISE = OperationSketch(
    name="stablehlo.reduce{add}_colsum",
    dialect="stablehlo",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",
        "(i, j) -> (j)",
    ],
    iterator_types=[IteratorType.REDUCTION, IteratorType.PARALLEL],
    compute_payload=ComputePayloadType.ACCUMULATE_ADD,
    num_inputs=1,
    num_outputs=1,
    description="B[j] = sum_i A[i,j]  — StableHLO column-wise reduce sum",
)

SKETCH_STABLEHLO_REDUCE_MAX = OperationSketch(
    name="stablehlo.reduce{max}",
    dialect="stablehlo",
    dim_names=["i", "j"],
    indexing_maps=[
        "(i, j) -> (i, j)",
        "(i, j) -> (i)",
    ],
    iterator_types=[IteratorType.PARALLEL, IteratorType.REDUCTION],
    compute_payload=ComputePayloadType.ACCUMULATE_MAX,
    num_inputs=1,
    num_outputs=1,
    description="B[i] = max_j A[i,j]  — StableHLO row-wise reduce max",
)

SKETCH_STABLEHLO_CONV = OperationSketch(
    name="stablehlo.convolution",
    dialect="stablehlo",
    dim_names=["n", "oh", "ow", "kh", "kw", "ic", "oc"],
    indexing_maps=[
        "(n, oh, ow, kh, kw, ic, oc) -> (n, oh + kh, ow + kw, ic)",
        "(n, oh, ow, kh, kw, ic, oc) -> (kh, kw, ic, oc)",
        "(n, oh, ow, kh, kw, ic, oc) -> (n, oh, ow, oc)",
    ],
    iterator_types=[
        IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.PARALLEL,
        IteratorType.REDUCTION, IteratorType.REDUCTION, IteratorType.REDUCTION,
        IteratorType.PARALLEL,
    ],
    compute_payload=ComputePayloadType.MULTIPLY_ACCUMULATE,
    num_inputs=2,
    num_outputs=1,
    description="StableHLO 2-D convolution in NHWC/HWCF layout",
)

# ---------------------------------------------------------------------------
# Master library list — ordered by specificity (more specific first)
# ---------------------------------------------------------------------------

SKETCH_LIBRARY: List[OperationSketch] = [
    # Convolution (most specific — check before simpler ops)
    SKETCH_CONV_2D_NHWC_HWCF,
    SKETCH_CONV_2D_NCHW_FCHW,
    SKETCH_CONV_2D_SIMPLE,
    SKETCH_CONV_1D_NWC_WCF,
    SKETCH_CONV_1D,
    # Pooling
    SKETCH_MAX_POOL_2D,
    # Batched matmul (before single matmul — more specific)
    SKETCH_BATCH_MATMUL,
    # Basic BLAS
    SKETCH_MATMUL,
    SKETCH_MATVEC,
    SKETCH_VECMAT,
    SKETCH_DOT_PRODUCT,
    # Reductions
    SKETCH_REDUCE_SUM_2D_TO_1D,
    SKETCH_REDUCE_SUM_COLWISE,
    SKETCH_REDUCE_MAX_2D_TO_1D,
    # Elementwise
    SKETCH_RELU,
    SKETCH_ELEMENTWISE_ADD,
    SKETCH_ELEMENTWISE_MUL,
    SKETCH_ELEMENTWISE_SUB,
    SKETCH_ELEMENTWISE_ADD_1D,
    SKETCH_SCALE_2D,
    # Copy / transpose
    SKETCH_TRANSPOSE_2D,
    SKETCH_COPY_2D,
    # StableHLO
    SKETCH_STABLEHLO_DOT,
    SKETCH_STABLEHLO_DOT_MATVEC,
    SKETCH_STABLEHLO_DOT_VECDOT,
    SKETCH_STABLEHLO_TRANSPOSE_2D,
    SKETCH_STABLEHLO_ELEMENTWISE_ADD,
    SKETCH_STABLEHLO_ELEMENTWISE_SUB,
    SKETCH_STABLEHLO_ELEMENTWISE_MUL,
    SKETCH_STABLEHLO_REDUCE_SUM,
    SKETCH_STABLEHLO_REDUCE_SUM_COLWISE,
    SKETCH_STABLEHLO_REDUCE_MAX,
    SKETCH_STABLEHLO_CONV,
]

# Convenience lookup by name
SKETCH_BY_NAME: Dict[str, OperationSketch] = {s.name: s for s in SKETCH_LIBRARY}

# Linalg-only and StableHLO-only subsets
LINALG_SKETCHES = [s for s in SKETCH_LIBRARY if s.dialect == "linalg"]
STABLEHLO_SKETCHES = [s for s in SKETCH_LIBRARY if s.dialect == "stablehlo"]
