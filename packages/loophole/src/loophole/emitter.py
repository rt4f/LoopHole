"""
emitter.py — MLIR text emitter for Linalg and StableHLO dialects.

Produces syntactically valid MLIR text (textual IR format) that can be
piped directly to `mlir-opt` for further compilation or verification.

Each emitter function accepts:
  - A matched OperationSketch
  - Concrete tensor shapes extracted from the source loop
  - The element type (f32, f64, etc.)
  - The function name (from the source)

And produces a complete `func.func` MLIR module as a string.

The emitter supports:
  Linalg named ops: linalg.matmul, linalg.transpose, linalg.conv_*,
                    linalg.dot, linalg.matvec, linalg.fill, etc.
  Linalg generic:   linalg.generic with fully-specified affine maps
  StableHLO:        stablehlo.dot_general, stablehlo.convolution,
                    stablehlo.reduce, stablehlo.map
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import re
from loophole.affine_extractor import LoopNestInfo
from loophole.sketch_library import ComputePayloadType, OperationSketch

try:
    import sympy as _sympy
except Exception:
    _sympy = None


class EmissionError(ValueError):
    """Raised when emitter cannot produce a valid artifact for a sketch."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _memref_type(shape: List[int], elem: str) -> str:
    if not shape:
        return f"memref<{elem}>"
    dims = "x".join(str(d) if d >= 0 else "?" for d in shape)
    return f"memref<{dims}x{elem}>"


def _tensor_type(shape: List[int], elem: str) -> str:
    if not shape:
        return f"tensor<{elem}>"
    dims = "x".join(str(d) if d >= 0 else "?" for d in shape)
    return f"tensor<{dims}x{elem}>"


def _indent(text: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


def _emit_module_header() -> str:
    return 'module {\n'


def _emit_module_footer() -> str:
    return '}\n'


def _map_elem_type(etype: str) -> str:
    """Normalise element type string."""
    mapping = {
        'float': 'f32', 'double': 'f64',
        'int': 'i32', 'long': 'i64',
    }
    return mapping.get(etype, etype)


def _require_element_type(loop: LoopNestInfo, op_name: str) -> str:
    et = (loop.element_type or "").strip()
    if not et:
        raise EmissionError(
            f"Missing element type metadata for '{op_name}'. "
            "Emission requires explicit tensor_types/element_type from extraction."
        )
    return et


def _require_output_tensor(loop: LoopNestInfo, op_name: str) -> str:
    out = loop.output_tensor
    if not out:
        raise EmissionError(
            f"Missing output tensor metadata for '{op_name}'. "
            "Extractor did not identify a writable output tensor."
        )
    return out


def _require_input_tensor(loop: LoopNestInfo, op_name: str, idx: int) -> str:
    if idx >= len(loop.input_tensors):
        raise EmissionError(
            f"Missing input tensor metadata for '{op_name}'. "
            f"Expected input at position {idx}, found {len(loop.input_tensors)} input tensors."
        )
    return loop.input_tensors[idx]


def _require_shape(
    shapes: Dict[str, List[int]],
    tensor_name: str,
    op_name: str,
    expected_rank: Optional[int] = None,
) -> List[int]:
    shape = shapes.get(tensor_name)
    if not shape:
        raise EmissionError(
            f"Missing tensor shape metadata for '{op_name}' tensor '{tensor_name}'. "
            "Emission requires explicit tensor_shapes from extraction."
        )
    if expected_rank is not None and len(shape) != expected_rank:
        raise EmissionError(
            f"Invalid tensor rank for '{op_name}' tensor '{tensor_name}'. "
            f"Expected rank {expected_rank}, got rank {len(shape)}."
        )
    return shape


def _require_iv(ivs: List[str], idx: int, role: str, op_name: str) -> str:
    if idx >= len(ivs):
        raise EmissionError(
            f"Convolution attr policy failure for '{op_name}': missing {role} iv at position {idx}."
        )
    return ivs[idx]


def _find_read_access_expr(loop: LoopNestInfo, tensor_name: str, dim: int, op_name: str) -> str:
    for read in loop.reads:
        if read.tensor_name == tensor_name and dim < len(read.index_exprs):
            return read.index_exprs[dim]
    raise EmissionError(
        f"Convolution attr policy failure for '{op_name}': cannot find read access for "
        f"tensor '{tensor_name}' at dimension {dim}."
    )


def _infer_linear_coeff(expr: str, var: str) -> Optional[int]:
    expr = expr.strip()
    if not expr or not var:
        return None

    if _sympy is None:
        # Regex fallback for simple linear patterns when sympy is unavailable.
        compact = expr.replace(" ", "")
        if compact == var:
            return 1

        coeff = 0
        for term in re.finditer(r"([+-]?)([^+-]+)", compact):
            term_sign = -1 if term.group(1) == "-" else 1
            part = term.group(2)
            if part == var:
                coeff += term_sign
                continue

            m_left = re.fullmatch(rf"(\d+)\*{re.escape(var)}", part)
            if m_left:
                coeff += term_sign * int(m_left.group(1))
                continue

            m_right = re.fullmatch(rf"{re.escape(var)}\*(\d+)", part)
            if m_right:
                coeff += term_sign * int(m_right.group(1))
                continue

        if coeff != 0:
            return coeff
        return None

    try:
        names = set(re.findall(r"\b[a-zA-Z_]\w*\b", expr))
        locals_map = {n: _sympy.Symbol(n, integer=True) for n in names}
        parsed = _sympy.expand(_sympy.sympify(expr, locals=locals_map))
        sym = _sympy.Symbol(var, integer=True)
        coeff = parsed.coeff(sym)
        if coeff is None or coeff == 0:
            return None
        if getattr(coeff, "free_symbols", None):
            return None
        if coeff.is_integer is True:
            return int(coeff)
        as_float = float(coeff)
        if as_float.is_integer():
            return int(as_float)
    except Exception:
        return None

    return None


def _infer_stride_dilation_from_expr(
    expr: str,
    out_var: str,
    ker_var: str,
    op_name: str,
    axis_label: str,
) -> Tuple[int, int]:
    stride = _infer_linear_coeff(expr, out_var)
    dilation = _infer_linear_coeff(expr, ker_var)

    if stride is None or dilation is None:
        raise EmissionError(
            f"Convolution attr policy failure for '{op_name}' on axis '{axis_label}': "
            f"cannot infer stride/dilation from expression '{expr}'."
        )
    if stride <= 0 or dilation <= 0:
        raise EmissionError(
            f"Convolution attr policy failure for '{op_name}' on axis '{axis_label}': "
            f"inferred non-positive stride/dilation ({stride}, {dilation}) from '{expr}'."
        )
    return stride, dilation


def _infer_conv2d_window_attrs(loop: LoopNestInfo, input_name: str, op_name: str) -> Tuple[int, int, int, int]:
    """Infer (stride_h, dilation_h, stride_w, dilation_w) from input access expressions.

    This supports multiple layouts (plain 2D, NHWC, NCHW) by inspecting which input
    dimensions depend on exactly one parallel IV and one reduction IV.
    """
    reads = [r for r in loop.reads if r.tensor_name == input_name]
    if not reads:
        raise EmissionError(
            f"Convolution attr policy failure for '{op_name}': cannot find read access for "
            f"tensor '{input_name}'."
        )

    axes: List[Tuple[int, int]] = []
    # Collect candidate spatial axes in input-dimension order.
    for expr in reads[0].index_exprs:
        par_hits: List[Tuple[str, int]] = []
        red_hits: List[Tuple[str, int]] = []

        for p in loop.parallel_vars:
            c = _infer_linear_coeff(expr, p)
            if c is not None and c > 0:
                par_hits.append((p, c))

        for r in loop.reduction_vars:
            c = _infer_linear_coeff(expr, r)
            if c is not None and c > 0:
                red_hits.append((r, c))

        if len(par_hits) == 1 and len(red_hits) == 1:
            axes.append((par_hits[0][1], red_hits[0][1]))

    if len(axes) == 1 and len(loop.reduction_vars) == 1:
        (stride_w, dilation_w) = axes[0]
        return 1, 1, stride_w, dilation_w

    if len(axes) < 2:
        raise EmissionError(
            f"Convolution attr policy failure for '{op_name}': "
            "cannot infer two spatial stride/dilation axes from input access expressions."
        )

    (stride_h, dilation_h), (stride_w, dilation_w) = axes[0], axes[1]
    return stride_h, dilation_h, stride_w, dilation_w


# ---------------------------------------------------------------------------
# Linalg Emitter
# ---------------------------------------------------------------------------

class LinalgEmitter:
    """
    Emits MLIR Linalg dialect code for a matched sketch.

    Preferred order:
      1. Named Linalg operation (e.g., linalg.matmul) when available.
      2. linalg.generic with fully-specified affine maps as fallback.
    """

    # Map of sketch name → emit method name
    _NAMED_OP_HANDLERS = {
        "linalg.matmul":               "_emit_matmul",
        "linalg.transpose":            "_emit_transpose",
        "linalg.dot":                  "_emit_dot",
        "linalg.matvec":               "_emit_matvec",
        "linalg.vecmat":               "_emit_vecmat",
        "linalg.batch_matmul":         "_emit_batch_matmul",
        "linalg.conv_1d_ncw_fcw":      "_emit_conv1d_ncw",
        "linalg.conv_1d_nwc_wcf":      "_emit_conv1d_nwc",
        "linalg.conv_2d":              "_emit_conv2d_simple",
        "linalg.conv_2d_nhwc_hwcf":    "_emit_conv2d_nhwc",
        "linalg.conv_2d_nchw_fchw":    "_emit_conv2d_nchw",
        "linalg.pooling_nhwc_max":     "_emit_pool2d_max",
        "linalg.copy":                 "_emit_copy",
        "linalg.map{arith.addf}":      "_emit_elementwise",
        "linalg.map{arith.addf}_1d":   "_emit_elementwise",
        "linalg.map{arith.mulf}":      "_emit_elementwise",
        "linalg.map{arith.subf}":      "_emit_elementwise",
        "linalg.map{arith.maxf_zero}": "_emit_relu",
        "linalg.map{arith.mulf_scalar}": "_emit_scale",
        "linalg.reduce{arith.addf}_rowsum": "_emit_reduce_sum",
        "linalg.reduce{arith.addf}_colsum": "_emit_reduce_sum",
        "linalg.reduce{arith.maxf}":   "_emit_reduce_max",
    }

    def _validate_required_metadata(self, sketch: OperationSketch, loop: LoopNestInfo) -> None:
        _require_element_type(loop, sketch.name)

        if not loop.tensor_shapes:
            raise EmissionError(
                f"Missing tensor_shapes metadata for '{sketch.name}'. "
                "Emitter no longer guesses default shapes."
            )

        if sketch.num_inputs > len(loop.input_tensors):
            raise EmissionError(
                f"Missing input tensor metadata for '{sketch.name}'. "
                f"Expected {sketch.num_inputs} input tensor(s), found {len(loop.input_tensors)}."
            )

        for idx in range(sketch.num_inputs):
            in_name = _require_input_tensor(loop, sketch.name, idx)
            _require_shape(loop.tensor_shapes, in_name, sketch.name)

        out_name = _require_output_tensor(loop, sketch.name)
        _require_shape(loop.tensor_shapes, out_name, sketch.name)

    def emit(
        self,
        sketch: OperationSketch,
        loop: LoopNestInfo,
        func_name: Optional[str] = None,
    ) -> str:
        """
        Emit a complete MLIR module containing the lifted function.
        Returns the MLIR text as a string.
        """
        self._validate_required_metadata(sketch, loop)
        name = func_name or f"lifted_{loop.func_name}"
        handler_method = self._NAMED_OP_HANDLERS.get(sketch.name)
        if handler_method and hasattr(self, handler_method):
            func_body = getattr(self, handler_method)(sketch, loop, name)
        else:
            if sketch.name.startswith("linalg."):
                raise EmissionError(
                    f"Unsupported Linalg sketch '{sketch.name}' for emission. "
                    "Add a dedicated emitter handler or map this sketch explicitly."
                )
            func_body = self._emit_generic(sketch, loop, name)

        lines = [
            "// LoopHole: Automatically lifted from scalar loop nest",
            f"// Source: {loop.func_name} -> {sketch.name}",
            f"// Description: {sketch.description}",
            "// Generated by LoopHole POC v0.1.0",
            "",
        ]
        lines.append(_emit_module_header())
        lines.append(func_body)
        lines.append(_emit_module_footer())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # BLAS ops
    # ------------------------------------------------------------------

    def _emit_matmul(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        A_name = _require_input_tensor(loop, sketch.name, 0)
        B_name = _require_input_tensor(loop, sketch.name, 1)
        out = _require_output_tensor(loop, sketch.name)

        A_shape = _require_shape(shapes, A_name, sketch.name, expected_rank=2)
        B_shape = _require_shape(shapes, B_name, sketch.name, expected_rank=2)
        C_shape = _require_shape(shapes, out, sketch.name, expected_rank=2)

        A_type = _memref_type(A_shape, et)
        B_type = _memref_type(B_shape, et)
        C_type = _memref_type(C_shape, et)

        return (
            f"  func.func @{name}(%A: {A_type}, %B: {B_type}, %C: {C_type}) {{\n"
            f"    linalg.matmul\n"
            f"      ins(%A, %B : {A_type}, {B_type})\n"
            f"      outs(%C : {C_type})\n"
            f"    return\n"
            f"  }}"
        )

    def _emit_transpose(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        A_name = _require_input_tensor(loop, sketch.name, 0)
        out = _require_output_tensor(loop, sketch.name)
        A_shape = _require_shape(shapes, A_name, sketch.name)
        B_shape = _require_shape(shapes, out, sketch.name)

        A_type = _memref_type(A_shape, et)
        B_type = _memref_type(B_shape, et)

        # Infer permutation from access pattern
        perm = self._infer_transpose_perm(loop, len(A_shape))
        self._validate_transpose_perm(perm, A_shape, B_shape, sketch.name)
        perm_str = "[" + ", ".join(str(p) for p in perm) + "]"

        return (
            f"  func.func @{name}(%A: {A_type}, %B: {B_type}) {{\n"
            f"    linalg.transpose\n"
            f"      ins(%A : {A_type})\n"
            f"      outs(%B : {B_type})\n"
            f"      permutation = {perm_str}\n"
            f"    return\n"
            f"  }}"
        )

    def _infer_transpose_perm(self, loop: LoopNestInfo, rank: int) -> List[int]:
        """Infer transpose permutation strictly from output index order."""
        if not loop.writes:
            raise EmissionError("Cannot infer transpose permutation: no write access pattern found.")

        w = loop.writes[0]
        if len(w.index_exprs) != rank:
            raise EmissionError(
                "Cannot infer transpose permutation: output index rank does not match input rank. "
                f"Expected {rank}, found {len(w.index_exprs)}."
            )

        all_ivs = loop.induction_vars
        iv_to_pos = {iv: i for i, iv in enumerate(all_ivs)}
        perm: List[int] = []

        for expr in w.index_exprs:
            token = expr.strip()
            if token not in iv_to_pos:
                raise EmissionError(
                    "Cannot infer transpose permutation: ambiguous output index expression "
                    f"'{expr}'. Expected direct IV references only."
                )
            perm.append(iv_to_pos[token])

        return perm

    def _validate_transpose_perm(
        self,
        perm: List[int],
        a_shape: List[int],
        b_shape: List[int],
        op_name: str,
    ) -> None:
        rank = len(a_shape)
        if len(b_shape) != rank:
            raise EmissionError(
                f"Invalid output rank for '{op_name}'. Expected rank {rank}, got {len(b_shape)}."
            )

        if len(perm) != rank:
            raise EmissionError(
                f"Invalid transpose permutation for '{op_name}'. Expected length {rank}, got {len(perm)}."
            )

        if sorted(perm) != list(range(rank)):
            raise EmissionError(
                f"Invalid transpose permutation for '{op_name}': {perm}. "
                "Permutation must contain each dimension index exactly once."
            )

        expected_b = [a_shape[p] for p in perm]
        if expected_b != b_shape:
            raise EmissionError(
                f"Transpose permutation does not match output shape for '{op_name}'. "
                f"Input shape {a_shape}, permutation {perm}, expected output {expected_b}, got {b_shape}."
            )

    def _emit_dot(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        A_name = _require_input_tensor(loop, sketch.name, 0)
        B_name = _require_input_tensor(loop, sketch.name, 1)
        out = _require_output_tensor(loop, sketch.name)

        A_shape = _require_shape(shapes, A_name, sketch.name, expected_rank=1)
        B_shape = _require_shape(shapes, B_name, sketch.name, expected_rank=1)
        C_shape = _require_shape(shapes, out, sketch.name, expected_rank=1)

        A_type = _memref_type(A_shape, et)
        B_type = _memref_type(B_shape, et)
        C_type = _memref_type(C_shape, et)

        return (
            f"  func.func @{name}(%A: {A_type}, %B: {B_type}, %c: {C_type}) {{\n"
            f"    linalg.dot\n"
            f"      ins(%A, %B : {A_type}, {B_type})\n"
            f"      outs(%c : {C_type})\n"
            f"    return\n"
            f"  }}"
        )

    def _emit_matvec(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        A_name = _require_input_tensor(loop, sketch.name, 0)
        x_name = _require_input_tensor(loop, sketch.name, 1)
        out = _require_output_tensor(loop, sketch.name)

        A_shape = _require_shape(shapes, A_name, sketch.name, expected_rank=2)
        x_shape = _require_shape(shapes, x_name, sketch.name, expected_rank=1)
        y_shape = _require_shape(shapes, out, sketch.name, expected_rank=1)

        A_type = _memref_type(A_shape, et)
        x_type = _memref_type(x_shape, et)
        y_type = _memref_type(y_shape, et)

        return (
            f"  func.func @{name}(%A: {A_type}, %x: {x_type}, %y: {y_type}) {{\n"
            f"    linalg.matvec\n"
            f"      ins(%A, %x : {A_type}, {x_type})\n"
            f"      outs(%y : {y_type})\n"
            f"    return\n"
            f"  }}"
        )

    def _emit_vecmat(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        x_name = _require_input_tensor(loop, sketch.name, 0)
        A_name = _require_input_tensor(loop, sketch.name, 1)
        out = _require_output_tensor(loop, sketch.name)

        x_shape = _require_shape(shapes, x_name, sketch.name, expected_rank=1)
        A_shape = _require_shape(shapes, A_name, sketch.name, expected_rank=2)
        y_shape = _require_shape(shapes, out, sketch.name, expected_rank=1)

        x_type = _memref_type(x_shape, et)
        A_type = _memref_type(A_shape, et)
        y_type = _memref_type(y_shape, et)

        return (
            f"  func.func @{name}(%x: {x_type}, %A: {A_type}, %y: {y_type}) {{\n"
            f"    linalg.vecmat\n"
            f"      ins(%x, %A : {x_type}, {A_type})\n"
            f"      outs(%y : {y_type})\n"
            f"    return\n"
            f"  }}"
        )

    def _emit_batch_matmul(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        A_name = _require_input_tensor(loop, sketch.name, 0)
        B_name = _require_input_tensor(loop, sketch.name, 1)
        out = _require_output_tensor(loop, sketch.name)

        A_shape = _require_shape(shapes, A_name, sketch.name, expected_rank=3)
        B_shape = _require_shape(shapes, B_name, sketch.name, expected_rank=3)
        C_shape = _require_shape(shapes, out, sketch.name, expected_rank=3)

        A_type = _memref_type(A_shape, et)
        B_type = _memref_type(B_shape, et)
        C_type = _memref_type(C_shape, et)

        return (
            f"  func.func @{name}(%A: {A_type}, %B: {B_type}, %C: {C_type}) {{\n"
            f"    linalg.batch_matmul\n"
            f"      ins(%A, %B : {A_type}, {B_type})\n"
            f"      outs(%C : {C_type})\n"
            f"    return\n"
            f"  }}"
        )

    # ------------------------------------------------------------------
    # Convolution ops
    # ------------------------------------------------------------------

    def _emit_conv1d_ncw(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        I_name = _require_input_tensor(loop, sketch.name, 0)
        K_name = _require_input_tensor(loop, sketch.name, 1)
        out = _require_output_tensor(loop, sketch.name)

        out_w_iv = _require_iv(loop.parallel_vars, 1, "parallel", sketch.name)
        ker_w_iv = _require_iv(loop.reduction_vars, 0, "reduction", sketch.name)
        i_expr_w = _find_read_access_expr(loop, I_name, 1, sketch.name)
        stride_w, dilation_w = _infer_stride_dilation_from_expr(
            i_expr_w, out_w_iv, ker_w_iv, sketch.name, "w"
        )

        # Infer shapes from bounds
        ivs = loop.induction_vars
        bounds = loop.bounds
        N = _bound_size(bounds, loop.parallel_vars, 0)
        W_out = _bound_size(bounds, loop.parallel_vars, 1)
        KW = _bound_size(bounds, loop.reduction_vars, 0)
        W_in = W_out + KW - 1

        I_shape = _require_shape(shapes, I_name, sketch.name, expected_rank=2)
        K_shape = _require_shape(shapes, K_name, sketch.name, expected_rank=1)
        O_shape = _require_shape(shapes, out, sketch.name, expected_rank=2)

        I_type = _memref_type(I_shape, et)
        K_type = _memref_type(K_shape, et)
        O_type = _memref_type(O_shape, et)

        return (
            f"  func.func @{name}(%I: {I_type}, %K: {K_type}, %O: {O_type}) {{\n"
            f"    linalg.conv_1d_ncw_fcw\n"
            f"      {{dilations = dense<{dilation_w}> : tensor<1xi64>,\n"
            f"       strides   = dense<{stride_w}> : tensor<1xi64>}}\n"
            f"      ins(%I, %K : {I_type}, {K_type})\n"
            f"      outs(%O : {O_type})\n"
            f"    return\n"
            f"  }}"
        )

    def _emit_conv1d_nwc(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        I_name = _require_input_tensor(loop, sketch.name, 0)
        K_name = _require_input_tensor(loop, sketch.name, 1)
        out = _require_output_tensor(loop, sketch.name)

        out_w_iv = _require_iv(loop.parallel_vars, 1, "parallel", sketch.name)
        ker_w_iv = _require_iv(loop.reduction_vars, 0, "reduction", sketch.name)
        i_expr_w = _find_read_access_expr(loop, I_name, 1, sketch.name)
        stride_w, dilation_w = _infer_stride_dilation_from_expr(
            i_expr_w, out_w_iv, ker_w_iv, sketch.name, "w"
        )
        bounds = loop.bounds
        par = loop.parallel_vars
        red = loop.reduction_vars

        N = _bound_size(bounds, par, 0)
        Wout = _bound_size(bounds, par, 1)
        F = _bound_size(bounds, par, 2) if len(par) > 2 else 1
        KW = _bound_size(bounds, red, 0)
        C = _bound_size(bounds, red, 1) if len(red) > 1 else 1
        Win = Wout + KW - 1

        I_shape = _require_shape(shapes, I_name, sketch.name, expected_rank=3)
        K_shape = _require_shape(shapes, K_name, sketch.name, expected_rank=3)
        O_shape = _require_shape(shapes, out, sketch.name, expected_rank=3)

        I_type = _memref_type(I_shape, et)
        K_type = _memref_type(K_shape, et)
        O_type = _memref_type(O_shape, et)

        return (
            f"  func.func @{name}(%I: {I_type}, %K: {K_type}, %O: {O_type}) {{\n"
            f"    linalg.conv_1d_nwc_wcf\n"
            f"      {{dilations = dense<{dilation_w}> : tensor<1xi64>,\n"
            f"       strides   = dense<{stride_w}> : tensor<1xi64>}}\n"
            f"      ins(%I, %K : {I_type}, {K_type})\n"
            f"      outs(%O : {O_type})\n"
            f"    return\n"
            f"  }}"
        )

    def _emit_conv2d_simple(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        I_name = _require_input_tensor(loop, sketch.name, 0)
        K_name = _require_input_tensor(loop, sketch.name, 1)
        out = _require_output_tensor(loop, sketch.name)

        stride_h, dilation_h, stride_w, dilation_w = _infer_conv2d_window_attrs(
            loop, I_name, sketch.name
        )
        bounds = loop.bounds
        par = loop.parallel_vars
        red = loop.reduction_vars

        OH = _bound_size(bounds, par, 0)
        OW = _bound_size(bounds, par, 1)
        KH = _bound_size(bounds, red, 0)
        KW = _bound_size(bounds, red, 1) if len(red) > 1 else KH
        IH = OH + KH - 1
        IW = OW + KW - 1

        I_shape = _require_shape(shapes, I_name, sketch.name, expected_rank=2)
        K_shape = _require_shape(shapes, K_name, sketch.name, expected_rank=2)
        O_shape = _require_shape(shapes, out, sketch.name, expected_rank=2)

        I_type = _memref_type(I_shape, et)
        K_type = _memref_type(K_shape, et)
        O_type = _memref_type(O_shape, et)

        return (
            f"  func.func @{name}(%I: {I_type}, %K: {K_type}, %O: {O_type}) {{\n"
            f"    linalg.conv_2d\n"
            f"      {{dilations = dense<[{dilation_h}, {dilation_w}]> : tensor<2xi64>,\n"
            f"       strides   = dense<[{stride_h}, {stride_w}]> : tensor<2xi64>}}\n"
            f"      ins(%I, %K : {I_type}, {K_type})\n"
            f"      outs(%O : {O_type})\n"
            f"    return\n"
            f"  }}"
        )

    def _emit_conv2d_nhwc(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        I_name = _require_input_tensor(loop, sketch.name, 0)
        K_name = _require_input_tensor(loop, sketch.name, 1)
        out = _require_output_tensor(loop, sketch.name)

        stride_h, dilation_h, stride_w, dilation_w = _infer_conv2d_window_attrs(
            loop, I_name, sketch.name
        )
        bounds = loop.bounds
        par = loop.parallel_vars
        red = loop.reduction_vars

        N  = _bound_size(bounds, par, 0)
        OH = _bound_size(bounds, par, 1)
        OW = _bound_size(bounds, par, 2)
        OC = _bound_size(bounds, par, 3) if len(par) > 3 else 1
        KH = _bound_size(bounds, red, 0)
        KW = _bound_size(bounds, red, 1) if len(red) > 1 else KH
        IC = _bound_size(bounds, red, 2) if len(red) > 2 else 1
        IH = OH + KH - 1
        IW = OW + KW - 1

        I_shape = _require_shape(shapes, I_name, sketch.name, expected_rank=4)
        K_shape = _require_shape(shapes, K_name, sketch.name, expected_rank=4)
        O_shape = _require_shape(shapes, out, sketch.name, expected_rank=4)

        I_type = _memref_type(I_shape, et)
        K_type = _memref_type(K_shape, et)
        O_type = _memref_type(O_shape, et)

        return (
            f"  func.func @{name}(%I: {I_type}, %K: {K_type}, %O: {O_type}) {{\n"
            f"    linalg.conv_2d_nhwc_hwcf\n"
            f"      {{dilations = dense<[{dilation_h}, {dilation_w}]> : tensor<2xi64>,\n"
            f"       strides   = dense<[{stride_h}, {stride_w}]> : tensor<2xi64>}}\n"
            f"      ins(%I, %K : {I_type}, {K_type})\n"
            f"      outs(%O : {O_type})\n"
            f"    return\n"
            f"  }}"
        )

    def _emit_conv2d_nchw(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        I_name = _require_input_tensor(loop, sketch.name, 0)
        K_name = _require_input_tensor(loop, sketch.name, 1)
        out = _require_output_tensor(loop, sketch.name)

        stride_h, dilation_h, stride_w, dilation_w = _infer_conv2d_window_attrs(
            loop, I_name, sketch.name
        )
        bounds = loop.bounds
        par = loop.parallel_vars
        red = loop.reduction_vars

        N  = _bound_size(bounds, par, 0)
        OC = _bound_size(bounds, par, 1)
        OH = _bound_size(bounds, par, 2)
        OW = _bound_size(bounds, par, 3) if len(par) > 3 else 1
        IC = _bound_size(bounds, red, 0)
        KH = _bound_size(bounds, red, 1) if len(red) > 1 else 3
        KW = _bound_size(bounds, red, 2) if len(red) > 2 else 3
        IH = OH + KH - 1
        IW = OW + KW - 1

        I_shape = _require_shape(shapes, I_name, sketch.name, expected_rank=4)
        K_shape = _require_shape(shapes, K_name, sketch.name, expected_rank=4)
        O_shape = _require_shape(shapes, out, sketch.name, expected_rank=4)

        I_type = _memref_type(I_shape, et)
        K_type = _memref_type(K_shape, et)
        O_type = _memref_type(O_shape, et)

        return (
            f"  func.func @{name}(%I: {I_type}, %K: {K_type}, %O: {O_type}) {{\n"
            f"    linalg.conv_2d_nchw_fchw\n"
            f"      {{dilations = dense<[{dilation_h}, {dilation_w}]> : tensor<2xi64>,\n"
            f"       strides   = dense<[{stride_h}, {stride_w}]> : tensor<2xi64>}}\n"
            f"      ins(%I, %K : {I_type}, {K_type})\n"
            f"      outs(%O : {O_type})\n"
            f"    return\n"
            f"  }}"
        )

    def _emit_pool2d_max(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        I_name = _require_input_tensor(loop, sketch.name, 0)
        out = _require_output_tensor(loop, sketch.name)
        bounds = loop.bounds
        par = loop.parallel_vars
        red = loop.reduction_vars

        N  = _bound_size(bounds, par, 0)
        OH = _bound_size(bounds, par, 1)
        OW = _bound_size(bounds, par, 2)
        C  = _bound_size(bounds, par, 3) if len(par) > 3 else 1
        KH = _bound_size(bounds, red, 0)
        KW = _bound_size(bounds, red, 1) if len(red) > 1 else KH
        IH, IW = OH + KH - 1, OW + KW - 1

        I_shape = _require_shape(shapes, I_name, sketch.name, expected_rank=4)
        W_shape = [KH, KW]
        O_shape = _require_shape(shapes, out, sketch.name, expected_rank=4)

        I_type = _memref_type(I_shape, et)
        W_type = _memref_type(W_shape, et)
        O_type = _memref_type(O_shape, et)

        return (
            f"  func.func @{name}(%I: {I_type}, %W: {W_type}, %O: {O_type}) {{\n"
            f"    linalg.pooling_nhwc_max\n"
            f"      {{dilations = dense<1> : tensor<2xi64>,\n"
            f"       strides   = dense<1> : tensor<2xi64>}}\n"
            f"      ins(%I, %W : {I_type}, {W_type})\n"
            f"      outs(%O : {O_type})\n"
            f"    return\n"
            f"  }}"
        )

    # ------------------------------------------------------------------
    # Elementwise and reduction ops
    # ------------------------------------------------------------------

    def _emit_elementwise(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        """Emit linalg.map for elementwise binary operations."""
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        A_name = _require_input_tensor(loop, sketch.name, 0)
        B_name = _require_input_tensor(loop, sketch.name, 1)
        out = _require_output_tensor(loop, sketch.name)

        A_shape = _require_shape(shapes, A_name, sketch.name)
        B_shape = _require_shape(shapes, B_name, sketch.name)
        C_shape = _require_shape(shapes, out, sketch.name)

        if A_shape != B_shape or C_shape != A_shape:
            raise EmissionError(
                f"Elementwise metadata gate failure for '{sketch.name}': "
                f"shape mismatch A={A_shape}, B={B_shape}, O={C_shape}."
            )

        A_type = _memref_type(A_shape, et)
        B_type = _memref_type(B_shape, et)
        C_type = _memref_type(C_shape, et)

        # Infer arith op from sketch name
        arith_op = "addf"
        if "mulf" in sketch.name:
            arith_op = "mulf"
        elif "subf" in sketch.name:
            arith_op = "subf"

        return (
            f"  func.func @{name}(%A: {A_type}, %B: {B_type}, %C: {C_type}) {{\n"
            f"    linalg.map\n"
            f"      ins(%A, %B : {A_type}, {B_type})\n"
            f"      outs(%C : {C_type})\n"
            f"      ({{%a: {et}, %b: {et}}} -> {{\n"
            f"        %res = arith.{arith_op} %a, %b : {et}\n"
            f"        linalg.yield %res : {et}\n"
            f"      }})\n"
            f"    return\n"
            f"  }}"
        )

    def _emit_relu(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        A_name = _require_input_tensor(loop, sketch.name, 0)
        out = _require_output_tensor(loop, sketch.name)
        A_shape = _require_shape(shapes, A_name, sketch.name)
        B_shape = _require_shape(shapes, out, sketch.name)

        A_type = _memref_type(A_shape, et)
        B_type = _memref_type(B_shape, et)

        return (
            f"  func.func @{name}(%A: {A_type}, %B: {B_type}) {{\n"
            f"    %zero = arith.constant 0.0 : {et}\n"
            f"    linalg.map\n"
            f"      ins(%A : {A_type})\n"
            f"      outs(%B : {B_type})\n"
            f"      ({{%a: {et}}} -> {{\n"
            f"        %res = arith.maxf %a, %zero : {et}\n"
            f"        linalg.yield %res : {et}\n"
            f"      }})\n"
            f"    return\n"
            f"  }}"
        )

    def _emit_scale(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        A_name = _require_input_tensor(loop, sketch.name, 0)
        out = _require_output_tensor(loop, sketch.name)
        A_shape = _require_shape(shapes, A_name, sketch.name)
        B_shape = _require_shape(shapes, out, sketch.name)

        A_type = _memref_type(A_shape, et)
        B_type = _memref_type(B_shape, et)

        return (
            f"  func.func @{name}(%A: {A_type}, %alpha: {et}, %B: {B_type}) {{\n"
            f"    linalg.map\n"
            f"      ins(%A : {A_type})\n"
            f"      outs(%B : {B_type})\n"
            f"      ({{%a: {et}}} -> {{\n"
            f"        %res = arith.mulf %a, %alpha : {et}\n"
            f"        linalg.yield %res : {et}\n"
            f"      }})\n"
            f"    return\n"
            f"  }}"
        )

    def _emit_copy(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        A_name = _require_input_tensor(loop, sketch.name, 0)
        out = _require_output_tensor(loop, sketch.name)
        A_shape = _require_shape(shapes, A_name, sketch.name)
        B_shape = _require_shape(shapes, out, sketch.name)

        A_type = _memref_type(A_shape, et)
        B_type = _memref_type(B_shape, et)

        return (
            f"  func.func @{name}(%A: {A_type}, %B: {B_type}) {{\n"
            f"    linalg.copy\n"
            f"      ins(%A : {A_type})\n"
            f"      outs(%B : {B_type})\n"
            f"    return\n"
            f"  }}"
        )

    def _emit_reduce_sum(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        A_name = _require_input_tensor(loop, sketch.name, 0)
        out = _require_output_tensor(loop, sketch.name)
        A_shape = _require_shape(shapes, A_name, sketch.name)

        # Determine reduction dimension
        red_dim = 1 if "rowsum" in sketch.name else 0
        B_shape = _require_shape(shapes, out, sketch.name)

        A_type = _memref_type(A_shape, et)
        B_type = _memref_type(B_shape, et)

        return (
            f"  func.func @{name}(%A: {A_type}, %B: {B_type}) {{\n"
            f"    linalg.reduce\n"
            f"      ins(%A : {A_type})\n"
            f"      outs(%B : {B_type})\n"
            f"      dimensions = [{red_dim}]\n"
            f"      ({{%a: {et}, %b: {et}}} -> {{\n"
            f"        %sum = arith.addf %a, %b : {et}\n"
            f"        linalg.yield %sum : {et}\n"
            f"      }})\n"
            f"    return\n"
            f"  }}"
        )

    def _emit_reduce_max(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        A_name = _require_input_tensor(loop, sketch.name, 0)
        out = _require_output_tensor(loop, sketch.name)
        A_shape = _require_shape(shapes, A_name, sketch.name)
        B_shape = _require_shape(shapes, out, sketch.name)

        A_type = _memref_type(A_shape, et)
        B_type = _memref_type(B_shape, et)

        return (
            f"  func.func @{name}(%A: {A_type}, %B: {B_type}) {{\n"
            f"    linalg.reduce\n"
            f"      ins(%A : {A_type})\n"
            f"      outs(%B : {B_type})\n"
            f"      dimensions = [1]\n"
            f"      ({{%a: {et}, %b: {et}}} -> {{\n"
            f"        %res = arith.maxf %a, %b : {et}\n"
            f"        linalg.yield %res : {et}\n"
            f"      }})\n"
            f"    return\n"
            f"  }}"
        )

    # ------------------------------------------------------------------
    # Generic fallback: linalg.generic
    # ------------------------------------------------------------------

    def _emit_generic(self, sketch: OperationSketch, loop: LoopNestInfo, name: str) -> str:
        """Emit a fully-specified linalg.generic for any operation."""
        et = _map_elem_type(loop.element_type)
        shapes = loop.tensor_shapes
        out = _require_output_tensor(loop, sketch.name)

        # Build affine map attributes
        map_attrs = []
        for i, map_str in enumerate(sketch.indexing_maps):
            # Convert "(m, n, k) -> (m, k)" to MLIR affine_map<(m,n,k) -> (m,k)>
            clean = map_str.replace(' ', '')
            map_attrs.append(f"    #map{i} = affine_map<{clean}>")

        maps_block = "\n".join(map_attrs)

        # Build iterator types
        it_str = ", ".join(f'"{t.value}"' for t in sketch.iterator_types)

        # Build operand types
        operand_types = []
        for i in range(sketch.num_inputs):
            t_name = _require_input_tensor(loop, sketch.name, i)
            shape = _require_shape(shapes, t_name, sketch.name)
            operand_types.append(_memref_type(shape, et))
        out_shape = _require_shape(shapes, out, sketch.name)
        out_type = _memref_type(out_shape, et)

        ins_parts = [f"%in{i}: {t}" for i, t in enumerate(operand_types)]
        ins_decl = ", ".join(f"%in{i}" for i in range(len(operand_types)))
        ins_types = ", ".join(operand_types)

        # Build compute body
        body_str = self._build_generic_body(sketch, et)

        # Build block args
        block_args_in = ", ".join(f"%a{i}: {et}" for i in range(sketch.num_inputs))
        block_args = f"{block_args_in}, %out: {et}"

        func_args = ", ".join(
            f"%in{i}: {t}" for i, t in enumerate(operand_types)
        ) + f", %out_buf: {out_type}"

        map_refs = "[" + ", ".join(f"#map{i}" for i in range(len(sketch.indexing_maps))) + "]"

        return (
            f"{maps_block}\n"
            f"  func.func @{name}({func_args}) {{\n"
            f"    linalg.generic {{\n"
            f"      indexing_maps = {map_refs},\n"
            f"      iterator_types = [{it_str}]\n"
            f"    }}\n"
            f"    ins({ins_decl} : {ins_types})\n"
            f"    outs(%out_buf : {out_type}) {{\n"
            f"    ^bb0({block_args}):\n"
            f"      {body_str}\n"
            f"    }}\n"
            f"    return\n"
            f"  }}"
        )

    def _build_generic_body(self, sketch: OperationSketch, et: str) -> str:
        """Build the linalg.generic body block."""
        cp = sketch.compute_payload
        if cp == ComputePayloadType.MULTIPLY_ACCUMULATE:
            return (
                f"%mul = arith.mulf %a0, %a1 : {et}\n"
                f"      %add = arith.addf %out, %mul : {et}\n"
                f"      linalg.yield %add : {et}"
            )
        elif cp == ComputePayloadType.ADD:
            return f"%res = arith.addf %a0, %a1 : {et}\n      linalg.yield %res : {et}"
        elif cp == ComputePayloadType.SUBTRACT:
            return f"%res = arith.subf %a0, %a1 : {et}\n      linalg.yield %res : {et}"
        elif cp == ComputePayloadType.MULTIPLY:
            return f"%res = arith.mulf %a0, %a1 : {et}\n      linalg.yield %res : {et}"
        elif cp == ComputePayloadType.COPY:
            return f"linalg.yield %a0 : {et}"
        elif cp == ComputePayloadType.ACCUMULATE_ADD:
            return f"%res = arith.addf %a0, %out : {et}\n      linalg.yield %res : {et}"
        elif cp == ComputePayloadType.ACCUMULATE_MAX:
            return f"%res = arith.maxf %a0, %out : {et}\n      linalg.yield %res : {et}"
        elif cp == ComputePayloadType.RELU:
            return (
                f"%zero = arith.constant 0.0 : {et}\n"
                f"      %res = arith.maxf %a0, %zero : {et}\n"
                f"      linalg.yield %res : {et}"
            )
        elif cp == ComputePayloadType.NEGATE:
            return f"%res = arith.negf %a0 : {et}\n      linalg.yield %res : {et}"
        else:
            return f"linalg.yield %a0 : {et}"


# ---------------------------------------------------------------------------
# StableHLO Emitter
# ---------------------------------------------------------------------------

class StableHLOEmitter:
    """Emits MLIR StableHLO dialect code for matched sketches."""

    _NAMED_OP_HANDLERS = {
        "stablehlo.dot_general": "_emit_dot_general",
        "stablehlo.dot_general_matvec": "_emit_dot_general",
        "stablehlo.dot_general_vecdot": "_emit_dot_general",
        "stablehlo.transpose": "_emit_transpose",
        "stablehlo.add": "_emit_elementwise",
        "stablehlo.subtract": "_emit_elementwise",
        "stablehlo.multiply": "_emit_elementwise",
        "stablehlo.reduce{add}": "_emit_reduce",
        "stablehlo.reduce{add}_colsum": "_emit_reduce",
        "stablehlo.reduce{max}": "_emit_reduce",
        "stablehlo.convolution_1d": "_emit_convolution",
        "stablehlo.convolution_2d": "_emit_convolution",
        "stablehlo.convolution": "_emit_convolution",
    }

    def _validate_required_metadata(self, sketch: OperationSketch, loop: LoopNestInfo) -> None:
        _require_element_type(loop, sketch.name)

        if not loop.tensor_shapes:
            raise EmissionError(
                f"Missing tensor_shapes metadata for '{sketch.name}'. "
                "Emitter no longer guesses default shapes."
            )

        if sketch.num_inputs > len(loop.input_tensors):
            raise EmissionError(
                f"Missing input tensor metadata for '{sketch.name}'. "
                f"Expected {sketch.num_inputs} input tensor(s), found {len(loop.input_tensors)}."
            )

        for idx in range(sketch.num_inputs):
            in_name = _require_input_tensor(loop, sketch.name, idx)
            _require_shape(loop.tensor_shapes, in_name, sketch.name)

        out_name = _require_output_tensor(loop, sketch.name)
        _require_shape(loop.tensor_shapes, out_name, sketch.name)

    def emit(
        self,
        sketch: OperationSketch,
        loop: LoopNestInfo,
        func_name: Optional[str] = None,
    ) -> str:
        self._validate_required_metadata(sketch, loop)
        name = func_name or f"lifted_{loop.func_name}"
        et = _map_elem_type(loop.element_type)

        handler_method = self._NAMED_OP_HANDLERS.get(sketch.name)
        if not handler_method:
            raise EmissionError(
                f"Unsupported StableHLO sketch '{sketch.name}' for emission. "
                "StableHLO sketch-to-emitter mapping is out of sync with sketch_library.py."
            )

        body = getattr(self, handler_method)(sketch, loop, name, et)

        lines = [
            "// LoopHole: Automatically lifted from scalar loop nest (StableHLO target)",
            f"// Source: {loop.func_name} -> {sketch.name}",
            f"// Description: {sketch.description}",
            "",
        ]
        lines.append(_emit_module_header())
        lines.append(body)
        lines.append(_emit_module_footer())
        return "\n".join(lines)

    def _emit_dot_general(
        self, sketch: OperationSketch, loop: LoopNestInfo, name: str, et: str
    ) -> str:
        shapes = loop.tensor_shapes
        A_name = _require_input_tensor(loop, sketch.name, 0)
        B_name = _require_input_tensor(loop, sketch.name, 1)
        out = _require_output_tensor(loop, sketch.name)

        A_shape = _require_shape(shapes, A_name, sketch.name)
        B_shape = _require_shape(shapes, B_name, sketch.name)
        C_shape = _require_shape(shapes, out, sketch.name)

        if len(A_shape) == 2 and len(B_shape) == 2:
            lhs_contract, rhs_contract = "[1]", "[0]"
        elif len(A_shape) == 2 and len(B_shape) == 1:
            lhs_contract, rhs_contract = "[1]", "[0]"
        elif len(A_shape) == 1 and len(B_shape) == 1:
            lhs_contract, rhs_contract = "[0]", "[0]"
        elif len(A_shape) == 1 and len(B_shape) == 2:
            lhs_contract, rhs_contract = "[0]", "[0]"
        else:
            raise EmissionError(
                f"Unsupported dot_general rank combination for '{sketch.name}': "
                f"lhs rank {len(A_shape)}, rhs rank {len(B_shape)}."
            )

        A_type = _tensor_type(A_shape, et)
        B_type = _tensor_type(B_shape, et)
        C_type = _tensor_type(C_shape, et)

        return (
            f"  func.func @{name}(%A: {A_type}, %B: {B_type}) -> {C_type} {{\n"
            f"    %result = stablehlo.dot_general %A, %B,\n"
            f"      contracting_dims = {lhs_contract} x {rhs_contract}\n"
            f"      : ({A_type}, {B_type}) -> {C_type}\n"
            f"    return %result : {C_type}\n"
            f"  }}"
        )

    def _emit_transpose(
        self, sketch: OperationSketch, loop: LoopNestInfo, name: str, et: str
    ) -> str:
        shapes = loop.tensor_shapes
        A_name = _require_input_tensor(loop, sketch.name, 0)
        out = _require_output_tensor(loop, sketch.name)

        A_shape = _require_shape(shapes, A_name, sketch.name)
        B_shape = _require_shape(shapes, out, sketch.name)

        if not loop.writes:
            raise EmissionError(
                f"Cannot infer transpose permutation for '{sketch.name}': no write access pattern found."
            )

        write_exprs = loop.writes[0].index_exprs
        if len(write_exprs) != len(A_shape):
            raise EmissionError(
                f"Cannot infer transpose permutation for '{sketch.name}': output index rank "
                f"{len(write_exprs)} does not match input rank {len(A_shape)}."
            )

        iv_to_pos = {iv: i for i, iv in enumerate(loop.induction_vars)}
        perm: List[int] = []
        for expr in write_exprs:
            token = expr.strip()
            if token not in iv_to_pos:
                raise EmissionError(
                    f"Cannot infer transpose permutation for '{sketch.name}': "
                    f"non-trivial output index expression '{expr}'."
                )
            perm.append(iv_to_pos[token])

        if sorted(perm) != list(range(len(A_shape))):
            raise EmissionError(
                f"Invalid transpose permutation for '{sketch.name}': {perm}."
            )

        expected_B = [A_shape[p] for p in perm]
        if expected_B != B_shape:
            raise EmissionError(
                f"Transpose permutation does not match output shape for '{sketch.name}'. "
                f"Expected {expected_B}, got {B_shape}."
            )

        A_type = _tensor_type(A_shape, et)
        B_type = _tensor_type(B_shape, et)
        perm_str = "[" + ", ".join(str(p) for p in perm) + "]"

        return (
            f"  func.func @{name}(%A: {A_type}) -> {B_type} {{\n"
            f"    %result = stablehlo.transpose %A, dims = {perm_str}\n"
            f"      : ({A_type}) -> {B_type}\n"
            f"    return %result : {B_type}\n"
            f"  }}"
        )

    def _emit_elementwise(
        self, sketch: OperationSketch, loop: LoopNestInfo, name: str, et: str
    ) -> str:
        shapes = loop.tensor_shapes
        A_name = _require_input_tensor(loop, sketch.name, 0)
        out = _require_output_tensor(loop, sketch.name)

        A_shape = _require_shape(shapes, A_name, sketch.name)
        O_shape = _require_shape(shapes, out, sketch.name)
        A_type = _tensor_type(A_shape, et)
        O_type = _tensor_type(O_shape, et)

        if sketch.compute_payload == ComputePayloadType.RELU:
            return (
                f"  func.func @{name}(%A: {A_type}) -> {O_type} {{\n"
                f"    %zero = stablehlo.constant dense<0.0> : {A_type}\n"
                f"    %result = stablehlo.maximum %A, %zero : {A_type}\n"
                f"    return %result : {O_type}\n"
                f"  }}"
            )

        B_name = _require_input_tensor(loop, sketch.name, 1)
        B_shape = _require_shape(shapes, B_name, sketch.name)
        B_type = _tensor_type(B_shape, et)

        if A_shape != B_shape:
            raise EmissionError(
                f"Elementwise operands must have equal shapes for '{sketch.name}'. "
                f"Got {A_shape} and {B_shape}."
            )
        if O_shape != A_shape:
            raise EmissionError(
                f"Elementwise output must match input shape for '{sketch.name}'. "
                f"Got output {O_shape}, input {A_shape}."
            )

        if sketch.compute_payload == ComputePayloadType.ADD:
            op = "stablehlo.add"
        elif sketch.compute_payload == ComputePayloadType.SUBTRACT:
            op = "stablehlo.subtract"
        elif sketch.compute_payload == ComputePayloadType.MULTIPLY:
            op = "stablehlo.multiply"
        else:
            raise EmissionError(
                f"Unsupported StableHLO elementwise payload for '{sketch.name}': "
                f"{sketch.compute_payload.value}."
            )

        return (
            f"  func.func @{name}(%A: {A_type}, %B: {B_type}) -> {O_type} {{\n"
            f"    %result = {op} %A, %B : {A_type}\n"
            f"    return %result : {O_type}\n"
            f"  }}"
        )

    def _emit_convolution(
        self, sketch: OperationSketch, loop: LoopNestInfo, name: str, et: str
    ) -> str:
        shapes = loop.tensor_shapes
        I_name = _require_input_tensor(loop, sketch.name, 0)
        K_name = _require_input_tensor(loop, sketch.name, 1)
        out = _require_output_tensor(loop, sketch.name)
        bounds = loop.bounds
        par = loop.parallel_vars
        red = loop.reduction_vars

        N = _bound_size(bounds, par, 0)
        OH = _bound_size(bounds, par, 1)
        OW = _bound_size(bounds, par, 2)
        OC = _bound_size(bounds, par, 3) if len(par) > 3 else 1
        KH = _bound_size(bounds, red, 0)
        KW = _bound_size(bounds, red, 1) if len(red) > 1 else KH
        IC = _bound_size(bounds, red, 2) if len(red) > 2 else 1
        IH, IW = OH + KH - 1, OW + KW - 1

        stride_h, dilation_h, stride_w, dilation_w = _infer_conv2d_window_attrs(
            loop, I_name, sketch.name
        )

        I_shape_raw = _require_shape(shapes, I_name, sketch.name)
        K_shape_raw = _require_shape(shapes, K_name, sketch.name)
        O_shape_raw = _require_shape(shapes, out, sketch.name)

        if len(I_shape_raw) == 2 and len(K_shape_raw) == 1 and len(O_shape_raw) == 2:
            I_shape = [I_shape_raw[0], 1, I_shape_raw[1], 1]
            K_shape = [1, K_shape_raw[0], 1, 1]
            O_shape = [O_shape_raw[0], 1, O_shape_raw[1], 1]
        elif len(I_shape_raw) == 2 and len(K_shape_raw) == 2 and len(O_shape_raw) == 2:
            I_shape = [1, I_shape_raw[0], I_shape_raw[1], 1]
            K_shape = [K_shape_raw[0], K_shape_raw[1], 1, 1]
            O_shape = [1, O_shape_raw[0], O_shape_raw[1], 1]
        elif len(I_shape_raw) == 4 and len(K_shape_raw) == 4 and len(O_shape_raw) == 4:
            I_shape = I_shape_raw
            K_shape = K_shape_raw
            O_shape = O_shape_raw
        else:
            raise EmissionError(
                f"Invalid tensor rank for '{sketch.name}'. "
                f"Expected either (2,1,2), (2,2,2), or (4,4,4) ranks for (I,K,O), got "
                f"({len(I_shape_raw)}, {len(K_shape_raw)}, {len(O_shape_raw)})."
            )

        I_type = _tensor_type(I_shape, et)
        K_type = _tensor_type(K_shape, et)
        O_type = _tensor_type(O_shape, et)

        return (
            f"  func.func @{name}(%I: {I_type}, %K: {K_type}) -> {O_type} {{\n"
            f"    %result = stablehlo.convolution(%I, %K)\n"
            f"      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],\n"
            f"      window = {{stride = [{stride_h}, {stride_w}], pad = [[0, 0], [0, 0]],\n"
            f"                lhs_dilate = [1, 1], rhs_dilate = [{dilation_h}, {dilation_w}]}}\n"
            f"      : ({I_type}, {K_type}) -> {O_type}\n"
            f"    return %result : {O_type}\n"
            f"  }}"
        )

    def _emit_reduce(
        self, sketch: OperationSketch, loop: LoopNestInfo, name: str, et: str
    ) -> str:
        shapes = loop.tensor_shapes
        A_name = _require_input_tensor(loop, sketch.name, 0)
        out = _require_output_tensor(loop, sketch.name)

        A_shape = _require_shape(shapes, A_name, sketch.name)
        B_shape = _require_shape(shapes, out, sketch.name)

        A_type = _tensor_type(A_shape, et)
        B_type = _tensor_type(B_shape, et)

        iv_to_pos = {iv: i for i, iv in enumerate(loop.induction_vars)}
        reduce_dims = sorted(iv_to_pos[iv] for iv in loop.reduction_vars if iv in iv_to_pos)
        if not reduce_dims:
            reduce_dims = [max(0, len(A_shape) - 1)] if A_shape else [0]
        dims_str = "[" + ", ".join(str(d) for d in reduce_dims) + "]"

        if sketch.compute_payload == ComputePayloadType.ACCUMULATE_MAX or "max" in sketch.name:
            reducer = "stablehlo.maximum"
            init_literal = "-3.4028235e38" if et == "f32" else "-1.7976931348623157e308"
        else:
            reducer = "stablehlo.add"
            init_literal = "0.0"

        return (
            f"  func.func @{name}(%A: {A_type}) -> {B_type} {{\n"
            f"    %init = stablehlo.constant dense<{init_literal}> : tensor<{et}>\n"
            f"    %result = stablehlo.reduce(%A init: %init) applies {reducer}\n"
            f"      across dimensions = {dims_str}\n"
            f"      : ({A_type}, tensor<{et}>) -> {B_type}\n"
            f"    return %result : {B_type}\n"
            f"  }}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bound_size(
    bounds: Dict[str, Tuple],
    iv_list: List[str],
    idx: int,
) -> int:
    """Get the size (hi - lo) for the iv at position idx in iv_list."""
    if idx >= len(iv_list):
        return 1
    iv = iv_list[idx]
    lo, hi = bounds.get(iv, (0, 1))
    lo_int = lo if isinstance(lo, int) else 0
    hi_int = hi if isinstance(hi, int) else 1
    return max(1, hi_int - lo_int)
