"""
sympy_tracer.py — Algebraic symbolic tracer for equivalence checking and fast sketch matching.

This module implements an alternative (and complementary) verification approach
to the Z3 SMT checker.  Instead of bit-level / real-arithmetic reasoning,
it uses SymPy's computer algebra system to:

  1. Symbolically trace the loop nest into a closed-form mathematical expression.
  2. Algebraically simplify the expression and compare it to each sketch.
  3. Return the matching sketch(es) without SMT overhead.

This is the approach inspired by Tensorize (CGO 2025) and is particularly
effective for:
  - Reduction loops (SymPy's Sum handles them natively).
  - Operations where the algebraic structure is unambiguous (matmul, dot, etc.).
  - Large tensor sizes where Z3 unrolling would be too slow.

Architecture:
  LoopNestInfo → [SympyTracer.trace()] → TraceResult
  TraceResult  → [matches_sketch()]     → bool
  TraceResult  → [infer_sketch()]       → OperationSketch | None
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp
from sympy import Symbol, Function, Sum, Piecewise, Max, Min, simplify, expand, Rational
from sympy import symbols, IndexedBase, Idx, factorial

from loophole.affine_extractor import AccessPattern, ComputeOp, LoopNestInfo
from loophole.sketch_library import (
    ComputePayloadType, IteratorType, OperationSketch, SKETCH_LIBRARY,
)


# ---------------------------------------------------------------------------
# Trace result
# ---------------------------------------------------------------------------

@dataclass
class TraceResult:
    """
    The symbolic mathematical representation of a loop nest's output.

    output_expr: A SymPy expression representing C[i,j,...] for general i,j,...
    index_syms: The free index symbols in output_expr (parallel IVs)
    reduction_syms: The bound (summed-over) symbols
    tensor_syms: {tensor_name -> IndexedBase symbol}
    matched_sketch: Set by infer_sketch() after matching
    """
    output_expr: Any                        # SymPy expression
    index_syms: Dict[str, Symbol]          # parallel IV → SymPy Symbol
    reduction_syms: Dict[str, Symbol]      # reduction IV → SymPy Symbol
    tensor_syms: Dict[str, Any]            # tensor name → IndexedBase
    matched_sketch: Optional[str] = None
    confidence: float = 0.0               # 0.0–1.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sympy_index_expr(expr_str: str, iv_syms: Dict[str, Any]) -> Any:
    """
    Convert an affine index expression string to a SymPy expression.
    e.g., 'i + kh' → iv_syms['i'] + iv_syms['kh']
    """
    expr_str = expr_str.strip()

    # Replace longest IV names first to avoid conflicts
    result_str = expr_str
    sorted_ivs = sorted(iv_syms.keys(), key=len, reverse=True)
    for iv in sorted_ivs:
        result_str = re.sub(r'\b' + re.escape(iv) + r'\b', f"__iv_{iv}__", result_str)

    # Now substitute back with actual SymPy symbols
    local_dict = {f"__iv_{iv}__": sym for iv, sym in iv_syms.items()}

    try:
        # Try sympy parsing
        parsed = sp.sympify(result_str.replace('__iv_', '').replace('__', ''),
                            locals=iv_syms)
        return parsed
    except Exception:
        pass

    # Manual tokenisation fallback
    parts = [p.strip() for p in result_str.replace(' - ', ' + -').split('+')]
    total = sp.Integer(0)
    for part in parts:
        part = part.strip()
        neg = part.startswith('-')
        if neg:
            part = part[1:].strip()
        replaced = part
        for iv in sorted_ivs:
            replaced = replaced.replace(f"__iv_{iv}__", str(iv_syms[iv]))
        try:
            val = sp.sympify(replaced)
        except Exception:
            val = sp.Integer(0)
        total = total - val if neg else total + val
    return total


# ---------------------------------------------------------------------------
# Main tracer
# ---------------------------------------------------------------------------

class SympyTracer:
    """
    Symbolically traces an affine loop nest and builds a SymPy closed-form
    expression for the output tensor element.

    Usage:
        tracer = SympyTracer()
        result = tracer.trace(loop_info)
        sketch = tracer.infer_sketch(result)
    """

    def trace(self, loop: LoopNestInfo) -> TraceResult:
        """
        Build a symbolic SymPy expression for the output of the loop nest.
        Returns a TraceResult with the expression and metadata.
        """
        # Create SymPy symbols for all IVs
        all_syms: Dict[str, Any] = {
            iv: sp.Symbol(iv, integer=True, nonnegative=True)
            for iv in loop.induction_vars
        }
        index_syms = {iv: all_syms[iv] for iv in loop.parallel_vars}
        reduction_syms = {iv: all_syms[iv] for iv in loop.reduction_vars}

        # Create IndexedBase symbols for tensors
        tensor_syms: Dict[str, Any] = {}
        for name in loop.tensor_shapes:
            clean = name.lstrip('%').replace('.', '_').replace('-', '_')
            tensor_syms[name] = sp.IndexedBase(clean)

        # Build the body expression (what gets written to C[i,j])
        body_expr = self._build_body_expr(loop, all_syms, tensor_syms)

        # Wrap in Sum over reduction IVs if present
        output_expr = body_expr
        for red_iv in loop.reduction_vars:
            sym = all_syms[red_iv]
            lo, hi = loop.bounds.get(red_iv, (0, 1))
            lo_v = sp.Integer(lo) if isinstance(lo, int) else sp.Symbol(str(lo))
            hi_v = sp.Integer(hi) if isinstance(hi, int) else sp.Symbol(str(hi))
            output_expr = sp.Sum(output_expr, (sym, lo_v, hi_v - 1)).doit()

        # Try to simplify
        try:
            output_expr = sp.expand(output_expr)
        except Exception:
            pass

        return TraceResult(
            output_expr=output_expr,
            index_syms=index_syms,
            reduction_syms=reduction_syms,
            tensor_syms=tensor_syms,
        )

    def _build_body_expr(
        self,
        loop: LoopNestInfo,
        all_syms: Dict[str, Any],
        tensor_syms: Dict[str, Any],
    ) -> Any:
        """
        Build the SymPy expression for the loop body computation.

        Strategy:
        1. Map each SSA value to a SymPy expression.
        2. Walk compute_ops in definition order to propagate values.
        3. The final expression is obtained from the SSA value that gets stored.
        """
        # Map of SSA name → SymPy expression
        ssa_vals: Dict[str, Any] = {}

        # Initialize reads
        for r in loop.reads:
            idx_exprs = [_sympy_index_expr(e, all_syms) for e in r.index_exprs]
            T = tensor_syms.get(r.tensor_name)
            if T is not None:
                if len(idx_exprs) == 0:
                    ssa_vals[r.ssa_name] = T
                elif len(idx_exprs) == 1:
                    ssa_vals[r.ssa_name] = T[idx_exprs[0]]
                else:
                    ssa_vals[r.ssa_name] = T[tuple(idx_exprs)]
            else:
                ssa_vals[r.ssa_name] = sp.Symbol(r.ssa_name.lstrip('%'))

        # Walk compute ops
        for op in loop.compute_ops:
            try:
                result = self._eval_compute_op(op, ssa_vals)
                ssa_vals[op.result] = result
            except Exception:
                ssa_vals[op.result] = sp.Symbol(op.result.lstrip('%'))

        # The written SSA value
        if loop.writes:
            written_ssa = loop.writes[0].ssa_name
            if written_ssa in ssa_vals:
                body = ssa_vals[written_ssa]
                # If accumulation: body = (out - out_init) → the delta written
                # For a reduce: C[i,j] = C_old[i,j] + delta → delta = body - C_old
                if loop.has_accumulation:
                    # The body expression is the full new value: out_new = out_old + delta
                    # Extract delta by subtracting the old output read
                    for r in loop.reads:
                        if r.tensor_name == loop.output_tensor:
                            old_val = ssa_vals.get(r.ssa_name, sp.Integer(0))
                            body = sp.expand(body - old_val)
                            break
                return body

        # Fallback: try to find any computed value
        if loop.compute_ops:
            last_result = loop.compute_ops[-1].result
            return ssa_vals.get(last_result, sp.Integer(0))
        return sp.Integer(0)

    def _eval_compute_op(self, op: ComputeOp, ssa_vals: Dict[str, Any]) -> Any:
        """Evaluate a single arithmetic op in SymPy."""
        def get(name: str) -> Any:
            return ssa_vals.get(name, sp.Symbol(name.lstrip('%')))

        if op.op_type == 'constant':
            return sp.Rational(op.value) if op.value is not None else sp.Integer(0)
        elif op.op_type in ('mulf', 'muli', 'mulsi', 'mulu'):
            return get(op.operands[0]) * get(op.operands[1])
        elif op.op_type in ('addf', 'addi', 'addui'):
            return get(op.operands[0]) + get(op.operands[1])
        elif op.op_type in ('subf', 'subi'):
            return get(op.operands[0]) - get(op.operands[1])
        elif op.op_type in ('divf', 'divi_signed'):
            denom = get(op.operands[1])
            if denom == sp.Integer(0):
                return sp.oo
            return get(op.operands[0]) / denom
        elif op.op_type in ('maxf', 'maxnumf', 'maxsi'):
            return sp.Max(get(op.operands[0]), get(op.operands[1]))
        elif op.op_type in ('minf', 'minnumf', 'minsi'):
            return sp.Min(get(op.operands[0]), get(op.operands[1]))
        elif op.op_type == 'negf':
            return -get(op.operands[0])
        else:
            return sp.Symbol(op.result.lstrip('%'))

    # ------------------------------------------------------------------
    # Sketch inference
    # ------------------------------------------------------------------

    def infer_sketch(
        self,
        result: TraceResult,
        loop: LoopNestInfo,
        target_dialect: str = "linalg",
    ) -> Optional[OperationSketch]:
        """
        Infer the best matching OperationSketch from a TraceResult
        using algebraic pattern recognition.

        Returns the matched sketch or None.
        """
        best_sketch: Optional[OperationSketch] = None
        best_confidence: float = 0.0

        for sketch in SKETCH_LIBRARY:
            if target_dialect != "all" and sketch.dialect != target_dialect:
                continue
            conf = self._sketch_confidence(result, loop, sketch)
            if conf > best_confidence:
                best_confidence = conf
                best_sketch = sketch

        if best_sketch and best_confidence >= 0.5:
            result.matched_sketch = best_sketch.name
            result.confidence = best_confidence
            return best_sketch
        return None

    def _sketch_confidence(
        self,
        result: TraceResult,
        loop: LoopNestInfo,
        sketch: OperationSketch,
    ) -> float:
        """
        Heuristic confidence score (0.0–1.0) that the loop matches the sketch.
        Uses structural + algebraic evidence.
        """
        score = 0.0
        max_score = 0.0

        # ---- Structural checks ----------------------------------------
        # Number of loops matches
        max_score += 2.0
        if len(loop.induction_vars) == sketch.num_loops:
            score += 2.0

        # Number of reductions matches
        max_score += 2.0
        if len(loop.reduction_vars) == sketch.num_reduction:
            score += 2.0

        # Compute payload matches inferred payload
        max_score += 3.0
        inferred_cp = self._infer_cp(loop)
        if inferred_cp == sketch.compute_payload:
            score += 3.0
        elif self._cp_compatible(inferred_cp, sketch.compute_payload):
            score += 1.5

        # Input count
        max_score += 1.0
        input_tensors_count = len(set(
            r.tensor_name for r in loop.reads if r.tensor_name != loop.output_tensor
        ))
        if input_tensors_count == sketch.num_inputs:
            score += 1.0

        # ---- Algebraic checks -----------------------------------------
        # Check if the trace expression has the right algebraic structure
        max_score += 2.0
        expr = result.output_expr
        alg_match = self._algebraic_structure_matches(expr, sketch, result)
        score += alg_match * 2.0

        # ---- Access pattern checks ------------------------------------
        # Check if reduction IVs appear in the right operand indices
        max_score += 2.0
        if self._access_pattern_matches(loop, sketch):
            score += 2.0

        return score / max_score if max_score > 0 else 0.0

    def _infer_cp(self, loop: LoopNestInfo) -> ComputePayloadType:
        ops = {op.op_type for op in loop.compute_ops}
        has_mul = bool(ops & {'mulf', 'muli', 'mulsi'})
        has_add = bool(ops & {'addf', 'addi'})
        has_sub = bool(ops & {'subf', 'subi'})
        has_max = bool(ops & {'maxf', 'maxnumf', 'maxsi'})
        if has_mul and has_add and loop.reduction_vars:
            return ComputePayloadType.MULTIPLY_ACCUMULATE
        if has_add and loop.reduction_vars:
            return ComputePayloadType.ACCUMULATE_ADD
        if has_add:
            return ComputePayloadType.ADD
        if has_sub:
            return ComputePayloadType.SUBTRACT
        if has_mul:
            return ComputePayloadType.MULTIPLY
        if has_max and loop.reduction_vars:
            return ComputePayloadType.ACCUMULATE_MAX
        if has_max:
            return ComputePayloadType.MAX
        return ComputePayloadType.COPY

    def _cp_compatible(self, a: ComputePayloadType, b: ComputePayloadType) -> bool:
        """Return True if two compute payloads are related."""
        related = {
            (ComputePayloadType.MULTIPLY_ACCUMULATE, ComputePayloadType.ACCUMULATE_ADD),
            (ComputePayloadType.ACCUMULATE_ADD, ComputePayloadType.MULTIPLY_ACCUMULATE),
            (ComputePayloadType.ADD, ComputePayloadType.ACCUMULATE_ADD),
            (ComputePayloadType.COPY, ComputePayloadType.SCALE),
        }
        return (a, b) in related

    def _algebraic_structure_matches(
        self,
        expr: Any,
        sketch: OperationSketch,
        result: TraceResult,
    ) -> float:
        """
        Check if the algebraic structure of the expression matches the sketch.
        Returns a score in [0, 1].
        """
        try:
            cp = sketch.compute_payload
            expr_atoms = expr.atoms(sp.Mul, sp.Add, sp.Symbol, sp.Max, sp.Min)

            if cp == ComputePayloadType.MULTIPLY_ACCUMULATE:
                # Should have products summed together
                has_mul = any(isinstance(a, sp.Mul) for a in expr_atoms)
                has_add = isinstance(expr, (sp.Add, sp.core.add.Add))
                return 1.0 if (has_mul and has_add) else (0.5 if has_mul else 0.0)

            elif cp == ComputePayloadType.COPY:
                # Expression should be a single indexed symbol
                syms = expr.free_symbols
                indexed = expr.atoms(sp.Indexed)
                return 1.0 if (len(indexed) == 1 and len(syms) <= 4) else 0.0

            elif cp in (ComputePayloadType.ADD, ComputePayloadType.ACCUMULATE_ADD):
                has_add = isinstance(expr, (sp.Add,))
                return 0.8 if has_add else 0.0

            elif cp == ComputePayloadType.ACCUMULATE_MAX:
                has_max = any(isinstance(a, sp.Max) for a in expr_atoms)
                return 1.0 if has_max else 0.0

            elif cp == ComputePayloadType.RELU:
                has_max = any(isinstance(a, sp.Max) for a in expr_atoms)
                return 1.0 if has_max else 0.0

            return 0.5

        except Exception:
            return 0.3

    def _access_pattern_matches(self, loop: LoopNestInfo, sketch: OperationSketch) -> bool:
        """
        Check that the access pattern of the loop is consistent with the sketch
        by verifying that reduction IVs appear in the input (not output) subscripts.
        """
        if not loop.reduction_vars:
            return sketch.num_reduction == 0

        red_ivs = set(loop.reduction_vars)
        out_tensor = loop.output_tensor

        # Reduction IVs should NOT appear in output subscripts
        for w in loop.writes:
            for idx_expr in w.index_exprs:
                for rv in red_ivs:
                    if rv in idx_expr:
                        return False  # reduction IV in output — not a standard reduction

        # Reduction IVs SHOULD appear in at least one input subscript
        for r in loop.reads:
            if r.tensor_name == out_tensor:
                continue
            for idx_expr in r.index_exprs:
                for rv in red_ivs:
                    if rv in idx_expr:
                        return True

        return False

    # ------------------------------------------------------------------
    # Convolution pattern recognizer
    # ------------------------------------------------------------------

    def recognise_convolution(self, loop: LoopNestInfo) -> Optional[str]:
        """
        Special-case recognizer for convolution patterns.
        Returns the sketch name if the access pattern matches a sliding window.
        """
        if not loop.reduction_vars:
            return None

        writes = loop.writes
        reads_no_out = [r for r in loop.reads if r.tensor_name != loop.output_tensor]

        if len(reads_no_out) < 2:
            return None

        # Look for 'w + kw' or 'h + kh' patterns in input reads
        # This is the hallmark of a convolution sliding window
        for r in reads_no_out:
            for idx in r.index_exprs:
                if '+' in idx and any(
                    (k in idx) for k in loop.reduction_vars
                ):
                    # Found sliding window pattern — determine dimensionality
                    n_loops = len(loop.induction_vars)
                    n_red = len(loop.reduction_vars)
                    if n_red == 1 and n_loops <= 3:
                        return "linalg.conv_1d_ncw_fcw"
                    elif n_red == 2 and n_loops == 4:
                        return "linalg.conv_2d"
                    elif n_red >= 2 and n_loops >= 6:
                        return "linalg.conv_2d_nhwc_hwcf"
        return None
