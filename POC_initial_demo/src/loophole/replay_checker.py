"""
replay_checker.py — Counterexample replay and triage for LoopHole verification failures (A-14).

Given a VerificationReport with non-empty counterexample_bindings, simulates both the source
loop nest and the target sketch at the concrete counterexample values to help users understand
WHY the proof failed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from loophole.affine_extractor import LoopNestInfo
from loophole.sketch_library import ComputePayloadType, OperationSketch
from loophole.z3_checker import VerificationReport


# ---------------------------------------------------------------------------
# Parsing Z3 model bindings
# ---------------------------------------------------------------------------

def parse_z3_binding(val_str: str) -> float:
    """
    Parse a Z3 model value string to a Python float.

    Handles: integers ('42', '-3'), rationals ('1/3', '-7/4'), floats ('5.0').
    Raises ValueError for values that cannot be parsed.
    """
    s = val_str.strip()
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        pass
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Cannot parse Z3 binding value: {val_str!r}")


def _is_scalar_iv_name(name: str) -> bool:
    """
    Heuristic: a binding entry is a scalar IV (not a tensor function) when its
    name consists of simple alphanumeric/underscore characters without digits in
    the middle (e.g. 'i', 'j', 'k', 'kh', 'kw') as opposed to function
    applications which appear in Z3 models as entries with array-like names.
    """
    return bool(re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', name))


# ---------------------------------------------------------------------------
# Float affine expression evaluator
# ---------------------------------------------------------------------------

def _eval_float_expr(expr: str, vals: Dict[str, float]) -> Optional[float]:
    """
    Evaluate a simple affine index expression at concrete float values.

    Supported: integer literals, variable names, +, -, unary minus,
    constant multiplication ('2 * i'), parentheses.
    Returns None if evaluation fails (e.g. unknown symbol).
    """
    expr = expr.strip().lstrip('%')

    try:
        return float(int(expr))
    except (ValueError, TypeError):
        pass

    if expr in vals:
        return vals[expr]

    # Handle simple binary expressions: a op b
    for op in ('+', '-', '*'):
        # find last occurrence so we split left-associatively
        idx = expr.rfind(op)
        if idx > 0:
            left = expr[:idx].strip()
            right = expr[idx + 1:].strip()
            lv = _eval_float_expr(left, vals)
            rv = _eval_float_expr(right, vals)
            if lv is not None and rv is not None:
                if op == '+':
                    return lv + rv
                if op == '-':
                    return lv - rv
                if op == '*':
                    return lv * rv

    # Unary minus
    if expr.startswith('-'):
        inner = _eval_float_expr(expr[1:].strip(), vals)
        if inner is not None:
            return -inner

    # Parentheses
    if expr.startswith('(') and expr.endswith(')'):
        return _eval_float_expr(expr[1:-1], vals)

    return None


# ---------------------------------------------------------------------------
# Access pattern simulation
# ---------------------------------------------------------------------------

def simulate_source(
    loop: LoopNestInfo,
    iv_values: Dict[str, float],
) -> Dict[str, List[tuple]]:
    """
    For each read/write access in the source loop, evaluate the index expressions
    at the given IV values and return the accessed index tuples per tensor.
    """
    result: Dict[str, List[tuple]] = {}

    for access in list(loop.reads) + list(loop.writes):
        indices: List[Optional[float]] = [
            _eval_float_expr(e, iv_values)
            for e in access.index_exprs
        ]
        idx_tuple = tuple(
            int(v) if v is not None and v == int(v) else v
            for v in indices
        )
        result.setdefault(access.tensor_name, [])
        if idx_tuple not in result[access.tensor_name]:
            result[access.tensor_name].append(idx_tuple)

    return result


def simulate_sketch(
    sketch: OperationSketch,
    loop: LoopNestInfo,
    iv_values: Dict[str, float],
) -> Dict[str, List[tuple]]:
    """
    Evaluate the sketch indexing maps at the given IV values and return accessed
    index tuples mapped to the source tensor names.
    """
    # Align sketch dim names to source IV names (mirrors _align_dims logic)
    src_parallel = loop.parallel_vars
    src_reduction = loop.reduction_vars
    sk_parallel = sketch.parallel_dims()
    sk_reduction = sketch.reduction_dims()

    sketch_to_source: Dict[str, str] = {}
    for sk_dim, src_iv in zip(sk_parallel, src_parallel):
        sketch_to_source[sk_dim] = src_iv
    for sk_dim, src_iv in zip(sk_reduction, src_reduction):
        sketch_to_source[sk_dim] = src_iv

    # Build IV values in sketch dimension name space
    sketch_iv_vals: Dict[str, float] = {}
    for sk_dim, src_iv in sketch_to_source.items():
        if src_iv in iv_values:
            sketch_iv_vals[sk_dim] = iv_values[src_iv]

    # Operand names (positional)
    input_tensors = [r.tensor_name for r in loop.reads if r.tensor_name != loop.output_tensor]
    unique_inputs = list(dict.fromkeys(input_tensors))
    output_tensor = loop.output_tensor or "%out"

    operand_names: List[str] = unique_inputs[:sketch.num_inputs] + [output_tensor]

    result: Dict[str, List[tuple]] = {}

    for op_idx, map_str in enumerate(sketch.indexing_maps):
        tensor_name = operand_names[op_idx] if op_idx < len(operand_names) else f"operand_{op_idx}"
        arrow_pos = map_str.find('->')
        if arrow_pos < 0:
            continue
        result_part = map_str[arrow_pos + 2:].strip().strip('(').strip(')')
        if not result_part:
            continue
        parts = [p.strip() for p in result_part.split(',')]
        indices: List[Optional[float]] = [_eval_float_expr(p, sketch_iv_vals) for p in parts]
        idx_tuple = tuple(
            int(v) if v is not None and v == int(v) else v
            for v in indices
        )
        result.setdefault(tensor_name, [])
        if idx_tuple not in result[tensor_name]:
            result[tensor_name].append(idx_tuple)

    return result


# ---------------------------------------------------------------------------
# Main replay entry point
# ---------------------------------------------------------------------------

@dataclass
class CounterexampleReport:
    """Structured counterexample triage report (A-14)."""
    iv_values: Dict[str, float]
    source_access_pattern: Dict[str, List[tuple]]
    sketch_access_pattern: Dict[str, List[tuple]]
    parse_errors: List[str]
    diverging_cells: List[str]
    max_abs_diff: Optional[float]
    verdict_summary: str


def replay_counterexample(
    loop: LoopNestInfo,
    sketch: OperationSketch,
    report: VerificationReport,
) -> Optional[CounterexampleReport]:
    """
    Main entry point for counterexample triage.

    Returns None if report.counterexample_bindings is empty.
    Otherwise returns a CounterexampleReport with the IV values at the
    counterexample, the tensor access patterns for both source and sketch,
    and a human-readable verdict summary.
    """
    if not report.counterexample_bindings:
        return None

    iv_values: Dict[str, float] = {}
    parse_errors: List[str] = []

    for name, val_str in report.counterexample_bindings.items():
        if not _is_scalar_iv_name(name):
            continue
        try:
            iv_values[name] = parse_z3_binding(val_str)
        except ValueError as exc:
            parse_errors.append(f"{name}: {exc}")

    source_pat = simulate_source(loop, iv_values)
    sketch_pat = simulate_sketch(sketch, loop, iv_values)

    # Identify diverging output cells
    diverging_cells: List[str] = []
    out_tensor = loop.output_tensor or "%out"
    src_out = set(source_pat.get(out_tensor, []))
    sk_out_keys = [k for k in sketch_pat if k == out_tensor or loop.output_tensor in k]
    sk_out: set = set()
    for k in sk_out_keys:
        sk_out.update(sketch_pat[k])

    symmetric_diff = src_out.symmetric_difference(sk_out)
    for cell in symmetric_diff:
        diverging_cells.append(
            f"output index {cell}: present in "
            f"{'source' if cell in src_out else 'sketch'} only"
        )

    # Build verdict summary
    direction = report.failed_implication or "unknown direction"
    iv_str = ", ".join(f"{k}={v}" for k, v in sorted(iv_values.items()))
    if not iv_str:
        iv_str = "(no scalar IV bindings parsed)"

    verdict_summary = (
        f"Counterexample at [{iv_str}] — "
        f"failed implication: {direction}. "
        f"Source accesses {len(source_pat)} tensor(s), "
        f"sketch accesses {len(sketch_pat)} tensor(s). "
    )
    if diverging_cells:
        verdict_summary += f"Diverging output cells: {diverging_cells[:3]}."
    else:
        verdict_summary += "Output cell sets match; divergence may be in values, not indices."

    if report.mismatch_summary:
        verdict_summary += f" Z3 mismatch summary: {report.mismatch_summary}."

    return CounterexampleReport(
        iv_values=iv_values,
        source_access_pattern=source_pat,
        sketch_access_pattern=sketch_pat,
        parse_errors=parse_errors,
        diverging_cells=diverging_cells,
        max_abs_diff=None,
        verdict_summary=verdict_summary,
    )
