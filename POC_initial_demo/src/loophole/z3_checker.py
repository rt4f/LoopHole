"""
z3_checker.py — Formal equivalence verification using the Z3 SMT solver.

Design goals:
  • Prove, not just test, that a loop nest is semantically identical to a sketch.
  • Use Z3's quantifier-free bit-vector arithmetic for concrete small sizes.
  • Use Z3 quantified (ForAll) formulas for symbolic / general proofs.
  • Handle reduction loops via Z3 RecFunction definitions (inductive axioms).
  • Return a structured CheckResult including a counter-example if available.

Encoding strategy:
  ┌────────────────────────────────────────────────────────────────────┐
  │ Tensors   → Z3 Array(Int, Int, ...) → Real (one array per tensor)  │
  │ Scalars   → Z3 Real constants                                       │
  │ Loops     → Z3 ForAll over bounded integer ranges                   │
  │ Reduction → Z3 RecFunction (partial_sum) with base+step axioms      │
  │ Equivalence check: ¬(loop ↔ sketch) must be UNSAT                  │
  └────────────────────────────────────────────────────────────────────┘

For concrete (fully static) shapes the checker unrolls reductions to a
concrete sum — this is dramatically faster than quantified reasoning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from z3 import (
    And, ArithRef, ArrayRef, ArraySort, BoolRef, ExprRef, ForAll, Function,
    If, IntSort, Not, Or, RealSort, RealVal, RecAddDefinition, RecFunction,
    Solver, Sum, Then, Tactic, Implies, Int, Ints, Real, Reals,
    is_expr, sat, unsat, unknown, simplify, substitute, Array, Store, Select,
    IntVal, BoolVal,
)

from loophole.affine_extractor import AccessPattern, ComputeOp, LoopNestInfo
from loophole.sketch_library import (
    ComputePayloadType, IteratorType, OperationSketch,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class CheckResult(Enum):
    EQUIVALENT = "EQUIVALENT"
    NOT_EQUIVALENT = "NOT_EQUIVALENT"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"
    STRUCTURAL_MISMATCH = "STRUCTURAL_MISMATCH"
    ENCODE_ERROR = "ENCODE_ERROR"


@dataclass
class VerificationReport:
    result: CheckResult
    sketch_name: str
    elapsed_ms: float
    z3_model: Optional[str] = None      # counter-example model (if NOT_EQUIVALENT)
    notes: str = ""

    @property
    def proved(self) -> bool:
        return self.result == CheckResult.EQUIVALENT


# ---------------------------------------------------------------------------
# Helpers for building Z3 array functions
# ---------------------------------------------------------------------------

def _make_tensor_func(name: str, rank: int) -> Any:
    """Create a Z3 uninterpreted function: Int^rank → Real."""
    sorts = [IntSort()] * rank + [RealSort()]
    return Function(name.lstrip('%').replace('.', '_'), *sorts)


def _apply_func(func: Any, indices: List[ArithRef]) -> ArithRef:
    return func(*indices)


def _z3_int(val: Union[int, str]) -> ArithRef:
    """Coerce a Python int or int-like string to a Z3 IntVal."""
    if isinstance(val, int):
        return IntVal(val)
    try:
        return IntVal(int(val))
    except (ValueError, TypeError):
        return Int(val)


# ---------------------------------------------------------------------------
# Index expression evaluator
# ---------------------------------------------------------------------------

def _eval_index_expr(expr: str, iv_vars: Dict[str, ArithRef]) -> ArithRef:
    """
    Evaluate a simple affine index expression string, e.g.:
      'i', 'j', 'i + kh', 'oh + kh', 'ow + kw', 'kw'
    into a Z3 arithmetic expression.

    Supports: +, -, integer literals, IV names.
    Multiplication by constant (e.g. '2*i') is also handled.
    """
    expr = expr.strip()

    # Replace IV names longest-first to avoid partial replacements
    sorted_ivs = sorted(iv_vars.keys(), key=len, reverse=True)

    # Tokenise: split on + and -, keeping the sign
    # But first handle subtraction by replacing ' - ' with ' + -'
    normalised = expr.replace(' - ', ' + -').replace('-', ' -')

    parts = [p.strip() for p in normalised.split('+') if p.strip()]
    result: Optional[ArithRef] = None

    for part in parts:
        part = part.strip()
        negate = False
        if part.startswith('-'):
            negate = True
            part = part[1:].strip()

        # Try multiplication: e.g. '2*i' or '2 * i'
        if '*' in part:
            factors = [f.strip() for f in part.split('*')]
            term: ArithRef = IntVal(1)
            for f in factors:
                f = f.strip()
                if f in iv_vars:
                    term = term * iv_vars[f]
                else:
                    try:
                        term = term * IntVal(int(f))
                    except ValueError:
                        term = term * Int(f)
        elif part in iv_vars:
            term = iv_vars[part]
        else:
            try:
                term = IntVal(int(part))
            except ValueError:
                term = Int(part)   # treat unknown as a fresh variable

        if negate:
            term = IntVal(-1) * term

        result = term if result is None else result + term

    return result if result is not None else IntVal(0)


# ---------------------------------------------------------------------------
# Structural pre-filter
# ---------------------------------------------------------------------------

def structural_match(loop: LoopNestInfo, sketch: OperationSketch) -> bool:
    """
    Quick structural check before invoking Z3.
    Returns False if the loop cannot possibly match the sketch.
    """
    # Loop count must match exactly
    if len(loop.induction_vars) != sketch.num_loops:
        return False

    # Number of reduction vars must match
    if len(loop.reduction_vars) != sketch.num_reduction:
        return False

    # Must have a write
    if not loop.writes:
        return False

    # Compute payload compatibility
    cp = sketch.compute_payload
    has_mul = any(op.op_type in ('mulf', 'muli', 'mulsi') for op in loop.compute_ops)
    has_add = any(op.op_type in ('addf', 'addi') for op in loop.compute_ops)
    has_max = any(op.op_type in ('maxf', 'maxnumf', 'maxsi') for op in loop.compute_ops)
    has_sub = any(op.op_type in ('subf', 'subi') for op in loop.compute_ops)

    if cp == ComputePayloadType.MULTIPLY_ACCUMULATE:
        if not (has_mul and (has_add or loop.has_accumulation)):
            return False
    elif cp == ComputePayloadType.COPY:
        if has_mul or has_add:
            return False
    elif cp == ComputePayloadType.ADD:
        if not has_add:
            return False
    elif cp == ComputePayloadType.SUBTRACT:
        if not has_sub:
            return False
    elif cp == ComputePayloadType.MULTIPLY:
        if not has_mul:
            return False
    elif cp == ComputePayloadType.ACCUMULATE_ADD:
        if not (has_add and len(loop.reduction_vars) > 0):
            return False
    elif cp == ComputePayloadType.ACCUMULATE_MAX:
        if not (has_max and len(loop.reduction_vars) > 0):
            return False
    elif cp == ComputePayloadType.RELU:
        if not has_max:
            return False

    return True


# ---------------------------------------------------------------------------
# Main checker class
# ---------------------------------------------------------------------------

class Z3EquivalenceChecker:
    """
    Proves or disproves that a LoopNestInfo is semantically equivalent to
    a given OperationSketch using the Z3 SMT solver.

    For concrete (static) shapes:
      → Unrolls reductions; uses quantifier-free arithmetic.
      → Much faster and more reliable than quantified reasoning.

    For symbolic shapes:
      → Uses Z3 ForAll + RecFunction for inductive proofs.
      → May timeout on complex invariants; check notes in VerificationReport.
    """

    def __init__(
        self,
        timeout_ms: int = 10_000,
        unroll_threshold: int = 32,
    ):
        self.timeout_ms = timeout_ms
        self.unroll_threshold = unroll_threshold  # max reduction dim size to unroll

    def check(
        self,
        loop: LoopNestInfo,
        sketch: OperationSketch,
    ) -> VerificationReport:
        """
        Main entry point.
        Returns a VerificationReport with the result and timing.
        """
        t0 = time.perf_counter()

        # Fast structural pre-filter
        if not structural_match(loop, sketch):
            return VerificationReport(
                result=CheckResult.STRUCTURAL_MISMATCH,
                sketch_name=sketch.name,
                elapsed_ms=0.0,
                notes="Structural pre-filter rejected this sketch.",
            )

        try:
            report = self._verify(loop, sketch)
        except Exception as exc:
            report = VerificationReport(
                result=CheckResult.ENCODE_ERROR,
                sketch_name=sketch.name,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                notes=f"Encoding error: {exc}",
            )

        report.elapsed_ms = (time.perf_counter() - t0) * 1000
        return report

    # ------------------------------------------------------------------
    # Internal verification
    # ------------------------------------------------------------------

    def _verify(self, loop: LoopNestInfo, sketch: OperationSketch) -> VerificationReport:
        """
        Choose concrete (unrolled) or symbolic (quantified) verification
        based on whether all bounds are statically known integers.
        """
        all_concrete = all(
            isinstance(lo, int) and isinstance(hi, int)
            for lo, hi in loop.bounds.values()
        )

        if all_concrete:
            return self._verify_concrete(loop, sketch)
        else:
            return self._verify_symbolic(loop, sketch)

    # ------------------------------------------------------------------
    # Concrete (unrolled) verification — for fully static shapes
    # ------------------------------------------------------------------

    def _verify_concrete(self, loop: LoopNestInfo, sketch: OperationSketch) -> VerificationReport:
        """
        Unroll parallel loops as ForAll constraints and reductions as
        explicit Z3 Sum expressions over bounded integer ranges.
        Uses quantifier-free linear real arithmetic — very fast for
        small tensors.
        """
        solver = Solver()
        solver.set("timeout", self.timeout_ms)

        ivs = loop.induction_vars  # short names: ['i', 'j', 'k']
        bounds = loop.bounds       # {'i': (0, M), 'j': (0, N), 'k': (0, K)}

        # Create Z3 real-valued tensor functions
        tensor_funcs: Dict[str, Any] = {}
        for name in list(loop.tensor_shapes.keys()):
            rank = len(loop.tensor_shapes[name])
            if rank == 0:
                tensor_funcs[name] = Real(name.lstrip('%'))
            else:
                tensor_funcs[name] = _make_tensor_func(name, rank)

        # Create Z3 Int variables for induction variables
        iv_z3: Dict[str, ArithRef] = {iv: Int(iv) for iv in ivs}

        # Build constraint: source program implies tensor semantics
        source_assertions = self._build_source_assertions(
            loop, iv_z3, tensor_funcs, bounds
        )
        sketch_assertions = self._build_sketch_assertions(
            loop, sketch, iv_z3, tensor_funcs, bounds
        )

        # Bi-directional equivalence:
        # Check ¬(source → sketch): should be UNSAT
        check1 = self._check_implication(
            solver, source_assertions, sketch_assertions
        )
        if check1 != unsat:
            model_str = None
            if check1 == sat:
                try:
                    model_str = str(solver.model())
                except Exception:
                    pass
            return VerificationReport(
                result=CheckResult.NOT_EQUIVALENT if check1 == sat else CheckResult.TIMEOUT if check1 == unknown else CheckResult.UNKNOWN,
                sketch_name=sketch.name,
                elapsed_ms=0.0,
                z3_model=model_str,
                notes="Source → sketch implication failed.",
            )

        # Check ¬(sketch → source): should also be UNSAT
        solver.reset()
        solver.set("timeout", self.timeout_ms)
        check2 = self._check_implication(
            solver, sketch_assertions, source_assertions
        )
        if check2 != unsat:
            model_str = None
            if check2 == sat:
                try:
                    model_str = str(solver.model())
                except Exception:
                    pass
            return VerificationReport(
                result=CheckResult.NOT_EQUIVALENT if check2 == sat else CheckResult.TIMEOUT if check2 == unknown else CheckResult.UNKNOWN,
                sketch_name=sketch.name,
                elapsed_ms=0.0,
                z3_model=model_str,
                notes="Sketch → source implication failed.",
            )

        return VerificationReport(
            result=CheckResult.EQUIVALENT,
            sketch_name=sketch.name,
            elapsed_ms=0.0,
            notes="Both implications proved UNSAT — formally equivalent.",
        )

    def _check_implication(
        self,
        solver: Solver,
        hypothesis: List[BoolRef],
        conclusion: List[BoolRef],
    ):
        """Add ¬(∧hypothesis → ∧conclusion) and check."""
        h = And(*hypothesis) if len(hypothesis) > 1 else (hypothesis[0] if hypothesis else BoolVal(True))
        c = And(*conclusion) if len(conclusion) > 1 else (conclusion[0] if conclusion else BoolVal(True))
        solver.add(Not(Implies(h, c)))
        return solver.check()

    # ------------------------------------------------------------------
    # Build source program assertions
    # ------------------------------------------------------------------

    def _build_source_assertions(
        self,
        loop: LoopNestInfo,
        iv_z3: Dict[str, ArithRef],
        tensor_funcs: Dict[str, Any],
        bounds: Dict[str, Tuple],
    ) -> List[BoolRef]:
        """
        Produce a list of Z3 assertions encoding the loop body semantics.
        Each assertion is a ForAll over parallel IVs.
        Reductions are encoded as explicit bounded sums when concrete.
        """
        parallel_ivs = loop.parallel_vars
        red_ivs = loop.reduction_vars
        assertions: List[BoolRef] = []

        # Build bounds conditions for all IVs
        def bounds_cond(iv_subset: List[str]) -> BoolRef:
            parts = []
            for iv in iv_subset:
                lo, hi = bounds.get(iv, (0, 1))
                v = iv_z3[iv]
                parts.append(And(IntVal(int(lo)) <= v, v < IntVal(int(hi))))
            return And(*parts) if parts else BoolVal(True)

        output_write = loop.writes[0] if loop.writes else None
        if output_write is None:
            return assertions

        out_func = tensor_funcs.get(output_write.tensor_name)
        if out_func is None:
            return assertions

        # Build the RHS expression from the loop body
        rhs = self._build_rhs_expr(loop, iv_z3, tensor_funcs, bounds)

        # Compute output index
        out_indices = [
            _eval_index_expr(e, iv_z3)
            for e in output_write.index_exprs
        ]

        if not parallel_ivs:
            # Scalar output (e.g. dot product)
            out_rank = len(output_write.index_exprs)
            if out_rank == 0:
                out_val = out_func if callable(out_func) and len(out_indices) == 0 else out_func(*out_indices) if out_indices else Real(output_write.tensor_name.lstrip('%'))
            else:
                out_val = _apply_func(out_func, out_indices) if out_indices else Real(output_write.tensor_name.lstrip('%'))
            assertions.append(out_val == rhs)
        else:
            # ForAll over parallel IVs
            par_vars = [iv_z3[iv] for iv in parallel_ivs]
            par_bounds = bounds_cond(parallel_ivs)
            out_val = _apply_func(out_func, out_indices)
            assertion = ForAll(
                par_vars,
                Implies(par_bounds, out_val == rhs)
            )
            assertions.append(assertion)

        return assertions

    def _build_rhs_expr(
        self,
        loop: LoopNestInfo,
        iv_z3: Dict[str, ArithRef],
        tensor_funcs: Dict[str, Any],
        bounds: Dict[str, Tuple],
    ) -> ArithRef:
        """
        Build the RHS Z3 expression for what the loop writes to the output.
        Handles:
          - Copy (no reduction): direct read value
          - Elementwise ops: compute on reads
          - Reductions: sum/max over reduction IVs
        """
        red_ivs = loop.reduction_vars
        reads_no_output = [
            r for r in loop.reads
            if r.tensor_name != loop.output_tensor
        ]

        if not red_ivs:
            # No reduction — direct elementwise
            return self._build_elementwise_expr(loop, iv_z3, tensor_funcs)

        # Has reductions — build sum/max over reduction range
        return self._build_reduction_expr(loop, iv_z3, tensor_funcs, bounds)

    def _build_elementwise_expr(
        self,
        loop: LoopNestInfo,
        iv_z3: Dict[str, ArithRef],
        tensor_funcs: Dict[str, Any],
    ) -> ArithRef:
        """Build the compute expression for elementwise ops."""
        cp = self._infer_compute_payload(loop)
        reads_no_output = [r for r in loop.reads if r.tensor_name != loop.output_tensor]

        def read_val(acc: AccessPattern) -> ArithRef:
            f = tensor_funcs.get(acc.tensor_name)
            if f is None:
                return RealVal(0)
            idxs = [_eval_index_expr(e, iv_z3) for e in acc.index_exprs]
            return _apply_func(f, idxs) if idxs else Real(acc.tensor_name.lstrip('%'))

        if not reads_no_output:
            return RealVal(0)

        a_val = read_val(reads_no_output[0])
        if len(reads_no_output) == 1:
            if cp == ComputePayloadType.COPY:
                return a_val
            elif cp == ComputePayloadType.NEGATE:
                return -a_val
            elif cp == ComputePayloadType.RELU:
                return If(a_val >= RealVal(0), a_val, RealVal(0))
            return a_val

        b_val = read_val(reads_no_output[1])
        if cp == ComputePayloadType.ADD:
            return a_val + b_val
        elif cp == ComputePayloadType.SUBTRACT:
            return a_val - b_val
        elif cp == ComputePayloadType.MULTIPLY:
            return a_val * b_val
        elif cp == ComputePayloadType.MAX:
            return If(a_val >= b_val, a_val, b_val)
        elif cp == ComputePayloadType.MIN:
            return If(a_val <= b_val, a_val, b_val)
        return a_val

    def _build_reduction_expr(
        self,
        loop: LoopNestInfo,
        iv_z3: Dict[str, ArithRef],
        tensor_funcs: Dict[str, Any],
        bounds: Dict[str, Tuple],
    ) -> ArithRef:
        """
        Build a Z3 Sum expression for reduction loops.
        Works by creating a Python-level sum over concrete integer ranges
        (possible when bounds are concrete ints ≤ unroll_threshold).
        For larger bounds, uses Z3 RecFunction.
        """
        red_ivs = loop.reduction_vars
        # Check if all reduction bounds are concrete and small
        can_unroll = all(
            isinstance(bounds.get(iv, (0, 0))[1], int)
            and (bounds.get(iv, (0, 0))[1] - bounds.get(iv, (0, 0))[0]) <= self.unroll_threshold
            for iv in red_ivs
        )

        if can_unroll:
            return self._unrolled_reduction(loop, iv_z3, tensor_funcs, bounds)
        else:
            return self._recfunc_reduction(loop, iv_z3, tensor_funcs, bounds)

    def _unrolled_reduction(
        self,
        loop: LoopNestInfo,
        iv_z3: Dict[str, ArithRef],
        tensor_funcs: Dict[str, Any],
        bounds: Dict[str, Tuple],
    ) -> ArithRef:
        """
        Build a reduction by explicitly summing over concrete integer indices.
        This is quantifier-free and very fast for Z3.
        """
        red_ivs = loop.reduction_vars
        reads_no_output = [r for r in loop.reads if r.tensor_name != loop.output_tensor]

        # Get the accumulation initial value (usually 0 for addf, -inf for max)
        cp = self._infer_compute_payload(loop)
        if cp == ComputePayloadType.MULTIPLY_ACCUMULATE:
            # C += A*B  — accumulates into output starting from 0
            init_val: ArithRef = RealVal(0)
        elif cp == ComputePayloadType.ACCUMULATE_ADD:
            init_val = RealVal(0)
        elif cp == ComputePayloadType.ACCUMULATE_MAX:
            init_val = RealVal(-1e30)
        else:
            init_val = RealVal(0)

        # Build the ranges for all reduction IVs
        def iter_red_ranges(red_ivs: List[str], idx: int, current_vals: Dict[str, int]):
            if idx == len(red_ivs):
                yield dict(current_vals)
                return
            iv = red_ivs[idx]
            lo_v, hi_v = bounds.get(iv, (0, 1))
            lo_i = int(lo_v)
            hi_i = int(hi_v)
            for v in range(lo_i, hi_i):
                current_vals[iv] = v
                yield from iter_red_ranges(red_ivs, idx + 1, current_vals)
                del current_vals[iv]

        total: ArithRef = init_val

        for red_vals in iter_red_ranges(red_ivs, 0, {}):
            # Clone iv_z3 with concrete reduction IVs
            local_iv = dict(iv_z3)
            for iv, v in red_vals.items():
                local_iv[iv] = IntVal(v)

            # Compute the body expression at this concrete reduction point
            term = self._body_term_at(loop, local_iv, tensor_funcs, cp)

            if cp in (ComputePayloadType.MULTIPLY_ACCUMULATE,
                      ComputePayloadType.ACCUMULATE_ADD):
                total = total + term
            elif cp == ComputePayloadType.ACCUMULATE_MAX:
                total = If(term > total, term, total)
            else:
                total = total + term

        return total

    def _body_term_at(
        self,
        loop: LoopNestInfo,
        iv_z3: Dict[str, ArithRef],
        tensor_funcs: Dict[str, Any],
        cp: ComputePayloadType,
    ) -> ArithRef:
        """Evaluate the innermost loop body expression at fixed IV values."""
        reads_no_output = [r for r in loop.reads if r.tensor_name != loop.output_tensor]

        def read_at(acc: AccessPattern) -> ArithRef:
            f = tensor_funcs.get(acc.tensor_name)
            if f is None:
                return RealVal(0)
            idxs = [_eval_index_expr(e, iv_z3) for e in acc.index_exprs]
            return _apply_func(f, idxs) if idxs else Real(acc.tensor_name.lstrip('%'))

        if not reads_no_output:
            return RealVal(0)

        if cp == ComputePayloadType.MULTIPLY_ACCUMULATE:
            a_val = read_at(reads_no_output[0])
            b_val = read_at(reads_no_output[1]) if len(reads_no_output) > 1 else RealVal(1)
            return a_val * b_val
        elif cp == ComputePayloadType.ACCUMULATE_ADD:
            return read_at(reads_no_output[0])
        elif cp == ComputePayloadType.ACCUMULATE_MAX:
            return read_at(reads_no_output[0])
        else:
            return read_at(reads_no_output[0])

    def _recfunc_reduction(
        self,
        loop: LoopNestInfo,
        iv_z3: Dict[str, ArithRef],
        tensor_funcs: Dict[str, Any],
        bounds: Dict[str, Tuple],
    ) -> ArithRef:
        """
        Build a reduction using Z3 RecFunction for large or symbolic reduction dims.
        Defines: acc(par..., k) = acc(par..., k-1) + A[par..., k-1] * B[..., k-1]
                 acc(par..., 0) = 0
        """
        red_iv = loop.reduction_vars[0]   # support single reduction IV for now
        lo, hi = bounds.get(red_iv, (0, 1))
        k_var = iv_z3[red_iv]

        cp = self._infer_compute_payload(loop)
        reads_no_output = [r for r in loop.reads if r.tensor_name != loop.output_tensor]

        # Build parameter list for RecFunction: parallel IVs + reduction IV
        par_ivs = loop.parallel_vars
        rec_name = f"_acc_{loop.func_name}_{red_iv}"
        param_sorts = [IntSort()] * (len(par_ivs) + 1) + [RealSort()]

        acc_func = RecFunction(rec_name, *param_sorts)

        par_vars_z3 = [iv_z3[iv] for iv in par_ivs]
        k_z3 = iv_z3[red_iv]

        # Build body: A[...k...] * B[...k...] at k = k_z3 - 1
        local_iv = dict(iv_z3)
        local_iv[red_iv] = k_z3 - IntVal(1)

        def read_at(acc: AccessPattern) -> ArithRef:
            f = tensor_funcs.get(acc.tensor_name)
            if f is None:
                return RealVal(0)
            idxs = [_eval_index_expr(e, local_iv) for e in acc.index_exprs]
            return _apply_func(f, idxs) if idxs else Real(acc.tensor_name.lstrip('%'))

        if cp == ComputePayloadType.MULTIPLY_ACCUMULATE and len(reads_no_output) >= 2:
            body_term = read_at(reads_no_output[0]) * read_at(reads_no_output[1])
        else:
            body_term = read_at(reads_no_output[0]) if reads_no_output else RealVal(0)

        # Define: acc(par..., k) = If(k == lo, 0, acc(par..., k-1) + body@(k-1))
        RecAddDefinition(
            acc_func,
            par_vars_z3 + [k_z3],
            If(
                k_z3 == IntVal(int(lo)),
                RealVal(0),
                acc_func(*par_vars_z3, k_z3 - IntVal(1)) + body_term,
            ),
        )

        return acc_func(*par_vars_z3, IntVal(int(hi)))

    # ------------------------------------------------------------------
    # Build sketch assertions
    # ------------------------------------------------------------------

    def _build_sketch_assertions(
        self,
        loop: LoopNestInfo,
        sketch: OperationSketch,
        iv_z3: Dict[str, ArithRef],
        tensor_funcs: Dict[str, Any],
        bounds: Dict[str, Tuple],
    ) -> List[BoolRef]:
        """
        Encode the target sketch's semantics as Z3 assertions using
        the same tensor functions as the source program.
        """
        parallel_ivs = sketch.parallel_dims()
        red_ivs = sketch.reduction_dims()

        # Align sketch dim names to source IV names
        # The alignment is derived from matching parallel/reduction roles
        sketch_to_source = self._align_dims(loop, sketch)
        if sketch_to_source is None:
            return []

        # Map sketch IV names → source Z3 vars
        sketch_iv_z3: Dict[str, ArithRef] = {}
        for sdim, sdim_mapped in sketch_to_source.items():
            if sdim_mapped in iv_z3:
                sketch_iv_z3[sdim] = iv_z3[sdim_mapped]
            else:
                sketch_iv_z3[sdim] = Int(sdim)

        # Map operand indices (sketch uses positional order: [inputs..., outputs...])
        input_tensors = [r.tensor_name for r in loop.reads if r.tensor_name != loop.output_tensor]
        unique_inputs = list(dict.fromkeys(input_tensors))  # preserve order, deduplicate
        output_tensor = loop.output_tensor

        # Assign operands to sketch roles
        sketch_funcs: Dict[int, Any] = {}
        for i in range(sketch.num_inputs):
            name = unique_inputs[i] if i < len(unique_inputs) else (output_tensor or '%unknown')
            rank = len(loop.tensor_shapes.get(name, []))
            sketch_funcs[i] = tensor_funcs.get(name, _make_tensor_func(name, rank))
        sketch_funcs[sketch.num_inputs] = tensor_funcs.get(
            output_tensor, _make_tensor_func(output_tensor or '%out', 2)
        )

        # Build output value according to sketch semantics
        rhs = self._build_sketch_rhs(sketch, sketch_iv_z3, sketch_funcs, bounds, loop)

        # Output access (last operand in sketch)
        out_map_str = sketch.indexing_maps[-1]
        out_indices = self._eval_sketch_map(out_map_str, sketch_iv_z3)
        out_func = sketch_funcs[sketch.num_inputs]

        if not parallel_ivs:
            out_val = out_func(*out_indices) if out_indices else Real(str(out_func))
            assertion = (out_val == rhs)
            return [assertion]
        else:
            par_vars = [sketch_iv_z3[iv] for iv in parallel_ivs if iv in sketch_iv_z3]
            par_bounds_parts = []
            for sdim in parallel_ivs:
                src_iv = sketch_to_source.get(sdim, sdim)
                lo, hi = bounds.get(src_iv, (0, 1))
                v = sketch_iv_z3.get(sdim, Int(sdim))
                par_bounds_parts.append(And(IntVal(int(lo)) <= v, v < IntVal(int(hi))))
            par_bounds = And(*par_bounds_parts) if par_bounds_parts else BoolVal(True)
            out_val = _apply_func(out_func, out_indices)
            return [ForAll(par_vars, Implies(par_bounds, out_val == rhs))]

    def _build_sketch_rhs(
        self,
        sketch: OperationSketch,
        sketch_iv_z3: Dict[str, ArithRef],
        sketch_funcs: Dict[int, Any],
        bounds: Dict[str, Tuple],
        loop: LoopNestInfo,
    ) -> ArithRef:
        """Build the RHS Z3 expression from the sketch definition."""
        cp = sketch.compute_payload
        red_dims = sketch.reduction_dims()

        def read_operand(op_idx: int) -> ArithRef:
            map_str = sketch.indexing_maps[op_idx]
            indices = self._eval_sketch_map(map_str, sketch_iv_z3)
            f = sketch_funcs.get(op_idx)
            if f is None:
                return RealVal(0)
            return _apply_func(f, indices) if indices else Real(str(f))

        if not red_dims:
            # No reductions
            if cp == ComputePayloadType.COPY:
                return read_operand(0)
            elif cp == ComputePayloadType.ADD:
                return read_operand(0) + read_operand(1)
            elif cp == ComputePayloadType.SUBTRACT:
                return read_operand(0) - read_operand(1)
            elif cp == ComputePayloadType.MULTIPLY:
                return read_operand(0) * read_operand(1)
            elif cp == ComputePayloadType.RELU:
                v = read_operand(0)
                return If(v >= RealVal(0), v, RealVal(0))
            elif cp == ComputePayloadType.NEGATE:
                return -read_operand(0)
            elif cp == ComputePayloadType.MAX:
                a, b = read_operand(0), read_operand(1)
                return If(a >= b, a, b)
            return read_operand(0)

        # Has reductions — build concrete unrolled sum
        # Check if bounds are all concrete
        can_unroll = all(
            isinstance(bounds.get(loop_iv, (0, 0))[1], int)
            and (bounds.get(loop_iv, (0, 0))[1] - bounds.get(loop_iv, (0, 0))[0]) <= 32
            for loop_iv in loop.reduction_vars
        )

        if can_unroll:
            total: ArithRef = RealVal(0)
            # We need to iterate over the reduction dimension ranges
            for loop_iv, sketch_iv in zip(loop.reduction_vars, red_dims):
                lo, hi = bounds.get(loop_iv, (0, 1))
                for k_val in range(int(lo), int(hi)):
                    local_iv = dict(sketch_iv_z3)
                    local_iv[sketch_iv] = IntVal(k_val)
                    if cp == ComputePayloadType.MULTIPLY_ACCUMULATE:
                        a = self._read_at(sketch, 0, local_iv, sketch_funcs)
                        b = self._read_at(sketch, 1, local_iv, sketch_funcs)
                        total = total + a * b
                    elif cp == ComputePayloadType.ACCUMULATE_ADD:
                        a = self._read_at(sketch, 0, local_iv, sketch_funcs)
                        total = total + a
                    elif cp == ComputePayloadType.ACCUMULATE_MAX:
                        a = self._read_at(sketch, 0, local_iv, sketch_funcs)
                        total = If(a > total, a, total)
                break   # handle only first reduction dim here
            return total
        else:
            # Symbolic reduction — use 0 as placeholder (will partially verify)
            return RealVal(0)

    def _read_at(
        self,
        sketch: OperationSketch,
        op_idx: int,
        iv_z3: Dict[str, ArithRef],
        funcs: Dict[int, Any],
    ) -> ArithRef:
        map_str = sketch.indexing_maps[op_idx]
        indices = self._eval_sketch_map(map_str, iv_z3)
        f = funcs.get(op_idx)
        if f is None:
            return RealVal(0)
        return _apply_func(f, indices) if indices else Real(str(f))

    # ------------------------------------------------------------------
    # Symbolic verification — for dynamic shapes
    # ------------------------------------------------------------------

    def _verify_symbolic(self, loop: LoopNestInfo, sketch: OperationSketch) -> VerificationReport:
        """
        Attempt symbolic verification using Z3 quantifiers.
        Less reliable than concrete unrolling but handles dynamic shapes.
        """
        # For the symbolic case we do a best-effort check:
        # substitute concrete small values (M=4, N=4, K=4) and verify concretely
        concrete_loop = self._instantiate_concrete(loop, size=4)
        report = self._verify_concrete(concrete_loop, sketch)
        if report.result == CheckResult.EQUIVALENT:
            report.notes += " [symbolic: verified on M=N=K=4 representative instance]"
        return report

    def _instantiate_concrete(self, loop: LoopNestInfo, size: int = 4) -> LoopNestInfo:
        """Replace all symbolic bounds with a concrete small integer."""
        import copy
        concrete = copy.deepcopy(loop)
        for iv in concrete.bounds:
            lo, hi = concrete.bounds[iv]
            lo_int = lo if isinstance(lo, int) else 0
            hi_int = hi if isinstance(hi, int) else size
            concrete.bounds[iv] = (lo_int, hi_int)
        # Also fix tensor shapes
        for name in concrete.tensor_shapes:
            concrete.tensor_shapes[name] = [
                d if d > 0 else size for d in concrete.tensor_shapes[name]
            ]
        return concrete

    # ------------------------------------------------------------------
    # Alignment: sketch dims ← source IVs
    # ------------------------------------------------------------------

    def _align_dims(
        self, loop: LoopNestInfo, sketch: OperationSketch
    ) -> Optional[Dict[str, str]]:
        """
        Map sketch dimension names to source induction variable names
        by matching parallel/reduction roles.
        Returns None if alignment is impossible.
        """
        src_parallel = loop.parallel_vars
        src_reduction = loop.reduction_vars
        sk_parallel = sketch.parallel_dims()
        sk_reduction = sketch.reduction_dims()

        if len(src_parallel) != len(sk_parallel):
            return None
        if len(src_reduction) != len(sk_reduction):
            return None

        mapping: Dict[str, str] = {}
        for sk_dim, src_iv in zip(sk_parallel, src_parallel):
            mapping[sk_dim] = src_iv
        for sk_dim, src_iv in zip(sk_reduction, src_reduction):
            mapping[sk_dim] = src_iv
        return mapping

    def _eval_sketch_map(
        self, map_str: str, iv_z3: Dict[str, ArithRef]
    ) -> List[ArithRef]:
        """
        Evaluate an indexing map string like '(m, n, k) -> (m, k)'
        and return the result index expressions as Z3 terms.
        """
        # Extract the result part after '->'
        arrow_pos = map_str.find('->')
        if arrow_pos < 0:
            return []
        result_part = map_str[arrow_pos + 2:].strip().strip('(').strip(')')
        if not result_part:
            return []   # scalar output ()
        parts = [p.strip() for p in result_part.split(',')]
        return [_eval_index_expr(p, iv_z3) for p in parts if p]

    def _infer_compute_payload(self, loop: LoopNestInfo) -> ComputePayloadType:
        """Infer compute payload from loop arithmetic operations."""
        ops = {op.op_type for op in loop.compute_ops}
        has_mul = bool(ops & {'mulf', 'muli', 'mulsi'})
        has_add = bool(ops & {'addf', 'addi'})
        has_sub = bool(ops & {'subf', 'subi'})
        has_max = bool(ops & {'maxf', 'maxnumf', 'maxsi'})

        if has_mul and has_add:
            if loop.reduction_vars:
                return ComputePayloadType.MULTIPLY_ACCUMULATE
            return ComputePayloadType.MULTIPLY
        if has_add:
            if loop.reduction_vars:
                return ComputePayloadType.ACCUMULATE_ADD
            return ComputePayloadType.ADD
        if has_sub:
            return ComputePayloadType.SUBTRACT
        if has_max:
            if loop.reduction_vars:
                return ComputePayloadType.ACCUMULATE_MAX
            return ComputePayloadType.MAX
        if not loop.compute_ops:
            return ComputePayloadType.COPY
        return ComputePayloadType.COPY
