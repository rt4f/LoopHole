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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from z3.z3 import (
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
    counterexample_bindings: Dict[str, str] = field(default_factory=dict)
    failed_implication: Optional[str] = None
    mismatch_summary: Optional[str] = None
    sympy_confidence: Optional[float] = None
    sympy_z3_disagreement: Optional[str] = None
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


def _sanitize_symbol_name(raw: str) -> str:
    """Create a Z3-friendly symbol name from parsed affine tokens."""
    name = raw.strip().lstrip('%')
    if not name:
        return "sym"

    cleaned = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in name)
    cleaned = cleaned.strip('_')
    if not cleaned:
        cleaned = "sym"
    if cleaned[0].isdigit():
        cleaned = f"s_{cleaned}"
    return cleaned


def _tokenize_affine_expr(expr: str) -> List[str]:
    """Tokenize a simple affine arithmetic expression."""
    tokens: List[str] = []
    i = 0
    n = len(expr)

    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in '+-*()':
            tokens.append(ch)
            i += 1
            continue
        if ch.isdigit():
            j = i
            while j < n and expr[j].isdigit():
                j += 1
            tokens.append(expr[i:j])
            i = j
            continue
        if ch.isalpha() or ch in {'_', '%'}:
            j = i
            while j < n and (expr[j].isalnum() or expr[j] in {'_', '%'}):
                j += 1
            tokens.append(expr[i:j])
            i = j
            continue
        raise ValueError(f"Unsupported character in affine expression: {ch!r}")

    return tokens


class _AffineExprParser:
    """Recursive-descent parser for affine-like expressions used by bounds/maps."""

    def __init__(self, tokens: List[str], symbol_lookup: Callable[[str], ArithRef]):
        self.tokens = tokens
        self.pos = 0
        self.symbol_lookup = symbol_lookup

    def _peek(self) -> Optional[str]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _consume(self, expected: Optional[str] = None) -> str:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        if expected is not None and tok != expected:
            raise ValueError(f"Expected '{expected}', got '{tok}'")
        self.pos += 1
        return tok

    def parse(self) -> ArithRef:
        value = self._parse_expr()
        if self._peek() is not None:
            raise ValueError(f"Unexpected trailing token: {self._peek()}")
        return value

    def _parse_expr(self) -> ArithRef:
        value = self._parse_term()
        while True:
            tok = self._peek()
            if tok == '+':
                self._consume('+')
                value = value + self._parse_term()
            elif tok == '-':
                self._consume('-')
                value = value - self._parse_term()
            else:
                return value

    def _parse_term(self) -> ArithRef:
        value = self._parse_factor()
        while self._peek() == '*':
            self._consume('*')
            value = value * self._parse_factor()
        return value

    def _parse_factor(self) -> ArithRef:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end while parsing factor")

        if tok == '-':
            self._consume('-')
            return IntVal(-1) * self._parse_factor()
        if tok == '(':
            self._consume('(')
            value = self._parse_expr()
            self._consume(')')
            return value

        token = self._consume()
        if token.isdigit():
            return IntVal(int(token))
        return self.symbol_lookup(token)


def _eval_affine_expr(
    expr: str,
    symbol_lookup: Callable[[str], ArithRef],
) -> ArithRef:
    """Parse and evaluate affine arithmetic into a Z3 arithmetic expression."""
    tokens = _tokenize_affine_expr(expr)
    if not tokens:
        return IntVal(0)
    return _AffineExprParser(tokens, symbol_lookup).parse()


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
    if not expr:
        return IntVal(0)

    def lookup(token: str) -> ArithRef:
        key = token.strip()
        if key in iv_vars:
            return iv_vars[key]
        key = key.lstrip('%')
        if key in iv_vars:
            return iv_vars[key]
        return Int(_sanitize_symbol_name(key))

    try:
        return _eval_affine_expr(expr, lookup)
    except Exception:
        # Keep best-effort behavior for unusual affine tokens we do not parse yet.
        return Int(_sanitize_symbol_name(expr))


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
        self._bound_symbols: Dict[str, ArithRef] = {}

    def _bound_expr(self, value: Union[int, str]) -> ArithRef:
        """Convert a loop bound into a Z3 arithmetic expression."""
        if isinstance(value, int):
            return IntVal(value)

        text = str(value).strip()
        if not text:
            return IntVal(0)

        try:
            return IntVal(int(text))
        except ValueError:
            pass

        def lookup(token: str) -> ArithRef:
            key = token.strip().lstrip('%')
            if key not in self._bound_symbols:
                self._bound_symbols[key] = Int(_sanitize_symbol_name(key))
            return self._bound_symbols[key]

        try:
            return _eval_affine_expr(text, lookup)
        except Exception:
            # Fallback to a stable symbolic variable if parsing fails.
            return lookup(text)

    def _bound_as_int(self, value: Union[int, str]) -> Optional[int]:
        """Return a concrete integer bound when available, otherwise None."""
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _extract_model_details(self, solver: Solver) -> Tuple[Optional[str], Dict[str, str]]:
        """Extract a stable string and key-value bindings from the current Z3 model."""
        try:
            model = solver.model()
        except Exception:
            return None, {}

        model_str = str(model)
        bindings: Dict[str, str] = {}
        try:
            for decl in model.decls():
                name = decl.name()
                try:
                    bindings[name] = str(model[decl])
                except Exception:
                    bindings[name] = "<unavailable>"
        except Exception:
            return model_str, {}

        return model_str, bindings

    def _format_mismatch_summary(self, bindings: Dict[str, str]) -> Optional[str]:
        if not bindings:
            return None
        keys = sorted(bindings.keys())[:6]
        summary = ", ".join(f"{k}={bindings[k]}" for k in keys)
        if len(bindings) > len(keys):
            summary += f", ... ({len(bindings)} bindings)"
        return summary

    def _failed_implication_report(
        self,
        sketch_name: str,
        implication_name: str,
        solver_result: Any,
        solver: Solver,
    ) -> VerificationReport:
        model_str: Optional[str] = None
        model_bindings: Dict[str, str] = {}
        if solver_result == sat:
            model_str, model_bindings = self._extract_model_details(solver)

        if solver_result == sat:
            result = CheckResult.NOT_EQUIVALENT
        elif solver_result == unknown:
            result = CheckResult.TIMEOUT
        else:
            result = CheckResult.UNKNOWN

        return VerificationReport(
            result=result,
            sketch_name=sketch_name,
            elapsed_ms=0.0,
            z3_model=model_str,
            counterexample_bindings=model_bindings,
            failed_implication=implication_name,
            mismatch_summary=self._format_mismatch_summary(model_bindings),
            notes=f"{implication_name} implication failed.",
        )

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
        self._bound_symbols = {}

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
        has_symbolic_bounds = any(
            self._bound_as_int(lo) is None or self._bound_as_int(hi) is None
            for lo, hi in bounds.values()
        )

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
        domain_assumptions = (
            self._build_symbolic_shape_assumptions(loop, bounds)
            if has_symbolic_bounds
            else []
        )

        # Bi-directional equivalence:
        # Check ¬(source → sketch): should be UNSAT
        check1 = self._check_implication(
            solver,
            domain_assumptions + source_assertions,
            domain_assumptions + sketch_assertions,
        )
        if check1 != unsat:
            return self._failed_implication_report(
                sketch_name=sketch.name,
                implication_name="source_to_sketch",
                solver_result=check1,
                solver=solver,
            )

        # Check ¬(sketch → source): should also be UNSAT
        solver.reset()
        solver.set("timeout", self.timeout_ms)
        check2 = self._check_implication(
            solver,
            domain_assumptions + sketch_assertions,
            domain_assumptions + source_assertions,
        )
        if check2 != unsat:
            return self._failed_implication_report(
                sketch_name=sketch.name,
                implication_name="sketch_to_source",
                solver_result=check2,
                solver=solver,
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

    def _parse_simple_iv_offset(self, expr: str) -> Optional[Tuple[str, int]]:
        """
        Parse a restricted affine index form to iv + offset.

        Supported shapes:
          iv
          iv + c
          iv - c
          c + iv
        """
        try:
            tokens = _tokenize_affine_expr(expr)
        except Exception:
            return None

        def as_var(tok: str) -> Optional[str]:
            if tok.isdigit() or tok in {'+', '-', '*', '(', ')'}:
                return None
            return tok.lstrip('%')

        if len(tokens) == 1:
            var = as_var(tokens[0])
            if var:
                return var, 0
            return None

        if len(tokens) == 3 and tokens[1] in {'+', '-'}:
            left_var = as_var(tokens[0])
            right_var = as_var(tokens[2])
            left_num = int(tokens[0]) if tokens[0].isdigit() else None
            right_num = int(tokens[2]) if tokens[2].isdigit() else None

            if left_var is not None and right_num is not None:
                offset = right_num if tokens[1] == '+' else -right_num
                return left_var, offset

            if left_num is not None and right_var is not None and tokens[1] == '+':
                return right_var, left_num

        return None

    def _build_symbolic_shape_assumptions(
        self,
        loop: LoopNestInfo,
        bounds: Dict[str, Tuple[Union[int, str], Union[int, str]]],
    ) -> List[BoolRef]:
        """
        Add conservative domain assumptions for symbolic bounds.

        These assumptions reflect static tensor extents for direct iv (+/- const)
        access forms, which helps the solver avoid exploring impossible domains.
        """
        assumptions: List[BoolRef] = []
        seen: set[str] = set()

        for access in list(loop.reads) + list(loop.writes):
            shape = loop.tensor_shapes.get(access.tensor_name, access.shape)
            if not shape:
                continue

            for dim_idx, idx_expr in enumerate(access.index_exprs):
                if dim_idx >= len(shape):
                    continue
                extent = shape[dim_idx]
                if extent <= 0:
                    continue

                parsed = self._parse_simple_iv_offset(idx_expr)
                if parsed is None:
                    continue

                iv_name, offset = parsed
                if iv_name not in bounds:
                    continue

                lo, hi = bounds[iv_name]
                lo_expr = self._bound_expr(lo)
                hi_expr = self._bound_expr(hi)

                lo_constraint = lo_expr + IntVal(offset) >= IntVal(0)
                hi_constraint = hi_expr + IntVal(offset) <= IntVal(extent)

                lo_key = f"{iv_name}:{offset}:lo"
                hi_key = f"{iv_name}:{offset}:hi:{extent}"

                if lo_key not in seen:
                    assumptions.append(lo_constraint)
                    seen.add(lo_key)
                if hi_key not in seen:
                    assumptions.append(hi_constraint)
                    seen.add(hi_key)

        return assumptions

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
                parts.append(And(self._bound_expr(lo) <= v, v < self._bound_expr(hi)))
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
            self._bound_as_int(bounds.get(iv, (0, 0))[0]) is not None
            and self._bound_as_int(bounds.get(iv, (0, 0))[1]) is not None
            and (
                self._bound_as_int(bounds.get(iv, (0, 0))[1])
                - self._bound_as_int(bounds.get(iv, (0, 0))[0])
            )
            <= self.unroll_threshold
            for iv in red_ivs
        )

        if can_unroll:
            return self._unrolled_reduction(loop, iv_z3, tensor_funcs, bounds)

        symbolic_limit = self._symbolic_reduction_upper_bound(loop, bounds)
        if symbolic_limit is not None:
            return self._guarded_symbolic_unroll_reduction(
                loop,
                iv_z3,
                tensor_funcs,
                bounds,
                symbolic_limit,
            )

        else:
            return self._recfunc_reduction(loop, iv_z3, tensor_funcs, bounds)

    def _symbolic_reduction_upper_bound(
        self,
        loop: LoopNestInfo,
        bounds: Dict[str, Tuple[Union[int, str], Union[int, str]]],
    ) -> Optional[int]:
        """
        Infer a finite symbolic reduction upper bound from static tensor extents.

        This is used to build guarded unrolled sums for symbolic bounds, which is
        typically more solver-friendly than recursive definitions.
        """
        if len(loop.reduction_vars) != 1:
            return None

        red_iv = loop.reduction_vars[0]
        lo_raw, hi_raw = bounds.get(red_iv, (0, 1))
        lo = self._bound_as_int(lo_raw)
        if lo is None:
            return None

        if self._bound_as_int(hi_raw) is not None:
            return None

        upper: Optional[int] = None
        for access in list(loop.reads) + list(loop.writes):
            shape = loop.tensor_shapes.get(access.tensor_name, access.shape)
            if not shape:
                continue

            for dim_idx, idx_expr in enumerate(access.index_exprs):
                if dim_idx >= len(shape):
                    continue
                extent = shape[dim_idx]
                if extent <= 0:
                    continue

                parsed = self._parse_simple_iv_offset(idx_expr)
                if parsed is None:
                    continue

                iv_name, offset = parsed
                if iv_name != red_iv:
                    continue

                candidate = extent - offset
                upper = candidate if upper is None else min(upper, candidate)

        if upper is None:
            return None

        if upper - lo > self.unroll_threshold:
            return None
        return upper

    def _guarded_symbolic_unroll_reduction(
        self,
        loop: LoopNestInfo,
        iv_z3: Dict[str, ArithRef],
        tensor_funcs: Dict[str, Any],
        bounds: Dict[str, Tuple[Union[int, str], Union[int, str]]],
        upper: int,
    ) -> ArithRef:
        """Build reduction as a finite guarded sum over a symbolic upper bound."""
        red_iv = loop.reduction_vars[0]
        lo_raw, hi_raw = bounds.get(red_iv, (0, 1))
        lo = self._bound_as_int(lo_raw)
        if lo is None:
            raise ValueError("Guarded symbolic unroll requires concrete lower bound.")

        hi_expr = self._bound_expr(hi_raw)
        cp = self._infer_compute_payload(loop)
        total: ArithRef = RealVal(-1e30) if cp == ComputePayloadType.ACCUMULATE_MAX else RealVal(0)

        for k in range(lo, upper):
            local_iv = dict(iv_z3)
            local_iv[red_iv] = IntVal(k)
            term = self._body_term_at(loop, local_iv, tensor_funcs, cp)
            active = IntVal(k) < hi_expr

            if cp in (ComputePayloadType.MULTIPLY_ACCUMULATE, ComputePayloadType.ACCUMULATE_ADD):
                total = If(active, total + term, total)
            elif cp == ComputePayloadType.ACCUMULATE_MAX:
                candidate = If(term > total, term, total)
                total = If(active, candidate, total)
            else:
                total = If(active, total + term, total)

        return total

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
        if len(loop.reduction_vars) != 1:
            raise ValueError(
                "Symbolic/large reduction path requires a single reduction dimension; "
                "use concrete bounds for multi-reduction kernels."
            )

        red_iv = loop.reduction_vars[0]
        lo, hi = bounds.get(red_iv, (0, 1))

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
                k_z3 == self._bound_expr(lo),
                RealVal(0),
                acc_func(*par_vars_z3, k_z3 - IntVal(1)) + body_term,
            ),
        )

        return acc_func(*par_vars_z3, self._bound_expr(hi))

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
        out_shape = loop.tensor_shapes.get(output_tensor or "", [])
        if not out_indices and len(out_shape) == 1 and out_shape[0] == 1:
            # Dot-like scalar outputs are often represented as memref<1xT>; map () to [0].
            out_indices = [IntVal(0)]

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
                par_bounds_parts.append(And(self._bound_expr(lo) <= v, v < self._bound_expr(hi)))
            par_bounds = And(*par_bounds_parts) if par_bounds_parts else BoolVal(True)
            out_val = _apply_func(out_func, out_indices) if out_indices else Real(str(out_func))
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
            self._bound_as_int(bounds.get(loop_iv, (0, 0))[0]) is not None
            and self._bound_as_int(bounds.get(loop_iv, (0, 0))[1]) is not None
            and (
                self._bound_as_int(bounds.get(loop_iv, (0, 0))[1])
                - self._bound_as_int(bounds.get(loop_iv, (0, 0))[0])
            )
            <= self.unroll_threshold
            for loop_iv in loop.reduction_vars
        )

        if can_unroll:
            total: ArithRef = RealVal(-1e30) if cp == ComputePayloadType.ACCUMULATE_MAX else RealVal(0)

            def iter_reduction_points(idx: int, current: Dict[str, int]):
                if idx == len(loop.reduction_vars):
                    yield dict(current)
                    return

                loop_iv = loop.reduction_vars[idx]
                sketch_iv = red_dims[idx]
                lo_raw, hi_raw = bounds.get(loop_iv, (0, 1))
                lo = self._bound_as_int(lo_raw)
                hi = self._bound_as_int(hi_raw)
                if lo is None or hi is None:
                    raise ValueError(
                        "Cannot unroll sketch reduction with symbolic bounds; "
                        "symbolic reduction requires a single reduction dimension."
                    )

                for value in range(lo, hi):
                    current[sketch_iv] = value
                    yield from iter_reduction_points(idx + 1, current)
                    del current[sketch_iv]

            for values in iter_reduction_points(0, {}):
                local_iv = dict(sketch_iv_z3)
                for sk_iv, value in values.items():
                    local_iv[sk_iv] = IntVal(value)

                if cp == ComputePayloadType.MULTIPLY_ACCUMULATE:
                    a = self._read_at(sketch, 0, local_iv, sketch_funcs)
                    b = self._read_at(sketch, 1, local_iv, sketch_funcs)
                    total = total + a * b
                elif cp == ComputePayloadType.ACCUMULATE_ADD:
                    total = total + self._read_at(sketch, 0, local_iv, sketch_funcs)
                elif cp == ComputePayloadType.ACCUMULATE_MAX:
                    a = self._read_at(sketch, 0, local_iv, sketch_funcs)
                    total = If(a > total, a, total)
                else:
                    total = total + self._read_at(sketch, 0, local_iv, sketch_funcs)

            return total

        symbolic_limit = self._symbolic_reduction_upper_bound(loop, bounds)
        if symbolic_limit is not None and len(loop.reduction_vars) == 1 and len(red_dims) == 1:
            red_loop_iv = loop.reduction_vars[0]
            red_sketch_iv = red_dims[0]
            lo_raw, hi_raw = bounds.get(red_loop_iv, (0, 1))
            lo = self._bound_as_int(lo_raw)
            if lo is None:
                raise ValueError("Guarded symbolic sketch reduction requires concrete lower bound.")

            hi_expr = self._bound_expr(hi_raw)
            total: ArithRef = RealVal(-1e30) if cp == ComputePayloadType.ACCUMULATE_MAX else RealVal(0)

            for k_val in range(lo, symbolic_limit):
                local_iv = dict(sketch_iv_z3)
                local_iv[red_sketch_iv] = IntVal(k_val)
                active = IntVal(k_val) < hi_expr

                if cp == ComputePayloadType.MULTIPLY_ACCUMULATE:
                    a = self._read_at(sketch, 0, local_iv, sketch_funcs)
                    b = self._read_at(sketch, 1, local_iv, sketch_funcs)
                    total = If(active, total + (a * b), total)
                elif cp == ComputePayloadType.ACCUMULATE_ADD:
                    a = self._read_at(sketch, 0, local_iv, sketch_funcs)
                    total = If(active, total + a, total)
                elif cp == ComputePayloadType.ACCUMULATE_MAX:
                    a = self._read_at(sketch, 0, local_iv, sketch_funcs)
                    total = If(active, If(a > total, a, total), total)
                else:
                    a = self._read_at(sketch, 0, local_iv, sketch_funcs)
                    total = If(active, total + a, total)

            return total

        if len(loop.reduction_vars) != 1:
            raise ValueError(
                "Symbolic sketch reduction requires a single reduction dimension; "
                "multi-reduction symbolic proofs are not yet supported."
            )

        red_loop_iv = loop.reduction_vars[0]
        red_sketch_iv = red_dims[0]
        lo, hi = bounds.get(red_loop_iv, (0, 1))
        lo_expr = self._bound_expr(lo)
        hi_expr = self._bound_expr(hi)

        acc_func = RecFunction(f"_sketch_acc_{loop.func_name}_{red_loop_iv}", IntSort(), RealSort())
        k = Int(f"{red_sketch_iv}_rf")

        local_iv = dict(sketch_iv_z3)
        local_iv[red_sketch_iv] = k - IntVal(1)

        if cp == ComputePayloadType.MULTIPLY_ACCUMULATE:
            step = self._read_at(sketch, 0, local_iv, sketch_funcs) * self._read_at(sketch, 1, local_iv, sketch_funcs)
        elif cp in (ComputePayloadType.ACCUMULATE_ADD, ComputePayloadType.ACCUMULATE_MAX):
            step = self._read_at(sketch, 0, local_iv, sketch_funcs)
        else:
            step = self._read_at(sketch, 0, local_iv, sketch_funcs)

        if cp == ComputePayloadType.ACCUMULATE_MAX:
            base = RealVal(-1e30)
            rec_step = If(step > acc_func(k - IntVal(1)), step, acc_func(k - IntVal(1)))
        else:
            base = RealVal(0)
            rec_step = acc_func(k - IntVal(1)) + step

        RecAddDefinition(
            acc_func,
            [k],
            If(k == lo_expr, base, rec_step),
        )
        return acc_func(hi_expr)

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
        Attempt direct symbolic verification using quantified constraints.
        This path avoids fixed representative-size substitution.
        """
        try:
            report = self._verify_concrete(loop, sketch)
        except ValueError as exc:
            return VerificationReport(
                result=CheckResult.UNKNOWN,
                sketch_name=sketch.name,
                elapsed_ms=0.0,
                notes=f"Symbolic verification unsupported for this kernel shape: {exc}",
            )

        if report.result == CheckResult.EQUIVALENT:
            report.notes += " [symbolic: quantified verification path]"
        return report

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
