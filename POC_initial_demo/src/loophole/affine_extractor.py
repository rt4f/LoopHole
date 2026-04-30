"""
affine_extractor.py — Parse MLIR Affine/SCF loop nests into LoopNestInfo.

This parser is intentionally implemented without regex-based extraction.
It uses deterministic token scanning and optional MLIR syntax validation via
MLIR Python bindings when they are available in the runtime environment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import re

try:
    from mlir import ir as _mlir_ir
except Exception:
    _mlir_ir = None

try:
    import sympy as _sympy
except Exception:
    _sympy = None


@dataclass
class AccessPattern:
    """Describes a single memref read or write with its subscript expression."""

    tensor_name: str
    index_exprs: List[str]
    is_affine: bool
    is_read: bool
    shape: List[int]
    element_type: str
    ssa_name: str


@dataclass
class ComputeOp:
    """A single arithmetic operation in the loop body."""

    op_type: str
    operands: List[str]
    result: str
    value: Optional[float] = None


@dataclass
class ParsingWarning:
    """A diagnostic warning/error from the parser with contextual information."""

    level: str  # 'debug', 'warning', 'error'
    location: str  # Parser method name or construct (e.g., '_parse_func_declaration')
    reason: str  # Why the parse failed (e.g., "missing closing paren")
    input_snippet: str  # Up to 80 chars of problematic input
    guidance: Optional[str] = None  # Optional hint for the user

    def __str__(self) -> str:
        msg = f"{self.level.upper()} at {self.location}:\n"
        msg += f"  Input: {self.input_snippet}\n"
        msg += f"  Reason: {self.reason}"
        if self.guidance:
            msg += f"\n  Guidance: {self.guidance}"
        return msg


@dataclass
class LoopNestInfo:
    """Full semantic description of an extracted loop nest."""

    func_name: str
    induction_vars: List[str]
    bounds: Dict[str, Tuple[Union[int, str], Union[int, str]]]
    loop_order: List[str]
    reads: List[AccessPattern]
    writes: List[AccessPattern]
    compute_ops: List[ComputeOp]
    tensor_shapes: Dict[str, List[int]]
    tensor_types: Dict[str, str]
    element_type: str
    reduction_vars: List[str]
    parallel_vars: List[str]
    func_args: Dict[str, str]
    has_accumulation: bool
    accumulation_op: Optional[str]
    diagnostics: List[ParsingWarning] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.diagnostics is None:
            self.diagnostics = []

    @property
    def output_tensor(self) -> Optional[str]:
        if self.writes:
            return self.writes[0].tensor_name
        return None

    @property
    def input_tensors(self) -> List[str]:
        out = self.output_tensor
        return [r.tensor_name for r in self.reads if r.tensor_name != out]


def _split_top_level(text: str, delimiter: str = ",") -> List[str]:
    """Split by delimiter while respecting nested (), [], <> depth."""
    parts: List[str] = []
    buf: List[str] = []
    depth_paren = 0
    depth_bracket = 0
    depth_angle = 0

    for ch in text:
        if ch == "(":
            depth_paren += 1
        elif ch == ")" and depth_paren > 0:
            depth_paren -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]" and depth_bracket > 0:
            depth_bracket -= 1
        elif ch == "<":
            depth_angle += 1
        elif ch == ">" and depth_angle > 0:
            depth_angle -= 1

        if (
            ch == delimiter
            and depth_paren == 0
            and depth_bracket == 0
            and depth_angle == 0
        ):
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _strip_inline_comment(line: str) -> str:
    idx = line.find("//")
    if idx >= 0:
        return line[:idx].rstrip()
    return line.rstrip()


def _find_matching(text: str, open_idx: int, open_ch: str, close_ch: str) -> int:
    """Find the matching close char index for the opener at open_idx."""
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != open_ch:
        return -1
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch in {"_", "%"}


def _replace_token_boundary(expr: str, old: str, new: str) -> str:
    """Replace old token with new only at identifier boundaries."""
    if not old:
        return expr

    out: List[str] = []
    i = 0
    n = len(expr)
    m = len(old)
    while i < n:
        if expr.startswith(old, i):
            prev = expr[i - 1] if i > 0 else ""
            nxt = expr[i + m] if i + m < n else ""
            if (not prev or not _is_ident_char(prev)) and (not nxt or not _is_ident_char(nxt)):
                out.append(new)
                i += m
                continue
        out.append(expr[i])
        i += 1
    return "".join(out)


def _strip_ssa_percent(expr: str) -> str:
    """Convert %foo tokens to foo for downstream symbolic consumers."""
    out: List[str] = []
    i = 0
    n = len(expr)
    while i < n:
        if expr[i] == "%" and i + 1 < n and (expr[i + 1].isalnum() or expr[i + 1] == "_"):
            i += 1
            continue
        out.append(expr[i])
        i += 1
    return "".join(out)


def _extract_memref_payload(type_str: str) -> Optional[str]:
    idx = type_str.find("memref<")
    if idx < 0:
        return None
    start = idx + len("memref<")
    depth = 1
    for i in range(start, len(type_str)):
        ch = type_str[i]
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
            if depth == 0:
                return type_str[start:i]
    return None


def _parse_memref_type(type_str: str) -> Tuple[List[int], str]:
    """
    Parse memref<MxNxf32> like syntax into (shape, elem_type).
    Unknown/dynamic dimensions are represented as -1.
    """
    raw = type_str.strip()
    payload = _extract_memref_payload(raw)
    if payload is None:
        payload = raw

    # Strip layout/affine details if present.
    main = payload.split(",", 1)[0].strip()
    if main == "*":
        return [], "f32"
    if main.startswith("*x"):
        return [], main.split("x", 1)[1].strip() or "f32"
    if "x" not in main:
        return [], main or "f32"

    parts = [p.strip() for p in main.split("x") if p.strip()]
    if not parts:
        return [], "f32"

    elem_type = parts[-1]
    shape_tokens = parts[:-1]
    shape: List[int] = []
    for token in shape_tokens:
        if token in {"?", "-1", "*"}:
            shape.append(-1)
            continue
        try:
            value = int(token, 10)
            shape.append(value if value >= 0 else -1)
        except ValueError:
            shape.append(-1)
    return shape, elem_type


def _apply_known_constants(
    expr: str,
    const_map: Optional[Dict[str, Union[int, float]]] = None,
) -> str:
    """Substitute known constant SSA names (for example %c1) with literal values."""
    if not const_map:
        return expr

    out = expr
    for name, value in const_map.items():
        value_str = str(int(value) if isinstance(value, (int, float)) and float(value).is_integer() else value)
        # Replace both %name and stripped name forms using token boundaries.
        out = _replace_token_boundary(out, str(name), value_str)
        out = _replace_token_boundary(out, str(name).lstrip("%"), value_str)
    return out


def _canonicalize_index_expr(expr: str) -> str:
    """
    Canonicalize arithmetic index expressions.

    Uses SymPy when available to normalize reordered/parenthesized equivalent forms.
    Falls back to whitespace normalization when symbolic parsing is unavailable.
    """
    expr = " ".join(expr.split())
    if not expr:
        return expr

    if _sympy is None:
        return expr

    try:
        symbol_names = set(re.findall(r"\b[a-zA-Z_]\w*\b", expr))
        locals_map = {name: _sympy.Symbol(name, integer=True) for name in symbol_names}
        parsed = _sympy.sympify(expr, locals=locals_map)
        canonical = _sympy.sstr(_sympy.expand(parsed), order="lex")
        return " ".join(canonical.split())
    except Exception:
        return expr


def _normalize_index(
    idx: str,
    iv_map: Dict[str, str],
    const_map: Optional[Dict[str, Union[int, float]]] = None,
    apply_map: Optional[Dict[str, str]] = None,
) -> str:
    expr = idx.strip()
    if apply_map:
        for ssa_name, symbolic_expr in sorted(apply_map.items(), key=lambda item: len(item[0]), reverse=True):
            expr = _replace_token_boundary(expr, ssa_name, f"({symbolic_expr})")
    for ssa_name, short_name in sorted(iv_map.items(), key=lambda item: len(item[0]), reverse=True):
        expr = _replace_token_boundary(expr, ssa_name, short_name)
    expr = _apply_known_constants(expr, const_map)
    expr = _strip_ssa_percent(expr)
    expr = _canonicalize_index_expr(expr)
    return " ".join(expr.split())


def _diagnostic_snippet(text: str, max_len: int = 80) -> str:
    """Create a truncated diagnostic snippet of input text with ellipsis if needed."""
    text = text.strip()
    if len(text) <= max_len:
        return f'"{text}"'
    return f'"{text[:max_len]}..."'


class AffineExtractor:
    """
    Parse MLIR Affine/SCF IR text and extract LoopNestInfo.

    The extractor avoids regex parsing. If MLIR Python bindings are available,
    syntax is validated through the MLIR parser before extraction.
    """

    def __init__(self, validate_with_mlir: bool = True, verbose_diagnostics: bool = False):
        self.validate_with_mlir = validate_with_mlir
        self.verbose_diagnostics = verbose_diagnostics
        self.diagnostics: List[ParsingWarning] = []

    def extract(self, mlir_text: str) -> LoopNestInfo:
        text = mlir_text.replace("\r\n", "\n").replace("\r", "\n")
        self.diagnostics = []  # Reset diagnostics for new extraction

        self._validate_mlir_syntax(text)

        func_name, func_args = self._parse_func_declaration(text)
        const_map = self._extract_constants(text)
        induction_vars, bounds_raw, loop_order, iv_map = self._parse_loop_structure(text)
        apply_map = self._extract_affine_apply_map(text, iv_map)
        reads, writes = self._collect_accesses(text, iv_map, const_map, apply_map)
        compute_ops = self._collect_compute_ops(text)
        self._add_unsupported_form_diagnostics(
            text=text,
            induction_vars=induction_vars,
            writes=writes,
        )
        bounds: Dict[str, Tuple[Union[int, str], Union[int, str]]] = {}
        for iv, (lo, hi) in bounds_raw.items():
            lo_r = self._resolve_bound(lo, const_map)
            hi_r = self._resolve_bound(hi, const_map)
            bounds[iv] = (lo_r, hi_r)

        tensor_shapes, tensor_types = self._resolve_tensor_metadata(func_args, reads, writes)
        element_type = self._dominant_element_type(tensor_types)
        reduction_vars, parallel_vars = self._classify_iv_roles(induction_vars, writes)
        has_accum, accum_op = self._detect_accumulation(compute_ops, reads, writes)

        return LoopNestInfo(
            func_name=func_name,
            induction_vars=induction_vars,
            bounds=bounds,
            loop_order=loop_order,
            reads=reads,
            writes=writes,
            compute_ops=compute_ops,
            tensor_shapes=tensor_shapes,
            tensor_types=tensor_types,
            element_type=element_type,
            reduction_vars=reduction_vars,
            parallel_vars=parallel_vars,
            func_args=func_args,
            has_accumulation=has_accum,
            accumulation_op=accum_op,
            diagnostics=self.diagnostics,
        )

    def _add_unsupported_form_diagnostics(
        self,
        text: str,
        induction_vars: List[str],
        writes: List[AccessPattern],
    ) -> None:
        """Emit explicit unsupported-form diagnostics for known corpus miss classes."""
        # scf.for with iter_args is still unsupported (result returned via SSA, no affine.store).
        # affine.for with iter_args is handled by the parser.
        if "scf.for" in text and "iter_args(" in text:
            self._add_diagnostic(
                "warning",
                "_add_unsupported_form_diagnostics",
                "Unsupported scf.for iter_args form detected",
                "iter_args(",
                guidance=(
                    "Rewrite iter_args reductions to explicit output memref accumulation "
                    "(load/add/store) before lifting."
                ),
            )

        if "affine.if" in text or "scf.if" in text:
            self._add_diagnostic(
                "warning",
                "_add_unsupported_form_diagnostics",
                "Conditional region form is not modeled by current parser",
                "affine.if / scf.if",
                guidance=(
                    "Specialize or predicatize conditionals before lifting, or split guarded "
                    "regions into separate loop kernels with explicit stores."
                ),
            )

        # affine.apply is handled by _extract_affine_apply_map; no warning needed

        if not induction_vars:
            self._add_diagnostic(
                "warning",
                "_add_unsupported_form_diagnostics",
                "No affine.for/scf.for loop nest detected",
                text[:120],
                guidance=(
                    "Lower vectorized/region-only frontend IR into explicit affine/scf loop "
                    "nests before parsing."
                ),
            )

        if not writes:
            self._add_diagnostic(
                "warning",
                "_add_unsupported_form_diagnostics",
                "No explicit output store detected",
                text[:120],
                guidance=(
                    "Ensure the kernel writes output via affine.store/memref.store, or lower "
                    "implicit return/iter_args updates into explicit stores."
                ),
            )

    def _validate_mlir_syntax(self, text: str) -> None:
        if not self.validate_with_mlir or _mlir_ir is None:
            return

        candidates = [text]
        if "module" not in text:
            candidates.append(f"module {{\n{text}\n}}")

        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                with _mlir_ir.Context() as ctx:
                    ctx.allow_unregistered_dialects = True
                    _mlir_ir.Module.parse(candidate)
                return
            except Exception as exc:
                last_error = exc

        raise ValueError(f"MLIR syntax validation failed: {last_error}")

    def _add_diagnostic(
        self,
        level: str,
        location: str,
        reason: str,
        input_text: str,
        guidance: Optional[str] = None,
    ) -> None:
        """Log a diagnostic warning/error with context."""
        snippet = _diagnostic_snippet(input_text)
        warning = ParsingWarning(
            level=level,
            location=location,
            reason=reason,
            input_snippet=snippet,
            guidance=guidance,
        )
        # Only include debug-level logs if verbose_diagnostics is True
        if level != 'debug' or self.verbose_diagnostics:
            self.diagnostics.append(warning)

    def _iter_clean_lines(self, text: str) -> List[str]:
        lines: List[str] = []
        for raw in text.split("\n"):
            line = _strip_inline_comment(raw).strip()
            if line:
                lines.append(line)
        return lines

    def _parse_func_declaration(self, text: str) -> Tuple[str, Dict[str, str]]:
        marker = "func.func @"
        idx = text.find(marker)
        if idx < 0:
            self._add_diagnostic(
                'error',
                '_parse_func_declaration',
                'Function marker not found',
                text[:100],
                guidance="Expected format: 'func.func @name(...) { ... }'",
            )
            return "unknown", {}

        name_start = idx + len(marker)
        name_end = text.find("(", name_start)
        if name_end < 0:
            func_line = text[idx:idx+200]
            self._add_diagnostic(
                'error',
                '_parse_func_declaration',
                'No opening paren for function arguments',
                func_line,
                guidance="Expected format: 'func.func @name(...)'",
            )
            return "unknown", {}

        func_name = text[name_start:name_end].strip().split()[0]
        args_end = _find_matching(text, name_end, "(", ")")
        if args_end < 0:
            func_line = text[idx:idx+200]
            self._add_diagnostic(
                'error',
                '_parse_func_declaration',
                'No closing paren for function arguments',
                func_line,
                guidance="Check for unmatched parentheses in argument list",
            )
            return func_name, {}

        args_str = text[name_end + 1:args_end].strip()
        func_args: Dict[str, str] = {}
        if not args_str:
            return func_name, func_args

        for segment in _split_top_level(args_str, ","):
            if ":" not in segment:
                continue
            arg_name, arg_type = segment.split(":", 1)
            arg_name = arg_name.strip()
            arg_type = arg_type.strip()
            if arg_name:
                func_args[arg_name] = arg_type

        return func_name, func_args

    def _extract_constants(self, text: str) -> Dict[str, Union[int, float]]:
        const_map: Dict[str, Union[int, float]] = {}
        for line in self._iter_clean_lines(text):
            if "=" not in line or "arith.constant" not in line:
                continue

            lhs, rhs = line.split("=", 1)
            ssa_name = lhs.strip()
            rhs = rhs.strip()
            marker = "arith.constant"
            marker_idx = rhs.find(marker)
            if marker_idx < 0:
                continue

            literal_part = rhs[marker_idx + len(marker):].strip()
            literal = literal_part.split(":", 1)[0].strip()
            if not literal:
                self._add_diagnostic(
                    'warning',
                    '_extract_constants',
                    f'Missing constant literal for {ssa_name}',
                    line,
                )
                continue
            
            if literal.startswith("dense<"):
                self._add_diagnostic(
                    'warning',
                    '_extract_constants',
                    f'Non-scalar constant (dense tensor) for {ssa_name}',
                    line,
                    guidance='Only scalar arith.constant values are supported',
                )
                continue

            try:
                if any(ch in literal for ch in (".", "e", "E")):
                    const_map[ssa_name] = float(literal)
                else:
                    const_map[ssa_name] = int(literal, 10)
            except ValueError:
                self._add_diagnostic(
                    'warning',
                    '_extract_constants',
                    f'Unparseable constant literal for {ssa_name}',
                    line,
                    guidance='Expected integer or float literal',
                )
                continue

        return const_map

    def _allocate_iv_name(self, iv_ssa: str, used: List[str]) -> str:
        base = iv_ssa.strip().lstrip("%")
        if not base or not base[0].isalpha():
            base = f"iv{len(used)}"

        name = base
        suffix = 0
        while name in used:
            suffix += 1
            name = f"{base}_{suffix}"
        return name

    def _parse_affine_for(self, line: str) -> Optional[Tuple[str, str, str]]:
        # Handles both:
        #   affine.for %iv = lb to ub { ... }
        #   %result = affine.for %iv = lb to ub iter_args(%acc = %init) -> (type) { ... }
        if "affine.for " not in line:
            return None
        idx = line.find("affine.for ")
        body = line[idx + len("affine.for "):].strip()
        if "=" not in body or " to " not in body:
            return None

        iv_ssa, rhs = body.split("=", 1)
        iv_ssa = iv_ssa.strip()

        lb_part, ub_part = rhs.split(" to ", 1)
        lb = lb_part.strip()
        ub = ub_part.strip()

        if " step " in ub:
            ub = ub.split(" step ", 1)[0].strip()
        # Strip iter_args(...) and return type annotations produced by Polygeist
        if " iter_args" in ub:
            ub = ub.split(" iter_args", 1)[0].strip()
        if " ->" in ub:
            ub = ub.split(" ->", 1)[0].strip()
        if ub.endswith("{"):
            ub = ub[:-1].strip()

        return iv_ssa, lb, ub

    def _parse_scf_for(self, line: str) -> Optional[Tuple[str, str, str]]:
        if not line.startswith("scf.for "):
            return None
        body = line[len("scf.for "):].strip()
        if "=" not in body or " to " not in body:
            return None

        iv_ssa, rhs = body.split("=", 1)
        iv_ssa = iv_ssa.strip()

        lb_part, ub_and_step = rhs.split(" to ", 1)
        lb = lb_part.strip()
        ub = ub_and_step.strip()
        if " step " in ub:
            ub = ub.split(" step ", 1)[0].strip()
        if ub.endswith("{"):
            ub = ub[:-1].strip()

        return iv_ssa, lb, ub

    def _parse_loop_structure(
        self,
        text: str,
    ) -> Tuple[List[str], Dict[str, Tuple[str, str]], List[str], Dict[str, str]]:
        induction_vars: List[str] = []
        bounds_raw: Dict[str, Tuple[str, str]] = {}
        loop_order: List[str] = []
        iv_map: Dict[str, str] = {}
        has_affine = False
        has_scf = False

        for line in self._iter_clean_lines(text):
            parsed = self._parse_affine_for(line)
            if parsed is None:
                parsed = self._parse_scf_for(line)
                if parsed is not None:
                    has_scf = True
            elif parsed is not None:
                has_affine = True
            
            if parsed is None:
                # Check if this looks like a loop-related line that failed parsing
                if 'for ' in line and ('=' in line or ' to ' in line):
                    self._add_diagnostic(
                        'debug',
                        '_parse_loop_structure',
                        'Unrecognized loop syntax',
                        line,
                        guidance='Expected "affine.for" or "scf.for"',
                    )
                continue

            iv_ssa, lb, ub = parsed
            short = self._allocate_iv_name(iv_ssa, induction_vars)
            induction_vars.append(short)
            loop_order.append(short)
            bounds_raw[short] = (lb, ub)
            iv_map[iv_ssa] = short

        # Mixed affine/scf nests are supported; keep a debug diagnostic for traceability.
        if has_affine and has_scf:
            self._add_diagnostic(
                'debug',
                '_parse_loop_structure',
                'Mixed affine.for and scf.for loops detected',
                text[:150],
                guidance='Mixed loop nest parsed with preserved source order',
            )

        return induction_vars, bounds_raw, loop_order, iv_map

    def _extract_affine_apply_map(
        self,
        text: str,
        iv_map: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Parse affine.apply statements and return a map from SSA result name to
        its substituted symbolic expression using short IV names.

        Example:
            %j = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
            with iv_map = {"%i": "i"}  ->  {"%j": "i + 1"}
        """
        apply_map: Dict[str, str] = {}

        for line in self._iter_clean_lines(text):
            if "affine.apply" not in line or "=" not in line:
                continue

            lhs, rhs = line.split("=", 1)
            result_ssa = lhs.strip()

            map_kw = "affine_map<"
            map_kw_idx = rhs.find(map_kw)
            if map_kw_idx < 0:
                continue

            # Parse affine_map<(dims)[syms] -> (result)> component by component
            # to avoid confusion between '->' and the map-closing '>'.
            pos = map_kw_idx + len(map_kw)  # right after 'affine_map<'

            # Dim params (d0, d1, ...)
            if pos >= len(rhs) or rhs[pos] != '(':
                continue
            dim_close = _find_matching(rhs, pos, '(', ')')
            if dim_close < 0:
                continue
            dim_names = [d.strip() for d in rhs[pos + 1:dim_close].split(',') if d.strip()]
            pos = dim_close + 1

            # Optional symbol params [s0, ...]
            while pos < len(rhs) and rhs[pos] == ' ':
                pos += 1
            if pos < len(rhs) and rhs[pos] == '[':
                sym_close = _find_matching(rhs, pos, '[', ']')
                if sym_close < 0:
                    continue
                pos = sym_close + 1

            # Arrow '->'
            arrow_idx = rhs.find('->', pos)
            if arrow_idx < 0:
                continue

            # Result expression (expr)
            res_open = rhs.find('(', arrow_idx + 2)
            if res_open < 0:
                continue
            res_close = _find_matching(rhs, res_open, '(', ')')
            if res_close < 0:
                continue
            expr_str = rhs[res_open + 1:res_close].strip()

            # Closing '>' of affine_map<
            map_close = rhs.find('>', res_close)
            if map_close < 0:
                continue

            # Operands (%i, ...) after the closing '>'
            op_open = rhs.find('(', map_close)
            if op_open < 0:
                continue
            op_close = _find_matching(rhs, op_open, '(', ')')
            if op_close < 0:
                continue
            operands = [o.strip() for o in rhs[op_open + 1:op_close].split(',') if o.strip()]

            if len(operands) != len(dim_names):
                self._add_diagnostic(
                    "debug",
                    "_extract_affine_apply_map",
                    f"Operand/dim count mismatch in affine.apply: {len(operands)} vs {len(dim_names)}",
                    line,
                )
                continue

            substituted = expr_str
            for dim_name, operand in zip(dim_names, operands):
                if operand in iv_map:
                    replacement = iv_map[operand]
                else:
                    replacement = operand.lstrip("%")
                substituted = _replace_token_boundary(substituted, dim_name, replacement)

            apply_map[result_ssa] = _canonicalize_index_expr(substituted)

        return apply_map

    def _parse_load(
        self,
        line: str,
        keyword: str,
        is_affine: bool,
        iv_map: Dict[str, str],
        const_map: Optional[Dict[str, Union[int, float]]] = None,
        apply_map: Optional[Dict[str, str]] = None,
    ) -> Optional[AccessPattern]:
        if "=" not in line or keyword not in line:
            return None

        lhs, rhs = line.split("=", 1)
        ssa_result = lhs.strip()

        kw_idx = rhs.find(keyword)
        if kw_idx < 0:
            return None

        body = rhs[kw_idx + len(keyword):].strip()
        lb = body.find("[")
        if lb < 0:
            self._add_diagnostic(
                'warning',
                '_parse_load',
                f'No opening bracket in load operands',
                line,
            )
            return None
        rb = _find_matching(body, lb, "[", "]")
        if rb < 0:
            self._add_diagnostic(
                'warning',
                '_parse_load',
                f'No closing bracket in load indices',
                line,
            )
            return None

        tensor = body[:lb].strip()
        indices_raw = body[lb + 1:rb]
        after = body[rb + 1:].strip()
        type_str = after.split(":", 1)[1].strip() if ":" in after else ""

        if not type_str:
            self._add_diagnostic(
                'warning',
                '_parse_load',
                f'Missing type annotation for load',
                line,
                guidance='Expected format: %result = affine.load %tensor[...] : memref<...>',
            )

        shape, etype = _parse_memref_type(type_str)
        idx_exprs = [_normalize_index(idx, iv_map, const_map, apply_map) for idx in _split_top_level(indices_raw, ",")]

        return AccessPattern(
            tensor_name=tensor,
            index_exprs=idx_exprs,
            is_affine=is_affine,
            is_read=True,
            shape=shape,
            element_type=etype,
            ssa_name=ssa_result,
        )

    def _parse_store(
        self,
        line: str,
        keyword: str,
        is_affine: bool,
        iv_map: Dict[str, str],
        const_map: Optional[Dict[str, Union[int, float]]] = None,
        apply_map: Optional[Dict[str, str]] = None,
    ) -> Optional[AccessPattern]:
        kw_idx = line.find(keyword)
        if kw_idx < 0:
            return None

        body = line[kw_idx + len(keyword):].strip()
        comma_idx = body.find(",")
        if comma_idx < 0:
            self._add_diagnostic(
                'warning',
                '_parse_store',
                f'No comma separator in store operands',
                line,
                guidance='Expected format: %value, %tensor[...]',
            )
            return None

        val_ssa = body[:comma_idx].strip()
        rest = body[comma_idx + 1:].strip()
        lb = rest.find("[")
        if lb < 0:
            self._add_diagnostic(
                'warning',
                '_parse_store',
                f'No opening bracket in store tensor',
                line,
            )
            return None
        rb = _find_matching(rest, lb, "[", "]")
        if rb < 0:
            self._add_diagnostic(
                'warning',
                '_parse_store',
                f'No closing bracket in store indices',
                line,
            )
            return None

        tensor = rest[:lb].strip()
        indices_raw = rest[lb + 1:rb]
        after = rest[rb + 1:].strip()
        type_str = after.split(":", 1)[1].strip() if ":" in after else ""

        if not type_str:
            self._add_diagnostic(
                'warning',
                '_parse_store',
                f'Missing type annotation for store',
                line,
                guidance='Expected format: affine.store %value, %tensor[...] : memref<...>',
            )

        shape, etype = _parse_memref_type(type_str)
        idx_exprs = [_normalize_index(idx, iv_map, const_map, apply_map) for idx in _split_top_level(indices_raw, ",")]

        return AccessPattern(
            tensor_name=tensor,
            index_exprs=idx_exprs,
            is_affine=is_affine,
            is_read=False,
            shape=shape,
            element_type=etype,
            ssa_name=val_ssa,
        )

    def _collect_accesses(
        self,
        text: str,
        iv_map: Dict[str, str],
        const_map: Optional[Dict[str, Union[int, float]]] = None,
        apply_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[AccessPattern], List[AccessPattern]]:
        reads: List[AccessPattern] = []
        writes: List[AccessPattern] = []

        for line in self._iter_clean_lines(text):
            if "affine.load" in line and "=" in line:
                parsed = self._parse_load(line, "affine.load", True, iv_map, const_map, apply_map)
                if parsed is not None:
                    reads.append(parsed)
                elif "affine.load" in line:
                    self._add_diagnostic(
                        'warning',
                        '_collect_accesses',
                        'Failed to parse affine.load operation',
                        line,
                    )
                continue

            if "memref.load" in line and "=" in line:
                parsed = self._parse_load(line, "memref.load", False, iv_map, const_map, apply_map)
                if parsed is not None:
                    reads.append(parsed)
                elif "memref.load" in line:
                    self._add_diagnostic(
                        'warning',
                        '_collect_accesses',
                        'Failed to parse memref.load operation',
                        line,
                    )
                continue

            if "affine.store" in line:
                parsed = self._parse_store(line, "affine.store", True, iv_map, const_map, apply_map)
                if parsed is not None:
                    writes.append(parsed)
                elif "affine.store" in line:
                    self._add_diagnostic(
                        'warning',
                        '_collect_accesses',
                        'Failed to parse affine.store operation',
                        line,
                    )
                continue

            if "memref.store" in line:
                parsed = self._parse_store(line, "memref.store", False, iv_map, const_map, apply_map)
                if parsed is not None:
                    writes.append(parsed)
                elif "memref.store" in line:
                    self._add_diagnostic(
                        'warning',
                        '_collect_accesses',
                        'Failed to parse memref.store operation',
                        line,
                    )

        return reads, writes

    def _collect_compute_ops(self, text: str) -> List[ComputeOp]:
        ops: List[ComputeOp] = []

        unary_ops = {
            "negf",
            "extf",
            "truncf",
        }

        for line in self._iter_clean_lines(text):
            if "= arith." not in line:
                continue

            lhs, rhs = line.split("=", 1)
            result_ssa = lhs.strip()
            rhs = rhs.strip()
            marker = "arith."
            marker_idx = rhs.find(marker)
            if marker_idx < 0:
                continue

            op_payload = rhs[marker_idx + len(marker):].strip()
            if not op_payload:
                continue

            parts = op_payload.split(None, 1)
            op_type = parts[0].strip()
            remainder = parts[1].strip() if len(parts) > 1 else ""
            operands_part = remainder.split(":", 1)[0].strip() if remainder else ""

            if op_type == "constant":
                value: Optional[float]
                literal = operands_part
                try:
                    value = float(literal)
                except ValueError:
                    self._add_diagnostic(
                        'warning',
                        '_collect_compute_ops',
                        f'Unparseable constant literal, defaulting to 0.0',
                        line,
                        guidance=f'Expected float literal, got: {literal}',
                    )
                    value = 0.0
                ops.append(
                    ComputeOp(
                        op_type="constant",
                        operands=[],
                        result=result_ssa,
                        value=value,
                    )
                )
                continue

            if not operands_part:
                ops.append(ComputeOp(op_type=op_type, operands=[], result=result_ssa))
                continue

            if op_type in unary_ops:
                operand = operands_part.split(",", 1)[0].strip()
                ops.append(ComputeOp(op_type=op_type, operands=[operand], result=result_ssa))
                continue

            operands = [o.strip() for o in _split_top_level(operands_part, ",") if o.strip()]
            if len(operands) > 2:
                self._add_diagnostic(
                    'warning',
                    '_collect_compute_ops',
                    f'N-ary operation truncated to first 2 operands',
                    line,
                    guidance=f'Found {len(operands)} operands, using first 2 for binary {op_type}',
                )
            ops.append(ComputeOp(op_type=op_type, operands=operands[:2], result=result_ssa))

        return ops

    def _resolve_tensor_metadata(
        self,
        func_args: Dict[str, str],
        reads: List[AccessPattern],
        writes: List[AccessPattern],
    ) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
        observed_shapes: Dict[str, List[List[int]]] = {}
        observed_types: Dict[str, List[str]] = {}
        observed_ranks: Dict[str, List[int]] = {}

        for arg_name, arg_type in func_args.items():
            if "memref<" not in arg_type:
                continue
            shape, elem = _parse_memref_type(arg_type)
            observed_shapes.setdefault(arg_name, []).append(shape)
            observed_types.setdefault(arg_name, []).append(elem)

        for acc in reads + writes:
            observed_shapes.setdefault(acc.tensor_name, []).append(acc.shape)
            observed_types.setdefault(acc.tensor_name, []).append(acc.element_type)
            if acc.index_exprs:
                observed_ranks.setdefault(acc.tensor_name, []).append(len(acc.index_exprs))

        all_tensors = set(observed_shapes.keys()) | set(observed_types.keys()) | set(observed_ranks.keys())
        shapes: Dict[str, List[int]] = {}
        types: Dict[str, str] = {}

        for tensor_name in sorted(all_tensors):
            shapes[tensor_name] = self._normalize_tensor_shape_metadata(
                tensor_name=tensor_name,
                observed_shapes=observed_shapes.get(tensor_name, []),
                observed_ranks=observed_ranks.get(tensor_name, []),
            )
            elem = self._normalize_tensor_element_type(
                tensor_name=tensor_name,
                observed_types=observed_types.get(tensor_name, []),
            )
            if elem:
                types[tensor_name] = elem

        return shapes, types

    def _normalize_tensor_shape_metadata(
        self,
        tensor_name: str,
        observed_shapes: List[List[int]],
        observed_ranks: List[int],
    ) -> List[int]:
        rank_sources = [len(shape) for shape in observed_shapes if shape] + [r for r in observed_ranks if r > 0]
        if not rank_sources:
            return []

        rank_frequency = Counter(rank_sources)
        rank = sorted(rank_frequency.items(), key=lambda item: (-item[1], -item[0]))[0][0]
        if len(rank_frequency) > 1:
            self._add_diagnostic(
                "warning",
                "_resolve_tensor_metadata",
                f"Ambiguous tensor rank observations for '{tensor_name}'",
                str(rank_sources),
                guidance=(
                    f"Using rank {rank} based on majority observation; conflicting ranks were "
                    f"{sorted(rank_frequency.keys())}."
                ),
            )

        normalized = [-1] * rank
        for dim_idx in range(rank):
            static_values = sorted(
                {
                    shape[dim_idx]
                    for shape in observed_shapes
                    if len(shape) > dim_idx and shape[dim_idx] >= 0
                }
            )

            if len(static_values) == 1:
                normalized[dim_idx] = static_values[0]
            elif len(static_values) > 1:
                self._add_diagnostic(
                    "warning",
                    "_resolve_tensor_metadata",
                    f"Conflicting static dimension annotations for '{tensor_name}'",
                    f"dim={dim_idx}, candidates={static_values}",
                    guidance="Keeping this dimension dynamic to avoid unstable matcher/emitter behavior.",
                )

        return normalized

    def _normalize_tensor_element_type(
        self,
        tensor_name: str,
        observed_types: List[str],
    ) -> str:
        cleaned = [t.strip() for t in observed_types if t and t.strip()]
        if not cleaned:
            return ""

        counts = Counter(cleaned)
        winner = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        if len(counts) > 1:
            self._add_diagnostic(
                "warning",
                "_resolve_tensor_metadata",
                f"Conflicting tensor element types for '{tensor_name}'",
                str(sorted(counts.keys())),
                guidance=f"Using dominant observed type '{winner}'.",
            )
        return winner

    def _dominant_element_type(self, tensor_types: Dict[str, str]) -> str:
        if not tensor_types:
            self._add_diagnostic(
                'warning',
                '_dominant_element_type',
                'No tensor element types discovered, defaulting to f32',
                '',
                guidance='Check that tensor arguments have proper type annotations',
            )
            return "f32"
        return Counter(tensor_types.values()).most_common(1)[0][0]

    def _classify_iv_roles(
        self,
        induction_vars: List[str],
        writes: List[AccessPattern],
    ) -> Tuple[List[str], List[str]]:
        if not writes:
            self._add_diagnostic(
                'warning',
                '_classify_iv_roles',
                'No writes found; classifying all induction variables as parallel',
                '',
                guidance='This is a conservative fallback for kernels with no explicit output writes',
            )
            return [], list(induction_vars)

        write_idx_pool = set()
        for write in writes:
            for expr in write.index_exprs:
                for iv in induction_vars:
                    if iv in expr:
                        write_idx_pool.add(iv)

        parallel_vars = [iv for iv in induction_vars if iv in write_idx_pool]
        reduction_vars = [iv for iv in induction_vars if iv not in write_idx_pool]
        return reduction_vars, parallel_vars

    def _detect_accumulation(
        self,
        compute_ops: List[ComputeOp],
        reads: List[AccessPattern],
        writes: List[AccessPattern],
    ) -> Tuple[bool, Optional[str]]:
        if not writes:
            return False, None

        write_tensor = writes[0].tensor_name
        write_ssa = writes[0].ssa_name

        output_read_exists = any(read.tensor_name == write_tensor for read in reads)
        if not output_read_exists:
            return False, None

        for op in compute_ops:
            if op.result == write_ssa and op.op_type in ("addf", "addi"):
                return True, op.op_type

        # Fallback: any add op is treated as accumulation
        for op in compute_ops:
            if op.op_type in ("addf", "addi"):
                self._add_diagnostic(
                    'debug',
                    '_detect_accumulation',
                    'Accumulation detected from any add op (not directly writing to output)',
                    '',
                    guidance='Aggressive heuristic: any addf/addi treated as reduction',
                )
                return True, op.op_type

        return False, None

    def _resolve_bound(
        self,
        bound: str,
        const_map: Dict[str, Union[int, float]],
    ) -> Union[int, str]:
        value = bound.strip()

        try:
            return int(value, 10)
        except ValueError:
            pass

        if value in const_map:
            return int(const_map[value])

        no_percent = value.lstrip("%")
        if no_percent in const_map:
            return int(const_map[no_percent])

        try:
            return int(no_percent, 10)
        except ValueError:
            return no_percent
