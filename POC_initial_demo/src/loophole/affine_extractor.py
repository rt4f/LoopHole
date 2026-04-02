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

try:
    from mlir import ir as _mlir_ir
except Exception:
    _mlir_ir = None


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
    if "x" not in main:
        return [], main or "f32"

    parts = [p.strip() for p in main.split("x") if p.strip()]
    if not parts:
        return [], "f32"

    elem_type = parts[-1]
    shape_tokens = parts[:-1]
    shape: List[int] = []
    for token in shape_tokens:
        if token == "?":
            shape.append(-1)
            continue
        try:
            shape.append(int(token, 10))
        except ValueError:
            shape.append(-1)
    return shape, elem_type


def _normalize_index(idx: str, iv_map: Dict[str, str]) -> str:
    expr = idx.strip()
    for ssa_name, short_name in sorted(iv_map.items(), key=lambda item: len(item[0]), reverse=True):
        expr = _replace_token_boundary(expr, ssa_name, short_name)
    expr = _strip_ssa_percent(expr)
    return " ".join(expr.split())


class AffineExtractor:
    """
    Parse MLIR Affine/SCF IR text and extract LoopNestInfo.

    The extractor avoids regex parsing. If MLIR Python bindings are available,
    syntax is validated through the MLIR parser before extraction.
    """

    def __init__(self, validate_with_mlir: bool = True):
        self.validate_with_mlir = validate_with_mlir

    def extract(self, mlir_text: str) -> LoopNestInfo:
        text = mlir_text.replace("\r\n", "\n").replace("\r", "\n")

        self._validate_mlir_syntax(text)

        func_name, func_args = self._parse_func_declaration(text)
        const_map = self._extract_constants(text)
        induction_vars, bounds_raw, loop_order, iv_map = self._parse_loop_structure(text)
        reads, writes = self._collect_accesses(text, iv_map)
        compute_ops = self._collect_compute_ops(text)
        tensor_shapes, tensor_types = self._resolve_tensor_metadata(func_args, reads, writes)
        element_type = self._dominant_element_type(tensor_types)
        reduction_vars, parallel_vars = self._classify_iv_roles(induction_vars, writes)
        has_accum, accum_op = self._detect_accumulation(compute_ops, reads, writes)

        bounds: Dict[str, Tuple[Union[int, str], Union[int, str]]] = {}
        for iv, (lo, hi) in bounds_raw.items():
            lo_r = self._resolve_bound(lo, const_map)
            hi_r = self._resolve_bound(hi, const_map)
            bounds[iv] = (lo_r, hi_r)

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
            return "unknown", {}

        name_start = idx + len(marker)
        name_end = text.find("(", name_start)
        if name_end < 0:
            return "unknown", {}

        func_name = text[name_start:name_end].strip().split()[0]
        args_end = _find_matching(text, name_end, "(", ")")
        if args_end < 0:
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
            if not literal or literal.startswith("dense<"):
                continue

            try:
                if any(ch in literal for ch in (".", "e", "E")):
                    const_map[ssa_name] = float(literal)
                else:
                    const_map[ssa_name] = int(literal, 10)
            except ValueError:
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
        if not line.startswith("affine.for "):
            return None
        body = line[len("affine.for "):].strip()
        if "=" not in body or " to " not in body:
            return None

        iv_ssa, rhs = body.split("=", 1)
        iv_ssa = iv_ssa.strip()

        lb_part, ub_part = rhs.split(" to ", 1)
        lb = lb_part.strip()
        ub = ub_part.strip()

        if " step " in ub:
            ub = ub.split(" step ", 1)[0].strip()
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

        for line in self._iter_clean_lines(text):
            parsed = self._parse_affine_for(line)
            if parsed is None and not induction_vars:
                parsed = self._parse_scf_for(line)
            if parsed is None:
                continue

            iv_ssa, lb, ub = parsed
            short = self._allocate_iv_name(iv_ssa, induction_vars)
            induction_vars.append(short)
            loop_order.append(short)
            bounds_raw[short] = (lb, ub)
            iv_map[iv_ssa] = short

        return induction_vars, bounds_raw, loop_order, iv_map

    def _parse_load(
        self,
        line: str,
        keyword: str,
        is_affine: bool,
        iv_map: Dict[str, str],
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
            return None
        rb = _find_matching(body, lb, "[", "]")
        if rb < 0:
            return None

        tensor = body[:lb].strip()
        indices_raw = body[lb + 1:rb]
        after = body[rb + 1:].strip()
        type_str = after.split(":", 1)[1].strip() if ":" in after else ""

        shape, etype = _parse_memref_type(type_str)
        idx_exprs = [_normalize_index(idx, iv_map) for idx in _split_top_level(indices_raw, ",")]

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
    ) -> Optional[AccessPattern]:
        kw_idx = line.find(keyword)
        if kw_idx < 0:
            return None

        body = line[kw_idx + len(keyword):].strip()
        comma_idx = body.find(",")
        if comma_idx < 0:
            return None

        val_ssa = body[:comma_idx].strip()
        rest = body[comma_idx + 1:].strip()
        lb = rest.find("[")
        if lb < 0:
            return None
        rb = _find_matching(rest, lb, "[", "]")
        if rb < 0:
            return None

        tensor = rest[:lb].strip()
        indices_raw = rest[lb + 1:rb]
        after = rest[rb + 1:].strip()
        type_str = after.split(":", 1)[1].strip() if ":" in after else ""

        shape, etype = _parse_memref_type(type_str)
        idx_exprs = [_normalize_index(idx, iv_map) for idx in _split_top_level(indices_raw, ",")]

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
    ) -> Tuple[List[AccessPattern], List[AccessPattern]]:
        reads: List[AccessPattern] = []
        writes: List[AccessPattern] = []

        for line in self._iter_clean_lines(text):
            if "affine.load" in line and "=" in line:
                parsed = self._parse_load(line, "affine.load", True, iv_map)
                if parsed is not None:
                    reads.append(parsed)
                continue

            if "memref.load" in line and "=" in line:
                parsed = self._parse_load(line, "memref.load", False, iv_map)
                if parsed is not None:
                    reads.append(parsed)
                continue

            if "affine.store" in line:
                parsed = self._parse_store(line, "affine.store", True, iv_map)
                if parsed is not None:
                    writes.append(parsed)
                continue

            if "memref.store" in line:
                parsed = self._parse_store(line, "memref.store", False, iv_map)
                if parsed is not None:
                    writes.append(parsed)

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
            ops.append(ComputeOp(op_type=op_type, operands=operands[:2], result=result_ssa))

        return ops

    def _resolve_tensor_metadata(
        self,
        func_args: Dict[str, str],
        reads: List[AccessPattern],
        writes: List[AccessPattern],
    ) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
        shapes: Dict[str, List[int]] = {}
        types: Dict[str, str] = {}

        for arg_name, arg_type in func_args.items():
            if "memref<" in arg_type:
                shape, elem = _parse_memref_type(arg_type)
                shapes[arg_name] = shape
                types[arg_name] = elem

        for acc in reads + writes:
            if acc.tensor_name not in shapes:
                shapes[acc.tensor_name] = acc.shape
            if acc.tensor_name not in types:
                types[acc.tensor_name] = acc.element_type

        return shapes, types

    def _dominant_element_type(self, tensor_types: Dict[str, str]) -> str:
        if not tensor_types:
            return "f32"
        return Counter(tensor_types.values()).most_common(1)[0][0]

    def _classify_iv_roles(
        self,
        induction_vars: List[str],
        writes: List[AccessPattern],
    ) -> Tuple[List[str], List[str]]:
        if not writes:
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

        for op in compute_ops:
            if op.op_type in ("addf", "addi"):
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
