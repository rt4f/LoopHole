"""
affine_extractor.py — Parse MLIR Affine IR text into LoopNestInfo data structures.

This module implements a hand-rolled recursive descent parser for a substantial
subset of the MLIR textual format, focused on the Affine, SCF, Arith, and MemRef
dialects that Polygeist produces as output.

Supported constructs:
  - func.func declarations with memref arguments
  - affine.for loops with integer-constant or symbolic bounds
  - affine.load / affine.store with affine map subscripts
  - arith.mulf, arith.addf, arith.subf, arith.divf, arith.maxf, arith.minf,
    arith.constant, arith.negf, arith.cmpf, arith.maxnumf, arith.minnumf
  - arith.extf, arith.truncf (float conversions, ignored for semantics)
  - scf.for as fallback loop representation
  - memref.load / memref.store (non-affine fallback)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AccessPattern:
    """Describes a single memref read or write with its subscript expression."""
    tensor_name: str                 # e.g., '%A'
    index_exprs: List[str]           # e.g., ['%i', '%k'] or ['%i + %p', '%j + %q']
    is_affine: bool                  # True if all indices are affine in loop IVs
    is_read: bool                    # False → write
    shape: List[int]                 # concrete dims from memref<MxNxf32>
    element_type: str                # 'f32' | 'f64' | 'i32' | 'i64'
    ssa_name: str                    # SSA result name for reads (e.g. '%val0')


@dataclass
class ComputeOp:
    """A single arithmetic operation in the loop body."""
    op_type: str                     # e.g., 'mulf', 'addf', 'constant'
    operands: List[str]              # SSA names of operand values
    result: str                      # SSA name of the result
    value: Optional[float] = None    # For arith.constant


@dataclass
class LoopNestInfo:
    """
    Full semantic description of an extracted affine loop nest.

    This is the central data structure passed between all pipeline stages.
    """
    func_name: str                             # e.g., 'matmul'
    induction_vars: List[str]                  # ordered outermost→innermost: ['i','j','k']
    bounds: Dict[str, Tuple[Union[int,str], Union[int,str]]]  # {'i': (0, 128), ...}
    loop_order: List[str]                      # same as induction_vars, for clarity
    reads: List[AccessPattern]
    writes: List[AccessPattern]
    compute_ops: List[ComputeOp]
    tensor_shapes: Dict[str, List[int]]        # {'%A': [128, 128], ...}
    tensor_types: Dict[str, str]               # {'%A': 'f32', ...}
    element_type: str                          # dominant element type
    reduction_vars: List[str]                  # IVs not appearing in output subscript
    parallel_vars: List[str]                   # IVs that DO appear in output subscript
    func_args: Dict[str, str]                  # {arg_name: mlir_type}
    has_accumulation: bool                     # whether body includes += style update
    accumulation_op: Optional[str]             # 'addf' | 'addi' for the += op

    # convenience: quick access to the single write tensor name
    @property
    def output_tensor(self) -> Optional[str]:
        if self.writes:
            return self.writes[0].tensor_name
        return None

    @property
    def input_tensors(self) -> List[str]:
        out = self.output_tensor
        return [r.tensor_name for r in self.reads if r.tensor_name != out]


# ---------------------------------------------------------------------------
# Regex patterns (pre-compiled for performance)
# ---------------------------------------------------------------------------

_RE_FUNC = re.compile(
    r'func\.func\s+@(\w+)\s*\(([^)]*)\)',
    re.MULTILINE
)
_RE_AFFINE_FOR = re.compile(
    r'affine\.for\s+(%\w+)\s*=\s*([\w%]+)\s+to\s+([\w%]+)(?:\s+step\s+(\d+))?',
)
_RE_SCF_FOR = re.compile(
    r'scf\.for\s+(%\w+)\s*=\s*(%\w+|[-\d]+)\s+to\s+(%\w+|[-\d]+)\s+step\s+(%\w+|[-\d]+)',
)
_RE_AFFINE_LOAD = re.compile(
    r'(%\w+)\s*=\s*affine\.load\s+(%\w+)\[([^\]]*)\]\s*:\s*memref<([^>]+)>',
)
_RE_AFFINE_STORE = re.compile(
    r'affine\.store\s+(%\w+),\s*(%\w+)\[([^\]]*)\]\s*:\s*memref<([^>]+)>',
)
_RE_MEMREF_LOAD = re.compile(
    r'(%\w+)\s*=\s*memref\.load\s+(%\w+)\[([^\]]*)\]\s*:\s*memref<([^>]+)>',
)
_RE_MEMREF_STORE = re.compile(
    r'memref\.store\s+(%\w+),\s*(%\w+)\[([^\]]*)\]\s*:\s*memref<([^>]+)>',
)
_RE_ARITH_BINOP = re.compile(
    r'(%\w+)\s*=\s*arith\.(mulf|addf|subf|divf|maxf|minf|maxnumf|minnumf|'
    r'muli|addi|subi|divi_signed|mulf|mulsi)\s+(%\w+),\s*(%\w+)\s*:\s*\S+',
)
_RE_ARITH_CONST = re.compile(
    r'(%\w+)\s*=\s*arith\.constant\s+([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*:\s*\S+',
)
_RE_ARITH_NEGF = re.compile(
    r'(%\w+)\s*=\s*arith\.negf\s+(%\w+)\s*:\s*\S+',
)
_RE_MEMREF_TYPE = re.compile(
    r'memref<([\dx]+x(?:f32|f64|i32|i64|bf16|f16))>',
)
_RE_CONST_INT = re.compile(
    r'(%\w+)\s*=\s*arith\.constant\s+(\d+)\s*:\s*index',
)


def _parse_memref_type(type_str: str) -> Tuple[List[int], str]:
    """
    Parse 'MxNxf32' from a memref<MxNxf32> type string.
    Returns (shape, element_type).  '?' becomes -1.
    """
    # Strip 'memref<' and '>' if present
    type_str = type_str.strip()
    if type_str.startswith('memref<'):
        type_str = type_str[7:]
    if type_str.endswith('>'):
        type_str = type_str[:-1]
    parts = type_str.split('x')
    elem_type = parts[-1]
    shape = []
    for p in parts[:-1]:
        if p == '?':
            shape.append(-1)
        else:
            try:
                shape.append(int(p))
            except ValueError:
                shape.append(-1)
    return shape, elem_type


def _normalize_index(idx: str, iv_map: Dict[str, str]) -> str:
    """
    Normalize an affine subscript expression: replace full SSA names
    with short induction variable names, collapse whitespace.
    """
    result = idx.strip()
    # Replace SSA iv names with short names
    for ssa_name, short_name in iv_map.items():
        result = result.replace(ssa_name, short_name)
    return result


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class AffineExtractor:
    """
    Parses MLIR Affine/SCF IR in textual format and extracts LoopNestInfo.

    Usage:
        extractor = AffineExtractor()
        info = extractor.extract(mlir_text)
    """

    def extract(self, mlir_text: str) -> LoopNestInfo:
        """
        Main entry point.  Takes an MLIR module or function as a string,
        returns a LoopNestInfo describing the first (or only) func.func.

        Raises ValueError if the text does not contain a parseable loop nest.
        """
        # Normalize line endings
        text = mlir_text.replace('\r\n', '\n').replace('\r', '\n')

        # 1. Find the function declaration
        func_name, func_args = self._parse_func_declaration(text)

        # 2. Extract constant definitions (for bounds that use %c0, %cN etc.)
        const_map = self._extract_constants(text)

        # 3. Parse all loop levels
        induction_vars, bounds_raw, loop_order = self._parse_loop_structure(text, const_map)

        # 4. Collect all memory accesses
        reads, writes = self._collect_accesses(text, induction_vars)

        # 5. Collect arithmetic ops
        compute_ops = self._collect_compute_ops(text)

        # 6. Resolve tensor shapes and types from func args and access sites
        tensor_shapes, tensor_types = self._resolve_tensor_metadata(func_args, reads, writes)

        # 7. Determine dominant element type
        element_type = self._dominant_element_type(tensor_types)

        # 8. Classify induction variables as parallel or reduction
        reduction_vars, parallel_vars = self._classify_iv_roles(
            induction_vars, writes
        )

        # 9. Detect accumulation pattern
        has_accum, accum_op = self._detect_accumulation(compute_ops, reads, writes)

        # Resolve symbolic bounds
        bounds = {}
        for iv, (lo, hi) in bounds_raw.items():
            lo_r = self._resolve_bound(lo, tensor_shapes, func_args, const_map)
            hi_r = self._resolve_bound(hi, tensor_shapes, func_args, const_map)
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_func_declaration(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Return (func_name, {arg_name: mlir_type})."""
        m = _RE_FUNC.search(text)
        if not m:
            # Bare function body without func.func declaration — use placeholder
            return "unknown", {}
        func_name = m.group(1)
        args_str = m.group(2)
        func_args: Dict[str, str] = {}
        if args_str.strip():
            for arg_segment in args_str.split(','):
                arg_segment = arg_segment.strip()
                colon_pos = arg_segment.find(':')
                if colon_pos < 0:
                    continue
                arg_name = arg_segment[:colon_pos].strip()
                arg_type = arg_segment[colon_pos + 1:].strip()
                func_args[arg_name] = arg_type
        return func_name, func_args

    def _extract_constants(self, text: str) -> Dict[str, Union[int, float]]:
        """Build a map from SSA constant name to its integer/float value."""
        const_map: Dict[str, Union[int, float]] = {}
        for m in _RE_CONST_INT.finditer(text):
            const_map[m.group(1)] = int(m.group(2))
        for m in _RE_ARITH_CONST.finditer(text):
            try:
                const_map[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
        # Common constant patterns like %c0 = arith.constant 0 : index
        pat2 = re.compile(r'(%\w+)\s*=\s*arith\.constant\s+(\d+)\s*:\s*(?:i\d+|index)')
        for m in pat2.finditer(text):
            const_map[m.group(1)] = int(m.group(2))
        return const_map

    def _parse_loop_structure(
        self,
        text: str,
        const_map: Dict[str, Union[int, float]],
    ) -> Tuple[List[str], Dict[str, Tuple], List[str]]:
        """
        Extract all induction variables, their bounds, and loop order.
        Returns (induction_vars_ordered, raw_bounds_dict, loop_order).
        """
        induction_vars: List[str] = []
        bounds_raw: Dict[str, Tuple] = {}
        loop_order: List[str] = []

        # Try affine.for first
        for m in _RE_AFFINE_FOR.finditer(text):
            iv = m.group(1)          # e.g. '%i'
            lb = m.group(2)          # lower bound literal or SSA
            ub = m.group(3)          # upper bound literal or SSA
            short = iv.lstrip('%')
            induction_vars.append(short)
            loop_order.append(short)
            bounds_raw[short] = (lb, ub)

        # Fallback to scf.for
        if not induction_vars:
            for m in _RE_SCF_FOR.finditer(text):
                iv = m.group(1)
                lb = m.group(2)
                ub = m.group(3)
                short = iv.lstrip('%')
                induction_vars.append(short)
                loop_order.append(short)
                bounds_raw[short] = (lb, ub)

        return induction_vars, bounds_raw, loop_order

    def _collect_accesses(
        self,
        text: str,
        induction_vars: List[str],
    ) -> Tuple[List[AccessPattern], List[AccessPattern]]:
        """Parse all affine.load / affine.store / memref.load / memref.store."""
        # Build IV SSA → short name map for index normalization
        iv_map: Dict[str, str] = {}
        for short in induction_vars:
            iv_map[f'%{short}'] = short

        reads: List[AccessPattern] = []
        writes: List[AccessPattern] = []

        # affine.load
        for m in _RE_AFFINE_LOAD.finditer(text):
            ssa_result = m.group(1)
            tensor = m.group(2)
            indices_raw = m.group(3)
            type_str = m.group(4)
            shape, etype = _parse_memref_type(type_str)
            idx_exprs = [_normalize_index(i, iv_map) for i in indices_raw.split(',')]
            reads.append(AccessPattern(
                tensor_name=tensor,
                index_exprs=idx_exprs,
                is_affine=True,
                is_read=True,
                shape=shape,
                element_type=etype,
                ssa_name=ssa_result,
            ))

        # affine.store
        for m in _RE_AFFINE_STORE.finditer(text):
            val_ssa = m.group(1)
            tensor = m.group(2)
            indices_raw = m.group(3)
            type_str = m.group(4)
            shape, etype = _parse_memref_type(type_str)
            idx_exprs = [_normalize_index(i, iv_map) for i in indices_raw.split(',')]
            writes.append(AccessPattern(
                tensor_name=tensor,
                index_exprs=idx_exprs,
                is_affine=True,
                is_read=False,
                shape=shape,
                element_type=etype,
                ssa_name=val_ssa,
            ))

        # memref.load (non-affine fallback)
        for m in _RE_MEMREF_LOAD.finditer(text):
            ssa_result = m.group(1)
            tensor = m.group(2)
            indices_raw = m.group(3)
            type_str = m.group(4)
            shape, etype = _parse_memref_type(type_str)
            idx_exprs = [_normalize_index(i, iv_map) for i in indices_raw.split(',')]
            reads.append(AccessPattern(
                tensor_name=tensor,
                index_exprs=idx_exprs,
                is_affine=False,
                is_read=True,
                shape=shape,
                element_type=etype,
                ssa_name=ssa_result,
            ))

        # memref.store (non-affine fallback)
        for m in _RE_MEMREF_STORE.finditer(text):
            val_ssa = m.group(1)
            tensor = m.group(2)
            indices_raw = m.group(3)
            type_str = m.group(4)
            shape, etype = _parse_memref_type(type_str)
            idx_exprs = [_normalize_index(i, iv_map) for i in indices_raw.split(',')]
            writes.append(AccessPattern(
                tensor_name=tensor,
                index_exprs=idx_exprs,
                is_affine=False,
                is_read=False,
                shape=shape,
                element_type=etype,
                ssa_name=val_ssa,
            ))

        return reads, writes

    def _collect_compute_ops(self, text: str) -> List[ComputeOp]:
        """Extract all arithmetic operations inside the loop body."""
        ops: List[ComputeOp] = []

        for m in _RE_ARITH_BINOP.finditer(text):
            ops.append(ComputeOp(
                op_type=m.group(2),
                operands=[m.group(3), m.group(4)],
                result=m.group(1),
            ))

        for m in _RE_ARITH_CONST.finditer(text):
            try:
                val = float(m.group(2))
            except ValueError:
                val = 0.0
            ops.append(ComputeOp(
                op_type='constant',
                operands=[],
                result=m.group(1),
                value=val,
            ))

        for m in _RE_ARITH_NEGF.finditer(text):
            ops.append(ComputeOp(
                op_type='negf',
                operands=[m.group(2)],
                result=m.group(1),
            ))

        return ops

    def _resolve_tensor_metadata(
        self,
        func_args: Dict[str, str],
        reads: List[AccessPattern],
        writes: List[AccessPattern],
    ) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
        """Build shape and type dicts for all tensors referenced."""
        shapes: Dict[str, List[int]] = {}
        types: Dict[str, str] = {}

        # From func args
        for arg_name, arg_type in func_args.items():
            if 'memref<' in arg_type:
                s, t = _parse_memref_type(arg_type)
                shapes[arg_name] = s
                types[arg_name] = t

        # From access sites (fills in any that were not in func args)
        for acc in reads + writes:
            if acc.tensor_name not in shapes:
                shapes[acc.tensor_name] = acc.shape
            if acc.tensor_name not in types:
                types[acc.tensor_name] = acc.element_type

        return shapes, types

    def _dominant_element_type(self, types: Dict[str, str]) -> str:
        """Return the most common element type, defaulting to 'f32'."""
        if not types:
            return 'f32'
        from collections import Counter
        cnt = Counter(types.values())
        return cnt.most_common(1)[0][0]

    def _classify_iv_roles(
        self,
        induction_vars: List[str],
        writes: List[AccessPattern],
    ) -> Tuple[List[str], List[str]]:
        """
        An IV is 'parallel' if it appears in at least one write subscript.
        An IV is a 'reduction' variable if it does NOT appear in any write.
        """
        if not writes:
            return [], list(induction_vars)

        # Collect all indices that appear in the write subscripts
        write_idx_pool: set = set()
        for w in writes:
            for expr in w.index_exprs:
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
        """
        Detect whether the loop body performs an accumulation into the output.
        An accumulation exists when the output tensor is read AND written,
        and the written value depends on the read value via an addf/addi.
        """
        if not writes:
            return False, None

        write_tensor = writes[0].tensor_name
        write_ssa = writes[0].ssa_name

        # Check whether any read accesses the output tensor
        out_read = any(r.tensor_name == write_tensor for r in reads)
        if not out_read:
            return False, None

        # Find the op that produces the written value
        for op in compute_ops:
            if op.result == write_ssa and op.op_type in ('addf', 'addi'):
                return True, op.op_type

        # Sometimes there are two ops: mulf then addf; check if write is any add chain
        for op in compute_ops:
            if op.op_type in ('addf', 'addi'):
                return True, op.op_type

        return False, None

    def _resolve_bound(
        self,
        bound: str,
        tensor_shapes: Dict[str, List[int]],
        func_args: Dict[str, str],
        const_map: Dict[str, Union[int, float]],
    ) -> Union[int, str]:
        """
        Try to resolve a loop bound to a concrete integer.
        - If already an integer literal, return int.
        - If an SSA constant, look up in const_map.
        - Otherwise return as string (symbolic).
        """
        bound = bound.strip()
        try:
            return int(bound)
        except ValueError:
            pass
        if bound in const_map:
            return int(const_map[bound])
        # Strip '%' and try
        bare = bound.lstrip('%')
        try:
            return int(bare)
        except ValueError:
            pass
        return bound  # symbolic, keep as str
