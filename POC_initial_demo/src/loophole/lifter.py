"""
lifter.py — Main synthesis pipeline orchestrator for LoopHole.

Coordinates the four-stage lifting pipeline:
  Stage 1: Parse — affine IR text → LoopNestInfo
  Stage 2: Match — LoopNestInfo → candidate OperationSketch(es)
  Stage 3: Verify — Z3 formal equivalence check + SymPy algebraic check
  Stage 4: Emit  — matched sketch + shapes → MLIR dialect text

Matching strategy:
  a. Structural pre-filter (fast, O(1) per sketch)
  b. SymPy algebraic confidence scoring (symbolic, O(N_sketches))
  c. Convolution-specific pattern recognizer
  d. Z3 formal verification for the top-k candidates (expensive)
  e. Best match selection by confidence + Z3 result

The lifter returns a LiftResult which contains:
  - The matched sketch name (or None if no match found)
  - The emitted MLIR text for the target dialect
  - The verification result (EQUIVALENT / TIMEOUT / etc.)
  - Timing information
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field, replace as _dc_replace
from enum import Enum
from typing import Dict, List, Mapping, Optional, Set, Tuple

from loophole.affine_extractor import AffineExtractor, LoopNestInfo
from loophole.emitter import LinalgEmitter, StableHLOEmitter, _infer_linear_coeff
from loophole.sketch_library import (
    SKETCH_LIBRARY, LINALG_SKETCHES, STABLEHLO_SKETCHES, OperationSketch,
)
from loophole.sympy_tracer import SympyTracer, TraceResult
from loophole.z3_checker import (
    CheckResult, VerificationReport, Z3EquivalenceChecker, structural_match,
)


# ---------------------------------------------------------------------------
# Strided convolution sketch builder
# ---------------------------------------------------------------------------


def _build_strided_conv_sketch(
    loop: LoopNestInfo,
    sketch: OperationSketch,
) -> Optional[OperationSketch]:
    """
    For a conv sketch that failed Z3 due to non-unit strides/dilations, build a
    modified sketch whose input indexing map uses the actual stride/dilation
    coefficients extracted from the loop access pattern.

    Returns None if the sketch is not a conv sketch, strides/dilations are already
    unit, or extraction fails.  On success returns a shallow copy of the sketch
    with updated indexing_maps (only the input map is changed).
    """
    is_conv = sketch.name.startswith("linalg.conv") or sketch.name.startswith("stablehlo.convolution")
    if not is_conv:
        return None

    src_parallel = loop.parallel_vars
    src_reduction = loop.reduction_vars
    sk_parallel = sketch.parallel_dims()
    sk_reduction = sketch.reduction_dims()
    if len(src_parallel) != len(sk_parallel) or len(src_reduction) != len(sk_reduction):
        return None

    dim_to_iv: Dict[str, str] = {}
    for sk_dim, src_iv in zip(sk_parallel, src_parallel):
        dim_to_iv[sk_dim] = src_iv
    for sk_dim, src_iv in zip(sk_reduction, src_reduction):
        dim_to_iv[sk_dim] = src_iv

    output_name = loop.output_tensor
    input_reads = [r for r in loop.reads if r.tensor_name != output_name]
    if not input_reads:
        return None
    input_exprs = input_reads[0].index_exprs

    # Per input dimension: source_iv → coefficient
    iv_coeffs_per_dim: List[Dict[str, int]] = []
    for expr in input_exprs:
        coeffs: Dict[str, int] = {}
        for iv in src_parallel + src_reduction:
            c = _infer_linear_coeff(expr, iv)
            if c is not None:
                coeffs[iv] = c
        iv_coeffs_per_dim.append(coeffs)

    # Parse and rewrite the input indexing map (first map in sketch.indexing_maps)
    input_map_str = sketch.indexing_maps[0]
    arrow_pos = input_map_str.find('->')
    if arrow_pos < 0:
        return None
    map_header = input_map_str[:arrow_pos].strip()
    result_part = input_map_str[arrow_pos + 2:].strip()
    inner = result_part.strip('(').rstrip(')')
    result_exprs = [p.strip() for p in inner.split(',')]

    modified_exprs: List[str] = []
    was_modified = False

    for i, sk_expr in enumerate(result_exprs):
        coeffs_this_dim = iv_coeffs_per_dim[i] if i < len(iv_coeffs_per_dim) else {}
        terms: List[str] = []
        for sk_dim in sketch.dim_names:
            if not re.search(rf'\b{re.escape(sk_dim)}\b', sk_expr):
                continue
            src_iv = dim_to_iv.get(sk_dim)
            if src_iv is None:
                terms.append(sk_dim)
                continue
            actual_coeff = coeffs_this_dim.get(src_iv)
            if actual_coeff is None or actual_coeff == 1:
                terms.append(sk_dim)
            elif actual_coeff > 1:
                terms.append(f"{actual_coeff}*{sk_dim}")
                was_modified = True
        modified_exprs.append(' + '.join(terms) if terms else sk_expr)

    if not was_modified:
        return None

    new_input_map = f"{map_header} -> ({', '.join(modified_exprs)})"
    new_maps = [new_input_map] + list(sketch.indexing_maps[1:])
    return _dc_replace(sketch, indexing_maps=new_maps)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyProfile:
    """Policy profile used to configure strictness and solver defaults."""

    name: str
    strict_mode: bool
    z3_timeout_ms: int


POLICY_PROFILE_ENV_VAR = "LOOPHOLE_POLICY_PROFILE"
DEFAULT_POLICY_PROFILE = "local-explore"

POLICY_PROFILES: Dict[str, PolicyProfile] = {
    "local-explore": PolicyProfile(
        name="local-explore",
        strict_mode=False,
        z3_timeout_ms=10_000,
    ),
    "ci-strict": PolicyProfile(
        name="ci-strict",
        strict_mode=True,
        z3_timeout_ms=15_000,
    ),
}

POLICY_PROFILE_ALIASES: Dict[str, str] = {
    "exploratory": "local-explore",
    "trusted": "ci-strict",
}

POLICY_PROFILE_CHOICES: Tuple[str, ...] = (
    "default",
) + tuple(POLICY_PROFILES.keys()) + tuple(POLICY_PROFILE_ALIASES.keys())


def resolve_policy_profile(
    profile_name: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> PolicyProfile:
    """
    Resolve the runtime policy profile.

    Selection order:
      1. Explicit profile_name unless it is "default".
      2. Environment variable LOOPHOLE_POLICY_PROFILE.
      3. DEFAULT_POLICY_PROFILE.
    """

    selected = (profile_name or "default").strip().lower()

    if selected == "default":
        env_map = env or os.environ
        env_selected = env_map.get(POLICY_PROFILE_ENV_VAR, "").strip().lower()
        selected = env_selected or DEFAULT_POLICY_PROFILE

    selected = POLICY_PROFILE_ALIASES.get(selected, selected)

    if selected not in POLICY_PROFILES:
        valid = ", ".join(POLICY_PROFILE_CHOICES)
        raise ValueError(
            f"Unknown policy profile '{selected}'. Valid profiles: {valid}."
        )

    return POLICY_PROFILES[selected]


class LiftResultState(Enum):
    PROVED = "proved"
    UNPROVED_TIMEOUT = "unproved_timeout"
    REFUTED = "refuted"


@dataclass
class LiftResult:
    """Complete result from the lifting pipeline for one input function."""
    func_name: str
    matched_sketch: Optional[OperationSketch]
    sketch_name: Optional[str]
    emitted_mlir: Optional[str]
    verification: Optional[VerificationReport]
    sympy_confidence: float
    total_elapsed_ms: float
    target_dialect: str
    loop_info: Optional[LoopNestInfo] = None
    candidates_tried: List[str] = field(default_factory=list)
    error: Optional[str] = None
    parser_diagnostics: List = field(default_factory=list)  # List[ParsingWarning]

    @property
    def success(self) -> bool:
        return (
            self.matched_sketch is not None
            and self.emitted_mlir is not None
            and self.verification is not None
            and self.verification.result == CheckResult.EQUIVALENT
        )

    @property
    def result_state(self) -> LiftResultState:
        if self.verification is None:
            return LiftResultState.REFUTED
        if self.verification.result == CheckResult.EQUIVALENT:
            return LiftResultState.PROVED
        if self.verification.result in (CheckResult.TIMEOUT, CheckResult.UNKNOWN):
            return LiftResultState.UNPROVED_TIMEOUT
        return LiftResultState.REFUTED

    @property
    def partial_success(self) -> bool:
        """True if a match was found but only timed out/unknown (never refuted)."""
        return (
            self.matched_sketch is not None
            and self.emitted_mlir is not None
            and self.result_state == LiftResultState.UNPROVED_TIMEOUT
        )

    def is_accepted(self, strict_mode: bool = False) -> bool:
        if self.success:
            return True
        if strict_mode:
            return False
        return self.partial_success

    def summary(self) -> str:
        if self.success:
            v = self.verification
            return (
                f"[OK] Lifted '{self.func_name}' -> {self.sketch_name}\n"
                f"     Z3: {v.result.value} in {v.elapsed_ms:.1f}ms | "
                f"SymPy conf: {self.sympy_confidence:.2f} | "
                f"Total: {self.total_elapsed_ms:.1f}ms"
            )
        if self.result_state == LiftResultState.UNPROVED_TIMEOUT:
            v = self.verification
            status = v.result.value if v else "no verification"
            return (
                f"[UNPROVED_TIMEOUT] '{self.func_name}' -> {self.sketch_name} "
                f"(Z3: {status}) SymPy conf: {self.sympy_confidence:.2f}"
            )
        if self.result_state == LiftResultState.REFUTED:
            v = self.verification
            status = v.result.value if v else "NO_VERDICT"
            return (
                f"[REFUTED] '{self.func_name}' -> {self.sketch_name or 'none'} "
                f"(Z3: {status}) {self.error or ''}".rstrip()
            )
        return (
            f"[FAIL] Could not lift '{self.func_name}': {self.error or 'no match found'}"
        )


# ---------------------------------------------------------------------------
# Main Lifter
# ---------------------------------------------------------------------------

class Lifter:
    """
    Orchestrates the full LoopHole lifting pipeline.

    Parameters:
      target         : "linalg" | "stablehlo" | "both"
      z3_timeout_ms  : Z3 solver timeout per check (default 10s)
      top_k          : Number of top SymPy candidates to verify with Z3
            strict_mode    : Accept only formally proved outputs
      verbose        : Print progress information
    """

    def __init__(
        self,
        target: str = "linalg",
        z3_timeout_ms: int = 10_000,
        top_k: int = 3,
        strict_mode: bool = False,
        verbose: bool = False,
        parametric_mode: bool = False,
    ):
        self.target = target
        self.z3_timeout_ms = z3_timeout_ms
        self.top_k = top_k
        self.strict_mode = strict_mode
        self.verbose = verbose
        self.parametric_mode = parametric_mode
        self.disagreement_confidence_threshold = 0.8

        self._extractor = AffineExtractor()
        self._tracer = SympyTracer()
        self._z3 = Z3EquivalenceChecker(timeout_ms=z3_timeout_ms)
        self._linalg_emitter = LinalgEmitter()
        self._stablehlo_emitter = StableHLOEmitter()

    @classmethod
    def from_policy_profile(
        cls,
        *,
        target: str = "linalg",
        profile_name: Optional[str] = None,
        z3_timeout_ms: Optional[int] = None,
        strict_mode: Optional[bool] = None,
        top_k: int = 3,
        verbose: bool = False,
        parametric_mode: bool = False,
        env: Optional[Mapping[str, str]] = None,
    ) -> "Lifter":
        """
        Build a Lifter from a named policy profile with explicit overrides.

        Explicit arguments override profile defaults.
        """
        profile = resolve_policy_profile(profile_name=profile_name, env=env)
        resolved_timeout = z3_timeout_ms if z3_timeout_ms is not None else profile.z3_timeout_ms
        resolved_strict = strict_mode if strict_mode is not None else profile.strict_mode
        return cls(
            target=target,
            z3_timeout_ms=resolved_timeout,
            top_k=top_k,
            strict_mode=resolved_strict,
            verbose=verbose,
            parametric_mode=parametric_mode,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lift(self, mlir_text: str) -> LiftResult:
        """
        Lift a single MLIR function (in textual Affine IR format) to the
        target tensor dialect.

        Returns a LiftResult even on failure (check result.success).
        """
        t0 = time.perf_counter()

        # Stage 1: parse
        try:
            loop = self._extractor.extract(mlir_text)
        except Exception as exc:
            return LiftResult(
                func_name="<parse error>",
                matched_sketch=None,
                sketch_name=None,
                emitted_mlir=None,
                verification=None,
                sympy_confidence=0.0,
                total_elapsed_ms=0.0,
                target_dialect=self.target,
                error=f"Parse error: {exc}",
                parser_diagnostics=[],
            )

        if self.verbose:
            print(f"[Lifter] Parsed '{loop.func_name}': "
                  f"{len(loop.induction_vars)} loops, "
                  f"{len(loop.reduction_vars)} reductions, "
                  f"{len(loop.reads)} reads, "
                  f"{len(loop.writes)} writes")

        # Stage 2: match candidates
        candidates = self._match_candidates(loop)

        if not candidates:
            return LiftResult(
                func_name=loop.func_name,
                matched_sketch=None,
                sketch_name=None,
                emitted_mlir=None,
                verification=None,
                sympy_confidence=0.0,
                total_elapsed_ms=(time.perf_counter() - t0) * 1000,
                target_dialect=self.target,
                loop_info=loop,
                error="No sketch candidates passed structural pre-filter.",
                parser_diagnostics=loop.diagnostics,
            )

        # Stage 3: verify top-k with Z3
        best_result = self._verify_candidates(loop, candidates)

        total_ms = (time.perf_counter() - t0) * 1000

        if best_result is None:
            return LiftResult(
                func_name=loop.func_name,
                matched_sketch=None,
                sketch_name=None,
                emitted_mlir=None,
                verification=None,
                sympy_confidence=0.0,
                total_elapsed_ms=total_ms,
                target_dialect=self.target,
                loop_info=loop,
                candidates_tried=[c[0].name for c in candidates],
                error=f"All {len(candidates)} candidates failed Z3 verification.",
                parser_diagnostics=loop.diagnostics,
            )

        sketch, report, sympy_conf = best_result

        # A-15: shape-parametric proof augmentation (non-blocking — never changes concrete result)
        if self.parametric_mode and report.result == CheckResult.EQUIVALENT:
            param_report = self._z3._verify_parametric(loop, sketch)
            report.parametric_result = param_report.result
            report.parametric_notes = param_report.notes

        if report.result == CheckResult.NOT_EQUIVALENT:
            return LiftResult(
                func_name=loop.func_name,
                matched_sketch=sketch,
                sketch_name=sketch.name,
                emitted_mlir=None,
                verification=report,
                sympy_confidence=sympy_conf,
                total_elapsed_ms=total_ms,
                target_dialect=self.target,
                loop_info=loop,
                candidates_tried=[c[0].name for c in candidates],
                error=(
                    f"Best candidate '{sketch.name}' was refuted by Z3 "
                    "(NOT_EQUIVALENT)."
                ),
                parser_diagnostics=loop.diagnostics,
            )

        if self.strict_mode and report.result in (CheckResult.TIMEOUT, CheckResult.UNKNOWN):
            return LiftResult(
                func_name=loop.func_name,
                matched_sketch=sketch,
                sketch_name=sketch.name,
                emitted_mlir=None,
                verification=report,
                sympy_confidence=sympy_conf,
                total_elapsed_ms=total_ms,
                target_dialect=self.target,
                loop_info=loop,
                candidates_tried=[c[0].name for c in candidates],
                error=f"Strict mode rejected unproved result ({report.result.value}).",
                parser_diagnostics=loop.diagnostics,
            )

        # Stage 4: emit MLIR
        try:
            emitted = self._emit(sketch, loop)
        except Exception as exc:
            return LiftResult(
                func_name=loop.func_name,
                matched_sketch=sketch,
                sketch_name=sketch.name,
                emitted_mlir=None,
                verification=report,
                sympy_confidence=sympy_conf,
                total_elapsed_ms=total_ms,
                target_dialect=self.target,
                loop_info=loop,
                candidates_tried=[c[0].name for c in candidates],
                error=(
                    f"Emission error while generating '{sketch.name}': {exc}. "
                    "Check extracted shapes/types and sketch-emitter mapping."
                ),
                parser_diagnostics=loop.diagnostics,
            )

        return LiftResult(
            func_name=loop.func_name,
            matched_sketch=sketch,
            sketch_name=sketch.name,
            emitted_mlir=emitted,
            verification=report,
            sympy_confidence=sympy_conf,
            total_elapsed_ms=total_ms,
            target_dialect=self.target,
            loop_info=loop,
            candidates_tried=[c[0].name for c in candidates],
            parser_diagnostics=loop.diagnostics,
        )

    def lift_many(self, mlir_texts: List[str]) -> List[LiftResult]:
        """Lift multiple MLIR functions. Returns one LiftResult per input."""
        return [self.lift(t) for t in mlir_texts]

    # ------------------------------------------------------------------
    # Stage 2: Candidate matching
    # ------------------------------------------------------------------

    def _match_candidates(
        self, loop: LoopNestInfo
    ) -> List[Tuple[OperationSketch, float]]:
        """
        Return (sketch, confidence) pairs that pass structural pre-filter,
        sorted by SymPy confidence (descending).
        Limits to the library subset for the target dialect.
        """
        # Select sketch library
        if self.target == "linalg":
            library = LINALG_SKETCHES
        elif self.target == "stablehlo":
            library = STABLEHLO_SKETCHES
        else:
            library = SKETCH_LIBRARY

        # Special fast path: convolution recognizer
        conv_name = self._tracer.recognise_convolution(loop)
        trace: Optional[TraceResult] = None
        trace_error: Optional[Exception] = None
        try:
            trace = self._tracer.trace(loop)
        except Exception as exc:
            trace_error = exc
            if self.verbose:
                print(f"[SymPy] Trace failed: {exc}")

        # Structural pre-filter
        ranked_candidates: List[Tuple[bool, float, OperationSketch]] = []
        for sketch in library:
            if not structural_match(loop, sketch):
                continue

            # SymPy confidence scoring
            if trace is not None:
                conf = self._tracer._sketch_confidence(trace, loop, sketch)
            else:
                conf = 0.0

            conv_match = bool(conv_name and sketch.name == conv_name)
            ranked_candidates.append((conv_match, conf, sketch))

        # Prefer recognizer-aligned candidate when confidence ties.
        ranked_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        candidates = [(sketch, conf) for _, conf, sketch in ranked_candidates]

        if self.verbose:
            print(f"[Lifter] {len(candidates)} structural candidates:")
            for sk, c in candidates[:5]:
                print(f"         {sk.name}: {c:.3f}")
            if trace_error is not None:
                print("         SymPy trace unavailable; candidates scored at 0.0")

        return candidates[: self.top_k * 2]  # take more than top_k for Z3 to try

    def _annotate_disagreement(self, report: VerificationReport, confidence: float) -> None:
        report.sympy_confidence = confidence

        if report.result == CheckResult.EQUIVALENT:
            report.sympy_z3_disagreement = None
            return

        if confidence < self.disagreement_confidence_threshold:
            report.sympy_z3_disagreement = None
            return

        if report.result in (CheckResult.TIMEOUT, CheckResult.UNKNOWN):
            report.sympy_z3_disagreement = "HIGH_CONFIDENCE_UNPROVED"
        elif report.result == CheckResult.NOT_EQUIVALENT:
            report.sympy_z3_disagreement = "HIGH_CONFIDENCE_REFUTED"
        else:
            report.sympy_z3_disagreement = None

        if report.sympy_z3_disagreement:
            marker = f" [sympy_z3_disagreement={report.sympy_z3_disagreement}]"
            report.notes = (report.notes + marker).strip()

    # ------------------------------------------------------------------
    # Stage 3: Z3 verification
    # ------------------------------------------------------------------

    def _verify_candidates(
        self,
        loop: LoopNestInfo,
        candidates: List[Tuple[OperationSketch, float]],
    ) -> Optional[Tuple[OperationSketch, VerificationReport, float]]:
        """
        Run Z3 verification on each candidate in order.
        Returns (sketch, report, confidence) for the first EQUIVALENT match.
        If none prove EQUIVALENT, returns the best TIMEOUT/UNKNOWN fallback.
        If no timeout fallback exists, returns a refuted candidate explicitly.
        """
        best_timeout: Optional[Tuple[OperationSketch, VerificationReport, float]] = None
        best_refuted: Optional[Tuple[OperationSketch, VerificationReport, float]] = None

        for sketch, conf in candidates[: self.top_k]:
            if self.verbose:
                print(f"[Z3] Checking {sketch.name} (sympy conf={conf:.3f})...")

            report = self._z3.check(loop, sketch)
            self._annotate_disagreement(report, conf)

            if self.verbose:
                print(f"     -> {report.result.value} in {report.elapsed_ms:.1f}ms")

            if report.result == CheckResult.EQUIVALENT:
                return (sketch, report, conf)

            if report.result in (CheckResult.TIMEOUT, CheckResult.UNKNOWN):
                if best_timeout is None:
                    best_timeout = (sketch, report, conf)
            elif report.result == CheckResult.NOT_EQUIVALENT:
                strided_sketch = _build_strided_conv_sketch(loop, sketch)
                if strided_sketch is not None:
                    if self.verbose:
                        print(f"[Z3] Strided conv fallback for {sketch.name}...")
                    strided_report = self._z3.check(loop, strided_sketch)
                    self._annotate_disagreement(strided_report, conf)
                    if self.verbose:
                        print(f"     -> {strided_report.result.value} in {strided_report.elapsed_ms:.1f}ms")
                    if strided_report.result == CheckResult.EQUIVALENT:
                        return (sketch, strided_report, conf)
                    if strided_report.result in (CheckResult.TIMEOUT, CheckResult.UNKNOWN):
                        if best_timeout is None:
                            best_timeout = (sketch, strided_report, conf)
                if best_refuted is None:
                    best_refuted = (sketch, report, conf)

        # No formal proof — return best timeout candidate (partial success),
        # otherwise return explicit refutation so callers can report it.
        return best_timeout or best_refuted

    # ------------------------------------------------------------------
    # Stage 4: Emit
    # ------------------------------------------------------------------

    def _emit(self, sketch: OperationSketch, loop: LoopNestInfo) -> str:
        """Emit MLIR for the matched sketch using the appropriate emitter."""
        if sketch.dialect == "stablehlo" or self.target == "stablehlo":
            return self._stablehlo_emitter.emit(sketch, loop)
        return self._linalg_emitter.emit(sketch, loop)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def lift(
    mlir_text: str,
    target: str = "linalg",
    z3_timeout_ms: int = 10_000,
    strict_mode: bool = False,
    verbose: bool = False,
) -> LiftResult:
    """
    One-shot lift: parse → match → verify → emit.

    Example:
        result = lift(open("matmul.mlir").read())
        if result.success:
            print(result.emitted_mlir)
    """
    lifter = Lifter(
        target=target,
        z3_timeout_ms=z3_timeout_ms,
        strict_mode=strict_mode,
        verbose=verbose,
    )
    return lifter.lift(mlir_text)
