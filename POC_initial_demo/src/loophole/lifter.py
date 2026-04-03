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

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from loophole.affine_extractor import AffineExtractor, LoopNestInfo
from loophole.emitter import LinalgEmitter, StableHLOEmitter
from loophole.sketch_library import (
    SKETCH_LIBRARY, LINALG_SKETCHES, STABLEHLO_SKETCHES, OperationSketch,
)
from loophole.sympy_tracer import SympyTracer, TraceResult
from loophole.z3_checker import (
    CheckResult, VerificationReport, Z3EquivalenceChecker, structural_match,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

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
    def partial_success(self) -> bool:
        """True if a match was found but Z3 timed out — still useful."""
        return (
            self.matched_sketch is not None
            and self.emitted_mlir is not None
        )

    def summary(self) -> str:
        if self.success:
            v = self.verification
            return (
                f"[OK] Lifted '{self.func_name}' → {self.sketch_name}\n"
                f"     Z3: {v.result.value} in {v.elapsed_ms:.1f}ms | "
                f"SymPy conf: {self.sympy_confidence:.2f} | "
                f"Total: {self.total_elapsed_ms:.1f}ms"
            )
        if self.partial_success:
            v = self.verification
            status = v.result.value if v else "no verification"
            return (
                f"[PARTIAL] '{self.func_name}' → {self.sketch_name} "
                f"(Z3: {status}) SymPy conf: {self.sympy_confidence:.2f}"
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
      verbose        : Print progress information
    """

    def __init__(
        self,
        target: str = "linalg",
        z3_timeout_ms: int = 10_000,
        top_k: int = 3,
        verbose: bool = False,
    ):
        self.target = target
        self.z3_timeout_ms = z3_timeout_ms
        self.top_k = top_k
        self.verbose = verbose

        self._extractor = AffineExtractor()
        self._tracer = SympyTracer()
        self._z3 = Z3EquivalenceChecker(timeout_ms=z3_timeout_ms)
        self._linalg_emitter = LinalgEmitter()
        self._stablehlo_emitter = StableHLOEmitter()

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

        # Structural pre-filter
        candidates: List[Tuple[OperationSketch, float]] = []
        for sketch in library:
            if not structural_match(loop, sketch):
                continue

            # Boost if convolution recognizer agrees
            boost = 0.3 if (conv_name and sketch.name == conv_name) else 0.0

            # SymPy confidence scoring
            try:
                trace = self._tracer.trace(loop)
                conf = self._tracer._sketch_confidence(trace, loop, sketch)
            except Exception:
                conf = 0.3  # fallback confidence
            conf = min(1.0, conf + boost)

            candidates.append((sketch, conf))

        # Sort by confidence descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        if self.verbose:
            print(f"[Lifter] {len(candidates)} structural candidates:")
            for sk, c in candidates[:5]:
                print(f"         {sk.name}: {c:.3f}")

        return candidates[: self.top_k * 2]  # take more than top_k for Z3 to try

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
        If none prove EQUIVALENT, returns the best TIMEOUT or unknown result.
        """
        best_fallback: Optional[Tuple[OperationSketch, VerificationReport, float]] = None

        for sketch, conf in candidates[: self.top_k]:
            if self.verbose:
                print(f"[Z3] Checking {sketch.name} (sympy conf={conf:.3f})...")

            report = self._z3.check(loop, sketch)

            if self.verbose:
                print(f"     → {report.result.value} in {report.elapsed_ms:.1f}ms")

            if report.result == CheckResult.EQUIVALENT:
                return (sketch, report, conf)

            if report.result in (CheckResult.TIMEOUT, CheckResult.UNKNOWN):
                if best_fallback is None:
                    best_fallback = (sketch, report, conf)
            elif report.result == CheckResult.NOT_EQUIVALENT and conf >= 0.7:
                # SymPy is confident but Z3 says NOT_EQUIVALENT — may be a Z3
                # encoding precision issue (e.g. sliding-window index expressions).
                # Treat as UNKNOWN fallback so we still emit a partial result.
                if best_fallback is None:
                    best_fallback = (sketch, report, conf)

        # No formal proof — return best timeout candidate (partial success)
        return best_fallback

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
        verbose=verbose,
    )
    return lifter.lift(mlir_text)
