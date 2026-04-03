"""
LoopHole — Automated Lifting of Legacy Scalar Loop Nests into MLIR Tensor Dialects

Package structure:
  affine_extractor  — Parse MLIR Affine IR text into LoopNestInfo data structures
  sketch_library    — Library of formal operation sketches for target Linalg/StableHLO ops
  z3_checker        — Z3 SMT solver based formal equivalence verification
  sympy_tracer      — SymPy algebraic symbolic tracer for verification and sketch matching
  emitter           — MLIR text emitter for Linalg and StableHLO dialects
  lifter            — Main synthesis pipeline orchestrator
  cli               — Command-line interface
"""

from loophole.affine_extractor import AffineExtractor, LoopNestInfo
from loophole.sketch_library import SKETCH_LIBRARY, OperationSketch
from loophole.z3_checker import Z3EquivalenceChecker, CheckResult
from loophole.sympy_tracer import SympyTracer
from loophole.emitter import LinalgEmitter, StableHLOEmitter
from loophole.lifter import Lifter, LiftResult
from loophole.polygeist_frontend import (
  PolygeistFrontend,
  PolygeistFrontendError,
  CgeistInvocation,
  CgeistResult,
)

__version__ = "0.1.0"
__all__ = [
    "AffineExtractor",
    "LoopNestInfo",
    "SKETCH_LIBRARY",
    "OperationSketch",
    "Z3EquivalenceChecker",
    "CheckResult",
    "SympyTracer",
    "LinalgEmitter",
    "StableHLOEmitter",
    "Lifter",
    "LiftResult",
    "PolygeistFrontend",
    "PolygeistFrontendError",
    "CgeistInvocation",
    "CgeistResult",
]
