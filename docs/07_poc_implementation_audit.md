# 07 - POC Implementation Audit (What Exists, What Is Hardcoded, What Breaks)

> Last updated: 2026-03-31
> Scope: This is a code-and-doc reality audit for the current LoopHole POC, based on the implementation under `POC_initial_demo/` and the planning docs under `docs/`.

---

## 1. Executive Summary

This POC is implemented and runnable, but it is still a research-grade prototype with heuristic matching, partial formal verification, and several hardcoded shortcuts.

Current state at a glance:
- Test suite status: `96 passed, 1 skipped` (`pytest tests -q -rs`)
- Linalg demo status (`loophole demo --target linalg`): `7/10` formally proved, `3/10` partial
- StableHLO demo status (`loophole demo --target stablehlo`): `2/10` formally proved, `2/10` partial, `6/10` failed
- Example script status (`python examples/run_demo.py --mode demo`): currently broken (NameError)

The POC does deliver end-to-end lifting for key kernels (especially matmul and basic elementwise/reduction patterns), but several kernels are accepted as "partial" even when Z3 reports `NOT_EQUIVALENT`. That is a major limitation if strict correctness is required.

---

## 2. What Is Actually Implemented

### 2.1 Implemented Pipeline (Code Reality)

The core lifter pipeline is implemented in `POC_initial_demo/src/loophole/lifter.py`:

1. Parse MLIR text into `LoopNestInfo`
2. Generate candidate sketches using structural filters + SymPy confidence
3. Run Z3 checks on top candidates
4. Emit lifted MLIR in Linalg or StableHLO

Primary modules:
- Parser: `POC_initial_demo/src/loophole/affine_extractor.py`
- Sketch catalog: `POC_initial_demo/src/loophole/sketch_library.py`
- Symbolic matcher: `POC_initial_demo/src/loophole/sympy_tracer.py`
- Formal checker: `POC_initial_demo/src/loophole/z3_checker.py`
- Emitters: `POC_initial_demo/src/loophole/emitter.py`
- Orchestrator: `POC_initial_demo/src/loophole/lifter.py`
- CLI: `POC_initial_demo/src/loophole/cli.py`

### 2.2 Command Surfaces That Work

Implemented command entrypoint (via `setup.py` console script):
- `loophole lift`
- `loophole verify`
- `loophole batch`
- `loophole sketches`
- `loophole demo`

Important detail:
- `python -m loophole.cli ...` is not wired (no `if __name__ == "__main__": main()` in `cli.py`). Use the installed `loophole` command instead.

---

## 3. Kernel Coverage Snapshot (Observed)

Observed from actual demo runs in current environment.

### 3.1 Linalg Target (`loophole demo --target linalg`)

| Kernel | Outcome | Notes |
|---|---|---|
| matmul_4x4 | PROVED | Good path |
| transpose_2d | PROVED | Good path |
| conv1d | PROVED | Good path |
| conv2d_simple | PARTIAL | Z3 reported `NOT_EQUIVALENT`, still emitted |
| conv2d_nhwc | PARTIAL | Z3 reported `NOT_EQUIVALENT`, still emitted |
| dot_product | PARTIAL | Z3 reported `NOT_EQUIVALENT`, still emitted |
| matvec | PROVED | Good path |
| elementwise_add | PROVED | Good path |
| reduce_sum | PROVED | Good path |
| relu | PROVED (but suspicious) | Matched as scale sketch in demo output |

Key takeaway: the Linalg path is useful for exploration but not yet strict-proof-only.

### 3.2 StableHLO Target (`loophole demo --target stablehlo`)

| Kernel | Outcome | Notes |
|---|---|---|
| matmul_4x4 | PROVED | `stablehlo.dot_general` |
| reduce_sum | PROVED | `stablehlo.reduce{add}` |
| conv2d_nhwc | PARTIAL | `stablehlo.convolution`, Z3 `NOT_EQUIVALENT` |
| matvec | PARTIAL | Mis-mapped to reduce sketch, Z3 `NOT_EQUIVALENT` |
| transpose_2d | FAIL | No candidate |
| conv1d | FAIL | No candidate |
| conv2d_simple | FAIL | No candidate |
| dot_product | FAIL | No candidate |
| elementwise_add | FAIL | No candidate |
| relu | FAIL | No candidate |

Key takeaway: StableHLO support is partial and narrow.

---

## 4. How the POC Works, Step by Step

### 4.1 Stage 1 - Parsing (`affine_extractor.py`)

Implemented:
- Parses function signature (`func.func`)
- Parses loops (`affine.for`, fallback `scf.for`)
- Parses reads/writes (`affine.load/store`, fallback `memref.load/store`)
- Parses basic arithmetic operations (add/mul/sub/div/max/min/constants)
- Builds `LoopNestInfo` with loop vars, bounds, reads/writes, and op list

Design choice:
- Parser is now a deterministic token scanner with optional MLIR syntax validation via Python bindings when available.

What this means:
- More robust than regex-only extraction for common syntax variation
- Still not a full operation-level MLIR AST walker for all dialect forms

### 4.2 Stage 2 - Candidate Matching (`sympy_tracer.py`, `lifter.py`)

Implemented:
- Symbolically traces loop body into SymPy expression
- Scores sketch candidates using heuristics
- Adds convolution-specific boost from a pattern recognizer

Reality:
- Confidence scoring is heuristic, not a proof
- Matching threshold and boosts are fixed constants

### 4.3 Stage 3 - Verification (`z3_checker.py`)

Implemented:
- Structural pre-filter before expensive checks
- Concrete verification path with bounded unrolling
- Basic symbolic fallback path

Critical caveat:
- Symbolic path is best-effort and substitutes concrete size (`size=4`) for dynamic cases
- For some cases, reduction handling is simplified and only partially modeled

### 4.4 Stage 4 - Emission (`emitter.py`)

Implemented:
- Named-op emitters for many Linalg operations
- StableHLO emitters for dot/reduce/convolution sketches
- `linalg.generic` fallback emitter

Caveat:
- Emitter often uses inferred or defaulted shapes when metadata is incomplete
- Syntax validity is string-tested, not validated by a downstream MLIR verifier in tests

---

## 5. Hardcoding, Stubs, and Heuristic Shortcuts

This section lists the most important non-generalized logic in the POC.

| Area | Where | What Is Hardcoded / Stubbed | Impact |
|---|---|---|---|
| Symbolic verification | `z3_checker.py` | Dynamic/symbolic verification instantiates concrete size (`size=4`) | Not a universal proof for dynamic shapes |
| Reduction handling | `z3_checker.py` | Symbolic reduction path can return placeholder `0` in some branches | False positives/negatives possible |
| Unrolling limit | `z3_checker.py` | `unroll_threshold=32` | Large reductions switch to weaker path |
| Candidate confidence fallback | `lifter.py` | Exception fallback confidence fixed at `0.3` | Ranking can be arbitrary on parse/trace failures |
| Convolution bias | `lifter.py` | Confidence boost fixed at `+0.3` when recognizer agrees | Biases ranking toward guessed conv pattern |
| Sketch acceptance threshold | `sympy_tracer.py` | Match acceptance threshold fixed at `0.5` | Behavior sensitive to hand-tuned score |
| Partial success semantics | `lifter.py` | `partial_success` only requires matched sketch + emitted MLIR, not proof | Emits results even when Z3 says NOT_EQUIVALENT |
| Default typing | `affine_extractor.py` | Dominant type defaults to `f32` if uncertain | May silently mis-type outputs |
| Shape defaults | `emitter.py` | Many emitters fallback to `[4,4]`, `[8,8]`, etc. | Can emit plausible but semantically wrong signatures |
| Transpose fallback perm | `emitter.py` | Default permutation is reverse dimensions | Wrong for non-reverse transposes |
| Conv attrs | `emitter.py` | Strides/dilations mostly hardcoded to ones | Ignores non-unit stride/dilation kernels |

---

## 6. Confirmed Functional Issues

### 6.1 Example Script Is Broken

File: `POC_initial_demo/examples/run_demo.py`

Confirmed runtime failure:
- Missing `import re` causes NameError in fancy demo mode
- Script references stale result fields (`linalg_mlir`, `stablehlo_mlir`) that no longer exist on `LiftResult`

Impact:
- The standalone example script is unreliable as shipped
- CLI demo (`loophole demo`) is the working path

### 6.2 Strict Correctness Contract Is Not Enforced

Files: `POC_initial_demo/src/loophole/lifter.py`, `POC_initial_demo/tests/integration/*`

Observed behavior:
- For some kernels (conv2d, dot), Z3 reports `NOT_EQUIVALENT`
- Pipeline can still return a partial lift and emit output
- Integration tests frequently accept `success or partial_success`

Impact:
- Tool is currently "best-effort synthesis" rather than "proof-only lifting"

### 6.3 Fixture Quality Issue

File: `POC_initial_demo/src/loophole/tests/fixtures.py`

Observed inconsistency:
- In `MATVEC_MLIR`, store type uses `memref<4x4xf32>` for `%y` which is declared `memref<4xf32>`

Impact:
- Parser/tests may pass while silently tolerating malformed fixture typing

---

## 7. Gaps Versus Project Docs and Roadmap

Comparing implementation to planning docs (`docs/01` to `docs/05` and `POC_DOCUMENTATION.md`):

Implemented as planned:
- End-to-end parse/match/verify/emit pipeline
- Core kernels for dense algebra and simple conv paths
- Z3 integration and SymPy tracer integration
- CLI for single-file, verify-only, and batch workflows

Partially implemented:
- StableHLO support exists but covers only a subset robustly
- Formal verification exists but degrades to best-effort paths

Not implemented (or intentionally deferred):
- Full MLIR Python bindings operation-level AST walker
- General dynamic-shape proof strategy
- Sparse tensor lifting
- Hardware-aware schedule synthesis (Transform dialect)
- Multi-function whole-program lifting

---

## 8. Test Coverage Quality (Not Just Count)

What is covered:
- Basic parser structure checks
- Sketch library integrity checks
- Emitter string-shape checks
- Integration paths for key fixture kernels

What is weak or missing:
- Strict proof-only assertion (many tests accept partial)
- No execution validation of emitted MLIR with `mlir-opt`/`iree`/`xla`
- No dedicated unit tests for `sympy_tracer.py` behavior
- No tests for CLI module invocation style (`python -m loophole.cli`)
- No tests covering `examples/run_demo.py` runtime

Observed skip:
- One skipped test in `test_z3_checker.py` due outdated search for an `elementwise_add` sketch key

---

## 9. Priority Fix Plan

### P0 (must do first)

1. Enforce correctness mode
- Add a strict mode where `NOT_EQUIVALENT` can never become partial success
- Default CI tests to strict mode for critical kernels

2. Fix broken demo script
- Add missing imports
- Replace stale fields with `emitted_mlir`
- Add a smoke test for `examples/run_demo.py`

3. Tighten partial semantics
- Split result states into: `proved`, `unproved`, `refuted`
- Treat `NOT_EQUIVALENT` as refuted, not partial success

### P1 (high value)

4. Improve Z3 modeling
- Remove placeholder reductions
- Support multi-reduction dimensions correctly
- Add targeted tests for conv and dot proof obligations

5. Remove silent defaulting in emitters
- Fail fast when required shape/type metadata is missing
- Add explicit diagnostics instead of `[4,4]`/`[8,8]` defaults

6. Strengthen parser robustness
- Migrate from token-scanner extraction to full MLIR AST traversal for operation-level extraction
- Preserve current syntax-validation path and add richer diagnostics for unsupported affine forms

### P2 (next evolution)

7. Expand StableHLO sketch coverage and alignment
- Add stablehlo transpose/elementwise/dot-specific sketches
- Add dialect-specific verification tests

8. Add artifact-level validation
- Compile emitted MLIR in CI where possible
- Add semantic round-trip tests on numeric examples

---

## 10. Practical Reading Guide for Contributors

If you want to understand this POC quickly and accurately:

1. Read orchestrator first: `POC_initial_demo/src/loophole/lifter.py`
2. Read parser data model: `POC_initial_demo/src/loophole/affine_extractor.py`
3. Read proof limitations: `POC_initial_demo/src/loophole/z3_checker.py`
4. Read emitter assumptions: `POC_initial_demo/src/loophole/emitter.py`
5. Run demos and tests yourself:
   - `pytest tests -q -rs`
   - `loophole demo --target linalg`
   - `loophole demo --target stablehlo`

---

## 11. Bottom Line

This POC is a strong implementation prototype for research iteration, not yet a strict-verification production lifter.

It is already useful for:
- quickly classifying common affine loop nests
- generating candidate Linalg/StableHLO IR
- experimentally validating lifting ideas

It is not yet reliable for:
- guaranteed formal equivalence in all reported "partial" cases
- broad StableHLO kernel coverage
- robust operation on arbitrary real-world MLIR text without parser edge cases

The fastest path to production-grade reliability is to harden verification semantics first (especially handling of `NOT_EQUIVALENT`), then remove emitter/parser defaulting that can silently produce plausible but wrong outputs.
