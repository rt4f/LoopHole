<<<<<<< HEAD
# 07 - POC Implementation Audit (What Exists, What Is Hardcoded, What Breaks)

> Last updated: 2026-03-31
> Scope: This is a code-and-doc reality audit for the current LoopHole POC, based on the implementation under `POC_initial_demo/` and the planning docs under `docs/`.
=======
# 07 - LoopHole POC Implementation Audit (Sequential Analysis)

**Date:** 2026-03-31  
**Scope:** Entire repository context with implementation deep-dive on `POC_initial_demo/`  
**Method:** Sequential pass over docs -> source -> tests -> runtime behavior -> limitations -> fixes
>>>>>>> e4273a6 (C-01, C-02, C-03: Fix demo script, add smoke test, wire strict-mode integration test)

---

## 1. Executive Summary

<<<<<<< HEAD
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
=======
This repository contains two major tracks:

1. **Research/design track** in `docs/01..06` (architecture, SOTA, roadmap, research direction).
2. **Working Python POC** in `POC_initial_demo/` implementing a lift pipeline:
   - parse affine-style MLIR text,
   - match to operation sketches,
   - attempt formal equivalence with Z3,
   - emit Linalg or StableHLO MLIR text.

Current implementation status:

- **Core POC is implemented and runnable**.
- **Unit/integration tests currently pass**: `96 passed, 1 skipped`.
- **CLI demo works**, but with partial results on several kernels.
- **Standalone example demo script is broken**.
- **There are important correctness and soundness limitations** in verification fallback logic and several hardcoded/stubbed code paths.

---

## 2. Sequential Findings (What Was Checked)

### Pass 1 - Project intent and architecture docs

Reviewed:
- `docs/01_project_overview.md`
- `docs/02_state_of_the_art.md`
- `docs/03_mlir_architecture.md`
- `docs/04_implementation_roadmap.md`
- `docs/05_novel_research_directions.md`
- `docs/06_references.md`
- `README.md`
- `PROBLEM_STATEMENT.md`
- `architecture.mmd`

Outcome:
- Clear target architecture and research framing are documented.
- `README.md` still describes project status as pre-implementation, while `POC_initial_demo/` already contains substantial implementation.

### Pass 2 - POC source audit

Reviewed all modules under `POC_initial_demo/src/loophole/`:
- `affine_extractor.py`
- `sketch_library.py`
- `sympy_tracer.py`
- `z3_checker.py`
- `emitter.py`
- `lifter.py`
- `cli.py`
- package exports in `__init__.py`

Also reviewed:
- Embedded fixtures: `POC_initial_demo/src/loophole/tests/fixtures.py`
- Example script: `POC_initial_demo/examples/run_demo.py`

### Pass 3 - Test and runtime validation

Executed:
- `pytest -q -rs` in `POC_initial_demo/` -> `96 passed, 1 skipped`.
- `loophole demo --target linalg` -> pipeline runs; `7/10` formally proved and `3` partial.
- `loophole lift tests/fixtures/matmul.mlir --target stablehlo --report` -> formally proved and emits StableHLO.
- `examples/run_demo.py --mode demo` -> fails at runtime (`NameError: re is not defined`).

---

## 3. What Has Been Implemented

## 3.1 Repository-level implementation map

| Area | Status | Notes |
|---|---|---|
| Research documentation (`docs/01..06`) | Implemented | Strong conceptual coverage and roadmap. |
| Environment/bootstrap scripts (`scripts/`) | Implemented | WSL-centric installation paths documented. |
| Full LLVM/Polygeist trees in workspace | Present | Heavy toolchain sources/build artifacts are included in workspace. |
| POC Python package (`POC_initial_demo`) | Implemented | Main functional deliverable. |

## 3.2 POC pipeline implementation

### Stage 1: MLIR text extraction (`affine_extractor.py`)

Implemented:
- Regex-based parsing of:
  - `func.func`,
  - `affine.for` and fallback `scf.for`,
  - `affine.load/store`, fallback `memref.load/store`,
  - key `arith.*` ops.
- Builds a structured `LoopNestInfo` object:
  - loop IVs/bounds/order,
  - reads/writes,
  - compute ops,
  - tensor shapes/types,
  - reduction/parallel classification,
  - accumulation detection.

### Stage 2: Sketch library (`sketch_library.py`)

Implemented:
- `OperationSketch` model with:
  - indexing maps,
  - iterator types,
  - compute payload classification,
  - operand counts.
- `SKETCH_LIBRARY` currently contains **25 sketches**:
  - **22 Linalg**
  - **3 StableHLO**
- Dialect subsets and lookup maps are exposed.

### Stage 3: SymPy symbolic tracer (`sympy_tracer.py`)

Implemented:
- Converts loop ops into symbolic SymPy expressions.
- Computes heuristic confidence scores per sketch.
- Includes convolution-specific recognizer for sliding-window patterns.

### Stage 4: Z3 checker (`z3_checker.py`)

Implemented:
- Structural pre-filter before expensive checking.
- Concrete verification path (unrolled where feasible).
- Reduction handling (explicit unroll and RecFunction fallback).
- Bidirectional implication checks for equivalence.
- Returns `VerificationReport` with enum status and timing.

### Stage 5: Emitters (`emitter.py`)

Implemented:
- Linalg emitter with named-op handlers (matmul, transpose, convs, reductions, etc).
- Generic fallback emission via `linalg.generic`.
- StableHLO emitter for dot/convolution/reduce variants.

### Orchestration (`lifter.py`)

Implemented:
- End-to-end pipeline orchestration.
- Candidate matching by structural filter + SymPy confidence.
- Top-k Z3 verification and fallback behavior.
- `LiftResult` status model with summary formatting.

### CLI (`cli.py`)

Implemented commands:
- `lift`
- `verify`
- `batch`
- `sketches`
- `demo`

Installed console entry point:
- `loophole=loophole.cli:main`

---

## 4. Current Behavior Snapshot

## 4.1 Test status

Command:

```bash
pytest -q -rs
```

Observed:
- `96 passed, 1 skipped`
- skipped case:
  - `tests/unit/test_z3_checker.py:77` skipped due name-based search for `elementwise_add` sketch string.

Interpretation:
- Core pipeline paths are tested and currently stable for tested kernels.
- There are still coverage gaps (see Section 6.5).

## 4.2 CLI demo behavior

Command:

```bash
loophole demo --target linalg
```

Observed summary:
- `7/10 formally proved`
- `3 partial`

Partial kernels shown by CLI run:
- `conv2d_simple` -> `NOT_EQUIVALENT` (still emitted as partial)
- `conv2d_nhwc` -> `NOT_EQUIVALENT` (still emitted as partial)
- `dot_product` -> `NOT_EQUIVALENT` (still emitted as partial)

Also observed:
- `relu` was matched as `linalg.map{arith.mulf_scalar}` and reported `EQUIVALENT`.
  - This is a **high-risk correctness signal** because ReLU semantics are `max(x,0)`, not scalar multiply.

## 4.3 Standalone demo script status

Command:

```bash
python examples/run_demo.py --mode demo
```

Observed:
- Runtime failure: `NameError: name 're' is not defined`.
- Additional stale API references still exist in script (`result.linalg_mlir`, `result.stablehlo_mlir`) that no longer match current `LiftResult` fields.

---

## 5. Limitations, Stubs, and Hardcoding Inventory

This section explicitly lists where hardcoding or placeholder behavior exists, with practical impact.

## 5.1 Verification status semantics (high impact)

### A) `partial_success` is too permissive

Location:
- `POC_initial_demo/src/loophole/lifter.py` (property `partial_success`)

Current behavior:
- Returns `True` whenever a sketch matched and MLIR was emitted.
- Does **not** require Z3 timeout/unknown.
- Includes cases where Z3 returned `NOT_EQUIVALENT`.

Impact:
- User-facing status can look acceptable even when formal check disproved equivalence.

Fix:
- Redefine `partial_success` to include only:
  - `verification is None` (if explicit no-verify mode), or
  - `verification.result in {TIMEOUT, UNKNOWN}`.
- Exclude `NOT_EQUIVALENT` from partial-success path.

### B) Non-equivalent candidates accepted as fallback

Location:
- `POC_initial_demo/src/loophole/lifter.py` in `_verify_candidates`

Current behavior:
- If Z3 says `NOT_EQUIVALENT` and SymPy confidence >= 0.7, candidate can still become fallback.

Impact:
- Can produce emitted IR tagged as partial despite failed formal check.

Fix:
- Remove `NOT_EQUIVALENT` fallback branch.
- Keep fallback only for `TIMEOUT`/`UNKNOWN`.

## 5.2 Z3 encoding stubs and approximations (high impact)

### C) RecFunction reduction supports only one reduction IV

Location:
- `POC_initial_demo/src/loophole/z3_checker.py` (`_recfunc_reduction`)

Current behavior:
- Explicit comment and logic: uses only `loop.reduction_vars[0]`.

Impact:
- Multi-dimensional reductions are not fully modeled in symbolic path.

Fix:
- Generalize reduction recurrence to nested dimensions, or
- Encode with quantified sums/auxiliary functions over tuples.

### D) Sketch reduction unroll handles only first reduction dim

Location:
- `POC_initial_demo/src/loophole/z3_checker.py` (`_build_sketch_rhs`)

Current behavior:
- Iterates zipped reduction dims then `break` after first dim.

Impact:
- Incomplete semantic encoding for multi-reduction ops (e.g., conv2d-like patterns).

Fix:
- Replace single-dim loop with cartesian-product iteration over all reduction dims.

### E) Symbolic reduction path returns zero placeholder

Location:
- `POC_initial_demo/src/loophole/z3_checker.py` (`_build_sketch_rhs`)

Current behavior:
- For non-unrollable symbolic reductions, returns `RealVal(0)`.

Impact:
- Symbolic verification can be under-constrained or misleading.

Fix:
- Replace placeholder with real symbolic reduction encoding (recursive or quantified).

### F) Symbolic verification is "representative instance" check

Location:
- `POC_initial_demo/src/loophole/z3_checker.py` (`_verify_symbolic`)

Current behavior:
- Instantiates symbolic bounds to size 4 and verifies concretely.

Impact:
- This is not a full symbolic proof; may miss size-dependent bugs.

Fix:
- Label as bounded check only, or
- Implement true universally quantified symbolic proof strategy.

### G) Elementwise MAX/ReLU semantics not robustly distinguished

Locations:
- `POC_initial_demo/src/loophole/z3_checker.py` (`_infer_compute_payload`, `_build_elementwise_expr`)

Current behavior:
- ReLU-like forms can be inferred as generic max or even collapse to pass-through under single-read handling.

Impact:
- Potential unsound equivalence results (as seen in demo: ReLU matched to scalar multiply).

Fix:
- Detect and track constants used in max operations.
- Add explicit ReLU pattern extraction (`max(read, 0)`).
- Add strict negative tests: ReLU must not verify as scale/copy.

## 5.3 Parser limitations and hardcoded assumptions (medium/high impact)

### H) Regex parser instead of MLIR AST parser

Location:
- `POC_initial_demo/src/loophole/affine_extractor.py`

Current behavior:
- Parsing is regex-driven with simplified syntax assumptions.

Impact:
- Fragile against valid MLIR syntactic variations.
- Complex affine maps/attributes/layouts may parse incorrectly.

Fix:
- Move to MLIR Python bindings once environment supports them.
- Until then, use a formal grammar parser for MLIR text subset.

### I) IV role classification uses substring checks

Location:
- `POC_initial_demo/src/loophole/affine_extractor.py` (`_classify_iv_roles`)

Current behavior:
- Determines IV presence via `if iv in expr` substring matching.

Impact:
- Risk of false positives (`i` in `i0`, etc.) and misclassification.

Fix:
- Tokenize index expressions and match identifiers, not substrings.

## 5.4 Emitter hardcoded defaults (medium impact)

### J) Default fallback shapes hardcoded across many handlers

Location:
- `POC_initial_demo/src/loophole/emitter.py`

Current behavior:
- Many `shapes.get(..., [4,4])`, `[8,8]`, `[16]`, etc.

Impact:
- If extraction misses shape metadata, emitted IR may be syntactically valid but semantically wrong.

Fix:
- Fail fast on missing required shape metadata for named ops.
- Keep fallback only for explicit debug mode.

### K) StableHLO reduction dimensions hardcoded

Location:
- `POC_initial_demo/src/loophole/emitter.py` (`_emit_reduce`)

Current behavior:
- Uses fixed `across dimensions = [1]`.

Impact:
- Wrong for many reduction layouts/ranks.

Fix:
- Infer reduction dims from loop roles and indexing maps.

## 5.5 CLI and tooling gaps (medium impact)

### L) `--no-verify` option does not actually skip verification

Location:
- `POC_initial_demo/src/loophole/cli.py` (`lift_cmd`)

Current behavior:
- Sets `z3_timeout_ms=0`; does not bypass checker.

Impact:
- Misleading CLI semantics.

Fix:
- Add explicit `verify: bool` flag into `Lifter`, and bypass `_verify_candidates` when false.

### M) `python -m loophole.cli ...` path is non-functional

Location:
- `POC_initial_demo/src/loophole/cli.py`

Current behavior:
- No `if __name__ == "__main__": main()` block.

Impact:
- Module execution gives no command dispatch.

Fix:
- Add module entry guard for developer convenience.

### N) Standalone demo script stale and broken

Location:
- `POC_initial_demo/examples/run_demo.py`

Current behavior:
- Missing `import re`.
- Uses stale fields (`result.linalg_mlir`, `result.stablehlo_mlir`) not present in current `LiftResult`.

Impact:
- Example script fails and misrepresents current API.

Fix:
- Add `import re`.
- Replace stale fields with `result.emitted_mlir` and target-aware logic.

## 5.6 Fixture and documentation drift (low/medium impact)

### O) Embedded fixture inconsistency

Location:
- `POC_initial_demo/src/loophole/tests/fixtures.py` (`MATVEC_MLIR`)

Current behavior:
- Store type for `%y` uses `memref<4x4xf32>` while argument type is `memref<4xf32>`.

Impact:
- Can mask parser/type checking issues in fixture-based tests.

Fix:
- Correct fixture type annotation.

### P) Documentation drift on sketch count

Locations:
- `POC_initial_demo/POC_DOCUMENTATION.md` (states 26 sketches)
- `POC_initial_demo/src/loophole/sketch_library.py` (actual 25 sketches)

Impact:
- Confusion in expected behavior/capability reporting.

Fix:
- Update documentation to real counts and regenerate diagrams/tables.

---

## 6. Coverage Gaps

Current tests are strong for core flow but still leave gaps:

1. No integration test for ReLU end-to-end correctness.
2. No tests guarding against false equivalence between different elementwise ops.
3. No tests for `--no-verify` behavior.
4. No tests for CLI module execution (`python -m loophole.cli`).
5. No tests for `examples/run_demo.py` script health.
6. Skip reason in z3 unit tests indicates sketch-name coupling mismatch (`elementwise_add` string expectation).

---

## 7. Prioritized Fix Plan

## P0 - Correctness and trustworthiness (must do first)

1. Tighten success model:
   - disallow `NOT_EQUIVALENT` as partial success.
2. Remove high-confidence override for `NOT_EQUIVALENT` in `_verify_candidates`.
3. Fix Z3 sketch reduction encoding to support all reduction dimensions.
4. Replace symbolic `0` placeholders with real symbolic encodings or fail explicitly.
5. Add strict negative tests (ReLU vs scale, dot vs matmul, etc.).

## P1 - Semantic robustness

1. Improve compute payload inference for constants and unary max forms.
2. Tokenize index expressions for robust IV classification.
3. Reduce shape hardcoding in emitter and fail fast on missing metadata.
4. Infer StableHLO reduction dimensions from extracted loop semantics.

## P2 - Developer UX and maintainability

1. Fix `examples/run_demo.py`.
2. Implement true `--no-verify` semantics.
3. Add `if __name__ == "__main__": main()` to CLI module.
4. Sync POC documentation counts and examples with current code.

## P3 - Strategic evolution toward planned architecture

1. Replace regex extractor with MLIR AST-driven parsing (when environment supports bindings).
2. Expand formal proof support for dynamic shapes and multi-dimensional reductions.
3. Add translation-validation step by running emitted MLIR through `mlir-opt` in CI.

---

## 8. Practical "How This POC Works" Summary

If you want to explain the POC in one paragraph:

> The POC takes affine-style loop MLIR text, extracts loop/memory/math structure with a regex parser, scores operation sketches using SymPy-based heuristics, checks top candidates with a Z3 equivalence engine, and emits higher-level Linalg or StableHLO MLIR. It works well for the tested dense kernels, but currently uses several approximation/fallback paths (especially in verification and status handling), so partial results and even some proved results should be interpreted carefully until the P0/P1 fixes are completed.

---

## 9. Recommended Immediate Actions (Next 1-2 Days)

1. Fix `run_demo.py` and add a regression test so the public demo path is always green.
2. Patch `partial_success` and `_verify_candidates` to prevent `NOT_EQUIVALENT` from being treated as acceptable output.
3. Add a dedicated ReLU integration test that fails if matched to scale/copy.
4. Correct `MATVEC_MLIR` fixture type inconsistency.
5. Update `POC_DOCUMENTATION.md` sketch counts and verification status semantics.

---

## 10. Final Status Statement

The POC is not a stub: it is a real, runnable lifting system with meaningful functionality, passing tests, and working CLI output. However, it is still a research-stage prototype with important correctness-risk shortcuts and hardcoded assumptions. Treat current results as strong feasibility evidence, not yet production-grade translation guarantees.
>>>>>>> e4273a6 (C-01, C-02, C-03: Fix demo script, add smoke test, wire strict-mode integration test)
