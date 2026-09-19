---
part: 2
title: Architecture and Pipeline
subtitle: How LoopHole is put together - system context, the four-stage pipeline, the complete data model, a fully worked matmul example with real intermediate values, the lifter's decision logic, configuration, and the verification, emission and toolchain architectures with their structural weaknesses.
audience: Anyone who will change code or needs to reason about what a result means.
covers: system context, pipeline stages and contracts, data classes and enums, matmul walkthrough, result states and exit codes, policy profiles and environment variables, Z3 encoding strategies, emitter design, frontend and validation paths, module dependency graph, architectural drawbacks
---

# System Context

LoopHole is a single Python package (`packages/loophole`) with a command-line interface. It relies on three kinds of external component:

1. **Python libraries** installed on the host: `z3-solver` (SMT solver), `sympy` (algebra), `click` (CLI framework) and `rich` (terminal tables and colours). They are declared in `pyproject.toml`.
2. **A Docker image** (`loophole-polygeist:llvm17`, also tagged `ghcr.io/schizoid-man/loophole-polygeist:llvm17`) containing LLVM/Clang/MLIR 17 from Ubuntu packages and a `cgeist` binary built from Polygeist. The host never needs LLVM installed; the CLI shells out to `docker run` whenever it needs `cgeist` or `mlir-opt`.
3. **GitHub infrastructure**: Actions for tests and image publishing, and the GitHub Container Registry (private) for the prebuilt image.

::diagram toolchain

The inputs to the system are C/C++ source files or MLIR text files. The outputs are MLIR text files in the Linalg or StableHLO dialect, terminal reports, and JSON/Markdown reports. There is no server, database or persistent state: every command is a stateless batch process, which keeps the operational picture simple.

## What the tool is not

- It is not an MLIR pass or plugin; it never links against MLIR and never builds MLIR objects in memory. Everything is text in, text out.
- It is not a general synthesiser: it can only produce the 35 operations in its sketch library, and only for loop nests whose shape matches one of them exactly (same number of loops, same parallel/reduction split).
- It does not run the lifted code or compare numbers. Downstream compilation (IREE, XLA) was exercised once by hand (see the lane A full-workflow log) but is not part of the tool or CI.

# The Four-Stage Pipeline

::diagram pipeline

All lifting goes through `Lifter.lift(mlir_text)` in `lifter.py`. The stages, with their contracts:

| Stage | Module and entry point | Input | Output | Failure behaviour |
|---|---|---|---|---|
| 0 (optional) Frontend | `polygeist_frontend.PolygeistFrontend.generate_mlir`, or `cli.compile_cmd` | C/C++ files | MLIR text | Raises `PolygeistFrontendError` (CLI prints a panel and exits 2) |
| 1 Parse | `affine_extractor.AffineExtractor.extract` | MLIR text (one function expected) | `LoopNestInfo` | Almost never raises; records `ParsingWarning` diagnostics and returns partial data (for example `func_name="unknown"`) |
| 2 Match | `Lifter._match_candidates` using `structural_match` and `SympyTracer` | `LoopNestInfo` | up to `2*top_k` (sketch, confidence) pairs | Empty list leads to a result with `verification=None` |
| 3 Verify | `Lifter._verify_candidates` using `Z3EquivalenceChecker.check` | `LoopNestInfo`, candidates | `_CandidateVerification`: `best` (sketch, `VerificationReport`, confidence) or `None`, plus `no_verdict` reports | ENCODE_ERROR and STRUCTURAL_MISMATCH reports surface only when no candidate reaches a verdict |
| 4 Emit | `LinalgEmitter.emit` or `StableHLOEmitter.emit` via `Lifter._emit` | sketch, `LoopNestInfo` | MLIR text | `EmissionError` becomes `LiftResult.error` |
| 5 (optional) Validate | `mlir_validator.validate_mlir_artifact` (only called by `batch`) | MLIR text | `MlirValidationResult` | Non-zero exit of the verifier marks failure; batch exits 1 at the end |

A crucial property of this design is that **stages communicate only through `LoopNestInfo` and `OperationSketch`**. Whatever the parser does not put into `LoopNestInfo` (loop steps, operand order, which value flows where, which function a loop belongs to) is invisible to every later stage. Most of the soundness problems in Part 5 are consequences of information that never makes it into this structure, or that is present but ignored by the verifier.

# Data Model

All data classes are plain `@dataclass` objects with no methods beyond small derived properties. They are defined in the module of the stage that produces them.

## Parser output (`affine_extractor.py`)

**`AccessPattern`**: one `affine.load`/`memref.load` (read) or `affine.store`/`memref.store` (write).

| Field | Type | Meaning |
|---|---|---|
| `tensor_name` | str | SSA name of the array, with `%`, for example `%A` |
| `index_exprs` | List[str] | Normalised index expressions using short IV names, for example `["i", "k"]` or `["2*oh + 3*kh", "ow + 2*kw"]` |
| `is_affine` | bool | True for `affine.*`, False for `memref.*` |
| `is_read` | bool | True for loads |
| `shape` | List[int] | Shape parsed from the type annotation on this access; -1 for dynamic |
| `element_type` | str | For example `f32` |
| `ssa_name` | str | For loads the result (`%a`); for stores the *stored value* (`%add`) |

**`ComputeOp`**: one `arith.*` operation.

| Field | Type | Meaning |
|---|---|---|
| `op_type` | str | Suffix after `arith.`, for example `mulf`, `addf`, `constant` |
| `operands` | List[str] | Operand SSA names, truncated to two |
| `result` | str | Result SSA name |
| `value` | Optional[float] | For constants; 0.0 when unparseable |

**`ParsingWarning`**: diagnostic with `level` (`debug`/`warning`/`error`), `location` (parser method name), `reason`, `input_snippet` (up to 80 characters) and optional `guidance`.

**`LoopNestInfo`**: the single contract between the parser and everything else.

| Field | Type | Meaning |
|---|---|---|
| `func_name` | str | From the first `func.func @name(` in the text |
| `induction_vars` | List[str] | Short IV names in source order |
| `bounds` | Dict[str, (lo, hi)] | Integers when resolvable, else strings such as `N` or unresolved text |
| `loop_order` | List[str] | Same as `induction_vars` today |
| `reads`, `writes` | List[AccessPattern] | In textual order across the whole input |
| `compute_ops` | List[ComputeOp] | In textual order |
| `tensor_shapes` | Dict[str, List[int]] | Reconciled shape per tensor (majority rank, conflicting static dims made dynamic) |
| `tensor_types` | Dict[str, str] | Dominant element type per tensor |
| `element_type` | str | Most common element type overall, default `f32` |
| `reduction_vars`, `parallel_vars` | List[str] | Classified by whether the IV name occurs in any write index |
| `func_args` | Dict[str, str] | Argument name to type string |
| `has_accumulation`, `accumulation_op` | bool, str | Output tensor both read and written, with an add |
| `diagnostics` | List[ParsingWarning] | Collected during extraction |
| `output_tensor` (property) | Optional[str] | Tensor of the **first** write |
| `input_tensors` (property) | List[str] | Read tensors other than the output, with duplicates |

Notably absent: loop steps, per-loop parent/child structure, which operations are inside which loop, `iter_args` and `affine.yield` links, conditionals, and any notion of multiple functions.

## Sketch library (`sketch_library.py`)

**`IteratorType`** (str enum): `PARALLEL`, `REDUCTION`, `WINDOW` (unused).

**`ComputePayloadType`** (str enum): `MULTIPLY_ACCUMULATE` (`C += A*B`), `COPY`, `ADD`, `SUBTRACT`, `MULTIPLY`, `MAX`, `MIN`, `RELU`, `SCALE`, `NEGATE`, `ACCUMULATE_ADD` (`C += A`), `ACCUMULATE_MAX` (`C = max(C, A)`).

**`OperationSketch`** fields: `name`, `dialect`, `dim_names`, `indexing_maps`, `iterator_types`, `compute_payload`, `num_inputs`, `num_outputs`, `description`. Derived: `num_loops`, `num_parallel`, `num_reduction`, `total_operands`, `parallel_dims()`, `reduction_dims()`.

Module-level collections: `SKETCH_LIBRARY` (ordered list of 35), `SKETCH_BY_NAME` (dict), `LINALG_SKETCHES` (22), `STABLEHLO_SKETCHES` (13). The full table is in Part 3.

## Tracer output (`sympy_tracer.py`)

**`TraceResult`**: `output_expr` (SymPy expression for one output element), `index_syms`, `reduction_syms`, `tensor_syms` (IndexedBase per tensor), `matched_sketch`, `confidence`, `notes`.

## Verifier output (`z3_checker.py`)

**`CheckResult`** (enum): `EQUIVALENT`, `NOT_EQUIVALENT`, `TIMEOUT`, `UNKNOWN`, `STRUCTURAL_MISMATCH`, `ENCODE_ERROR`.

**`ReductionPatternKind`** (enum, diagnostics only): `CONCRETE_UNROLL`, `SINGLE_SYMBOLIC_INFERRED`, `MULTI_SYMBOLIC_INFERRED`, `SINGLE_SYMBOLIC_UNINFERRED`, `MULTI_SYMBOLIC_UNINFERRED`.

**`VerificationReport`**:

| Field | Meaning |
|---|---|
| `result`, `sketch_name`, `elapsed_ms` | Outcome and timing |
| `z3_model` | Full model text when a counterexample was found |
| `counterexample_bindings` | Dict of declaration name to value string |
| `failed_implication` | `source_to_sketch` or `sketch_to_source` |
| `mismatch_summary` | First six bindings as text |
| `sympy_confidence`, `sympy_z3_disagreement` | Set by the lifter; disagreement is `HIGH_CONFIDENCE_UNPROVED` or `HIGH_CONFIDENCE_REFUTED` when confidence is at least 0.8 but not proved |
| `notes` | Free text; carries ENCODE_ERROR details |
| `parametric_result`, `parametric_notes` | Filled only with `--parametric` |
| `proved` (property) | `result == EQUIVALENT` |

## Lifter output (`lifter.py`)

**`PolicyProfile`** (frozen): `name`, `strict_mode`, `z3_timeout_ms`.

**`LiftResultState`** (enum): `PROVED`, `UNPROVED_TIMEOUT`, `REFUTED`.

**`LiftResult`**:

| Field or property | Meaning |
|---|---|
| `func_name`, `target_dialect`, `total_elapsed_ms` | Identification and timing |
| `matched_sketch`, `sketch_name` | Chosen sketch (may be set even when refuted) |
| `emitted_mlir` | Output text, or `None` |
| `verification` | The chosen candidate's report, or `None` |
| `sympy_confidence` | Confidence of the chosen candidate |
| `loop_info`, `candidates_tried`, `error`, `parser_diagnostics` | Context |
| `success` | sketch and emitted text and verification `EQUIVALENT` |
| `result_state` | `None` verification gives REFUTED; `EQUIVALENT` gives PROVED; `TIMEOUT`/`UNKNOWN` give UNPROVED_TIMEOUT; anything else REFUTED |
| `partial_success` | sketch and emitted text and state UNPROVED_TIMEOUT |
| `is_accepted(strict)` | success, or partial_success when not strict |
| `summary()` | One-line text used by scripts and the demo |

## Frontend, validation, replay and corpus types

| Class | Module | Fields |
|---|---|---|
| `CgeistInvocation` | `polygeist_frontend.py` | `source_files`, `language` (`c`/`cpp`), `function`, `include_dirs`, `defines`, `std`, `extra_clang_args`, `output_file`, `timeout_sec`, `cwd` |
| `CgeistResult` | `polygeist_frontend.py` | `command`, `returncode`, `elapsed_ms`, `stdout`, `stderr`, `mlir_text` |
| `PolygeistFrontendError` | `polygeist_frontend.py` | exception with `command`, `returncode`, `stderr`, `stdout` |
| `MlirValidationResult` | `mlir_validator.py` | `ok`, `command`, `returncode`, `stdout`, `stderr` |
| `CounterexampleReport` | `replay_checker.py` | `iv_values`, `source_access_pattern`, `sketch_access_pattern`, `parse_errors`, `diverging_cells`, `max_abs_diff` (always None), `verdict_summary` |
| `CorpusFileResult`, `CorpusCompatibilitySummary` | `parser_corpus_compat.py` | per-file compatibility classes and counts |
| `EmissionError` | `emitter.py` | `ValueError` subclass raised on missing or inconsistent metadata |

# Worked Example: Matrix Multiply End to End

This chapter traces the committed fixture `tests/fixtures/matmul.mlir` through every stage. All values below were captured from the real code in September 2026.

## Input

```
func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      affine.for %k = 0 to 4 {
        %a = affine.load %A[%i, %k] : memref<4x4xf32>
        %b = affine.load %B[%k, %j] : memref<4x4xf32>
        %c = affine.load %C[%i, %j] : memref<4x4xf32>
        %mul = arith.mulf %a, %b : f32
        %add = arith.addf %c, %mul : f32
        affine.store %add, %C[%i, %j] : memref<4x4xf32>
      }
    }
  }
  return
}
```

## Stage 1: extraction

The extractor scans the text line by line (after stripping `//` comments). It finds the function declaration, three `affine.for` lines, three loads, one store and two `arith` operations:

```
func_name        = matmul
induction_vars   = ['i', 'j', 'k']
bounds           = {'i': (0, 4), 'j': (0, 4), 'k': (0, 4)}
tensor_shapes    = {'%A': [4, 4], '%B': [4, 4], '%C': [4, 4]}
element_type     = f32
reads            = %A[i, k] -> %a,  %B[k, j] -> %b,  %C[i, j] -> %c
writes           = %C[i, j] <- %add
compute_ops      = mulf(%a, %b) -> %mul,  addf(%c, %mul) -> %add
parallel_vars    = ['i', 'j']        (they appear in the write index)
reduction_vars   = ['k']             (it does not)
has_accumulation = True, accumulation_op = addf   (C is read and written, and %add is stored)
```

## Stage 2: candidate matching

For the `linalg` target the lifter iterates over the 22 Linalg sketches. `structural_match` keeps sketches with 3 loops, 1 reduction IV, and a payload compatible with "has multiply and add". Two survive: `linalg.conv_1d_ncw_fcw` (3 loops `n, w, kw`, reduction `kw`) and `linalg.matmul`.

`SympyTracer.trace` builds a symbolic expression by substituting reads into the op chain, subtracting the old output value for accumulations and summing over `k = 0..3`:

```
A[i, 0]*B[0, j] + A[i, 1]*B[1, j] + A[i, 2]*B[2, j] + A[i, 3]*B[3, j]
```

Both candidates score confidence **1.00**. The scoring cannot tell a 1-D convolution from a matrix multiply because it looks at counts and payload categories, not at which indices appear where. The convolution recogniser returns nothing (no `+` in any index), so the tie is broken by library order and `linalg.conv_1d_ncw_fcw` is first.

## Stage 3: verification

The lifter checks candidates in order. The convolution sketch's indexing maps do not fit (its input is indexed `(n, w + kw)`), so it is not proved, and the strided-convolution fallback does not apply because all coefficients are 1. Then `linalg.matmul` is checked.

For matmul the checker first aligns sketch dimensions to source IVs by role: parallel dims `m, n` map to `i, j` in order, reduction dim `k` maps to `k`. Every tensor becomes a Z3 uninterpreted function from integers to reals. Because all bounds are concrete and the reduction has 4 iterations (at most 32), the reduction is unrolled into an explicit sum. The two formulas built are:

```
source: ForAll([i, j], Implies(0 <= i < 4 and 0 <= j < 4,
          C(i, j) == 0 + A(i,0)*B(0,j) + A(i,1)*B(1,j) + A(i,2)*B(2,j) + A(i,3)*B(3,j)))
sketch: ForAll([i, j], Implies(0 <= i < 4 and 0 <= j < 4,
          C(i, j) == 0 + A(i,0)*B(0,j) + A(i,1)*B(1,j) + A(i,2)*B(2,j) + A(i,3)*B(3,j)))
```

Z3 is asked whether `not (source implies sketch)` is satisfiable, and then the reverse. Both are `unsat`, so the report is `EQUIVALENT` ("Both implications proved UNSAT - formally equivalent."), in about 15 ms.

> [!WARN] Look at where the source formula came from
> The source formula above was *not* built by following `%a`, `%b`, `%mul`, `%add`. It was built by inferring the payload category `MULTIPLY_ACCUMULATE` from the presence of `mulf` and `addf`, then multiplying "the first non-output read" by "the second". For this kernel that happens to coincide with the real computation. For `C += 2*A*B` or `C = B - A` it does not, and the proof still succeeds. See the verification architecture chapter and Part 5, F-01.

## Stage 4: emission

`LinalgEmitter` looks up `linalg.matmul` in its handler table, checks that element type, input and output tensors and rank-2 shapes are known, and fills a template:

```
// LoopHole: Automatically lifted from scalar loop nest
// Source: matmul -> linalg.matmul
// Description: C[m,n] += A[m,k] * B[k,n]  - dense matrix multiplication
// Generated by LoopHole POC v0.1.0

module {
  func.func @lifted_matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
    linalg.matmul
      ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
      outs(%C : memref<4x4xf32>)
    return
  }
}
```

`mlir-opt` 17 accepts this. For the StableHLO target the same kernel becomes a function returning a fresh tensor, `%result = stablehlo.dot_general %A, %B, contracting_dims = [1] x [0]`, which drops the old contents of `C` (Part 5, F-07) and cannot be validated with the image's `mlir-opt`.

## The same kernel from real C code

`loophole compile examples/smoke_matmul.c` produces the following (the long `dlti` module attribute is shortened):

```
module attributes {dlti.dl_spec = ...} {
  func.func @matmul_2x2(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>) attributes {llvm.linkage = ...} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        %0 = affine.for %arg5 = 0 to 2 iter_args(%arg6 = %cst) -> (f32) {
          %1 = affine.load %arg0[%arg3, %arg5] : memref<2x2xf32>
          %2 = affine.load %arg1[%arg5, %arg4] : memref<2x2xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %arg6, %3 : f32
          affine.yield %4 : f32
        }
        affine.store %0, %arg2[%arg3, %arg4] : memref<2x2xf32>
      }
    }
    return
  }
}
```

Differences from the fixture: numbered SSA names, the accumulator is a loop block argument (`iter_args`) rather than a load of `C`, and the result leaves the inner loop through `affine.yield`. The parser does not model `iter_args`; the kernel still lifts to `linalg.matmul` (PROVED, 8 ms) because the store index lacks `arg5` (so it is a reduction) and the op set contains a multiply and an add. That the right answer comes out is again a consequence of the category-based encoding, not of the parser understanding the accumulator.

# Lifter Decision Flow and Result States

::diagram lift_decision

## Candidate verification order

`_verify_candidates` walks the first `top_k` (default 3) candidates in ranking order:

1. Run `Z3EquivalenceChecker.check(loop, sketch)` and annotate SymPy/Z3 disagreement.
2. If `EQUIVALENT`, return immediately with this candidate.
3. If `TIMEOUT` or `UNKNOWN`, remember the first such candidate as the timeout fallback.
4. If `NOT_EQUIVALENT` and the sketch is a convolution, build a strided variant (coefficients from the input index expressions) and check again; an `EQUIVALENT` result returns the *original* sketch with the strided report. Remember the first refuted candidate.
5. Any other result (`ENCODE_ERROR`, `STRUCTURAL_MISMATCH`) is kept in `no_verdict`.
6. After the loop, `best` is the timeout fallback if any, otherwise the refuted candidate, otherwise `None`. When it is `None`, `lift` sets `error` to "None of the N checked candidates reached a Z3 verdict." followed by each candidate's sketch, result and notes. When `best` is set, the `no_verdict` reports are not shown anywhere.

Because the first `EQUIVALENT` wins, **a wrong sketch that proves beats a right sketch that is refuted**. Combined with an unsound encoding, this ordering is how the `elementwise_mul` fixture ends up as a scale operation (Part 5, F-02).

## Acceptance and exit codes

| Situation | `result_state` | `success` | `partial_success` | `lift` exit | `batch` counts as | `batch` exit (strict) | `demo` (strict) |
|---|---|---|---|---|---|---|---|
| EQUIVALENT and emitted | PROVED | True | False | 0 | proved | 0 | ok |
| TIMEOUT/UNKNOWN, not strict, emitted | UNPROVED_TIMEOUT | False | True | 0 | unproved_timeout | n/a | n/a |
| TIMEOUT/UNKNOWN, strict | UNPROVED_TIMEOUT | False | False | 1 (error set) | unproved_timeout | 1 | exit 1 |
| NOT_EQUIVALENT best candidate | REFUTED | False | False | 1 | refuted | 1 | exit 1 |
| No candidates or all ENCODE_ERROR | REFUTED | False | False | 1 | refuted | 1 | exit 1 |
| Emission error after a proof | PROVED | False | False | 1 | proved (no artifact) | 0 | ok if not refuted |
| Artifact validation failed | any | any | any | n/a | `validation_failures` | 1 (always) | n/a |

Two contract problems stand out. A lift that was proved but could not be emitted counts as PROVED in batch totals. The code path that makes `lift` exit 2 for a strict unproved result is effectively unreachable, because the lifter already rejected it and set an error (Part 5, F-13).

# Configuration: Profiles, Flags and Environment

## Policy profiles

| Profile | Alias | `strict_mode` | `z3_timeout_ms` | Extra behaviour |
|---|---|---|---|---|
| `local-explore` (default) | `exploratory` | False | 10,000 | none |
| `ci-strict` | `trusted` | True | 15,000 | `batch` forces `--validate-emitted` |

Resolution (`resolve_policy_profile`): explicit `--profile` unless it is `default`; else `LOOPHOLE_POLICY_PROFILE`; else `local-explore`. Aliases are applied after resolution. Explicit flags override profile values in `Lifter.from_policy_profile`, but the CLI's own defaults interfere: `lift` and `lift-c` always pass `--z3-timeout` (default 10,000), so the `ci-strict` 15,000 ms timeout never applies to them, while `batch` and `demo` pass `None` and do get it.

## Environment variables

| Variable | Read by | Effect |
|---|---|---|
| `LOOPHOLE_POLICY_PROFILE` | `lifter.resolve_policy_profile`, `conftest.py` | Default profile; `ci-strict`/`trusted` also makes artifact validation mandatory in tests |
| `LOOPHOLE_CGEIST_DOCKER_IMAGE` | `polygeist_frontend` | Image for `cgeist` and `lift-c` |
| `LOOPHOLE_MLIR_DOCKER_IMAGE` | `mlir_validator`, and `polygeist_frontend` as a fallback | Image for Docker `mlir-opt` |
| `LOOPHOLE_DISABLE_DOCKER_CGEIST` | `polygeist_frontend` | `1`/`true`/`yes` disables the Docker cgeist path |
| `LOOPHOLE_DISABLE_DOCKER_MLIR_VERIFY` | `mlir_validator` | Disables the Docker verifier; falls through to PATH, then to the internal fake verifier |
| `LOOPHOLE_MLIR_VERIFY_CMD` | `mlir_validator` | Explicit verifier command (first word must be on PATH) |
| `LOOPHOLE_REQUIRE_MLIR_VERIFY` | `conftest.py` only | Tests fail instead of skipping when no external verifier exists |
| `LOOPHOLE_DOCKER_IMAGE` | `cli.batch` report metadata only | Recorded in `run_metadata.docker_image`; set by the PowerShell wrappers |
| `LOOPHOLE_DEMO_Z3_TIMEOUT_MS`, `LOOPHOLE_DEMO_FIXTURE_LIMIT` | `examples/run_demo.py` | Speed up the standalone demo |
| `PYTHONUTF8` | Python | Needed on Windows to avoid `charmap` decode errors on Docker output |

There is no single configuration object. Profile, flags and variables are resolved separately in `cli.py` (per command), `lifter.py`, `mlir_validator.py`, `polygeist_frontend.py` and `conftest.py`, which is why defaults drift between commands.

# Verification Architecture

## Overall encoding

`Z3EquivalenceChecker.check(loop, sketch)`:

1. Resets the per-check symbol cache and runs `structural_match` again (returns `STRUCTURAL_MISMATCH` if it fails).
2. Calls `_verify`, which picks `_verify_concrete` if every bound is an integer, else `_verify_symbolic`. **`_verify_symbolic` just calls `_verify_concrete`** inside exception handlers that convert unsupported reduction forms to ENCODE_ERROR and `ValueError` to UNKNOWN. There is one encoding path.
3. `_verify_concrete` creates one uninterpreted function `Int^rank -> Real` per tensor (or a `Real` constant for rank 0), one `Int` per IV, then builds:
   - **source assertions** (`_build_source_assertions`): `ForAll(parallel IVs, bounds implies out(write index) == rhs)`, or a plain equality when there are no parallel IVs;
   - **sketch assertions** (`_build_sketch_assertions`): the same shape using the sketch's indexing maps after aligning sketch dimensions to source IVs;
   - **domain assumptions** for symbolic bounds (`_build_symbolic_shape_assumptions`): for each access of the form `iv + c` into a dimension of static extent `E`, assume `lo + c >= 0` and `hi + c <= E`.
4. Checks `not(assumptions and source implies assumptions and sketch)` and the reverse, each with the configured timeout. `sat` gives NOT_EQUIVALENT with a model; `unknown` gives TIMEOUT; anything else non-`unsat` gives UNKNOWN.

::diagram verification_gap

## How the right-hand side is built

This is the heart of the soundness issue and worth understanding precisely.

- **Source side.** `_build_rhs_expr` calls `_infer_compute_payload(loop)`, which looks only at the set of `op_type` strings: multiply and add with reductions gives `MULTIPLY_ACCUMULATE`, multiply and add without gives `MULTIPLY`, add gives `ACCUMULATE_ADD`/`ADD`, subtract gives `SUBTRACT`, max gives `ACCUMULATE_MAX`/`MAX`, everything else `COPY`. The expression is then that payload applied to the first one or two reads that are not the output tensor, in textual order. Constants, the order of operands inside `subf`, additional reads, `divf`, `negf` and the actual SSA dataflow are never consulted.
- **Sketch side.** `_build_sketch_rhs` applies the sketch's declared payload to the operands selected by its indexing maps. Operand *i* of the sketch is bound to the *i*-th unique non-output read tensor of the source.

So both sides are "payload applied to reads at indices"; they differ only in which indices and which payload label. The proof really establishes "the index pattern of the source matches the sketch, assuming the source computes what its op-type set suggests".

## Reduction strategies

| Strategy | Used when | How | Limits |
|---|---|---|---|
| Concrete unroll (`_unrolled_reduction`) | All reduction bounds are integers and each range is at most `unroll_threshold` (32) | Python loop builds `term(k=lo) + ... + term(k=hi-1)` | Product of ranges grows fast (a 3x3x2 conv is 18 terms per output) |
| Guarded symbolic unroll (single IV) | One reduction IV, symbolic upper bound, maximum inferable from a static tensor extent | Sum up to the inferred maximum with `If(k < hi, total + term, total)` | Proves only for bounds up to the static extent |
| Guarded multi-reduction unroll (A-13) | Several reduction IVs, each upper bound inferable, total iterations at most 32 squared | Cartesian product with conjunctive guards | Same; cartesian blow-up guard |
| Recursive function (`_recfunc_reduction`) | Single reduction IV, not inferable or larger than 32 | `RecFunction` with `acc(k) = If(k == lo, 0, acc(k-1) + body(k-1))` | Multi-reduction raises `_UnsupportedReductionForm` |

Max reductions start from the constant `-1e30` instead of a true minus infinity (F-07).

## Dimension alignment

`_align_dims` maps sketch parallel dimensions to source parallel IVs *in order*, and likewise for reduction dimensions. There is no search over permutations. If a source loop nest is written `j, i, k` (outer loop over columns), alignment maps `m` to `j` and the proof fails or, for symmetric shapes, may succeed with a transposed meaning. Loop order sensitivity is therefore part of the matching behaviour.

## Parametric proof pilot (A-15)

With `--parametric`, after a concrete proof the lifter calls `_verify_parametric`: every distinct concrete upper bound value (at most 32) is replaced by a symbol (`M`, `N`, `K`, ...) and `_verify_concrete` runs again, which then takes the guarded-unroll paths. Equal bounds share a symbol and tensor shapes stay concrete, so the result covers "all square sizes up to the fixture size" rather than "all shapes" (F-14).

## Numeric semantics

All values are reals. There is no modelling of IEEE-754 (rounding, non-associativity, NaN, signed zero), of integer width or overflow, or of out-of-bounds reads. Any use of the word "proved" externally must carry that qualification.

# Emission Architecture

## Handler tables

Each emitter has a `_NAMED_OP_HANDLERS` dict from sketch name to method name. `emit()` validates metadata (element type present, tensor shapes present, enough input tensors, input and output shapes non-empty), calls the handler to produce the `func.func` text, and wraps it with a comment header and `module { ... }`. An unmapped Linalg sketch raises `EmissionError`; an unmapped StableHLO sketch raises as well. The `linalg.generic` fallback (`_emit_generic`) is therefore unreachable.

## Types and semantics

| Aspect | Linalg emitter | StableHLO emitter |
|---|---|---|
| Argument types | `memref<...>` (copied from the source) | `tensor<...>` |
| Output | Written into an `outs(...)` argument, function returns nothing | Returned as a fresh value, output buffer not an argument |
| Accumulation into existing output | Named ops like `linalg.matmul` add to `outs` (matches `C +=`) | Dropped: result ignores old `C` |
| Dynamic dims | `?` in `memref` | `?` in `tensor` |
| Convolution shapes | Rank 2 (1-D NCW), 2 (simple 2-D), 4 (NHWC, NCHW) | Reshaped to rank-4 NHWC/HWIO from (2,1,2), (2,2,2) or (4,4,4) |
| Attribute inference | Strides and dilations from linear coefficients of input index expressions (`_infer_linear_coeff`, SymPy or regex fallback) | Same helpers |
| Transpose permutation | From positions of IVs in the write index; validated against shapes | Same logic, duplicated |

## Validity

Only some templates produce valid MLIR for `mlir-opt` 17: matmul, matvec, transpose, conv_2d, conv_2d_nhwc_hwcf and copy were accepted; map, reduce, dot and conv_1d_ncw_fcw were rejected; StableHLO cannot be checked with that tool. Handlers that no fixture reaches (vecmat, batch_matmul, conv_1d_nwc_wcf, conv_2d_nchw_fchw, pooling_nhwc_max, map for relu/scale/1-D add) are untested against a real parser. See the per-handler table in Part 3.

## The missing link between sketch and emitter

The sketch defines what an operation means for verification. The emitter independently decides what text to write for the same name. Nothing checks that the emitted operation actually has the semantics encoded in the sketch (for example, that `linalg.conv_1d_ncw_fcw` indexes its input as `(n, w + kw)` with rank 2, which it does not). A structural improvement would be to validate emitted artifacts with the real toolchain and a numerical oracle for every sketch, as proposed in Part 5.

# Frontend and Validation Architecture

## Two separate paths into Polygeist

| Aspect | `loophole compile` (`cli.compile_cmd`) | `loophole cgeist` / `lift-c` (`PolygeistFrontend`) |
|---|---|---|
| Invocation | `docker run ... bash -lc "cgeist ... | sed ... | mlir-opt --canonicalize > out"` | `cgeist` on PATH first (unless `prefer_docker`, which both commands set), otherwise `docker run ... cgeist ...` |
| Default image | `ghcr.io/schizoid-man/loophole-polygeist:llvm17` (hard-coded default) | explicit flag, then `LOOPHOLE_CGEIST_DOCKER_IMAGE`, then `LOOPHOLE_MLIR_DOCKER_IMAGE`, then first local image of `loophole-polygeist:llvm17` or the GHCR name |
| Function selection | Always `--function=*` | `--function` option |
| Post-processing | `sed` rewrites legacy `-1` dims to `?`, then `mlir-opt --canonicalize` | None |
| Timeout | None | `--timeout-sec` (default 60) |
| Include dirs, defines, clang args | Not supported | Supported and mapped into the container |
| Mount | Common parent of source and output | Common parent of sources, includes and output |

Both require all paths on one drive (single volume mount on Windows).

## Verifier discovery

`find_mlir_verifier(preferred)`:

1. An explicit `--mlir-verifier` command if its executable is on PATH, else `None`.
2. `LOOPHOLE_MLIR_VERIFY_CMD` if its executable is on PATH.
3. The Docker verifier sentinel `loophole-docker-mlir-verify`, if Docker is running and the image exists (unless disabled by environment variable).
4. `mlir-opt`, `mlir-opt-18`, `mlir-opt-17` on PATH.
5. The internal sentinel `loophole-internal-mlir-verify`, a brace-and-keyword check.

`validate_mlir_artifact` dispatches on the sentinel: Docker runs `mlir-opt /work/artifact.mlir` in a container (timeout at least 30 s, errors captured); internal runs the text checks; anything else runs the command with the artifact path appended (a timeout here raises out of `batch`). There is no per-dialect selection: StableHLO goes to `mlir-opt`, which does not know it.

# Module Dependency Graph

::diagram module_deps

Layering, from bottom to top:

1. **Leaves:** `affine_extractor.py` (data model and parser) and `sketch_library.py` (sketches). Neither imports other project modules.
2. **Engines:** `sympy_tracer.py`, `z3_checker.py` and `emitter.py`, each importing both leaves. `lifter.py` additionally imports a private helper `_infer_linear_coeff` from `emitter.py` for strided convolution sketches, which couples the orchestrator to emitter internals.
3. **Orchestration:** `lifter.py`; `replay_checker.py` (imports `z3_checker` types); `parser_corpus_compat.py` (imports the extractor).
4. **Surfaces:** `cli.py` (imports lifter, frontend, validator, sketch library, and lazily replay checker, extractor and test fixtures); scripts (import lifter and each other).
5. **Tooling adapters:** `polygeist_frontend.py` and `mlir_validator.py`, which only use the standard library.

There are no import cycles. The main coupling problems are the CLI's dependency on `loophole.tests.fixtures`, the lifter's import of an emitter private function, the script importing another script's private helpers (`generate_proof_quality_summary.py` imports underscore functions from `generate_weekly_benchmark_report.py`), and `conftest.py` sitting at package root outside `tests/`.

# Architectural Drawbacks Summary

| Drawback | Consequence | Part 5 reference |
|---|---|---|
| Parser output lacks steps, dataflow, loop nesting and function boundaries | Later stages cannot be correct even if they try | F-03, F-04, F-05, F-20, F-21 |
| Verifier re-derives source semantics from op-type sets | False proofs | F-01, F-02, F-18 |
| First-EQUIVALENT-wins ordering with heuristic ranking | Wrong sketch can beat right one | F-02, F-19 |
| Sketch semantics and emitter templates not cross-checked | Invalid or mismatched artifacts | F-08 |
| Text-based parsing and emission instead of MLIR APIs | Fragile to formatting and real frontend output | F-16, F-34 |
| One verifier for all dialects, with a fake fallback | Gates that cannot fail | F-09, F-10 |
| Result states conflate errors with refutations | Misleading metrics | F-12, F-28 |
| Configuration resolved in many places | Inconsistent defaults | F-24 |
| Two Polygeist invocation paths | Different outputs for the same source | F-23 |
| Test data and demo data in the runtime package | Packaging and layering confusion | F-26 |
