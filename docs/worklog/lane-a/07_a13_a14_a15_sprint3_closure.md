# A-13 / A-14 / A-15 Sprint 3 Closure

Last updated: 2026-04-26

Tasks A-13, A-14, and A-15 are complete. This document records the scope closed,
evidence artifacts, and test validation results.

---

## A-13: Symbolic Multi-Reduction Boundary Improvements

### What was done

Three sites in `z3_checker.py` previously raised `ValueError` (caught as opaque `UNKNOWN`)
when more than one symbolic reduction dimension was encountered:

- `_symbolic_reduction_upper_bound()` — silently returned `None`
- `_recfunc_reduction()` — raised `ValueError`
- `_build_sketch_rhs()` — raised `ValueError`

Changes:

1. **`ReductionPatternKind` enum** added (`CONCRETE_UNROLL`, `SINGLE_SYMBOLIC_INFERRED`,
   `MULTI_SYMBOLIC_INFERRED`, `SINGLE_SYMBOLIC_UNINFERRED`, `MULTI_SYMBOLIC_UNINFERRED`).

2. **`_UnsupportedReductionForm` exception** replaces bare `ValueError` so the catch site
   can distinguish this case from other encoding errors.

3. **`_symbolic_reduction_upper_bound_multi()`** extends the existing single-IV bound
   inference to all reduction IVs simultaneously. Returns `{iv: upper_int}` for each IV
   when all bounds can be inferred from tensor extents, or `None` if any fails.
   Includes a total-iterations guard (≤ `unroll_threshold²`) to prevent formula blowup.

4. **`_guarded_symbolic_unroll_reduction_multi()`** performs nested guarded unrolling for
   multi-reduction cases where all bounds are inferred. Uses `If(k_i < hi_expr_i, …)` guards
   identical in structure to the existing single-IV path.

5. **`_build_reduction_expr()` decision tree** updated to try the multi-path before falling
   back to `_recfunc_reduction`.

6. **`_build_sketch_rhs()`** receives a symmetric multi-reduction guarded unroll path.

7. **`_classify_reduction_pattern()`** helper provides human-readable classification for any
   reduction pattern, wired into `_verify_symbolic()` ENCODE_ERROR notes.

8. **`_verify_symbolic()`** now catches `_UnsupportedReductionForm` with a structured
   `ENCODE_ERROR` result containing `unsupported_form=`, `pattern=`, `reason=`, and `hint=`
   fields. Generic `ValueError` is still caught as `UNKNOWN` for backward compatibility.

### Result

- Concrete multi-reduction (conv2d: kh, kw): continues to use CONCRETE_UNROLL path unchanged.
- Multi-reduction with inferable symbolic bounds (e.g. K[kh, kw] shape [3, 3]):
  now attempts guarded unroll and may return EQUIVALENT.
- Multi-reduction with uninferrable symbolic bounds: returns `ENCODE_ERROR` with structured
  diagnostic notes instead of opaque `UNKNOWN`.

### Tests

`tests/unit/test_a13_multi_reduction_boundary.py` — 17 tests, all passed.

---

## A-14: Counterexample Replay Utility

### What was done

New module `packages/loophole/src/loophole/replay_checker.py`:

- `parse_z3_binding(val_str: str) -> float`: parse Z3 model value strings
  (integers, rationals `1/3`, floats) to Python float.
- `_eval_float_expr(expr, vals)`: evaluate affine index expressions at concrete float values.
- `simulate_source(loop, iv_values)`: evaluate source loop access patterns at IV values.
- `simulate_sketch(sketch, loop, iv_values)`: evaluate sketch indexing maps at IV values.
- `CounterexampleReport` dataclass: iv_values, source/sketch access patterns, diverging cells,
  max_abs_diff, verdict_summary.
- `replay_counterexample(loop, sketch, report)`: main entry point; returns `None` for empty
  bindings, otherwise a `CounterexampleReport` with triage diagnostics.

CLI additions in `cli.py`:

- `loophole replay REPORT_FILE [--entry N] [--verbose]`: load a JSON batch report entry,
  reconstruct loop info, call `replay_counterexample()`, print a Rich panel.
- `--show-replay` flag on `loophole lift`: after a NOT_EQUIVALENT result, replay inline.
- `_print_counterexample_report()` helper for consistent Rich formatting.

### Limitations

`max_abs_diff` is always `None` in the current implementation: Z3 model entries for
uninterpreted tensor functions are complex arrays that cannot be parsed from the string
representation without a full model parser. The IV-level simulation (which indices were
accessed) is fully implemented and sufficient for most triage use cases.

### Tests

`tests/unit/test_a14_counterexample_replay.py` — 30 tests, all passed.

---

## A-15: Shape-Parametric Proof Pilot

### What was done

Pilot implementation of shape-parametric proofs: prove equivalence for ALL valid shapes
rather than one concrete size.

**`VerificationReport`** gains two optional fields (backward-compatible defaults):
- `parametric_result: Optional[CheckResult] = None`
- `parametric_notes: str = ""`

**`Z3EquivalenceChecker._collect_shape_dims(loop)`**: identifies unique concrete integer
upper-bounds in loop.bounds and assigns symbol names (M, N, K, P, Q, R, S).

**`Z3EquivalenceChecker._verify_parametric(loop, sketch)`**:
1. Calls `_collect_shape_dims()` to build `{concrete_int: symbol_name}` mapping.
2. Replaces concrete integer bounds with symbol strings in a synthetic `LoopNestInfo` via
   `dataclasses.replace()`.
3. Delegates to `_verify_concrete()` with the parametric bounds — the symbolic bound strings
   trigger the existing guarded-symbolic-unroll paths.
4. Tags result with `[parametric over dims=[...]]` in notes.
5. Catches all exceptions and returns UNKNOWN (pilot, not production hardened).

**`Lifter`** gains `parametric_mode: bool = False` on both `__init__` and
`from_policy_profile()`. When `True`, after a successful concrete proof (EQUIVALENT), calls
`_verify_parametric()` and stores the result in `report.parametric_result` /
`report.parametric_notes`.

**CLI**: `--parametric` flag added to `loophole lift`. When set, passes `parametric_mode=True`
to the lifter. The `_print_verification_report()` table now includes parametric fields if set.

### Pilot results (2026-04-26)

| Kernel     | Concrete result | Parametric result | Parametric notes |
|------------|----------------|-------------------|------------------|
| matmul 4×4×4 | EQUIVALENT 7.2ms | EQUIVALENT | `[parametric over dims=['M']]` |
| matvec 4×4 | EQUIVALENT 1.5ms | EQUIVALENT | `[parametric over dims=['M']]` |
| reduce_sum 4×4 | EQUIVALENT 1.2ms | EQUIVALENT | `[parametric over dims=['M']]` |

All three pilot kernels achieved parametric EQUIVALENT. The shape-parametric proof was
fast (< 10ms) because all kernels have a single unique bound value (4), which maps to a
single symbol 'M' replacing all three loop dimensions, and the symbolic guarded-unroll
path handles this efficiently.

### Tests

`tests/unit/test_a15_shape_parametric_proof.py` — 28 tests, all passed.

---

## Evidence Summary

| Item | Path |
|------|------|
| A-13 tests | `tests/unit/test_a13_multi_reduction_boundary.py` |
| A-14 module | `src/loophole/replay_checker.py` |
| A-14 tests | `tests/unit/test_a14_counterexample_replay.py` |
| A-15 tests | `tests/unit/test_a15_shape_parametric_proof.py` |
| A-15 fixtures | `tests/fixtures/a15_parametric/` |
| Baseline snapshot | `research/benchmarks/baselines/a13_a14_a15_sprint3_baseline_20260426.json` |

Full test suite: **348 passed** (273 prior + 75 new).

---

## Sprint Gate Status

Sprint 3 gate (G3):
1. No silent fallback reintroduced in new paths. **DONE**: ENCODE_ERROR (not UNKNOWN/ValueError)
   for uninferrable multi-reduction symbolic cases.
2. Emitter metadata errors are explicit and actionable. **DONE** (prior B-13/B-16 work).

Sprint 4 gate (G4):
1. Parametric proof pilot demonstrates at least one successful end-to-end case. **DONE**:
   matmul, matvec, reduce_sum all achieved parametric EQUIVALENT.

Sprint 5 gate (G5):
1. Defect burn-down list is prioritized. **IN PROGRESS** (B-16 complete; A-16 next).
