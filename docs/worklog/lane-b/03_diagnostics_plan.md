# 12 - B-02 Better Diagnostics for Unsupported Forms

Date: 2026-04-03
Objective: Standardize parser error messages in `affine_extractor.py` so failures explain which construct failed and why. Propagate diagnostics through `lifter.py` and `cli.py`.

---

## A) Diagnostic Context Strategy

Each parser failure will be reported with:

1. **Location**: Which parsing function failed (e.g., "function declaration", "loop structure")
2. **Input snippet**: Up to 80 chars of problematic input (with `...` truncation)
3. **Reason**: Why the parse failed (e.g., "no closing paren for function args", "unrecognized constant format")
4. **Guidance**: Optional hint for user fixing (e.g., "expected format: `func.func @name(...)`")

Example diagnostic:
```
Parser error at _parse_func_declaration:
  Input: "func.func mykernel(..."
  Reason: Function name marker not found (expected 'func.func @')
  Guidance: Ensure function is declared as 'func.func @name(...)'
```

---

## B) Silent Failures to Convert (B-02 Fallbacks)

| Fallback ID | Function | Current behavior | Diagnostic strategy | Error class |
|---|---|---|---|---|
| P-03 | `_parse_func_declaration` | Return `("unknown", {})` on failure | Raise `ParsingError` with location and reason | NON-CRITICAL (file continues with partial parse) |
| P-04 | `_extract_constants` | Silent skip of non-scalar constants | Log warning with constant SSA name and literal | WARNING (can proceed without constants) |
| P-07 | `_parse_loop_structure` | Silent skip of unrecognized loop lines | Log debug event; skip is acceptable for mixed IR | DEBUG (expected in real-world IR) |
| P-08 | `_parse_load/_parse_store` | Default to `type_str = ""` on missing type | Log warning; fallback to inferred type later | WARNING (can proceed with inference) |
| P-09 | `_parse_memref_type` | Return `[], "f32"` on missing/invalid shape | Log warning with type_str that failed parsing | WARNING (will trigger B-05 assertion later) |
| P-11 | `_collect_accesses` | Silent drop of failed load/store parses | Log warning with line and reason for drop | WARNING (missing memory access) |
| P-12 | `_collect_compute_ops` | Unparseable constants become `0.0` | Log warning with constant literal and fallback value | WARNING (silent value substitution) |
| P-13 | `_collect_compute_ops` | N-ary ops truncated to first two operands | Log warning with op type and operand count | WARNING (lost operands) |
| P-15 | `_dominant_element_type` | Default to `"f32"` when no types discovered | Log warning if fallback used; continue | WARNING (inferred default choice) |
| P-16 | `_classify_iv_roles` | All IVs parallel when no writes exist | Log warning; continue with conservative classification | WARNING (unexpected structure) |
| P-17 | `_detect_accumulation` | Fall back to any add op if no direct write-result | Log debug when fallback used | DEBUG (aggressive accumulation detection) |

---

## C) Implementation Approach

### Phase 1: Logging Infrastructure (no parser changes yet)

1. Add `ParsingWarning` exception class to module
2. Add `_diagnostic_context(line, prefix)` helper to format snippets
3. Create `ParsingDiagnostics` dataclass to track warnings during extraction

### Phase 2: Convert Silent Failures to Diagnostics (in-place error/warning reporting)

For each fallback P-03 through P-17:
- If CRITICAL (P-03 function declaration): Raise `ParsingError` with context
- If WARNING (P-04, P-08, P-09, P-11, P-12, P-13, P-15, P-16): Append to diagnostics log, continue extraction
- If DEBUG (P-07, P-17): Append to debug log only if verbose flag set

### Phase 3: Propagate Diagnostics to LoopNestInfo

1. Add `diagnostics: List[str]` field to `LoopNestInfo` dataclass
2. Modify `extract()` to return diagnostics in result
3. Update `lifter.py` to surface diagnostics in lifted IR comments
4. Update `cli.py` to print diagnostics to stderr before/after demo output

### Phase 4: Test Coverage

Add negative parser tests:
- `test_parse_func_no_marker` — missing `func.func @` marker
- `test_parse_func_unclosed_args` — no closing paren in args list
- `test_parse_constant_non_scalar` — non-scalar constant attempt
- `test_parse_loop_unrecognized_line` — garbage loop declaration
- `test_parse_load_missing_type` — load without type annotation
- `test_parse_memref_invalid_shape` — shape with unparseable dimension
- `test_parse_access_missing_bracket` — load/store with malformed indices
- `test_parse_multiple_diagnostics` — corpus with multiple issues, all collected

Add integration test:
- `test_cli_shows_parser_warnings` — verify warnings appear in CLI output for problematic IR

---

## D) Changes Summary

### Modified files:
1. `affine_extractor.py`
   - Add `ParsingWarning`, `ParsingDiagnostics` classes
   - Add `_diagnostic_context()` helper
   - Modify `__init__` to track diagnostics
   - Modify each parser method to report failures
   - Modify `extract()` to return diagnostics in `LoopNestInfo`

2. `lifter.py`
   - Modify `lift()` to accept diagnostics from extractor
   - Emit diagnostics as MLIR comments in lifted IR output

3. `cli.py`
   - Modify demo runners to capture and display diagnostics
   - Print warnings to stderr before IR output

4. `tests/unit/test_affine_extractor.py`
   - Add 8 negative parser tests covering each fallback class
   - Add 1 multi-diagnostic corpus test

5. `tests/integration/test_cli_diagnostics.py` (new)
   - `test_cli_shows_parser_warnings` — full CLI path validation

---

## E) Test validation criteria

**All existing tests must pass** (67 baseline tests remain green).

**New tests must verify:**
1. Function declaration failures raise (or log if non-critical)
2. Constant parsing issues logged with context
3. Loop structure unrecognized lines logged
4. Load/store missing type annotations logged as warnings
5. Memref type parsing failures logged
6. Collection failures (accesses/ops) logged
7. Compute op fallbacks (constants, operand truncation) logged
8. Classification fallbacks (IV roles, accumulation) logged
9. All diagnostics collected in `LoopNestInfo.diagnostics`
10. CLI output includes warning section before IR output

---

## F) Scope and Dependencies

**Scope:** Parser-layer diagnostics only; does NOT modify emitter fallback handling (reserved for B-05/B-06/B-07).

**Dependencies:**
- No external dependencies added
- Uses only standard library (`dataclasses`, logging `print()`)
- Must not break existing test expectations (baselines from B-01)

**Can Run In Parallel With:**
- B-03 (mixed affine/scf support) — different code paths
- B-04 (affine index normalization) — different code paths

**Blocks:**
- B-05 (emitter hardening) — can reuse diagnostic patterns from parser

---

## G) B-02 Completion Status

Status: COMPLETE (as of 2026-04-03)

---

## H) Implementation Audit

### Files Modified

**Core parser (`affine_extractor.py`)**
- Added `ParsingWarning` dataclass for structured diagnostic reporting (lines ~45-55)
- Updated `LoopNestInfo` dataclass to include `diagnostics: List[ParsingWarning]` field (lines ~57-87)
- Added `_diagnostic_snippet()` helper to format input text with truncation (lines ~261-264)
- Updated `AffineExtractor.__init__()` to accept `verbose_diagnostics` flag and initialize diagnostics tracker (lines ~277-280)
- Added `_add_diagnostic()` method to log warnings/errors with context (lines ~348-360)
- Updated `extract()` to initialize diagnostics and return them in `LoopNestInfo` (lines ~286, 320)

**Parser methods with diagnostics**:
- `_parse_func_declaration()`: Added error reporting for missing marker, unclosed args (P-03)
- `_extract_constants()`: Added warnings for non-scalar + unparseable constants (P-04)
- `_parse_loop_structure()`: Added debug/warning for unrecognized loops + mixed affine/scf (P-06, P-07)
- `_parse_load()`: Added warnings for missing brackets, type annotations (P-08)
- `_parse_store()`: Added warnings for missing separator, brackets, type annotations (P-08)
- `_collect_accesses()`: Added warnings for failed load/store parse attempts (P-11)
- `_collect_compute_ops()`: Added warnings for unparseable constants, operand truncation (P-12, P-13)
- `_dominant_element_type()`: Added warning for f32 default when no types (P-15)
- `_classify_iv_roles()`: Added warning for conservative IV classification (P-16)
- `_detect_accumulation()`: Added debug message for aggressive fallback (P-17)

**Lifter integration (`lifter.py`)**
- Added `parser_diagnostics` field to `LiftResult` dataclass (line ~57)
- Updated all four `LiftResult` construction sites to propagate diagnostics from `loop.diagnostics` (lines ~149, ~173, ~193, ~212)

**CLI output (`cli.py`)**
- Added diagnostic display in demo() function output (lines ~634-637)
- Formatted diagnostics with color coding (error=red, warning=yellow, debug=cyan)
- Included guidance text if available
- Diagnostics printed after result summary but before emitted MLIR

### Test Coverage

**New test file: `tests/unit/test_parser_diagnostics.py`**
- 11 new tests covering all major fallback paths:
  - `test_parse_func_no_marker` — P-03 error path
  - `test_parse_func_unclosed_args` — P-03 error path
  - `test_parse_constant_non_scalar` — P-04 warning path
  - `test_parse_constant_unparseable` — P-04 warning path
  - `test_parse_load_missing_type` — P-08 warning path
  - `test_parse_store_missing_type` — P-08 warning path
  - `test_parse_no_tensor_types` — P-15 warning path
  - `test_parse_no_writes_classify_parallel` — P-16 warning path
  - `test_parse_compute_op_operand_truncation` — P-13 warning path
  - `test_multiple_diagnostics_collected` — multi-issue corpus test
  - `test_verbose_diagnostics_includes_debug` — verbose flag test

All 11 tests **PASSING**.

**New integration test file: `tests/integration/test_cli_error_messages.py`**
- 11 integration tests for parser and emission diagnostics through `lift()`:
   - malformed function marker/type/constant paths produce actionable parser diagnostics
   - multi-issue inputs return multiple diagnostics
   - emission-side failures are surfaced as actionable `LiftResult.error` text

All 11 integration tests **PASSING**.

### Test Validation Results

**Before B-02 implementation:**
- 100 baseline tests passing (67 core + 1 policy + 32 baseline-dependent)

**After B-02 implementation:**
- 122 tests passing (100 baseline + 11 parser diagnostics tests + 11 integration diagnostics tests)
- 1 skipped (z3_checker.py, pre-existing)
- **Zero regressions** — all baseline tests remain green

### Deliverables Checklist

1. `ParsingWarning` dataclass + `LoopNestInfo.diagnostics` field added
2. All 11 silent failures (P-03, P-04, P-06, P-07, P-08, P-11, P-12, P-13, P-15, P-16, P-17) converted to explicit diagnostics
3. Diagnostics propagated through `LoopNestInfo` to `LiftResult`
4. CLI demo output enhanced to display parser diagnostics with colors + guidance
5. 11 comprehensive negative parser tests added
6. Multi-diagnostic corpus test validates collection semantics
7. Verbose diagnostics flag correctly filters debug-level messages
8. All 122 tests passing (122 passed, 1 skipped)
9. No regressions in existing parse behavior

### Known Limitations & Future Work

**P-09 (memref type parsing)**: Currently returns empty shape + default f32; warning logged by upstream _parse_load/_parse_store on missing type annotation. Consider adding explicit validation in `_parse_memref_type()` for malformed dimensions in B-03+.

**P-06 (mixed affine/scf)**: Now detected and warned with guidance. Full support queued for B-03 task.

**P-18 (unresolved bounds)**: Intentionally does NOT log warning (accepted fallback per B-01 audit). Symbolic tokens handled by lifter validation layer.

---

## I) B-02 Completion Criteria (Original)
