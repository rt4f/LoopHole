# 11 - Parser/Emitter Fallback Audit (B-01)

Date: 2026-04-03
Scope: Complete fallback/default behavior inventory for:

- `POC_initial_demo/src/loophole/affine_extractor.py`
- `POC_initial_demo/src/loophole/emitter.py`

Classification legend:

- `SAFE`: acceptable default for robustness/readability
- `RISKY`: can hide correctness or diagnostics issues; acceptable short-term with explicit tracking
- `MUST-REMOVE`: should be eliminated by planned hardening tasks

Disposition legend:

- `Accepted`: keep as-is for now
- `Queued B-05`: remove silent shape/type defaults in emitter
- `Queued B-06`: improve transpose permutation inference/validation
- `Queued B-07`: infer/propagate convolution attrs with explicit policy
- `Queued B-02`: improve diagnostics, reduce silent ignores
- `Queued B-03`: mixed affine/scf support hardening

---

## A) Parser inventory (`affine_extractor.py`)

| ID | Location | Fallback/default behavior | Class | Disposition |
|---|---|---|---|---|
| P-01 | `_validate_mlir_syntax` | If MLIR bindings unavailable or validation disabled, parser proceeds without syntax validation. | RISKY | Accepted (documented runtime limitation) |
| P-02 | `_validate_mlir_syntax` | If input has no module wrapper, parser retries by wrapping text in `module { ... }`. | SAFE | Accepted |
| P-03 | `_parse_func_declaration` | Returns `("unknown", {})` if function marker/signature parse fails. | RISKY | Queued B-02 |
| P-04 | `_extract_constants` | Non-scalar/unparseable constants silently skipped. | RISKY | Queued B-02 |
| P-05 | `_allocate_iv_name` | Invalid/non-alpha IV names replaced with generated `iv{n}`. | SAFE | Accepted |
| P-06 | `_parse_loop_structure` | `scf.for` parsing only attempted before first loop is seen; mixed affine/scf nests can be under-parsed. | MUST-REMOVE | Queued B-03 |
| P-07 | `_parse_loop_structure` | Unrecognized loop lines silently ignored. | RISKY | Queued B-02 |
| P-08 | `_parse_load`/`_parse_store` | If type annotation missing, type string falls back to empty and downstream parser defaults element type. | RISKY | Queued B-02 |
| P-09 | `_parse_memref_type` | If memref payload parse fails or has no `x`, returns empty shape and default element type `f32`. | RISKY | Queued B-02 |
| P-10 | `_parse_memref_type` | Dynamic/unknown shape tokens represented as `-1`. | SAFE | Accepted |
| P-11 | `_collect_accesses` | Failed load/store parses are silently dropped. | RISKY | Queued B-02 |
| P-12 | `_collect_compute_ops` | Constants with unparseable literal become `0.0`. | RISKY | Queued B-02 |
| P-13 | `_collect_compute_ops` | For n-ary ops, operands truncated to first two (`operands[:2]`). | RISKY | Queued B-02 |
| P-14 | `_resolve_tensor_metadata` | Missing arg metadata filled from observed accesses. | SAFE | Accepted |
| P-15 | `_dominant_element_type` | If no types discovered, defaults dominant type to `f32`. | RISKY | Queued B-02 |
| P-16 | `_classify_iv_roles` | If no writes found, classifies all IVs as parallel and none as reduction. | RISKY | Queued B-02 |
| P-17 | `_detect_accumulation` | If direct write-result add not found, any add op can trigger accumulation classification. | RISKY | Queued B-02 |
| P-18 | `_resolve_bound` | Unresolved bounds return symbolic token strings instead of hard error. | SAFE | Accepted |

Parser summary:

- SAFE accepted: `P-02`, `P-05`, `P-10`, `P-14`, `P-18`
- RISKY tracked for diagnostics hardening: `P-01`, `P-03`, `P-04`, `P-07`, `P-08`, `P-09`, `P-11`, `P-12`, `P-13`, `P-15`, `P-16`, `P-17`
- MUST-REMOVE: `P-06` (mixed-loop gap)

---

## B) Emitter inventory (`emitter.py`)

### B1. Top-level and dispatch fallbacks

| ID | Location | Fallback/default behavior | Class | Disposition |
|---|---|---|---|---|
| E-01 | `LinalgEmitter.emit` | Unknown/unmapped sketch name falls back to `linalg.generic`. | RISKY | Accepted (with explicit tests) |
| E-02 | `StableHLOEmitter.emit` | Unknown stablehlo sketch falls back to dot_general emitter. | MUST-REMOVE | Queued B-05 |
| E-03 | `_map_elem_type` | Unknown element type passes through unchanged. | SAFE | Accepted |

### B2. Placeholder tensor names when inputs are missing

| ID | Location | Fallback/default behavior | Class | Disposition |
|---|---|---|---|---|
| E-04 | Multiple named emitters | Missing input tensors replaced with placeholders like `%A`, `%B`, `%I`, `%K`, `%x`, `%y`, `%c`, `%O`. | RISKY | Queued B-05 |

### B3. Guessed shape fallbacks in Linalg named emitters

| ID | Location | Fallback/default behavior | Class | Disposition |
|---|---|---|---|---|
| E-05 | `_emit_matmul` | Defaults to `A:[4,4], B:[4,4], C:[A0,B1]`. | MUST-REMOVE | Queued B-05 |
| E-06 | `_emit_transpose` | Defaults to `A:[4,4], B:[A1,A0]`. | MUST-REMOVE | Queued B-05 |
| E-07 | `_emit_dot` | Defaults `A:[16], B:[16], C:[1]` when missing/empty. | MUST-REMOVE | Queued B-05 |
| E-08 | `_emit_matvec` | Defaults `A:[8,8], x:[A1], y:[A0]`. | MUST-REMOVE | Queued B-05 |
| E-09 | `_emit_vecmat` | Defaults `x:[8], A:[8,8], y:[A1]`. | MUST-REMOVE | Queued B-05 |
| E-10 | `_emit_batch_matmul` | Defaults `A:[2,4,4], B:[2,4,4], C:[A0,A1,B2]`. | MUST-REMOVE | Queued B-05 |
| E-11 | `_emit_conv1d_ncw` | Defaults from inferred bounds and shape guesses `I:[N,Win], K:[KW], O:[N,Wout]`. | RISKY | Queued B-05/B-07 |
| E-12 | `_emit_conv1d_nwc` | Defaults inferred for `F`, `C`, then guessed `I/K/O` shapes. | RISKY | Queued B-05/B-07 |
| E-13 | `_emit_conv2d_simple` | Defaults inferred `KW=KH`; guessed `I/K/O` shapes. | RISKY | Queued B-05/B-07 |
| E-14 | `_emit_conv2d_nhwc` | Defaults inferred `OC=1`, `KW=KH`, `IC=1`; guessed `I/K/O` shapes. | RISKY | Queued B-05/B-07 |
| E-15 | `_emit_conv2d_nchw` | Defaults inferred `OW=1`, `KH=3`, `KW=3`; guessed `I/K/O` shapes. | RISKY | Queued B-05/B-07 |
| E-16 | `_emit_pool2d_max` | Defaults inferred `C=1`, `KW=KH`; guessed `I/O` shapes. | RISKY | Queued B-05/B-07 |
| E-17 | `_emit_elementwise` | Defaults `A:[8,8], B:[8,8], C:A`. | MUST-REMOVE | Queued B-05 |
| E-18 | `_emit_relu` | Defaults `A:[8,8], B:A`. | MUST-REMOVE | Queued B-05 |
| E-19 | `_emit_scale` | Defaults `A:[8,8], B:A`. | MUST-REMOVE | Queued B-05 |
| E-20 | `_emit_copy` | Defaults `A:[8,8], B:A`. | MUST-REMOVE | Queued B-05 |
| E-21 | `_emit_reduce_sum` | Defaults `A:[8,8]`, reduction dim heuristic, `B:[A(1-red)]` or `[1]`. | MUST-REMOVE | Queued B-05 |
| E-22 | `_emit_reduce_max` | Defaults `A:[8,8], B:[A0]`, hardcoded dimension `[1]`. | RISKY | Queued B-05 |
| E-23 | `_emit_generic` | Defaults each input/output memref shape to `[4,4]` when missing. | MUST-REMOVE | Queued B-05 |
| E-24 | `_build_generic_body` | Unknown compute payload defaults to yielding first input `%a0`. | MUST-REMOVE | Queued B-05 |

### B4. Permutation fallback

| ID | Location | Fallback/default behavior | Class | Disposition |
|---|---|---|---|---|
| E-25 | `_infer_transpose_perm` | If inference incomplete, defaults to reverse order permutation. | MUST-REMOVE | Queued B-06 |

### B5. Convolution/pooling attribute fallbacks

| ID | Location | Fallback/default behavior | Class | Disposition |
|---|---|---|---|---|
| E-26 | Linalg conv/pool emitters | Hardcoded `dilations = 1` and `strides = 1`. | MUST-REMOVE | Queued B-07 |
| E-27 | StableHLO convolution | Hardcoded window attrs: stride/pad/lhs_dilate/rhs_dilate constants. | MUST-REMOVE | Queued B-07 |

### B6. StableHLO shape/reduction defaults

| ID | Location | Fallback/default behavior | Class | Disposition |
|---|---|---|---|---|
| E-28 | `_emit_dot_general` | Defaults `A/B` to `[4,4]`; output shape heuristic `[A0,B1]` else `[A0,1]`. | MUST-REMOVE | Queued B-05 |
| E-29 | `_emit_convolution` | Uses inferred defaults for missing dims/shape metadata. | RISKY | Queued B-05/B-07 |
| E-30 | `_emit_reduce` | Defaults `A:[4,4]`, `B:[A0]`, fixed reduce dimension `[1]`, fixed zero init style. | RISKY | Queued B-05 |

### B7. Bound-size fallback helper (shared across emitters)

| ID | Location | Fallback/default behavior | Class | Disposition |
|---|---|---|---|---|
| E-31 | `_bound_size` | Missing IV or missing bound defaults to `(0,1)` size 1. | RISKY | Queued B-05 |
| E-32 | `_bound_size` | Non-integer bounds coerced to `lo=0, hi=1`. | RISKY | Queued B-05 |

Emitter summary:

- SAFE accepted: `E-03`
- RISKY short-term accepted with hardening queue: `E-01`, `E-04`, `E-11`, `E-12`, `E-13`, `E-14`, `E-15`, `E-16`, `E-22`, `E-29`, `E-30`, `E-31`, `E-32`
- MUST-REMOVE queued: `E-02`, `E-05`, `E-06`, `E-07`, `E-08`, `E-09`, `E-10`, `E-17`, `E-18`, `E-19`, `E-20`, `E-21`, `E-23`, `E-24`, `E-25`, `E-26`, `E-27`, `E-28`

---

## C) B-01 completion status

Status: Done

---

## D) Test additions for fallback detection (B-01)

Added in `POC_initial_demo/tests/unit/test_emitter.py`:

1. `test_emit_matmul_detects_guessed_shape_fallback`
- Detects guessed-shape fallback by clearing `tensor_shapes` and asserting `memref<4x4xf32>` appears.

2. `test_emit_transpose_detects_reverse_permutation_fallback`
- Detects reverse-permutation fallback by forcing rank/input mismatch and asserting `permutation = [2, 1, 0]`.

These tests intentionally lock current fallback behavior so later B-05/B-06 changes can flip expectations safely.
