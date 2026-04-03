# LoopHole Baseline Report (Locked)

Date: 2026-04-03
Owner: Person B lane (baseline freeze)

## Scope

This baseline locks current behavior for:

- `tests/unit/test_affine_extractor.py`
- `tests/unit/test_emitter.py`
- `tests/integration/test_conv1d.py`
- `tests/integration/test_conv2d.py`
- `tests/integration/test_matmul.py`
- `tests/integration/test_transpose.py`
- Demo behavior from both:
  - `python examples/run_demo.py --mode demo`
  - `loophole demo --target linalg`



## Baseline Commands and Results

### 1) Targeted test suite

Command:

```powershell
pytest tests/unit/test_affine_extractor.py tests/unit/test_emitter.py tests/integration/test_conv1d.py tests/integration/test_conv2d.py tests/integration/test_matmul.py tests/integration/test_transpose.py -q -rs
```

Result:

- `65 passed in 7.30s`
- No skips in this targeted set

### 2) Standalone demo script behavior

Command:

```powershell
python examples/run_demo.py --mode demo
```

Observed behavior:

- Demo table prints kernel-level results before crash.
- Summary from printed table:
  - Success: `7`
  - Partial: `3`
  - Failed before summary completion due to runtime error after table stage
- Runtime failure:
  - `AttributeError: 'LiftResult' object has no attribute 'linalg_mlir'`
  - Failing location: `examples/run_demo.py` in `run_demo_fancy()`

Interpretation:

- Standalone demo script is currently not baseline-clean (known stale field usage).

### 3) CLI demo behavior

Command:

```powershell
loophole demo --target linalg
```

Observed behavior:

- Command executes successfully.
- Demo summary printed by CLI:
  - `7/10 formally proved`
  - `3 partial`
- Partial kernels shown by CLI output:
  - `conv2d_simple`
  - `conv2d_nhwc`
  - `dot_product`

Interpretation:

- CLI demo is the baseline-healthy demo path.

## Fixture Source Lock (Parser + Emitter Hardening)

Policy locked as of this baseline:

1. `POC_initial_demo/src/loophole/tests/fixtures.py` is the single fixture source for parser and emitter hardening tests.
2. Parser hardening tests must import fixtures from `loophole.tests.fixtures` and avoid inline MLIR fixtures.
3. Emitter hardening tests must import fixtures from `loophole.tests.fixtures` and avoid inline MLIR fixtures.
4. New fixture scenarios for parser/emitter hardening must be added to `fixtures.py` first, then referenced in tests.

Current compliance check:

- `tests/unit/test_affine_extractor.py` imports fixture constants from `loophole.tests.fixtures`.
- `tests/unit/test_emitter.py` imports fixture constants from `loophole.tests.fixtures`.

## Baseline Status

- Targeted tests: LOCKED (pass)
- CLI demo behavior: LOCKED (7 proved / 3 partial)
- Standalone demo script behavior: LOCKED as failing due to stale `LiftResult` fields

## Change Control Note

Any future Person B parser/emitter hardening change should be compared against this baseline before merge.
