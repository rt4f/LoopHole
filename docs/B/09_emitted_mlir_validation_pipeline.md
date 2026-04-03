# B-08: Artifact-Level Emitted MLIR Validation Pipeline

## Goal

Add an artifact-level validation stage so emitted MLIR is checked by an external MLIR verifier command, and make this enforceable in CI.

## Implemented Changes

### 1. Shared validation utility

Added a reusable validation module:

- `POC_initial_demo/src/loophole/mlir_validator.py`

Capabilities:

- discovers verifier command from:
  - explicit CLI arg
  - `LOOPHOLE_MLIR_VERIFY_CMD`
  - fallback binaries (`mlir-opt`, `mlir-opt-18`, `mlir-opt-17`)
- validates emitted artifact text by writing a temporary `.mlir` file and invoking verifier command
- returns structured validation result (`ok`, command, return code, stdout/stderr)

### 2. CLI validation hook (optional local checks)

Wired optional validation in `cli.py`:

- `loophole lift`:
  - `--validate-emitted`
  - `--mlir-verifier`
- `loophole lift-c`:
  - `--validate-emitted`
  - `--mlir-verifier`
- `loophole batch`:
  - `--validate-emitted`
  - `--mlir-verifier`

Behavior:

- when validation requested and no verifier exists, command exits non-zero with explicit guidance
- when emitted artifact fails verifier, command exits non-zero and prints verifier diagnostics

### 3. Test workflow validation stage

Added verifier-aware pytest fixture in `POC_initial_demo/conftest.py`:

- `mlir_verifier_cmd` fixture
- strict mode env gate: `LOOPHOLE_REQUIRE_MLIR_VERIFY=1`
  - if set and no verifier is found, tests fail immediately
  - otherwise verifier-dependent tests are skipped

This creates a clean local-dev behavior while allowing strict CI enforcement.

### 4. Unit test upgrades

Updated `POC_initial_demo/tests/unit/test_emitter.py`:

- added artifact validation tests for emitted Linalg artifacts:
  - matmul
  - transpose
  - conv2d

These tests now run verifier-backed validity checks where toolchain is available.

### 5. Integration test additions

Added emitted artifact validation checks in:

- `POC_initial_demo/tests/integration/test_matmul.py`
- `POC_initial_demo/tests/integration/test_conv2d.py`
- `POC_initial_demo/tests/integration/test_transpose.py`

Each test validates emitted Linalg artifact text through external verifier command.

### 6. CI enforcement

Added CI workflow:

- `.github/workflows/ci.yml`

Pipeline:

- installs Python deps
- installs MLIR tool package (`mlir-tools`/`mlir-18-tools`/`mlir-17-tools` fallback)
- resolves verifier binary to env (`LOOPHOLE_MLIR_VERIFY_CMD`)
- runs full pytest with `LOOPHOLE_REQUIRE_MLIR_VERIFY=1`

Result:

- CI is now configured to fail if verifier tool is missing, or if emitted artifacts are invalid.

## Validation

Executed locally in `POC_initial_demo`:

- `pytest tests/unit/test_emitter.py tests/integration/test_matmul.py tests/integration/test_conv2d.py tests/integration/test_transpose.py -q`
- `pytest tests/ -q`

Latest result:

- `174 passed, 1 skipped`

## Acceptance Mapping (B-08)

- Add validation stage in test workflow: Done.
- Wire validation invocation from CLI for optional local checks: Done.
- Upgrade `test_emitter.py` to artifact validity checks where toolchain is available: Done.
- Add integration-level artifact verification in `test_matmul.py`, `test_conv2d.py`, `test_transpose.py`: Done.
- CI fails on invalid emitted MLIR artifacts: Done via enforced verifier-backed pytest stage.
