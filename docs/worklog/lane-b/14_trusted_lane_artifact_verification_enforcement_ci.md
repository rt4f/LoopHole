# B-14: Trusted-Lane Artifact Verification Enforcement in CI

## Goal

Make artifact validation mandatory in trusted CI lanes, provide explicit failure signaling when verifier tooling is unavailable, and add CI-facing checks for deterministic pass/fail behavior.

## Implementation Summary

### 1. Mandatory validation in trusted batch lane (no optional bypass)

Updated:
- `packages/loophole/src/loophole/cli.py`

Changes:
- Added trusted-lane detection helper:
  - `_requires_trusted_artifact_validation(profile_name)`
- In `batch` command:
  - if resolved profile is trusted (`ci-strict`, including `trusted` alias), emitted artifact validation is forced on even when `--validate-emitted` is not passed
  - this removes optional bypass for required trusted-lane suites

### 2. Explicit trusted-lane failure when verifier toolchain is unavailable

Updated:
- `packages/loophole/src/loophole/cli.py`
- `packages/loophole/conftest.py`

Changes:
- `batch` now fails with explicit trusted-lane messaging when verifier command is missing under trusted profile
- `mlir_verifier_cmd` fixture now treats trusted profile env (`LOOPHOLE_POLICY_PROFILE=ci-strict|trusted`) as requiring verifier availability, not just `LOOPHOLE_REQUIRE_MLIR_VERIFY`
- failure messaging explicitly states trusted artifact verification is required

### 3. Validation cannot be bypassed by disabling output writes

Updated:
- `packages/loophole/src/loophole/cli.py`

Change:
- artifact verification now runs whenever emitted MLIR exists and validation is enabled, regardless of `--write-emitted/--no-write-emitted`
- this closes the previous bypass path where `--no-write-emitted` skipped validation

### 4. CI workflow enforcement and guard checks

Updated:
- `.github/workflows/ci.yml`

Changes:
- Added trusted-lane batch smoke check using trusted profile without explicit `--validate-emitted`, relying on mandatory trusted-lane enforcement
- Added CI guard test step running trusted-lane integration checks for:
  - mandatory validation activation
  - missing verifier failure
  - invalid artifact failure

### 5. CI-facing test additions

Updated:
- `packages/loophole/tests/integration/test_cli_batch_report.py`

New tests:
1. `test_trusted_batch_requires_validation_even_without_flag`
2. `test_trusted_batch_fails_when_verifier_missing`
3. `test_trusted_batch_fails_on_invalid_artifact`

These tests assert deterministic trusted-lane failure semantics and required validation behavior.

## Acceptance Mapping (B-14)

- Action: Make artifact validation mandatory in trusted CI lanes (no optional bypass for required suites).
  - Done: trusted profile forces validation in batch path; `--no-write-emitted` no longer bypasses verification.
- Action: Add explicit failure signaling when verifier toolchain is unavailable in trusted lane.
  - Done: explicit trusted-lane failure in CLI + fixture-level enforcement for trusted profile env.
- Test addition: Add CI-facing checks that assert trusted lane fails on invalid artifacts or missing verifier.
  - Done: three dedicated integration tests + CI guard step.
- Done when: Trusted CI always enforces artifact validity and reports pass/fail deterministically.
  - Done: trusted smoke check + guard tests + deterministic validation failure exit paths.

## Validation

Executed in `POC_initial_demo`:

- Trusted-lane CI-facing tests:
  - `PYTHONPATH=src python -m pytest tests/integration/test_cli_batch_report.py::test_trusted_batch_requires_validation_even_without_flag tests/integration/test_cli_batch_report.py::test_trusted_batch_fails_when_verifier_missing tests/integration/test_cli_batch_report.py::test_trusted_batch_fails_on_invalid_artifact -q`
- Broader CLI regression:
  - `PYTHONPATH=src python -m pytest tests/integration/test_cli_batch_report.py tests/integration/test_cli_strict_policy.py -q`
- Full regression:
  - `PYTHONPATH=src python -m pytest tests/ -q`

Latest result:
- `242 passed, 1 skipped`
