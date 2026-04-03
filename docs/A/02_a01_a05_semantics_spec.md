# A-01 to A-05 Semantics Specification

Date: 2026-04-03
Scope: Trust-boundary behavior and strict policy

## Trust-State Model

External trust states used for reporting and acceptance:
1. proved
2. unproved_timeout
3. refuted

State mapping source:
1. EQUIVALENT -> proved
2. TIMEOUT or UNKNOWN -> unproved_timeout
3. NOT_EQUIVALENT, structural failures, or missing verification -> refuted

## Acceptance Rules

Default mode (non-strict):
1. proved is accepted.
2. unproved_timeout is accepted as partial output.
3. refuted is rejected.

Strict mode:
1. proved is accepted.
2. unproved_timeout is rejected.
3. refuted is rejected.

## Fallback Policy

Previous behavior allowed a high-confidence NOT_EQUIVALENT fallback to be emitted as partial output.
This path is removed.

Current behavior:
1. EQUIVALENT returns immediately as success candidate.
2. TIMEOUT or UNKNOWN may be returned as unproved fallback.
3. NOT_EQUIVALENT is retained only for explicit refuted reporting and is never partial success.

## CLI Policy

Strict mode options:
1. Added --strict to lift.
2. Added --strict to lift-c.

Option compatibility:
1. --strict cannot be combined with --no-verify.

Strict exit codes:
1. 0 for proved
2. 1 for refuted or failed
3. 2 for unproved_timeout

## Test Policy

Required checks added:
1. Refuted cannot be partial_success.
2. Strict mode rejects unproved_timeout and refuted outcomes.
3. CLI enforces strict/no-verify option validation.
4. CLI strict exit code mapping is deterministic.
