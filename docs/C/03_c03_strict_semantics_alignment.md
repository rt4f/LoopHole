# C-03 Strict Semantics Alignment

Date: 2026-04-03
Scope: Integration test tightening with Person A dependency handling

## Goal

Reduce broad success-or-partial assertions for kernels where behavior is deterministic and proofs are stable, while remaining compatible with pending strict-semantics API work from Person A (A-02).

## Files adjusted

1. `POC_initial_demo/tests/integration/test_matmul.py`
2. `POC_initial_demo/tests/integration/test_conv1d.py`
3. `POC_initial_demo/tests/integration/test_transpose.py`
4. `POC_initial_demo/tests/integration/test_conv2d.py`

## What was tightened

1. Matmul, conv1d, and transpose deterministic checks now require proved equivalence where baseline behavior is stable.
2. Assertions now check explicit verification state where relevant.
3. Conv2d strict behavior is encoded as dependency-aware behavior, not a permanent broad pass condition.

## Dependency-aware strategy for A-02

Conv2d strict test behavior is intentionally adaptive:
1. If strict-mode constructor API is not available yet, the strict test is xfailed with clear reason.
2. If strict-mode API exists, the same test auto-runs strict assertions.
3. In strict path, refuted (`NOT_EQUIVALENT`) results must not be reported as partial-success.

This approach prevents test churn and avoids manual rewrites once A-02 lands.

## Why this was chosen

1. It allows Person C to progress now without blocking on Person A merge timing.
2. It creates an executable contract for trust-boundary semantics.
3. It avoids regressions back to permissive status handling in critical kernels.

## Validation

Focused integration validation command:
1. `c:/Users/Yasho/LoopHole/.venv/Scripts/python.exe -m pytest tests/integration/test_conv2d.py tests/integration/test_matmul.py tests/integration/test_conv1d.py tests/integration/test_transpose.py -q`

Expected result while A-02 API is absent:
1. Pass with one explicit xfail tied to strict API availability.

Expected result after A-02 API lands:
1. Strict test path auto-activates and enforces non-permissive handling for refuted conv2d results.
