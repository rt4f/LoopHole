# Person A Execution Documentation

This folder contains documentation for Person A work in the three-person parallel execution plan.

## Task Index

### Phase 1: Stabilize the Trust Boundary

| Doc | Task | Status |
|---|---|---|
| [01_phase1_person_a_summary.md](01_phase1_person_a_summary.md) | A-01 through A-05 completion summary | Done |
| [02_a01_a05_semantics_spec.md](02_a01_a05_semantics_spec.md) | Trust-state and strict-mode semantics specification | Done |
| [03_validation_and_handoff.md](03_validation_and_handoff.md) | Validation evidence and commit scope | Done |

### Phase 2: Harden Core Engine

| Doc | Task | Status |
|---|---|---|
| [04_phase2_person_a_summary.md](04_phase2_person_a_summary.md) | A-06 through A-10 implementation, diagnostics, and validation | Done |

### Sprint 1 (12-Week Plan V2): Trust Gate Consolidation

| Doc | Task | Status |
|---|---|---|
| [05_sprint1_a11_a12_closure.md](05_sprint1_a11_a12_closure.md) | A-11 contract parity and A-12 week-1 dual-target baseline evidence | Done |

### Sprint 2 (12-Week Plan V2): Weak-Kernel Hardening

| Doc | Task | Status |
|---|---|---|
| [06_a12_weak_kernel_hardening_completion.md](06_a12_weak_kernel_hardening_completion.md) | A-12 weak-kernel hardening completion and week-2 evidence | Done |

### Sprint 3–5 (12-Week Plan V2): Symbolic Boundary, Replay, Parametric Proof

| Doc | Task | Status |
|---|---|---|
| [07_a13_a14_a15_sprint3_closure.md](07_a13_a14_a15_sprint3_closure.md) | A-13 multi-reduction boundary, A-14 counterexample replay, A-15 shape-parametric proof | Done |

### Sprint 6 (12-Week Plan V2): Proof-Quality Reporting

| Doc | Task | Status |
|---|---|---|
| [08_a16_proof_quality_summary_closure.md](08_a16_proof_quality_summary_closure.md) | A-16 proof-quality summary artifact integration into weekly reports | Done |

### Full Workflow Test

| Doc | Task | Status |
|---|---|---|
| [fullworkflowtest/README.md](fullworkflowtest/README.md) | End-to-end C to StableHLO to GPU benchmark runbook and evidence log | Done |

## Scope Clarification

Person A scope for Phase 1:
1. A-01: Introduce explicit result states (proved, unproved_timeout, refuted).
2. A-02: Redefine partial success to exclude refuted results.
3. A-03: Add strict mode control for verification acceptance behavior.
4. A-04: Remove high-confidence not-equivalent fallback path.
5. A-05: Update tests to enforce strict semantics.

## Implementation Files

1. POC_initial_demo/src/loophole/lifter.py
2. POC_initial_demo/src/loophole/cli.py
3. POC_initial_demo/src/loophole/__init__.py
4. POC_initial_demo/tests/unit/test_lifter_semantics.py
5. POC_initial_demo/tests/integration/test_conv2d.py
6. POC_initial_demo/tests/integration/test_transpose.py
7. POC_initial_demo/tests/integration/test_cli_strict_policy.py

## Validation Snapshot

1. Focused Phase 1 suite: 22 passed.
2. Full suite: 140 passed, 1 skipped.
