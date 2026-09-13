# LoopHole Expansion Plan: Detailed 3-Person Parallel Execution Guide

Date: April 2, 2026

This document is a shareable, execution-ready roadmap for improving and expanding the LoopHole project with three people working in parallel.

---

## 1) Project Objective

LoopHole aims to automatically lift legacy scalar loop programs (represented in MLIR Affine or SCF style) into higher-level tensor dialects (Linalg and StableHLO), with verification strong enough to trust outputs before downstream compilation to modern accelerators.

In practical terms, this means:

1. Recognize loop semantics reliably.
2. Match those semantics to high-level tensor sketches.
3. Prove equivalence where possible.
4. Emit valid, trustworthy MLIR for toolchains.

---

## 2) Current Baseline (Reality Check)

Current known state from code, docs, and tests:

- Test baseline: 96 passed, 1 skipped.
- End-to-end pipeline exists and is runnable.
- Linalg path is stronger than StableHLO.
- Some outputs are still accepted as partial even when not fully proved equivalent.
- Demo script has known breakage history and stale field references.
- Parser and emitter still contain fallback/default behavior that can hide correctness gaps.

Implication: the project is a strong POC, but reliability guarantees and coverage breadth must improve before claiming production-grade trust.

---

## 3) Strategy (Execution Order)

Work must happen in this order to avoid rework:

1. Correctness semantics first (trust boundary).
2. Parser/emitter hardening (output validity and diagnostics).
3. StableHLO and coverage expansion.
4. Research expansion (sparse, dynamic shape, transform scheduling).

If we skip step 1, all other progress rests on weak guarantees.

---

## 4) Team Split (Parallel Workstreams)

### Person A: Verification and Correctness Semantics

Primary outcomes:

- Strict interpretation of proved vs unproved vs refuted.
- Better Z3 and SymPy alignment.
- CI semantics that enforce proof quality where required.

### Person B: Parser and Emitter Robustness

Primary outcomes:

- More reliable extraction across real MLIR variants.
- Fail-fast emitter when metadata is insufficient.
- Artifact-level validation to detect invalid MLIR early.

### Person C: Dialect Coverage, CLI/UX, Documentation

Primary outcomes:

- Broader StableHLO coverage.
- Better CLI reporting and batch workflows.
- Docs, demos, and benchmark reporting that make the project easy to adopt.

---

## 5) Detailed Backlog (Fine-Grained)

Task ID format:

- A-* for Person A
- B-* for Person B
- C-* for Person C

Each task includes dependencies and done criteria.

### Phase 1: Stabilize the Trust Boundary (Week 1)

#### Person A

- A-01: Introduce explicit result states: proved, unproved_timeout, refuted.
	- Depends on: none
	- Done when: Lift result model exposes state unambiguously.

- A-02: Redefine partial success to exclude refuted results.
	- Depends on: A-01
	- Done when: not-equivalent no longer appears as acceptable partial output.

- A-03: Add strict mode control for verification behavior.
	- Depends on: A-01
	- Done when: strict mode prevents output acceptance unless proof criteria are met.

- A-04: Remove high-confidence not-equivalent fallback path.
	- Depends on: A-02
	- Done when: confidence cannot override explicit refutation.

- A-05: Update unit tests to enforce new semantics.
	- Depends on: A-01, A-02, A-03
	- Done when: tests fail if refuted output is treated as partial or success.

#### Person B

- B-01: Audit fallback/default behavior in parser and emitter.
	- Depends on: none
	- Done when: inventory of all defaults and hidden assumptions is documented.

- B-02: Add clear diagnostic messages for unsupported forms.
	- Depends on: B-01
	- Done when: user-facing errors explain why extraction or emission failed.

#### Person C

- C-01: Fix standalone demo script compatibility with current LiftResult fields.
	- Depends on: none
	- Done when: demo runs in both rich and plain mode without runtime errors.

- C-02: Add smoke test for demo script.
	- Depends on: C-01
	- Done when: CI executes demo smoke path.

- C-03: Tighten integration tests currently allowing broad success-or-partial behavior.
	- Depends on: A-02
	- Done when: critical kernels are asserted in strict mode with deterministic expected states.

### Phase 2: Harden Core Engine (Weeks 2-4)

#### Person A

- A-06: Improve reduction modeling in verifier for multi-reduction kernels.
	- Depends on: A-05
	- Done when: targeted conv and reduction proofs are stable.

- A-07: Improve symbolic path handling for dynamic-like expressions.
	- Depends on: A-06
	- Done when: fewer unknown outcomes caused by weak substitution shortcuts.

- A-08: Add richer counterexample reporting for refuted cases.
	- Depends on: A-05
	- Done when: report includes actionable mismatch clues.

- A-09: Add SymPy vs Z3 disagreement diagnostics.
	- Depends on: A-06, A-07
	- Done when: mismatch class is visible in logs or report objects.

- A-10: Add regression suite for known weak kernels (conv2d, dot, matvec edge cases).
	- Depends on: A-06
	- Done when: known regressions are reproducible and guarded.

#### Person B

- B-03: Expand parser support for mixed affine/scf patterns and variants.
	- Depends on: B-01
	- Done when: parser handles broader loop forms with deterministic extraction.

- B-04: Improve affine index normalization for edge expressions.
	- Depends on: B-03
	- Done when: index expressions remain consistent for verifier and tracer.

- B-05: Remove silent shape/type defaults in emitter.
	- Depends on: B-01
	- Done when: emission fails clearly if required metadata is missing.

- B-06: Improve transpose permutation inference and validation.
	- Depends on: B-05
	- Done when: non-trivial transpose patterns emit correct permutation metadata.

- B-07: Infer or propagate conv attrs (strides/dilations) with explicit fallback policy.
	- Depends on: B-05
	- Done when: conv emission no longer relies on hardcoded assumptions without signal.

- B-08: Add artifact-level validation pipeline (mlir-opt checks on emitted outputs).
	- Depends on: B-05
	- Done when: CI fails on syntactically invalid emitted MLIR artifacts.

#### Person C

- C-04: Improve CLI result reporting to surface proved/unproved/refuted states.
	- Depends on: A-01
	- Done when: CLI users can immediately distinguish trusted vs untrusted outputs.

- C-05: Add machine-readable report output (json summary for batch runs).
	- Depends on: C-04
	- Done when: batch results can be ingested by dashboards or scripts.

- C-06: Add baseline benchmark report script for weekly tracking.
	- Depends on: none
	- Done when: project can generate repeatable metrics snapshot.

- C-07: Update docs and usage walkthroughs to match strict-mode semantics.
	- Depends on: A-03, C-04
	- Done when: contributor docs no longer describe outdated behavior.

### Phase 3: Coverage Expansion (Weeks 5-7)

#### Person A

- A-11: Extend verifier templates for newly added StableHLO patterns.
	- Depends on: C-08 to C-10
	- Done when: added StableHLO kernels are checked with meaningful criteria.

- A-12: Add strict-mode policy profile for CI vs exploratory local runs.
	- Depends on: A-03
	- Done when: policy can be tuned without changing core semantics.

#### Person B

- B-09: Add emitter support hooks required by new StableHLO operations.
	- Depends on: C-09
	- Done when: emission pathways exist and pass artifact validation.

- B-10: Expand fixture realism for parser and emitter stress testing.
	- Depends on: B-03
	- Done when: fixture set reflects expected real-world variants.

#### Person C

- C-08: Expand StableHLO sketch library coverage (transpose, elementwise, dot/reduce variants).
	- Depends on: none
	- Done when: sketch catalog includes agreed coverage targets.

- C-09: Implement emitter and mapping logic for newly added StableHLO sketches.
	- Depends on: C-08
	- Done when: output generation exists for all newly introduced sketches.

- C-10: Add integration tests for each new StableHLO operation.
	- Depends on: C-09, A-11
	- Done when: CI includes strict coverage tests for these kernels.

- C-11: Improve batch workflow UX for larger fixture directories.
	- Depends on: C-05
	- Done when: users can run and share structured project-wide reports quickly.

### Phase 4: Research Track Selection and First Novel Milestone (Week 8+)

Choose one primary innovation track after Phase 3 stabilizes.

Preferred first choice: Sparse reverse lifting.
Fallback/secondary: Transform dialect schedule synthesis.

#### If Sparse Track is selected

- A-13: Define sparse verification envelope and acceptance criteria.
- B-11: Prototype non-affine sparse pattern extraction.
- C-12: Build sparse benchmark and fixture corpus.

#### If Transform Track is selected

- A-14: Define correctness checks for synthesized schedules.
- B-12: Integrate transform artifact validation flow.
- C-13: Build schedule generation and performance comparison harness.

---

## 6) Parallelism and Dependency Graph

Maximize parallel work by keeping three lanes mostly independent.

Hard sync checkpoints:

1. Sync-1 (end of Week 1): result-state model and strict semantics are frozen.
2. Sync-2 (end of Week 4): parser/emitter diagnostics and artifact validation are frozen.
3. Sync-3 (end of Week 7): StableHLO expansion merged and tested.
4. Sync-4 (Week 8): research track selected with one concrete first milestone.

Rules for smooth parallel integration:

1. No silent behavior changes in shared models without changelog note.
2. New feature must include tests and report output updates.
3. Any fallback path must be explicit in logs and state model.

---

## 7) Weekly Execution Cadence

Weekly ritual:

1. Monday planning sync (30 minutes)
	 - Confirm blockers and dependency handoffs.

2. Mid-week integration check
	 - Merge completed atomic tasks.
	 - Run full test and artifact validation.

3. Friday quality gate
	 - Publish metrics snapshot.
	 - Decide carry-over items.

Suggested branch policy:

1. One branch per task ID.
2. Merge only with task-level tests green.
3. Use feature flags where semantics are evolving.

---

## 8) Success Metrics (Track Weekly)

Core correctness metrics:

1. Proof rate on canonical fixtures.
2. Refuted-but-emitted rate (target: zero in strict mode).
3. Unknown/timeout rate over time.

Coverage metrics:

1. StableHLO kernel coverage count.
2. Integration test strict-pass rate by kernel family.

Output quality metrics:

1. Emitted artifact validation pass rate.
2. Demo script pass rate.

Developer velocity metrics:

1. Median task cycle time.
2. Reopen rate after merge.

---

## 9) Risks and Mitigations

Risk: Verifier improvements increase runtime significantly.

- Mitigation: keep fast smoke checks for PRs and deep checks nightly.

Risk: Parser hardening introduces regressions on existing fixtures.

- Mitigation: lock existing fixture expectations before parser expansion.

Risk: StableHLO expansion diverges from verifier capability.

- Mitigation: pair C-08/C-09 with A-11 checkpoints, not end-loaded.

Risk: Team gets split across too many research ideas.

- Mitigation: single primary innovation track until first publishable milestone.

---

## 10) Kickoff Checklist (Immediate)

Do this in order:

1. Confirm owners for Person A, Person B, Person C lanes.
2. Start Phase 1 tasks A-01, B-01, C-01 in parallel.
3. Run baseline test + report snapshot before first merges.
4. Freeze strict-mode semantics at Sync-1.
5. Publish first weekly status with metrics and blocker list.

---

## 11) Expected Outcome

After Phase 3, the project should move from research-grade POC behavior to a much more trustworthy lifting engine with:

1. Explicit correctness semantics.
2. Better parser/emitter reliability and diagnostics.
3. Broader StableHLO support with strict validation.
4. A clear launchpad for a genuinely novel research contribution.

