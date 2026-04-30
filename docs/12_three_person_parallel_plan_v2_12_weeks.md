# 12 - Three-Person Parallel Plan V2 (12 Weeks)

Last updated: 2026-04-16
Supersedes execution details in: [08](./08_three_person_parallel_execution_plan.md)

This plan assumes current completion status from Team A, Team B, and Team C lane docs, and focuses on the next 12 weeks of improvement work.

---

## 12.1 Plan Goals

1. Close trust-envelope gaps first (strict policy and artifact validity).
2. Increase proof reliability on known weak kernels.
3. Improve parser/emitter behavior on real frontend-generated MLIR.
4. Expand StableHLO confidence without silent fallbacks.
5. Produce reliable weekly quality signals and a clear research-track decision package.

---

## 12.2 Team Lanes

## Person A - Verification and Trust Semantics

Primary mission:
- proof quality, strictness policy correctness, and weak-kernel verifier hardening.

## Person B - Parser/Emitter and Validation Infrastructure

Primary mission:
- parser robustness, emitter safety, verifier pipeline stability, artifact validity.

## Person C - Workflow UX, Reporting, and Coverage Operations

Primary mission:
- benchmark reporting, CLI/profile usability, docs and adoption workflow, StableHLO coverage operations.

---

## 12.3 Task Backlog for This 12-Week Cycle

Task IDs continue from previously completed lanes.

## Person A tasks

- A-11: Strict-profile contract tests across lift, batch, and demo.
- A-12: Weak-kernel proof hardening pack (conv2d, dot, matvec, reductions).
- A-13: Symbolic multi-reduction boundary improvements and clearer unsupported envelopes.
- A-14: Counterexample replay utility and triage workflow.
- A-15: Shape-parametric proof pilot for selected kernels.
- A-16: Proof-quality summary artifact for weekly reports.

## Person B tasks

- B-11: Real-world MLIR corpus parser compatibility pass.
- B-12: Unsupported-form diagnostics closure for corpus misses.
- B-13: Emitter validation consistency pass (all required metadata gates).
- B-14: Trusted-lane artifact verification enforcement in CI.
- B-15: Dynamic memref normalization hardening toward IR-aware handling.
- B-16: Defect burn-down and parser/emitter reliability polish.

## Person C tasks

- C-12: Quickstart and runbook consolidation for trusted and exploratory profiles.
- C-13: Weekly benchmark dashboard hardening and trend summaries.
- C-14: StableHLO canonical-suite expansion and report integration.
- C-15: Batch profile UX presets and reproducibility metadata improvements.
- C-16: Release-readiness documentation and troubleshooting matrix.
- C-17: Research-track decision package preparation with evidence.

---

## 12.4 Sprint Calendar (6 x 2-week sprints)

## Sprint 1 (Weeks 1-2): Trust Gate Consolidation

Person A:
1. A-11 strict-profile contract tests.
2. Begin A-12 weak-kernel baseline measurement.

Person B:
1. Start B-11 corpus parser pass.
2. Start B-14 trusted-lane artifact verification enforcement.

Person C:
1. C-12 quickstart and runbook draft.
2. C-13 weekly dashboard baseline refresh.

Sprint 1 gate (G1):
1. Trusted profile rejects refuted outputs.
2. Required artifact verification runs in CI.
3. Baseline week-1 metrics report committed.

## Sprint 2 (Weeks 3-4): Weak-Kernel and Diagnostics Hardening

Person A:
1. Complete A-12.
2. Start A-13 symbolic boundary improvements.

Person B:
1. Complete B-11 parser corpus compatibility summary.
2. Start B-12 diagnostics closure.

Person C:
1. Complete C-13 weekly dashboard hardening.
2. Start C-15 profile UX presets.

Sprint 2 gate (G2):
1. Weak-kernel trend improves from week-1 baseline.
2. Unsupported corpus misses produce explicit diagnostics.

## Sprint 3 (Weeks 5-6): Emitter and Policy Reliability

Person A:
1. Continue A-13.
2. Start A-14 counterexample replay utility.

Person B:
1. Complete B-12 diagnostics closure.
2. Complete B-13 emitter validation consistency.

Person C:
1. Complete C-15 profile UX presets.
2. Start C-14 StableHLO canonical-suite expansion.

Sprint 3 gate (G3):
1. No silent fallback reintroduced in new paths.
2. Emitter metadata errors are explicit and actionable.

## Sprint 4 (Weeks 7-8): Parametric Proof Pilot and StableHLO Confidence

Person A:
1. Start A-15 shape-parametric proof pilot.
2. Finalize A-14 triage workflow.

Person B:
1. Start B-15 dynamic memref normalization hardening.
2. Maintain B-14 CI enforcement and fix regressions.

Person C:
1. Complete C-14 suite expansion and reporting.
2. Start C-16 release-readiness docs.

Sprint 4 gate (G4):
1. StableHLO failure trend decreases on canonical suite.
2. Parametric proof pilot demonstrates at least one successful end-to-end case.

## Sprint 5 (Weeks 9-10): Productization Prep

Person A:
1. Complete A-15 pilot scope.
2. Start A-16 proof-quality weekly artifact integration.

Person B:
1. Complete B-15 hardening path and tests.
2. Start B-16 defect burn-down.

Person C:
1. Continue C-16 release-readiness docs.
2. Start C-17 research-track decision package draft.

Sprint 5 gate (G5):
1. Trusted release candidate workflow is documented and reproducible.
2. Defect burn-down list is prioritized and measurable.

## Sprint 6 (Weeks 11-12): Finalization and Track Selection

Person A:
1. Complete A-16 reporting integration.
2. Final weak-kernel and strict-profile summary.

Person B:
1. Complete B-16 burn-down within agreed scope.
2. Final parser/emitter reliability report.

Person C:
1. Complete C-16 docs and troubleshooting matrix.
2. Complete C-17 research-track decision package.

Sprint 6 gate (G6):
1. 12-week metrics package complete.
2. Next research track selected with milestone-1 definition.

---

## 12.5 Dependency and Handoff Rules

Hard dependencies:
1. A-11 must complete before final trusted-profile lock.
2. B-14 must complete before trusted CI sign-off.
3. C-13 weekly dashboard baseline must be live before trend-based gates are used.
4. C-17 decision package starts only after A/B week-10 technical status is available.

Handoff artifacts required at sprint end:
1. Lane summary note in docs/A, docs/B, or docs/C.
2. Updated benchmark/report snapshots.
3. Updated known-risk table if severity changed.

---

## 12.6 Weekly Cadence and Governance

1. Monday 30-minute planning and dependency sync.
2. Wednesday integration checkpoint.
3. Friday quality gate with metrics publication.

Merge policy:
1. One branch per task ID.
2. No merge without task-level tests and trusted-lane checks.
3. Any semantics-impacting change requires report-schema compatibility check.

Escalation:
1. If gate fails, pause feature expansion and run a stabilization mini-sprint.
2. If two consecutive weekly trends regress in trust metrics, freeze non-critical work.

---

## 12.7 Definition of Done

A task is done only when all conditions hold:
1. Code and tests are merged.
2. Diagnostics/report outputs are updated.
3. Documentation is updated for user-facing behavior.
4. Weekly metrics reflect the change or explicitly explain why they do not.

A sprint is done only when:
1. Gate criteria pass.
2. Dependency handoff artifacts are published.
3. Risk table is updated.

---

## 12.8 Metrics for Plan Success

Core trust metrics:
1. Refuted-but-accepted rate in trusted profile: 0.
2. Emitted-artifact verification pass rate in trusted CI lane: 100% on required suites.

Quality trend metrics:
1. Weak-kernel unproved/refuted trend decreases from week-1 baseline.
2. StableHLO canonical-suite failure trend decreases.
3. Unsupported-form diagnostics coverage reaches all known misses in tracked corpus.

Operations metrics:
1. Weekly report generation continuity: 12/12.
2. Reproducible trusted quickstart validated on clean environment.

---

## 12.9 Expected Outputs at Week 12

1. Stronger trusted profile with consistent semantics and artifact verification.
2. Improved weak-kernel reliability with richer triage tooling.
3. Better parser/emitter behavior on realistic frontend corpus.
4. Better StableHLO confidence envelope and reporting.
5. Clear evidence-backed decision for the next research track.

Related strategy document:
- [11 - Future Plan and Improvement Strategy](./11_future_plan_and_improvements.md)

---

## 12.10 Sprint 1 Execution Snapshot (2026-04-17)

This section records execution evidence for Sprint 1 gate tracking.

Person A status:
1. A-11 complete: strict-profile contract behavior is validated across lift, lift-c, batch, and demo surfaces.
2. A-12 Sprint 1 scope complete: week-1 baseline artifacts generated for both linalg and stablehlo targets with fixed timeout (10000 ms).
3. Lane summary artifact published: [docs/A/05_sprint1_a11_a12_closure.md](./A/05_sprint1_a11_a12_closure.md).

Generated baseline artifacts:
1. [A/baselines/sprint1_week1_linalg_20260417.json](./A/baselines/sprint1_week1_linalg_20260417.json)
2. [A/baselines/sprint1_week1_stablehlo_20260417.json](./A/baselines/sprint1_week1_stablehlo_20260417.json)

G1 status update (partial):
1. Trusted profile rejects refuted outputs: done for Person A lane.
2. Required artifact verification runs in CI: pending Person B B-14 final sign-off.
3. Baseline week-1 metrics report committed: Person A baseline snapshots complete; Person C dashboard hardening remains gate dependency.

---

## 12.11 Implementation Update (2026-04-20)

Person C deliverables implemented for C-12 through C-17:

1. C-12 quickstart and runbook consolidation:
- `docs/C/06_c12_quickstart_runbook_profiles.md`

2. C-13 weekly dashboard hardening and trend summaries:
- `docs/C/07_c13_weekly_dashboard_hardening.md`
- `POC_initial_demo/scripts/generate_weekly_benchmark_report.py` (schema 1.1 trend support)

3. C-14 StableHLO canonical-suite expansion and report integration:
- `docs/C/08_c14_stablehlo_canonical_suite.md`
- canonical summary fields integrated into batch and weekly reports

4. C-15 batch profile UX presets and reproducibility metadata:
- aliases `trusted` and `exploratory` accepted by policy profile resolution
- report `run_metadata` and per-file source hashes added
- `docs/C/09_c15_batch_profile_presets_reproducibility.md`

5. C-16 release-readiness documentation and troubleshooting matrix:
- `docs/C/10_c16_release_readiness_troubleshooting_matrix.md`

6. C-17 research-track decision package with ranked recommendation:
- `docs/C/11_c17_research_track_decision_package.md`

---

## 12.12 A-12 Completion Update (2026-04-21)

Person A A-12 weak-kernel hardening pack is complete.

Scope closed:
1. conv2d, dot, matvec, and reduction weak-kernel hardening.
2. StableHLO weak-kernel convolution fixtures from Sprint-1 baseline (`conv1d`, `conv2d`) moved from refuted to proved.

Evidence artifacts:
1. `docs/A/06_a12_weak_kernel_hardening_completion.md`
2. `docs/A/baselines/a12_week2_weakkernels_linalg_20260421.json`
3. `docs/A/baselines/a12_week2_weakkernels_stablehlo_20260421.json`

---

## 12.13 B-16 Completion Update (2026-04-23)

Person B B-16 defect burn-down and parser/emitter reliability polish is complete.

Scope closed:
1. Fixed static zero-dimension normalization so parser metadata and emitted MLIR preserve `0` dimensions (no silent dynamic substitution).
2. Hardened no-SymPy convolution coefficient fallback parsing for `N*var`, `var*N`, and signed linear-term forms.
3. Added dedicated regression fixtures and tests for each fix.
4. Added trusted-lane CI guard checks for B-16 regression tests.

Evidence artifacts:
1. `docs/B/17_defect_burndown_parser_emitter_reliability_polish.md`
2. `.github/workflows/ci.yml` (`B-16 parser/emitter reliability guard checks` step)

Validation snapshot:
1. Focused parser/emitter tests: `91 passed`
2. Full suite: `273 passed`

---

## 12.14 A-13 / A-14 / A-15 Completion Update (2026-04-26)

Person A Sprint 3–5 tasks A-13, A-14, and A-15 are complete.

Scope closed:
1. A-13: Multi-reduction symbolic boundary improvements — `ReductionPatternKind` enum, `_UnsupportedReductionForm` exception, multi-IV guarded unroll, structured ENCODE_ERROR diagnostics replacing opaque UNKNOWN/ValueError.
2. A-14: Counterexample replay utility — new `replay_checker.py` module, `replay_counterexample()` entry point, `loophole replay` CLI subcommand, `--show-replay` flag on `loophole lift`.
3. A-15: Shape-parametric proof pilot — `_verify_parametric()` on Z3EquivalenceChecker, `parametric_mode` flag on Lifter, `--parametric` CLI flag. All three pilot kernels (matmul, matvec, reduce_sum) achieved parametric EQUIVALENT.

Evidence artifacts:
1. `docs/A/07_a13_a14_a15_sprint3_closure.md`
2. `docs/A/baselines/a13_a14_a15_sprint3_baseline_20260426.json`
3. `tests/unit/test_a13_multi_reduction_boundary.py`
4. `tests/unit/test_a14_counterexample_replay.py`
5. `tests/unit/test_a15_shape_parametric_proof.py`

Validation snapshot:
1. A-13 tests: `17 passed`
2. A-14 tests: `30 passed`
3. A-15 tests: `28 passed`
4. Full suite: `348 passed` (273 prior + 75 new)

Benchmark snapshot (linalg fixtures, 2026-04-26):
- Total kernels: 23 (includes 3 new A-15 parametric fixtures)
- PROVED: 18 / REFUTED: 5 / TIMEOUT: 0
- Accepted strict: 18
- Average elapsed ms: 5.88

---

## 12.15 A-16 Completion Update (2026-04-29)

Person A A-16 proof-quality summary artifact integration is complete. All Person A tasks for the 12-week cycle are now done.

Scope closed:
1. `_build_proof_quality_summary(rows)` added to `generate_weekly_benchmark_report.py` — computes proof_rate, strict_acceptance_rate, confidence distribution, per-sketch breakdown, corpus vs unexpected refuted counts, and proof_quality_grade (A/B/C/D/F).
2. `_render_proof_quality_section(pq)` added — renders the proof-quality block into the weekly Markdown report.
3. Both wired into `generate_report` payload and `_render_markdown`; report `schema_version` bumped to `"1.2"`.
4. New standalone script `scripts/generate_proof_quality_summary.py` for emitting the proof-quality artifact independently with optional baseline trend delta.

Evidence artifacts:
1. `docs/A/08_a16_proof_quality_summary_closure.md`
2. `docs/A/baselines/proof_quality_a16_pq_20260429_20260429T110512Z.json`
3. `docs/A/baselines/proof_quality_a16_pq_20260429_20260429T110512Z.md`
4. `tests/unit/test_a16_proof_quality_summary.py`

Validation snapshot:
1. A-16 tests: `15 passed`
2. Full suite: `365 passed` (348 prior + 17 new, including schema-version test updates)

Proof-quality baseline (linalg fixtures, 2026-04-29):
- Grade: **B**
- Total kernels: 23
- Proof rate: 78.3% (18 proved / 5 refuted)
- Strict acceptance rate: 78.3%
- Avg confidence (proved only): 0.993
- Fully confident (conf=1.0): 14
- Corpus refuted (expected): 5
- Unexpected refuted: **0**
