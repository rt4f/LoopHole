# 10 - Project Status: Achievements, Limitations, and Failure Analysis

Last updated: 2026-04-16
Owner: Project-level synthesis

This document consolidates status across core docs and all execution lanes (A, B, C), then normalizes what is complete, what is limited, and what still fails.

---

## 10.1 Scope and Method

This is a synthesis of:
- Core docs: [01](../background/project-overview.md), [02](../background/state-of-the-art.md), [03](../architecture/mlir-architecture.md), [04](../planning/implementation-roadmap.md), [05](../background/novel-research-directions.md), [07](./poc-implementation-audit.md), [08](../planning/three-person-execution-plan-v1.md), [09](./docker-polygeist-delivery-handoff.md)
- Team A lane docs: [A/README](../worklog/lane-a/README.md), [A/01](../worklog/lane-a/01_phase1_person_a_summary.md), [A/04](../worklog/lane-a/04_phase2_person_a_summary.md)
- Team B lane docs: [B/README](../worklog/lane-b/README.md), [B/01](../worklog/lane-b/01_baseline_report.md), [B/11](../worklog/lane-b/11_fixture_realism_stress_coverage.md)
- Team C lane docs: [C/README](../worklog/lane-c/README.md), [C/01](../worklog/lane-c/01_phase2_person_c_summary.md), [C/03](../worklog/lane-c/03_weekly_benchmark_report_workflow.md), [C/04](../worklog/lane-c/04_phase3_stablehlo_coverage_expansion.md), [C/05](../worklog/lane-c/05_phase3_batch_workflow_ux.md)

Normalization rule:
- When counts differ, this document keeps date and scope labels rather than forcing one value.

---

## 10.2 Normalized Status Timeline

| Date | Source | Snapshot |
|---|---|---|
| 2026-03-31 | [07](./poc-implementation-audit.md) | 96 passed, 1 skipped. Linalg demo 7/10 proved, 3/10 partial. StableHLO demo 2/10 proved, 2/10 partial, 6/10 failed. |
| 2026-04-02 | [09](./docker-polygeist-delivery-handoff.md) | Dockerized Polygeist + MLIR workflow delivered (llvm17 image), compile-lift flow validated for C and C++. |
| 2026-04-03 | [A/01](../worklog/lane-a/01_phase1_person_a_summary.md) | Trust-state semantics hardened (proved, unproved_timeout, refuted) and strict mode introduced. |
| 2026-04-03 | [B/01](../worklog/lane-b/01_baseline_report.md) | Targeted test baseline lock: 65 passed. CLI demo stable; standalone demo script had stale field failure at that time. |
| 2026-04-04 | [A/04](../worklog/lane-a/04_phase2_person_a_summary.md) | A-06 to A-10 completed: stronger symbolic handling, richer diagnostics, weak-kernel regression coverage. |
| 2026-04-xx | [C/04](../worklog/lane-c/04_phase3_stablehlo_coverage_expansion.md) | StableHLO coverage expanded; suite snapshot listed as 201 passed, 7 skipped for that phase scope. |
| 2026-04-xx | [B/11](../worklog/lane-b/11_fixture_realism_stress_coverage.md) | Stress realism expansion completed; suite snapshot listed as 232 passed, 1 skipped for that phase scope. |

Interpretation:
- The project moved from POC baseline into significant hardening.
- Test counts differ because reports were captured at different times and scopes.

---

## 10.3 What Has Been Achieved

## 10.3.1 End-to-End Lifting Pipeline Exists and Runs

Delivered pipeline stages:
1. Parser extraction from MLIR loop nests.
2. Sketch candidate generation and ranking.
3. Z3 verification path.
4. Linalg and StableHLO emission.
5. CLI and batch execution surfaces.

Evidence:
- [07](./poc-implementation-audit.md)
- [POC docs](../architecture/poc-documentation.md)

## 10.3.2 Trust Semantics and Strictness Boundary Were Hardened (Team A)

Major completed outcomes:
1. Explicit result states were formalized: proved, unproved_timeout, refuted.
2. Partial-success semantics were tightened so refuted outcomes are not treated as partial.
3. Strict mode and CLI policy guards were added.
4. High-confidence fallback over explicit refutation was removed.
5. Phase-2 verifier diagnostics were improved (counterexample details, disagreement metadata).

Evidence:
- [A/01](../worklog/lane-a/01_phase1_person_a_summary.md)
- [A/04](../worklog/lane-a/04_phase2_person_a_summary.md)

## 10.3.3 Parser and Emitter Hardening Was Substantial (Team B)

Major completed outcomes:
1. Fallback behavior was audited and classified.
2. Unsupported-form diagnostics were surfaced explicitly.
3. Mixed affine/scf parsing support was added.
4. Index normalization improved via canonicalization.
5. Silent shape/type emitter defaults were removed in critical paths.
6. Transpose permutation inference and validation improved.
7. Convolution attribute inference policy was implemented.
8. Artifact-level MLIR validation workflow was added.
9. StableHLO emitter hook sync and fixture realism stress coverage were completed.

Evidence:
- [B/README](../worklog/lane-b/README.md)
- [B/11](../worklog/lane-b/11_fixture_realism_stress_coverage.md)

## 10.3.4 StableHLO Coverage and Workflow UX Improved (Team C)

Major completed outcomes:
1. Unified CLI state vocabulary: PROVED, UNPROVED_TIMEOUT, REFUTED.
2. Batch JSON reporting schema and report-path controls.
3. Weekly benchmark report script for repeatable metrics.
4. StableHLO sketch and emitter expansion for transpose/elementwise/dot/reduce variants.
5. Batch UX scaling controls: include/exclude glob, max-files, table-limit, no-write-emitted.

Evidence:
- [C/01](../worklog/lane-c/01_phase2_person_c_summary.md)
- [C/03](../worklog/lane-c/03_weekly_benchmark_report_workflow.md)
- [C/04](../worklog/lane-c/04_phase3_stablehlo_coverage_expansion.md)
- [C/05](../worklog/lane-c/05_phase3_batch_workflow_ux.md)

## 10.3.5 Docker + Polygeist Toolchain Is Delivered and Reproducible

Major completed outcomes:
1. Dockerized Polygeist build and runtime path.
2. Two-step compile then lift workflow.
3. C/C++ frontend handling with fallback behavior.
4. GHCR publish workflow.

Evidence:
- [09](./docker-polygeist-delivery-handoff.md)

## 10.3.6 Strategic Positioning Is Mature

The project has already documented where novelty is and is not likely, reducing research misalignment risk.

Evidence:
- [02](../background/state-of-the-art.md)
- [05](../background/novel-research-directions.md)

---

## 10.4 Limitations and Failure Modes

## 10.4.1 Critical Limitations (Trust and Correctness)

1. Strict proof-only behavior is not yet globally guaranteed across all workflows by default.
- Why it matters: any path that accepts outputs after explicit refutation weakens trust claims.
- Evidence: [07](./poc-implementation-audit.md), [A/01](../worklog/lane-a/01_phase1_person_a_summary.md)

2. Symbolic multi-reduction verification remains explicitly unsupported in some cases.
- Why it matters: certain kernels remain unproved even after hardening.
- Evidence: [A/04](../worklog/lane-a/04_phase2_person_a_summary.md)

## 10.4.2 High Limitations (Coverage and Soundness Envelope)

1. StableHLO parity is improved but still incomplete relative to dense Linalg maturity.
- Evidence: [07](./poc-implementation-audit.md), [C/04](../worklog/lane-c/04_phase3_stablehlo_coverage_expansion.md)

2. Dynamic-shape universal proof remains an open area.
- Evidence: [07](./poc-implementation-audit.md), [05](../background/novel-research-directions.md)

3. Sparse lifting is not implemented.
- Evidence: [05](../background/novel-research-directions.md)

## 10.4.3 Medium Limitations (Tooling and Operational)

1. Parser architecture is more robust than earlier regex-only work, but full operation-level MLIR AST traversal is still a future-strength area.
- Evidence: [07](./poc-implementation-audit.md), [04](../planning/implementation-roadmap.md)

2. Docker toolchain includes compatibility patches and text normalization steps that are practical but fragile to upstream shifts.
- Evidence: [09](./docker-polygeist-delivery-handoff.md)

3. Platform friction remains for Windows users outside WSL/Docker-first workflows.
- Evidence: [04](../planning/implementation-roadmap.md), [09](./docker-polygeist-delivery-handoff.md)

---

## 10.5 Historical Failures and Current Disposition

| Failure Theme | Historical Signal | Current Disposition |
|---|---|---|
| Refuted outputs accepted as partial | Identified in [07](./poc-implementation-audit.md) | Mitigated by Team A semantics hardening; must still be made policy-default across every execution mode. |
| Silent parser/emitter fallback behavior | Audited in Team B docs | Strongly mitigated through diagnostics and fail-fast work; continue closing remaining edge cases. |
| Standalone demo script stale fields | Captured in [B/01](../worklog/lane-b/01_baseline_report.md) | Historical issue; current standalone demo path is covered by integration smoke tests and strict-profile contract checks. Keep CI smoke as guardrail. |
| Weak-kernel proof fragility | Documented in [A/04](../worklog/lane-a/04_phase2_person_a_summary.md) | Better diagnosed and tested; still a targeted hardening area. |
| StableHLO under-coverage | Documented in [07](./poc-implementation-audit.md) | Improved in [C/04](../worklog/lane-c/04_phase3_stablehlo_coverage_expansion.md), not yet feature-complete parity. |

---

## 10.6 Readiness Assessment

Readiness rating by objective:

1. Research POC viability: High.
- Reason: full pipeline exists, multiple lanes hardened it, broad tests documented.

2. Strictly trusted verification-grade output for broad kernel families: Medium.
- Reason: major trust fixes landed, but unresolved symbolic and coverage boundaries remain.

3. Production-grade operator toolchain for diverse real-world kernels: Low to Medium.
- Reason: significant progress, but fragility and feature-envelope limits remain.

4. Novel research contribution pathway: Medium.
- Reason: novelty opportunities are clear, but implementation in sparse/dynamic/transform tracks remains future work.

---

## 10.7 Bottom Line

LoopHole has moved well beyond a concept note and into a credible, multi-lane hardened prototype with meaningful tooling and documentation maturity. The largest remaining improvement opportunity is not basic implementation existence; it is finishing the trust envelope (strict policy by default plus broader proof coverage) and converting strong research-grade progress into release-grade reliability.

Next documents:
- [11 - Future Plan and Improvement Strategy](../planning/future-plan.md)
- [12 - Three-Person Parallel Plan V2 (12 Weeks)](../planning/three-person-execution-plan-v2-12-weeks.md)
