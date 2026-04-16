# 11 - Future Plan and Improvement Strategy

Last updated: 2026-04-16
Owner: Project-level strategy synthesis

This document defines how to improve LoopHole from current hardened POC status to a stronger trust envelope, broader coverage, and clearer path to research novelty.

---

## 11.1 Strategy Principles

1. Trust before breadth.
- No new coverage should weaken strict correctness policy.

2. Explicit failure over silent fallback.
- Unsupported patterns should produce diagnostics, never hidden defaults.

3. Reproducibility as a first-class requirement.
- Every weekly claim should be traceable in machine-readable reports.

4. Separate engineering reliability from research novelty.
- Reliability work and novelty work should progress in parallel but with clear gates.

---

## 11.2 Improvement Objectives (12-Week Horizon)

Primary objectives:
1. Make strict policy enforceable and observable across all workflows.
2. Increase proof reliability on weak kernel families.
3. Expand StableHLO confidence envelope without reintroducing hidden fallback behavior.
4. Improve parser/emitter behavior on real-world MLIR outputs from C/C++ frontends.
5. Strengthen operational adoption via reports, docs, and reproducible runbooks.

Secondary objective:
6. Prepare a clean decision point for the next novelty track (sparse, transform scheduling, or dynamic shapes).

---

## 11.3 Prioritized Improvement Backlog

## P0 (must-do first)

### P0-1: Strict policy by default for CI and release profiles

Why:
- Prevent accidental acceptance of refuted outputs in trusted workflows.

Success criteria:
1. CI strict profile rejects any refuted output as accepted.
2. Batch and single-file commands expose strictness clearly in reports.
3. Release notes define strict and exploratory profiles explicitly.

Depends on:
- Existing Team A semantics and Team C reporting work.

### P0-2: Unified trust contract tests across CLI, batch, and demo surfaces

Why:
- Semantics can drift between command paths.

Success criteria:
1. Shared regression suite verifies identical result-state handling.
2. Exit code and report-state behavior are consistent.

Depends on:
- P0-1.

### P0-3: Mandatory emitted-artifact verification in CI trusted lane

Why:
- Detect invalid MLIR artifacts before merge.

Success criteria:
1. Trusted lane runs emitted-artifact verification for representative kernels.
2. Failing verifier output blocks merge.

Depends on:
- Existing Team B validation pipeline and toolchain availability.

---

## P1 (high value, 12-week target)

### P1-1: Weak-kernel verifier hardening

Scope:
- conv2d variants, dot variants, matvec edge cases, reductions with more complex symbolic behavior.

Success criteria:
1. Reduced unproved/refuted rates on canonical weak-kernel suite.
2. Rich mismatch diagnostics for unresolved cases.

### P1-2: Parser robustness on real frontend-generated MLIR

Scope:
- Polygeist-generated IR corpus, mixed loop patterns, dynamic memref signatures.

Success criteria:
1. Corpus parsing success rate increases week over week.
2. Unsupported forms always include diagnostics with actionable context.

### P1-3: StableHLO parity and confidence expansion

Scope:
- close high-impact op gaps and tighten proof/diagnostic quality.

Success criteria:
1. StableHLO demo failure count reduced on canonical fixture set.
2. No silent fallback behavior introduced.

### P1-4: Weekly benchmark and quality dashboard hardening

Scope:
- make weekly trend reporting stable for decision-making.

Success criteria:
1. JSON + markdown reports generated every week from a fixed fixture set.
2. trend deltas for proved/unproved/refuted and latency are visible.

---

## P2 (medium-term, start now and continue)

### P2-1: IR-aware dynamic memref canonicalization

Why:
- Current text rewrite is practical but fragile.

Success criteria:
1. Replace or encapsulate text rewrite with parser/IR-aware transform path.
2. Add regression tests for tricky signatures.

### P2-2: Unified diagnostics taxonomy

Why:
- Different layers currently report differently.

Success criteria:
1. parser, verifier, emitter diagnostics share stable categories/codes.
2. Reports and CLI map these consistently.

### P2-3: Release-grade operator runbook

Why:
- Adoption depends on clear and fast workflow guidance.

Success criteria:
1. Quickstart from clean machine to first verified lift.
2. Troubleshooting matrix for top failure patterns.

---

## P3 (research direction preparation)

### P3-1: Track-selection gate with evidence package

Candidates:
1. Sparse reverse lifting.
2. Transform dialect schedule synthesis.
3. Dynamic shape lifting and proof.

Decision inputs:
1. engineering readiness after 12 weeks.
2. benchmark quality trend.
3. novelty and feasibility from [05](./05_novel_research_directions.md).

Success criteria:
1. One selected primary track.
2. 1-page scope boundary and first milestone spec.

---

## 11.4 Architecture Improvement Plan

## 11.4.1 Verification Layer

Improve:
1. symbolic reduction handling and bounded proof strategies.
2. disagreement handling between heuristic and formal signals.
3. counterexample usability for debugging and triage.

Expected effect:
- Higher trust in accepted outputs and faster root-cause cycles.

## 11.4.2 Parser Layer

Improve:
1. real-world frontend IR handling and edge-case extraction.
2. explicit unsupported reporting.
3. optional long-term migration toward stronger operation-level traversal.

Expected effect:
- Better recall on realistic kernels with less hidden behavior.

## 11.4.3 Emitter Layer

Improve:
1. fail-fast validation for required metadata.
2. better dialect-specific attribute inference where safe.
3. stronger emitted artifact validation hooks.

Expected effect:
- fewer invalid artifacts and clearer failure surfaces.

## 11.4.4 Workflow and UX Layer

Improve:
1. policy profiles for trusted versus exploratory runs.
2. benchmark/report pipelines for trend visibility.
3. contributor-facing docs for fast onboarding.

Expected effect:
- smoother team operations and reproducible quality tracking.

---

## 11.5 Metrics and Targets

Use relative targets to avoid snapshot confusion from mixed test scopes.

Core targets for 12 weeks:
1. Strict accepted-refuted rate: 0 in trusted profile.
2. Emitted artifact verifier pass rate in trusted CI lane: 100% on required suites.
3. Weak-kernel unproved/refuted rate: decreasing trend across weekly reports.
4. StableHLO canonical fixture failure count: decreasing trend.
5. Unsupported-pattern diagnostic coverage: 100% for known unsupported forms in tracked corpus.
6. Weekly report continuity: 12/12 weekly runs generated without schema break.

---

## 11.6 Risks and Mitigations

1. Risk: solver timeouts block proof-rate improvements.
- Mitigation: split trusted and exploratory profiles, keep explicit unproved classification, prioritize diagnostic clarity.

2. Risk: coverage expansion reintroduces fallback shortcuts.
- Mitigation: fail-fast policy checks and regression gates before merges.

3. Risk: toolchain drift in Docker and verifier availability.
- Mitigation: pin tags, keep lock files current, maintain smoke tests for C and C++.

4. Risk: metric ambiguity from mixed scopes.
- Mitigation: every report includes fixture set, profile, timeout, and date metadata.

---

## 11.7 Deliverables by End of Horizon

By end of this strategy cycle, expected deliverables are:
1. Trusted CI profile with strict policy and artifact verification.
2. Improved weak-kernel reliability with explicit diagnostics.
3. Expanded and better-tested StableHLO path.
4. Real-world corpus trend report with parser/emitter diagnostics.
5. Decision memo for next research track with first milestone.

Execution details are formalized in:
- [12 - Three-Person Parallel Plan V2 (12 Weeks)](./12_three_person_parallel_plan_v2_12_weeks.md)
