# C-17 Research-Track Decision Package (Evidence-Based)

Last updated: 2026-04-20
Status: Provisional recommendation pending final A/B week-10 evidence handoff.

## Decision Inputs

Person C evidence inputs:
1. Weekly report trend deltas (`trend_summary`).
2. Canonical StableHLO coverage summaries (`canonical_stablehlo_summary`).
3. Trusted vs exploratory acceptance behavior from batch reports.

Required A/B inputs before final lock:
1. Person A: weak-kernel proof-quality trend and strict-profile stability.
2. Person B: parser/emitter reliability trend and trusted verifier pass rate.

## Scoring Rubric

Each candidate track is scored on:
1. Trust impact (30%)
2. Feasibility in next 12 weeks (25%)
3. Evidence readiness from existing infra (20%)
4. Novelty potential (15%)
5. Integration risk (10%, inverse)

## Candidate Ranking (Current)

1. Dynamic shape lifting and proof envelope expansion.
2. Transform-dialect schedule synthesis for lifted kernels.
3. Sparse reverse lifting track.

## Recommended Primary Track

Primary recommendation: Dynamic shape lifting and proof envelope expansion.

Why this is ranked first now:
1. Highest direct impact on current trust and coverage limitations.
2. Strongest reuse of existing verifier and report instrumentation.
3. Lower integration risk than sparse-first path under current parser/emitter maturity.

## Milestone-1 Definition (Proposed)

Within first milestone window:
1. Define bounded dynamic-shape canonical kernel subset.
2. Add explicit dynamic-shape support diagnostics in reports.
3. Demonstrate at least one end-to-end trusted proof case in subset.
4. Publish baseline and week-over-week trend package using new dashboard fields.

## Finalization Gate

Before declaring final selection complete:
1. Incorporate A/B week-10 evidence package.
2. Recompute ranking with final scores.
3. Record final sign-off in sprint-6 handoff notes.
