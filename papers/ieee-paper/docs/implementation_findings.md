# LoopHole POC Implementation Findings (April 2026)

This document records what is currently implemented in the repository and what should be treated as in-progress or planned work. It is intended to be the factual source for the IEEE paper draft.

## 1. Scope and Method

- Scope reviewed:
  - `packages/loophole/src/loophole/*`
  - `packages/loophole/tests/integration/*`
  - `packages/loophole/reports/stablehlo_batch_lift_report_2026-04-06.json`
  - project docs in `docs/`
- Method: static source and artifact inspection (no new feature implementation in this pass).
- Writing policy for paper: only include statements directly supported by code/tests/reports.

## 2. Implemented and Verifiable (Current State)

### 2.1 End-to-end lifting pipeline exists

The POC contains an end-to-end pipeline that:

1. Parses Affine-style MLIR loop nests.
2. Performs symbolic candidate matching.
3. Verifies candidate equivalence with Z3.
4. Emits target dialect MLIR (Linalg or StableHLO).

Primary implementation anchors:

- `packages/loophole/src/loophole/lifter.py`
- `packages/loophole/src/loophole/affine_extractor.py`
- `packages/loophole/src/loophole/sympy_tracer.py`
- `packages/loophole/src/loophole/z3_checker.py`
- `packages/loophole/src/loophole/emitter.py`

### 2.2 CLI workflow exists and is usable

The CLI includes practical commands for compile/lift/verify/batch flows (including reporting).

Implementation anchor:

- `packages/loophole/src/loophole/cli.py`

Observed command surface includes:

- `compile`
- `lift`
- `cgeist`
- `lift-c`
- `verify`
- `batch`
- `sketches`
- `demo`

### 2.3 Bounded sketch coverage exists

The implemented sketch library is finite and manually encoded, not open-ended synthesis.

Implementation anchor:

- `packages/loophole/src/loophole/sketch_library.py`

Covered families include:

- Linalg: matmul, matvec/vecmat, dot, transpose/copy, elementwise ops, reductions, selected conv and pooling variants.
- StableHLO: dot_general variants, transpose, add/subtract/multiply, reduce sum/max, convolution.

### 2.4 Structured reporting artifact exists

A concrete batch report artifact is present:

- `packages/loophole/reports/stablehlo_batch_lift_report_2026-04-06.json`

From this artifact:

- Total files: 7
- PROVED: 5
- REFUTED: 2
- UNPROVED_TIMEOUT: 0
- Reported refuted fixtures: conv1d, conv2d

### 2.5 Test evidence exists for current behavior

Relevant integration tests validate key pathways and reported behavior.

Examples:

- `packages/loophole/tests/integration/test_stablehlo_phase3_ops.py`
- `packages/loophole/tests/integration/test_cli_batch_report.py`
- `packages/loophole/tests/integration/test_matmul.py`
- `packages/loophole/tests/integration/test_demo_script_smoke.py`

## 3. Known Limitations (Current State)

### 3.1 Not a completed general-purpose lifter yet

The current system should be described as a POC milestone. Coverage is intentionally bounded by the hand-authored sketch set and current matching logic.

### 3.2 Some high-level docs are broader than implemented state

Some project docs discuss broader coverage and benchmark scope than is currently evidenced by checked-in POC artifacts. These should be treated as roadmap statements unless backed by code/tests/reports.

Relevant docs to qualify carefully:

- `docs/background/project-overview.md`
- `docs/architecture/mlir-architecture.md`
- `docs/planning/implementation-roadmap.md`

### 3.3 Result claims must stay conservative

Current quantitative claims should be limited to:

- test behavior currently present in repository tests
- the available batch report artifact

Claims about broad Polybench-scale outcomes or full operator completeness should be deferred to future-work language unless new evidence is added in the next pass.

## 4. Paper Writing Guidance for This Pass

Use a three-tier statement style in the paper:

1. Implemented now: directly supported by source/tests/reports.
2. Current limitations: directly supported by source/tests/reports.
3. Planned next pass: explicitly marked as in-progress and not yet fully validated.

Recommended wording style:

- Prefer: "currently supports", "in this POC", "under active refinement", "preliminary artifact evidence".
- Avoid: "fully supports", "complete", "production-ready", "comprehensive benchmarked".

## 5. Delta Checklist for Next Revision Pass

When bugs are fixed and coverage expands, update this findings file first, then update the paper.

Update points:

- New operator coverage (with exact sketch/emitter/test evidence)
- New batch or benchmark report artifacts
- Any change in acceptance policy (strict vs loose)
- Any verified robustness improvements in parser/verification/emission
