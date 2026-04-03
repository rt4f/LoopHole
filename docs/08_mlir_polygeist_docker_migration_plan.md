# 08 - MLIR + Polygeist Docker Migration Plan (Sequential Implementation)

**Date:** 2026-03-31
**Status:** Implementation started
**Primary objective:** Replace regex-based parsing with MLIR-native parsing and run the full pipeline in a Dockerized LLVM/MLIR/Polygeist environment that works from both Linux and Windows hosts.

---

## 1. Why this migration is necessary

The current POC proves feasibility, but its parsing stage is regex-driven and fragile for broader MLIR syntax and real-world C/C++ inputs. The project needs:

1. A real frontend from C/C++ to MLIR using Polygeist (`cgeist`).
2. A real MLIR parser/traversal path in Python instead of regex extraction.
3. A reproducible platform-independent runtime (Docker) usable from Linux and Windows.

This document defines the implementation sequence, bottlenecks, mitigation steps, and acceptance criteria.

---

## 2. Scope and non-scope

## 2.1 In scope

1. Build a self-contained Docker image that compiles LLVM/MLIR/Clang + Polygeist from source.
2. Add a C/C++ ingestion frontend in the POC that invokes `cgeist`.
3. Hard cutover from regex parser to MLIR-native parsing APIs.
4. Add one-step and two-step CLI flows.
5. Add tests and smoke checks in containerized runs.
6. After migration, address current correctness limitations identified in audit doc 07.

## 2.2 Out of scope for this phase

1. Full production-grade formal proofs for arbitrary dynamic-shape kernels.
2. GPU backend codegen optimization.
3. End-to-end deployment packaging beyond Docker workflow.

---

## 3. Current baseline and constraints

## 3.1 Baseline summary

1. POC pipeline currently works with MLIR-text input and regex extraction.
2. Existing scripts already install and smoke-test Polygeist in WSL:
   - `scripts/wsl_install_polygeist_native.sh`
   - `scripts/wsl_smoke_test.sh`
3. Tests currently pass for the existing behavior.

## 3.2 Key constraints

1. LLVM/Polygeist builds are heavy (time, RAM, disk).
2. MLIR Python APIs can be version-sensitive.
3. Windows host support should be achieved via Docker Desktop + Linux containers (not native Windows LLVM build path).
4. Existing tests depend on current extraction contracts (`LoopNestInfo`, `AccessPattern`, `ComputeOp`) and must remain compatible during parser replacement.

---

## 4. Target architecture after migration

## 4.1 End-to-end data flow

1. C/C++ source files -> `cgeist` -> Affine/SCF/Arith MLIR.
2. MLIR text -> MLIR parser (`mlir.ir.Module.parse`) -> operation walk.
3. MLIR semantic extraction -> `LoopNestInfo` contract.
4. Sketch matching + SymPy confidence -> candidate list.
5. Z3 verification -> accept/reject/partial states.
6. Linalg or StableHLO emission.

## 4.2 Execution modes

1. One-step: C/C++ directly to lifted dialect output.
2. Two-step: C/C++ to MLIR only, then MLIR to lifted output.
3. Existing MLIR-file mode remains available.

---

## 5. Sequential implementation plan

## Phase A - Docker toolchain foundation

### A.1 Deliverables

1. New Docker assets in repository (for example `docker/` folder):
   - `Dockerfile`
   - optional `entrypoint.sh`
   - optional helper scripts for build/run
2. Image contains:
   - LLVM + Clang + MLIR + LLD (pinned revisions)
   - Polygeist build from source against same LLVM revision
   - Python runtime + POC dependencies
   - `cgeist`, `mlir-opt`, `polygeist-opt`, and `loophole` CLI

### A.2 Implementation notes

1. Reuse logic from existing WSL scripts as source of truth.
2. Prefer multi-stage build to reduce runtime image size.
3. Use `Ninja` and `ccache` to speed incremental rebuilds.
4. Pin revisions (commit SHA or tags) to avoid API drift.

### A.3 Bottlenecks and rectification

1. **Bottleneck:** Very long compile times.
   - **Rectify:** BuildKit cache mounts, `ccache`, and pinned toolchain layer reuse.
2. **Bottleneck:** OOM during LLVM link stage.
   - **Rectify:** Lower parallelism (`-j`), use `lld`, tune Docker memory, split build stages.
3. **Bottleneck:** Large final image size.
   - **Rectify:** Keep build artifacts in build stage only, copy runtime binaries/libraries only.
4. **Bottleneck:** Version mismatch between Polygeist and LLVM.
   - **Rectify:** Lock compatible SHAs and add startup version check command.

### A.4 Exit criteria

1. `docker build` succeeds from clean host.
2. `cgeist --help` and `mlir-opt --help` work inside container.
3. `loophole --help` works inside container.

---

## Phase B - C/C++ frontend integration

### B.1 Deliverables

1. New frontend module in POC (proposed):
   - `POC_initial_demo/src/loophole/polygeist_frontend.py`
2. Module supports:
   - language selection (`c`, `cpp`)
   - multi-file inputs
   - include directories
   - macro defines
   - standard selection (`c11`, `c17`, `c++17`, `c++20`)
   - extra compile flags
   - target function filtering (if needed)
3. Returns generated MLIR text and structured diagnostics.

### B.2 CLI extension (proposed)

1. Add `lift-c` command for one-step flow.
2. Add `cgeist` command for explicit two-step generation.
3. Keep current `lift` command for direct MLIR input.

### B.3 Bottlenecks and rectification

1. **Bottleneck:** `cgeist` flag complexity for real C++ projects.
   - **Rectify:** Add pass-through `--clang-arg` support and provide profile presets.
2. **Bottleneck:** Header resolution failures.
   - **Rectify:** Explicit `-I` management and diagnostics that print unresolved includes.
3. **Bottleneck:** Unsupported C++ constructs in Polygeist path.
   - **Rectify:** Document supported subset and add graceful error messages with fallback advice.
4. **Bottleneck:** Multi-file linkage behavior confusion.
   - **Rectify:** Define deterministic compilation model and require explicit entry function.

### B.4 Exit criteria

1. One-step command produces lifted output from sample C and C++ kernels.
2. Two-step command generates MLIR file and can be consumed by `lift`.
3. Errors are actionable and include tool command context.

---

## Phase C - Hard parser cutover to MLIR APIs

### C.1 Deliverables

1. Replace internals of extraction stage to use MLIR operation traversal.
2. Preserve `LoopNestInfo` contract so downstream modules continue functioning.
3. Remove regex parser path entirely.

### C.2 Implementation strategy

1. Parse module with MLIR Python APIs.
2. Walk operations and collect:
   - loop IVs and bounds (`affine.for`, `scf.for`)
   - load/store index maps and tensors
   - arithmetic payload graph
   - reduction vs parallel iterator semantics
3. Normalize gathered semantics into existing dataclasses.

### C.3 Bottlenecks and rectification

1. **Bottleneck:** MLIR Python API differences across revisions.
   - **Rectify:** Freeze toolchain revision in Docker and run parser contract tests in CI.
2. **Bottleneck:** Affine map and symbol extraction complexity.
   - **Rectify:** Add helper utilities with narrow responsibilities and golden tests per op kind.
3. **Bottleneck:** Dynamic shape and symbolic bounds extraction edge cases.
   - **Rectify:** Introduce explicit unsupported-case diagnostics instead of silent fallbacks.
4. **Bottleneck:** Potential regression in sketch matching due to changed normalization.
   - **Rectify:** Snapshot current behavior on fixtures and enforce parity gates.

### C.4 Exit criteria

1. Existing extractor tests pass after adaptation.
2. No regex parsing path remains in source.
3. Integration tests remain green for existing fixture kernels.

---

## Phase D - Orchestration and status semantics hardening

### D.1 Deliverables

1. Integrate new frontend and parser into `lifter.py` and `cli.py`.
2. Ensure verification status semantics are strict and user-trustworthy.
3. Keep output reporting clear across equivalent, unknown, timeout, not-equivalent.

### D.2 Bottlenecks and rectification

1. **Bottleneck:** Backward compatibility concerns in CLI behavior.
   - **Rectify:** Keep old command paths functional and deprecate with clear warnings only if needed.
2. **Bottleneck:** Partial-success semantics currently permissive.
   - **Rectify:** Tighten acceptance logic to reject `NOT_EQUIVALENT` for partial success.
3. **Bottleneck:** User confusion over no-verify behavior.
   - **Rectify:** Implement true verification bypass flag and explicit output annotation.

### D.3 Exit criteria

1. Existing CLI commands still work.
2. New C/C++ commands work in one-step and two-step modes.
3. Status and reports match strict semantics.

---

## Phase E - Validation matrix and cross-platform checks

### E.1 Test layers

1. Unit tests for frontend command generation and diagnostics parsing.
2. Parser contract tests with affine/scf/load/store/arith combinations.
3. Integration tests for C and C++ sample kernels.
4. Container smoke tests for one-step and two-step flows.

### E.2 Host matrix

1. Linux host -> Docker engine.
2. Windows host -> Docker Desktop with WSL2 backend.

### E.3 Bottlenecks and rectification

1. **Bottleneck:** Windows path mount issues.
   - **Rectify:** Normalize path handling and provide tested run scripts for PowerShell/Bash.
2. **Bottleneck:** CRLF line ending effects in mounted scripts.
   - **Rectify:** enforce LF for container scripts and use `.gitattributes` where necessary.
3. **Bottleneck:** Non-deterministic timing in heavy builds/tests.
   - **Rectify:** separate long-running toolchain build from quick functional tests.

### E.4 Exit criteria

1. Full test suite passes in container.
2. Smoke flows pass from both Linux and Windows hosts.
3. Reproducible instructions produce same outcomes on clean machines.

---

## Phase F - Post-migration limitation fixes (from audit 07)

## F.1 P0 correctness and trustworthiness

1. Disallow `NOT_EQUIVALENT` from partial success classification.
2. Remove high-confidence fallback acceptance for `NOT_EQUIVALENT`.
3. Fix reduction encoding to cover all reduction dimensions.
4. Remove symbolic zero placeholders in reduction encoding.
5. Add strict negative tests (for example ReLU must not validate as scale/copy).

## F.2 P1 semantic robustness

1. Improve payload inference for max/relu constants.
2. Replace substring IV role checks with token-level matching.
3. Reduce shape hardcoding in emitters; fail fast when required metadata is missing.
4. Infer StableHLO reduction dimensions from extracted semantics.

## F.3 P2 developer UX and maintainability

1. Fix standalone demo script drift.
2. Implement true no-verify semantics.
3. Add module execution entrypoint in CLI.
4. Sync documentation and sketch counts.

### F.4 Bottlenecks and rectification

1. **Bottleneck:** Fixes may alter expected outputs and break tests.
   - **Rectify:** update tests with explicit behavioral intent and migration notes.
2. **Bottleneck:** Z3 encoding changes can increase solver cost.
   - **Rectify:** cache sub-formulas, use structural pre-checks, tune timeout strategy.
3. **Bottleneck:** More strict behavior may reduce apparent success rate.
   - **Rectify:** clearly distinguish soundness-first status categories in reports.

### F.5 Exit criteria

1. New strict tests are green.
2. Known high-risk mismatches no longer pass as partial/equivalent.
3. Audit 07 P0/P1/P2 issues are tracked as closed or deferred with rationale.

---

## 6. Probable bottleneck ledger (condensed)

| Area | Risk | Impact | Likelihood | Rectification |
|---|---|---|---|---|
| Toolchain build | LLVM build time too high | Delays all phases | High | Build cache, ccache, pinned layers |
| Toolchain memory | OOM during link | Build failures | Medium/High | Lower -j, more Docker RAM, lld |
| Version compatibility | Polygeist/LLVM mismatch | Runtime failures | Medium | Lock SHAs, startup sanity checks |
| Frontend flags | C++ include and macro complexity | Parse failures | High | Pass-through clang args, clear diagnostics |
| Parser migration | Behavior regressions | Wrong extraction | Medium | Contract tests + fixture parity gates |
| Verification semantics | False confidence states | Trust erosion | High | Strict status logic, remove bad fallback |
| Windows host UX | Path and mount issues | Developer friction | Medium | PowerShell wrappers and tested examples |

---

## 7. Work breakdown structure (implementation checklist)

## 7.1 Docker and infra

1. Add Docker build files and scripts.
2. Pin LLVM/Polygeist refs.
3. Add smoke command script in container.
4. Document host usage for Linux and Windows.

## 7.2 Code changes

1. Add Polygeist frontend module.
2. Extend CLI with one-step and two-step commands.
3. Replace extractor internals with MLIR APIs.
4. Integrate flow with lifter and reports.

## 7.3 Tests

1. New frontend unit tests.
2. Parser contract parity tests.
3. C/C++ integration tests.
4. Docker smoke test command in CI/local run guide.

## 7.4 Post-migration quality fixes

1. Apply P0 fixes from audit.
2. Apply P1 fixes from audit.
3. Apply P2 fixes from audit.

---

## 8. Definition of done

Migration is complete when all conditions are true:

1. Regex parser path has been removed and replaced by MLIR-native traversal.
2. C and C++ kernels can be ingested via Polygeist inside Docker.
3. One-step and two-step CLI flows both work.
4. End-to-end tests pass in container from Linux and Windows hosts.
5. P0 correctness issues from audit are fixed and covered by tests.
6. Updated documentation reflects actual commands, counts, and behavior.

---

## 9. Immediate next implementation actions

1. Create Docker assets and perform first successful toolchain build.
2. Introduce Polygeist frontend module and wire a minimal CLI command (`cgeist`) for two-step flow.
3. Add parser adapter skeleton using MLIR APIs while preserving `LoopNestInfo` contract.
4. Add smoke tests for C and C++ sample kernels in container.
5. Begin P0 limitation fixes once migration path is green.

---

## 10. Traceability to existing docs

1. Problem framing and architecture intent: docs 01-06.
2. Current implemented limitations and risk hotspots: `docs/07_poc_implementation_audit.md`.
3. This plan is the executable migration bridge from current POC to MLIR/Polygeist-first implementation.
