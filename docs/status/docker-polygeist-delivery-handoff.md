# 09 - Docker + Polygeist Delivery Handoff

Date: April 2, 2026

This document is the complete handoff for the Dockerized Polygeist + MLIR integration work completed in the current cycle.

---

## 1) Executive Summary

The project now has a reproducible Docker toolchain that compiles C/C++ kernels to MLIR using Polygeist, then feeds the result into the existing LoopHole lifting flow.

Delivered outcomes:

- Built and validated image: `loophole-polygeist:llvm17`
- Added two-step workflow:
  1. `loophole compile` (C/C++ -> MLIR)
  2. `loophole lift` (MLIR -> Linalg/StableHLO)
- Added resilient C++ frontend handling (`cgeist++` fallback symlink)
- Added dynamic memref normalization in compile pipeline (`-1` -> `?`) so `mlir-opt` accepts pointer-heavy C++ outputs
- Added repository docs and a publish workflow so teammates can pull prebuilt images from `ghcr.io`

---

## 2) What Was Implemented

### 2.1 Docker Toolchain and Build System

Primary files:

- `docker/polygeist/Dockerfile`
- `docker/polygeist/toolchain.lock`
- `packages/loophole/.dockerignore`
- `docs/guides/docker-polygeist.md`

Highlights:

- Base image: Ubuntu 24.04
- LLVM/MLIR toolchain: apt LLVM 17 packages
- Polygeist source build: branch `llvm17bump`
- Build output target: `cgeist` (with `cgeist++` installed if present, symlink fallback otherwise)

### 2.2 CLI Integration

Primary file:

- `packages/loophole/src/loophole/cli.py`

Delivered behavior:

- New `loophole compile` command (Docker-backed)
- C vs C++ frontend auto-select (`cgeist` / `cgeist++`)
- Compile pipeline:
  - frontend emit
  - dynamic-shape normalization (`sed` rewrite)
  - `mlir-opt --canonicalize`
- Default compile image updated to `loophole-polygeist:llvm17`

### 2.3 Documentation and Planning

Primary files:

- `docs/status/poc-implementation-audit.md`
- `docs/planning/three-person-execution-plan-v1.md`
- `README.md` (docs index updated)

New handoff file:

- `docs/status/docker-polygeist-delivery-handoff.md` (this file)

### 2.4 Distribution Automation

New workflow file:

- `.github/workflows/publish-docker-image.yml`

Purpose:

- Build and publish the Docker image to GHCR on push to `main` (Docker-related paths) or manual dispatch

---

## 3) Validation Performed

Build and runtime validations completed in local environment:

- Docker image rebuild completed successfully for `loophole-polygeist:llvm17`
- In-container checks confirmed binaries:
  - `cgeist`
  - `cgeist++`
- C smoke compile validated:
  - input: `packages/loophole/examples/smoke_matmul.c`
  - output: affine MLIR with nested `affine.for` and expected loads/stores
- C++ smoke compile validated:
  - fixed-shape kernel: success through `cgeist++ | mlir-opt`
  - pointer-arg kernel: success after `-1` to `?` normalization
- End-to-end CLI verification:
  - `loophole compile ... --docker-image loophole-polygeist:llvm17` produces valid MLIR

---

## 4) Drawbacks and Limitations

### 4.1 Toolchain Fragility

- The Dockerfile includes targeted Polygeist/LLVM17 compatibility patches.
- Upstream Polygeist or LLVM package changes can break this patch set.
- This is currently a practical compatibility bridge, not a clean upstreamed solution.

### 4.2 Feature Surface Is Intentionally Reduced

- Optional GPU-related passes and some conversion paths were removed/disabled to stabilize the build.
- Result: better reliability for current CPU-centric compile flow, but less pass coverage.

### 4.3 Dynamic-Shape Normalization Is a Text Rewrite

- The compile pipeline normalizes legacy `memref<-1x...>` syntax to `memref<?x...>` with `sed`.
- This is robust for the observed output format but still a textual transform, not an AST-level canonicalization pass.

### 4.4 Platform and Runtime Constraints

- Assumes Linux containers under Docker Desktop on Windows.
- Compile command requires input/output to share a common mount root (same drive on Windows).
- Build time and image size remain significant due to source build steps.

### 4.5 Current Tag Strategy

- Team usage should treat `:llvm17` as the stable functional tag for this cycle.
- `:latest` is convenient but can drift if future major toolchain changes land.

---

## 5) Operational Recommendations

### 5.1 For Daily Use

- Prefer pinned image tag: `ghcr.io/schizoid-man/loophole-polygeist:llvm17`
- Keep `--print-command` enabled during debugging for reproducibility
- Use `loophole compile` and `loophole lift` as separate steps to isolate failures

### 5.2 For Team Reproducibility

- Keep Docker refs in `toolchain.lock` synchronized with Dockerfile defaults
- Treat Dockerfile patch blocks as audited compatibility policy and review them on each LLVM bump
- Keep at least one C and one C++ smoke kernel in CI validation

---

## 6) How To Publish So Friends Can Reuse the Image

Two options are supported.

### Option A (Recommended): GitHub Actions -> GHCR

1. Push this repository to `main` (includes workflow file).
2. GitHub Actions runs `.github/workflows/publish-docker-image.yml`.
3. Image is published to:
   - `ghcr.io/<owner>/loophole-polygeist:llvm17`
   - `ghcr.io/<owner>/loophole-polygeist:latest`
   - `ghcr.io/<owner>/loophole-polygeist:sha-<shortsha>`
4. In GitHub Packages, set package visibility to Public once.

Friend pull command:

```bash
docker pull ghcr.io/schizoid-man/loophole-polygeist:llvm17
```

Friend compile command example:

```bash
loophole compile path/to/kernel.cpp \
  --docker-image ghcr.io/schizoid-man/loophole-polygeist:llvm17 \
  --std c++17 \
  -o path/to/kernel.mlir
```

### Option B: Manual Local Push to Registry

If you prefer manual publication from your machine:

```bash
docker tag loophole-polygeist:llvm17 ghcr.io/schizoid-man/loophole-polygeist:llvm17
docker push ghcr.io/schizoid-man/loophole-polygeist:llvm17
```

### Option C: Private Repo, No PAT for Collaborators (Artifact Download)

If collaborators are repo members and you want zero PAT setup for them:

1. Run `.github/workflows/export-docker-image-artifact.yml` via `workflow_dispatch`.
2. The workflow logs into GHCR, pulls the already-built image, and uploads a downloadable image archive artifact.
3. Collaborators download from the Actions run page and run `docker load` locally.

This option does not rebuild the image and keeps distribution private to repo collaborators.

---

## 7) Repository Push Checklist

Use this sequence:

```bash
git add README.md docs/07_poc_implementation_audit.md docs/08_three_person_parallel_execution_plan.md docs/09_docker_polygeist_delivery_handoff.md

git add POC_initial_demo/.dockerignore POC_initial_demo/docker/Dockerfile POC_initial_demo/docker/README.md POC_initial_demo/docker/toolchain.lock

git add POC_initial_demo/examples/smoke_matmul.c

git add POC_initial_demo/src/loophole/affine_extractor.py POC_initial_demo/src/loophole/cli.py POC_initial_demo/POC_DOCUMENTATION.md

git add .github/workflows/publish-docker-image.yml

git commit -m "Add Dockerized Polygeist LLVM17 pipeline, docs handoff, and GHCR publish workflow"

git push origin main
```

---

## 8) Known Next Steps

High priority:

1. Track and reduce patch delta against upstream Polygeist.
2. Add CI smoke test that runs `loophole compile` with both C and C++ kernels.
3. Add strict checks for emitted MLIR validity on representative kernels.

Medium priority:

1. Evaluate a cleaner dynamic-shape handling path than text normalization.
2. Split Docker patch logic into scripted, testable patch files for easier maintenance.
3. Add image SBOM and vulnerability scan to publish workflow.
