# Polygeist + MLIR Docker Toolchain

This folder contains a reproducible Docker toolchain for compiling C/C++ into MLIR and then lifting it with LoopHole.

## Pinned Revisions

Pinned versions are stored in `toolchain.lock`:

- `LLVM_TAG`
- `LLVM_REF`
- `POLYGEIST_BRANCH`
- `POLYGEIST_REF`

The Dockerfile defaults to those pinned refs, and can be overridden with `--build-arg` if needed.

## Build Image

From the repository root:

```bash
docker build \
  -f POC_initial_demo/docker/Dockerfile \
  -t loophole-polygeist:llvm17 \
  POC_initial_demo
```

## Use Prebuilt Image (No Local Build)

If a prebuilt image is published, contributors can pull it directly:

```bash
docker pull ghcr.io/schizoid-man/loophole-polygeist:llvm17
```

Then run compile by pointing LoopHole at that image:

```bash
loophole compile path/to/kernel.cpp \
  --docker-image ghcr.io/schizoid-man/loophole-polygeist:llvm17 \
  --std c++17 \
  -o path/to/kernel.mlir
```

## Two-Step Workflow (Recommended)

1. Compile C/C++ into MLIR using Polygeist in Docker.
2. Lift the generated MLIR to Linalg or StableHLO.

### Step 1: Compile

```bash
loophole compile path/to/kernel.c -o path/to/kernel.mlir
```

For C++:

```bash
loophole compile path/to/kernel.cpp -o path/to/kernel.mlir --std c++17
```

Useful options:

- `--docker-image loophole-polygeist:llvm17`
- `--frontend-binary cgeist` or `cgeist++`
- `--no-affine-raise` to keep SCF form
- `--print-command` to show the exact docker invocation

### Step 2: Lift

```bash
loophole lift path/to/kernel.mlir --target linalg -o path/to/kernel_linalg.mlir
```

Or StableHLO:

```bash
loophole lift path/to/kernel.mlir --target stablehlo -o path/to/kernel_stablehlo.mlir
```

## Notes

- The `compile` command mounts the common root of input/output paths into the container and writes output directly on the host.
- Source and output must be on the same drive (Windows Docker volume constraint for a single mount).
- This setup removes regex dependence in extraction and is designed for MLIR/Polygeist-based workflows.

## Publishing for Team Access

A workflow is provided at `.github/workflows/publish-docker-image.yml`.

- Trigger: push to `main` affecting Docker files, or manual `workflow_dispatch`
- Registry: GitHub Container Registry (`ghcr.io`)
- Published tags:
  - `ghcr.io/<owner>/loophole-polygeist:llvm17`
  - `ghcr.io/<owner>/loophole-polygeist:latest`
  - `ghcr.io/<owner>/loophole-polygeist:sha-<shortsha>`

After the first publish, set the package visibility to public in GitHub Packages so collaborators can pull without authentication.

## Private Repo: No-PAT Download Option

If you want collaborators to avoid PAT setup entirely, use the export workflow:

- Workflow: `.github/workflows/export-docker-image-artifact.yml`
- Behavior: pulls existing `ghcr.io/.../loophole-polygeist:<tag>` and uploads a downloadable `.tar.gz` artifact
- Important: it does not rebuild the image

Collaborator flow (browser-only):

1. Open the Actions run for `Export Existing Docker Image Artifact`.
2. Download the artifact from the run summary.
3. Extract/reassemble if split (instructions are included in `README_LOAD_IMAGE.txt`).
4. Load the image locally:

```bash
docker load -i loophole-polygeist-llvm17.tar.gz
```

This works for private repositories as long as collaborators have access to the repo and Actions artifacts.
