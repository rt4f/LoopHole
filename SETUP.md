# LoopHole Setup (Docker + Polygeist)

This is the do-it-yourself guide for getting LoopHole running with Polygeist and
MLIR inside Docker. You do **not** need to install LLVM, MLIR or Polygeist on
your machine — they live in a Docker image. Only the LoopHole Python CLI runs
on the host.

```
C/C++ source ──(docker: cgeist + mlir-opt)──▶ affine MLIR ──(host: loophole lift)──▶ Linalg / StableHLO
```

---

## 1. Prerequisites

| Tool | Version used | Check |
|---|---|---|
| Docker Desktop (Linux containers, WSL2 backend) | 4.79 / Engine 29.5 | `docker version` |
| Python | 3.11+ (3.13 works) | `python --version` |
| Git | any | `git --version` |

Docker Desktop resources: the build uses `BUILD_THREADS=4` by default and was
verified with 12 CPUs / ~8 GB RAM. With less than ~6 GB RAM, lower the thread
count (see step 2). Free disk: ~6 GB for the image.

Make sure Docker Desktop is **running** before any step below (`docker info`
must succeed).

---

## 2. Get the Polygeist image

There are two ways. **Option A (build locally) needs no credentials.**

### Option A — build locally (recommended)

From the repository root:

```powershell
docker build -f docker/polygeist/Dockerfile `
  -t loophole-polygeist:llvm17 `
  -t ghcr.io/schizoid-man/loophole-polygeist:llvm17 `
  packages/loophole
```

- The Dockerfile lives in `docker/polygeist/`, but the build context is `packages/loophole` (the image installs the package's `requirements.txt`).
- First build took **~5 minutes** on a 12-CPU / 8 GB Docker Desktop (apt LLVM 17 packages + building `cgeist` from Polygeist branch `llvm17bump`); expect longer on slower machines/networks. Rebuilds are cached. Resulting image: ~3.4 GB.
- Two tags are applied so that every LoopHole command and helper script finds the image without passing `--docker-image` (some default to the short name, some to the `ghcr.io/...` name).
- Low on RAM? Add `--build-arg BUILD_THREADS=2`.
- Pinned toolchain refs live in `docker/polygeist/toolchain.lock`.

> An older, much heavier Dockerfile that builds all of LLVM from source is kept in `tools/legacy/docker-source-build/`. It is unsupported and you don't need it.

### Option B — pull the prebuilt image from GHCR

The package `ghcr.io/schizoid-man/loophole-polygeist` is **private**, so an
anonymous `docker pull` fails with `denied`. You must log in first:

1. Create a GitHub Personal Access Token (classic) with the `read:packages` scope.
2. Log in and pull:

```powershell
docker login ghcr.io -u Schizoid-man     # paste the PAT when prompted for password
docker pull ghcr.io/schizoid-man/loophole-polygeist:llvm17
docker tag  ghcr.io/schizoid-man/loophole-polygeist:llvm17 loophole-polygeist:llvm17
```

(Alternative without a PAT: run the *Export Existing Docker Image Artifact*
GitHub Action, download the `.tar.gz`, and `docker load -i <file>`.)

### Verify the image

```powershell
docker run --rm loophole-polygeist:llvm17 bash -lc "which cgeist cgeist++ mlir-opt && mlir-opt --version"
```

Expected:

```
/usr/local/bin/cgeist
/usr/local/bin/cgeist++
/usr/lib/llvm-17/bin/mlir-opt
Ubuntu LLVM version 17.0.6
```

---

## 3. Install the LoopHole CLI on the host

From the repository root (PowerShell):

```powershell
cd packages/loophole
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
loophole --help
```

`.venv/` is already git-ignored.

On Windows, also enable Python UTF-8 mode (avoids `UnicodeDecodeError: 'charmap' codec`
when LoopHole reads Docker output):

```powershell
$env:PYTHONUTF8 = "1"     # per session; or setx PYTHONUTF8 1 to make it permanent
```

---

## 4. Run it

All commands below are run from `packages/loophole/` with the venv active.

### 4.1 Compile C → MLIR (Polygeist in Docker)

```powershell
loophole compile examples/smoke_matmul.c -o out/smoke_matmul.mlir --print-command
```

Expect a `COMPILE SUCCEEDED` panel, and `out/smoke_matmul.mlir` containing
`func.func @matmul_2x2` with three nested `affine.for` loops, `affine.load` and
`affine.store`. For C++ add `--std c++17`.

### 4.2 Lift MLIR → Linalg / StableHLO (host)

```powershell
loophole lift out/smoke_matmul.mlir --target linalg    -o out/smoke_matmul_linalg.mlir
loophole lift out/smoke_matmul.mlir --target stablehlo -o out/smoke_matmul_stablehlo.mlir
```

Expect `FORMALLY PROVED - EQUIVALENT`, lifted to `linalg.matmul` and
`stablehlo.dot_general` respectively. (`out/` is git-ignored scratch space.)

### 4.3 One-step C → lifted tensor IR

```powershell
loophole lift-c examples/smoke_matmul.c --target linalg
```

Runs `cgeist` in Docker (auto-detected image) then lifts; expect
`FORMALLY PROVED - EQUIVALENT` → `linalg.matmul`.

### 4.4 Batch over fixtures with MLIR verification (mlir-opt in Docker)

```powershell
loophole batch tests/fixtures --profile trusted --include-glob matmul.mlir --max-files 1 --no-write-emitted
```

Expect `1 PROVED / 0 UNPROVED_TIMEOUT / 0 REFUTED`.

The trusted profile validates emitted MLIR with `mlir-opt`; with no local
`mlir-opt` installed, LoopHole automatically runs it inside the Docker image.
Check which verifier it picked:

```powershell
python -c "from loophole import mlir_validator as m; print(m.find_mlir_verifier())"
# loophole-docker-mlir-verify  -> mlir-opt runs in Docker
```

### 4.5 Tests

```powershell
python -m pytest tests -q
```

Current baseline (Sept 2026): **364 passed, 1 failed**. The known failure is
`tests/integration/test_matmul.py::TestMatmulLiftPipeline::test_lift_larger_matmul`
— Z3 rejects both candidates for the 128×128 kernel, and the test expects an
"Emission error" message instead. It is unrelated to Docker.

---

## 5. How LoopHole finds the image (reference)

| Setting | Used by | Default |
|---|---|---|
| `--docker-image` flag | `compile`, `cgeist`, `lift-c` | `compile`: `ghcr.io/schizoid-man/loophole-polygeist:llvm17`; others: auto-detect |
| `LOOPHOLE_CGEIST_DOCKER_IMAGE` | `cgeist`, `lift-c` | first existing of `loophole-polygeist:llvm17`, `ghcr.io/schizoid-man/loophole-polygeist:llvm17` |
| `LOOPHOLE_MLIR_DOCKER_IMAGE` | MLIR verifier (`batch --profile trusted`, `--validate-emitted`) | same auto-detect |
| `LOOPHOLE_MLIR_VERIFY_CMD` | MLIR verifier | unset → local `mlir-opt`, else Docker |
| `LOOPHOLE_DISABLE_DOCKER_CGEIST=1` / `LOOPHOLE_DISABLE_DOCKER_MLIR_VERIFY=1` | opt out of Docker | unset |

Docker report helpers (run everything inside the container), from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File packages/loophole/scripts/run_batch_report_in_docker.ps1 -Target stablehlo -Profile trusted
powershell -ExecutionPolicy Bypass -File packages/loophole/scripts/run_weekly_report_in_docker.ps1 -Target stablehlo -Label week_01
```

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `error from registry: denied` on pull | GHCR package is private — use Option A, or `docker login ghcr.io` with a `read:packages` PAT. |
| `Unable to find image ... locally` then denied | Image not built/tagged — rerun step 2 with both `-t` tags. |
| `error during connect` / `docker info` fails | Docker Desktop isn't running. |
| Build killed / `c++: fatal error: Killed` | Out of memory — `--build-arg BUILD_THREADS=2` or raise Docker Desktop memory. |
| `COPY requirements.txt ... not found` during build | Wrong build context — the last argument must be `packages/loophole`. |
| `Input/output/include paths must be on the same drive` | Source and output must share a drive (single volume mount). |
| `loophole` not found | Venv not activated — `.\.venv\Scripts\Activate.ps1`. |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or call `.\.venv\Scripts\loophole.exe` directly. |
| Venv broken after pulling the restructure | The old venv under `POC_initial_demo/` points at moved code — delete it and redo step 3. |

More background: [Docker Polygeist toolchain guide](docs/guides/docker-polygeist.md),
[Docker/Polygeist delivery handoff](docs/status/docker-polygeist-delivery-handoff.md).
