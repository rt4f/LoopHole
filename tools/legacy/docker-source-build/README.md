# Docker Workflow for LoopHole + Polygeist

This container builds LLVM/MLIR/Clang and Polygeist from source and installs the LoopHole POC so you can run the same flow from Linux or Windows hosts.

## 1) Build image

Run from repository root:

```bash
docker build -f docker/Dockerfile -t loophole-polygeist:dev .
```

Optional build overrides:

```bash
docker build -f docker/Dockerfile \
  --build-arg POLYGEIST_REF=main \
  --build-arg JOBS=8 \
  -t loophole-polygeist:dev .
```

## 2) Start container (Linux host)

```bash
docker run --rm -it \
  -v "$(pwd):/workspace/src" \
  loophole-polygeist:dev bash
```

## 3) Start container (Windows PowerShell host)

```powershell
docker run --rm -it `
  -v "${PWD}:/workspace/src" `
  loophole-polygeist:dev bash
```

## 4) Verify toolchain inside container

```bash
cgeist --help
mlir-opt --help
polygeist-opt --help
loophole --help
```

## 5) One-step C/C++ flow

```bash
loophole lift-c /workspace/src/path/to/kernel.c --function=kernel --target=linalg
```

For C++:

```bash
loophole lift-c /workspace/src/path/to/kernel.cpp -x cpp --std c++17 -I /workspace/src/include --function=kernel
```

## 6) Two-step flow

```bash
loophole cgeist /workspace/src/path/to/kernel.c --function=kernel -o /tmp/kernel.mlir
loophole lift /tmp/kernel.mlir --target stablehlo
```

## 7) Container smoke test

```bash
loophole-smoke-test
```

## Notes

1. LLVM/Polygeist builds are compute-heavy; allocate enough Docker memory/CPU.
2. Keep `JOBS` conservative on smaller machines to avoid OOM.
3. The image includes MLIR Python bindings via `PYTHONPATH` from the LLVM build tree.
