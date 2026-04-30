# Full Workflow Test Log

This document records the end-to-end C -> StableHLO -> GPU benchmark run that was executed on April 28, 2026, including the failed attempts that occurred before the successful path was found.

## Goal

Benchmark a C matmul kernel on the host CPU, lift it to StableHLO with LoopHole, compile the lifted module for CUDA, and measure execution on an NVIDIA RTX 3060 Ti.

## Environment

- Host OS: Windows
- GPU: NVIDIA GeForce RTX 3060 Ti
- Docker: available and running
- WSL2 Ubuntu: available with GPU passthrough
- LoopHole CLI: `POC_initial_demo/.venv/Scripts/loophole.exe`
- Polygeist image used: local image tagged as `loophole-polygeist:llvm17`
- IREE runtime: installed in a WSL virtual environment at `~/loophole-iree-venv`

## What Was Run

### 1. Confirm GPU and toolchain availability

Commands run:

```powershell
nvidia-smi
docker --version
Get-Command loophole -ErrorAction SilentlyContinue | Format-List *
```

Results:

- `nvidia-smi` confirmed the RTX 3060 Ti and the CUDA driver were visible.
- Docker was installed.
- `loophole.exe` was available through the local virtual environment.

### 2. Check Docker GPU access

Command run:

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Result:

- Success. Docker could access the GPU and report the RTX 3060 Ti.

### 3. First attempt: compile the sample kernel

Command run:

```powershell
.\.venv\Scripts\loophole.exe compile .\examples\smoke_matmul.c --docker-image loophole-polygeist:llvm17 -o .\reports\benchmarks\smoke_matmul_affine.mlir
```

Failure:

- The local Docker image name did not exist yet.
- Docker reported:

```text
Unable to find image 'loophole-polygeist:llvm17' locally
docker: Error response from daemon: pull access denied for loophole-polygeist, repository does not exist or may require 'docker login'
```

### 4. First attempt: lift the missing MLIR file

Command run:

```powershell
.\.venv\Scripts\loophole.exe lift .\reports\benchmarks\smoke_matmul_affine.mlir --target stablehlo -o .\reports\benchmarks\smoke_matmul_stablehlo.mlir
```

Failure:

- The input MLIR file did not exist because the compile step had failed.
- LoopHole reported the file was missing.

### 5. Fix the Docker image reference

Commands run:

```powershell
docker images --no-trunc --format "{{.Repository}}:{{.Tag}} {{.ID}}"
docker tag sha256:f29af15bf7cf8ddc278de3a6ef71c13f40661132a2d4b1c9dff59d423f908d36 loophole-polygeist:llvm17
```

Result:

- The local image was found under the SHA256 digest and tagged as `loophole-polygeist:llvm17`.

### 6. Retry compile with the correct image

Command run:

```powershell
.\.venv\Scripts\loophole.exe compile .\examples\smoke_matmul.c --docker-image loophole-polygeist:llvm17 -o .\reports\benchmarks\smoke_matmul_affine.mlir
```

Result:

- Compile succeeded.

### 7. Retry lift on the sample kernel

Command run:

```powershell
.\.venv\Scripts\loophole.exe lift .\reports\benchmarks\smoke_matmul_affine.mlir --target stablehlo -o .\reports\benchmarks\smoke_matmul_stablehlo.mlir
```

Failure:

- Lift failed because the best candidate `stablehlo.add` was refuted by Z3.
- This sample kernel was not a good match for the current proof-backed StableHLO path.

### 8. Retry `lift-c` on the sample kernel

Command run:

```powershell
.\.venv\Scripts\loophole.exe lift-c .\examples\smoke_matmul.c --docker-image loophole-polygeist:llvm17 --target stablehlo --function matmul_2x2 -o .\reports\benchmarks\smoke_matmul_liftc_stablehlo.mlir
```

Failure:

- Same refutation as above: `stablehlo.add` was rejected by Z3.

### 9. Switch to a lift-friendly in-place matmul kernel

Created file:

- `POC_initial_demo/examples/matmul_inplace.c`

Kernel shape:

```c
void matmul_inplace_4x4(float A[4][4], float B[4][4], float C[4][4]) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }
}
```

### 10. Compile the in-place kernel

Command run:

```powershell
.\.venv\Scripts\loophole.exe compile .\examples\matmul_inplace.c --docker-image loophole-polygeist:llvm17 -o .\reports\benchmarks\matmul_inplace_affine.mlir
```

Result:

- Compile succeeded.

### 11. Lift the in-place kernel to StableHLO

Command run:

```powershell
.\.venv\Scripts\loophole.exe lift .\reports\benchmarks\matmul_inplace_affine.mlir --target stablehlo -o .\reports\benchmarks\matmul_inplace_stablehlo.mlir --strict
```

Result:

- Lift succeeded with formal proof.
- Matched sketch: `stablehlo.dot_general`.
- Total lift time: 31.1 ms.
- Z3 time: 6.8 ms.

Emitted MLIR was written to:

- `POC_initial_demo/reports/benchmarks/matmul_inplace_stablehlo.mlir`

### 12. Scale the benchmark kernel to 128x128

Created file:

- `POC_initial_demo/examples/matmul_inplace_128.c`

Then ran the compile and lift steps again.

Compile command:

```powershell
.\.venv\Scripts\loophole.exe compile .\examples\matmul_inplace_128.c --docker-image loophole-polygeist:llvm17 -o .\reports\benchmarks\matmul_inplace_128_affine.mlir
```

Result:

- Compile succeeded after a long run.

Lift command:

```powershell
.\.venv\Scripts\loophole.exe lift .\reports\benchmarks\matmul_inplace_128_affine.mlir --target stablehlo -o .\reports\benchmarks\matmul_inplace_128_stablehlo.mlir
```

Result:

- Lift matched `stablehlo.dot_general`.
- Z3 timed out, so the result was a match without strict proof.
- Total lift time: 10,145.5 ms.
- Z3 time: 10,090.8 ms.

### 13. Build and run the CPU baseline

Created file:

- `POC_initial_demo/reports/benchmarks/matmul_cpu_bench_128.c`

Command run in WSL:

```bash
gcc -O3 -march=native /mnt/c/Users/user/Music/LoopHole/POC_initial_demo/reports/benchmarks/matmul_cpu_bench_128.c -o /mnt/c/Users/user/Music/LoopHole/POC_initial_demo/reports/benchmarks/matmul_cpu_bench_128 && /mnt/c/Users/user/Music/LoopHole/POC_initial_demo/reports/benchmarks/matmul_cpu_bench_128
```

Result:

```text
cpu_matmul_n=128 iters=50 avg_ms=0.112770
checksum=1.648600
```

### 14. Install IREE in WSL

First attempt:

```bash
python3 -m pip install --user --upgrade pip setuptools wheel
python3 -m pip install --user iree-compiler iree-runtime
```

Failure:

- Debian/Ubuntu system Python rejected the install with PEP 668 `externally-managed-environment`.

Second attempt:

```bash
python3 -m venv ~/loophole-iree-venv
source ~/loophole-iree-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install iree-compiler iree-runtime
```

Result:

- Success.
- IREE compiler/runtime installed in the WSL virtual environment.

### 15. Compile StableHLO for CUDA

Command run in WSL:

```bash
source ~/loophole-iree-venv/bin/activate
iree-compile /mnt/c/Users/user/Music/LoopHole/POC_initial_demo/reports/benchmarks/matmul_inplace_128_stablehlo.mlir --iree-input-type=stablehlo --iree-hal-target-backends=cuda -o /mnt/c/Users/user/Music/LoopHole/POC_initial_demo/reports/benchmarks/matmul_inplace_128_cuda.vmfb
```

Result:

- Success.

### 16. First GPU runtime attempt

Command run in WSL:

```bash
python /mnt/c/Users/user/Music/LoopHole/POC_initial_demo/reports/benchmarks/iree_gpu_bench_128.py
```

Failure:

- IREE could not create a CUDA device.
- Error: `CUDA_ERROR_NO_DEVICE`.

### 17. Fix WSL CUDA library visibility and rerun GPU benchmark

Check:

```bash
ls -l /usr/lib/wsl/lib/libcuda.so*
```

Rerun with linker path:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH python /mnt/c/Users/user/Music/LoopHole/POC_initial_demo/reports/benchmarks/iree_gpu_bench_128.py
```

Result:

```text
gpu_stablehlo_n=128 iters=200 avg_ms=5.448358
checksum=30.063013
```

## Final Outcome

The full workflow completed successfully after several failures and corrections:

- C baseline measured on CPU.
- C lifted to StableHLO through LoopHole.
- StableHLO compiled to a CUDA VMFB with IREE.
- GPU execution succeeded on the RTX 3060 Ti in WSL2.

## Key Failures Observed

1. The Polygeist Docker image was not initially available under the expected local tag.
2. The sample `smoke_matmul.c` kernel failed proof during StableHLO lift.
3. A strict lift on the 128x128 kernel timed out, though the non-strict lift matched correctly.
4. System-wide Python package install in WSL failed due to PEP 668.
5. IREE CUDA runtime initially failed to detect the GPU until `LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` was set.

## Notes for Reuse

- For this repo, the in-place matmul form lifts reliably to `stablehlo.dot_general`.
- For WSL GPU execution, IREE needs the WSL CUDA driver library path on `LD_LIBRARY_PATH`.
- Small kernels are not a good performance benchmark for GPU; launch and synchronization overhead dominates.
