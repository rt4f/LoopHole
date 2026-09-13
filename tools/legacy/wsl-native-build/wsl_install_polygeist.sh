#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/wsl_install_polygeist.sh [repo_root]
# Example:
#   ./scripts/wsl_install_polygeist.sh /mnt/c/Users/Yasho/LoopHole

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LLVM_SRC_DIR="$REPO_ROOT/Polygeist/llvm-project/llvm"
LLVM_BUILD_DIR="$REPO_ROOT/Polygeist/llvm-project/build"
POLYGEIST_SRC_DIR="$REPO_ROOT/Polygeist"
POLYGEIST_BUILD_DIR="$REPO_ROOT/Polygeist/build"
JOBS="${JOBS:-$(nproc)}"

if [[ ! -d "$LLVM_SRC_DIR" ]]; then
  echo "[error] LLVM source directory not found: $LLVM_SRC_DIR"
  exit 1
fi

if [[ ! -d "$POLYGEIST_SRC_DIR" ]]; then
  echo "[error] Polygeist source directory not found: $POLYGEIST_SRC_DIR"
  exit 1
fi

echo "[1/5] Installing Ubuntu build prerequisites"
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  python3 \
  python3-venv \
  python3-pip \
  libxml2-dev \
  libedit-dev \
  libncurses-dev \
  lld

echo "[1.5/5] Normalizing line endings for LLVM helper scripts"
# Windows checkouts can convert shell helpers to CRLF and break CMake probes.
sed -i 's/\r$//' "$REPO_ROOT/Polygeist/llvm-project/llvm/cmake/config.guess"
if [[ -f "$REPO_ROOT/Polygeist/llvm-project/llvm/cmake/config.sub" ]]; then
  sed -i 's/\r$//' "$REPO_ROOT/Polygeist/llvm-project/llvm/cmake/config.sub"
fi

echo "[2/5] Configuring LLVM+MLIR build"
mkdir -p "$LLVM_BUILD_DIR"
cmake -S "$LLVM_SRC_DIR" -B "$LLVM_BUILD_DIR" -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;lld" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON

echo "[3/5] Building LLVM+MLIR (jobs=$JOBS)"
cmake --build "$LLVM_BUILD_DIR" -- -j"$JOBS"

echo "[4/5] Configuring Polygeist"
mkdir -p "$POLYGEIST_BUILD_DIR"
cmake -S "$POLYGEIST_SRC_DIR" -B "$POLYGEIST_BUILD_DIR" -G Ninja \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DCLANG_DIR="$LLVM_BUILD_DIR/lib/cmake/clang" \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON

echo "[5/5] Building Polygeist binaries"
cmake --build "$POLYGEIST_BUILD_DIR" -- -j"$JOBS"

echo "\n[done] Build artifacts:"
echo "  LLVM/MLIR:     $LLVM_BUILD_DIR/bin"
echo "  Polygeist:     $POLYGEIST_BUILD_DIR/bin"
echo "  cgeist binary: $POLYGEIST_BUILD_DIR/bin/cgeist"
echo "  polygeist-opt: $POLYGEIST_BUILD_DIR/bin/polygeist-opt"
