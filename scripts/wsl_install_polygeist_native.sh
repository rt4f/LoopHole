#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/wsl_install_polygeist_native.sh [toolchain_root]
# Example:
#   ./scripts/wsl_install_polygeist_native.sh ~/loophole-toolchain

TOOLCHAIN_ROOT="${1:-$HOME/loophole-toolchain}"
LLVM_REPO_DIR="$TOOLCHAIN_ROOT/llvm-project"
LLVM_SRC_DIR="$LLVM_REPO_DIR/llvm"
LLVM_BUILD_DIR="$LLVM_REPO_DIR/build"
POLYGEIST_SRC_DIR="$TOOLCHAIN_ROOT/polygeist"
POLYGEIST_BUILD_DIR="$POLYGEIST_SRC_DIR/build"
JOBS="${JOBS:-$(nproc)}"

echo "[1/6] Installing Ubuntu build prerequisites"
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

echo "[2/6] Preparing source trees under $TOOLCHAIN_ROOT"
mkdir -p "$TOOLCHAIN_ROOT"
if [[ ! -d "$LLVM_REPO_DIR/.git" ]]; then
  git clone https://github.com/llvm/llvm-project.git "$LLVM_REPO_DIR"
fi
if [[ ! -d "$POLYGEIST_SRC_DIR/.git" ]]; then
  git clone --recursive https://github.com/llvm/Polygeist.git "$POLYGEIST_SRC_DIR"
fi

echo "[3/6] Configuring LLVM+MLIR"
mkdir -p "$LLVM_BUILD_DIR"
cmake -S "$LLVM_SRC_DIR" -B "$LLVM_BUILD_DIR" -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;lld" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON

echo "[4/6] Building LLVM+MLIR (jobs=$JOBS)"
cmake --build "$LLVM_BUILD_DIR" -- -j"$JOBS"

echo "[5/6] Configuring Polygeist"
mkdir -p "$POLYGEIST_BUILD_DIR"
cmake -S "$POLYGEIST_SRC_DIR" -B "$POLYGEIST_BUILD_DIR" -G Ninja \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DCLANG_DIR="$LLVM_BUILD_DIR/lib/cmake/clang" \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON

echo "[6/6] Building Polygeist binaries"
cmake --build "$POLYGEIST_BUILD_DIR" -- -j"$JOBS"

echo "\n[done] Build artifacts:"
echo "  LLVM/MLIR:     $LLVM_BUILD_DIR/bin"
echo "  Polygeist:     $POLYGEIST_BUILD_DIR/bin"
echo "  cgeist binary: $POLYGEIST_BUILD_DIR/bin/cgeist"
echo "  polygeist-opt: $POLYGEIST_BUILD_DIR/bin/polygeist-opt"
