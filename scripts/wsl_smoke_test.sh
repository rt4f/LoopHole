
#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/wsl_smoke_test.sh [repo_root] [polygeist_bin_dir]
# Example:
#   ./scripts/wsl_smoke_test.sh /mnt/c/Users/Yasho/LoopHole /home/user/loophole-toolchain/polygeist/build/bin

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
POC_DIR="$REPO_ROOT/POC_initial_demo"
POLYGEIST_BIN_DIR="${2:-${POLYGEIST_BIN_DIR:-$REPO_ROOT/Polygeist/build/bin}}"
CGEIST_BIN="$POLYGEIST_BIN_DIR/cgeist"
TMP_DIR="$REPO_ROOT/.tmp_smoke"
INPUT_C="$TMP_DIR/add_kernel.c"
GENERATED_MLIR="$TMP_DIR/add_kernel.mlir"
GENERATED_LIFTED="$TMP_DIR/add_kernel.lifted.mlir"

if [[ ! -x "$CGEIST_BIN" ]]; then
  echo "[error] cgeist not found or not executable: $CGEIST_BIN"
  echo "Run scripts/wsl_install_polygeist.sh first."
  exit 1
fi

if [[ ! -d "$POC_DIR" ]]; then
  echo "[error] POC directory not found: $POC_DIR"
  exit 1
fi

mkdir -p "$TMP_DIR"

echo "[1/5] Creating smoke-test C kernel"
cat > "$INPUT_C" <<'EOF'
void add_kernel(float *A, float *B, float *C, int n) {
  for (int i = 0; i < n; ++i) {
    C[i] = A[i] + B[i];
  }
}
EOF

echo "[2/5] Generating MLIR with cgeist"
"$CGEIST_BIN" "$INPUT_C" -function=add_kernel -S > "$GENERATED_MLIR"

echo "[3/5] Creating Python virtual environment for LoopHole"
cd "$POC_DIR"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]

echo "[4/5] Running LoopHole lift on generated MLIR"
loophole lift "$GENERATED_MLIR" --output "$GENERATED_LIFTED"

echo "[5/5] Running baseline POC tests"
pytest tests/unit/test_affine_extractor.py -q

echo "\n[done] Smoke artifacts:"
echo "  Input C:       $INPUT_C"
echo "  Generated MLIR:$GENERATED_MLIR"
echo "  Lifted MLIR:   $GENERATED_LIFTED"
