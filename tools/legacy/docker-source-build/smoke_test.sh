#!/usr/bin/env bash
set -euo pipefail

TMP_DIR="${TMP_DIR:-/tmp/loophole-smoke}"
mkdir -p "${TMP_DIR}"

INPUT_C="${TMP_DIR}/add_kernel.c"
GENERATED_MLIR="${TMP_DIR}/add_kernel.mlir"
GENERATED_LIFTED="${TMP_DIR}/add_kernel.lifted.mlir"

cat > "${INPUT_C}" <<'EOF'
void add_kernel(float *A, float *B, float *C, int n) {
  for (int i = 0; i < n; ++i) {
    C[i] = A[i] + B[i];
  }
}
EOF

cgeist "${INPUT_C}" -function=add_kernel -S > "${GENERATED_MLIR}"
loophole lift "${GENERATED_MLIR}" --output "${GENERATED_LIFTED}" --target linalg

echo "[ok] smoke test artifacts:"
echo "  C source:      ${INPUT_C}"
echo "  MLIR source:   ${GENERATED_MLIR}"
echo "  Lifted output: ${GENERATED_LIFTED}"
