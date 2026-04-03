#!/usr/bin/env bash
set -euo pipefail

LLVM_BUILD_DIR="/home/yash/build"
POLY_BUILD_DIR="/home/yash/Polygeist/build"

for i in $(seq 1 40); do
  echo "[loop $i] try cgeist"
  if ninja -C "$POLY_BUILD_DIR" cgeist >/tmp/cgeist.log 2>&1; then
    echo "cgeist-built"
    tail -n 10 /tmp/cgeist.log || true
    exit 0
  fi

  missing=$(grep -oE "/home/yash/build/lib/lib[^ ]+\\.a" /tmp/cgeist.log | head -n 1 || true)
  if [[ -z "$missing" ]]; then
    echo "could-not-parse-missing-lib"
    tail -n 80 /tmp/cgeist.log || true
    exit 1
  fi

  base=$(basename "$missing" .a)
  target="${base#lib}"
  echo "building target: $target"
  cmake --build "$LLVM_BUILD_DIR" --target "$target" -j8

done

echo "loop-limit-reached"
exit 1
