#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUTLASS_DIR="$(cd "$SCRIPT_DIR/../cutlass" && pwd)"

if [ ! -f "$CUTLASS_DIR/include/cutlass/cutlass.h" ]; then
    echo "ERROR: CUTLASS not found at $CUTLASS_DIR"
    echo "Run: git submodule update --init src/cuda_extension/cutlass"
    exit 1
fi

NUM_CPU="$(nproc)"
export MAX_JOBS=$(( NUM_CPU < 12 ? NUM_CPU : 12 ))

cd "$SCRIPT_DIR"
python setup.py build_ext --inplace -f
