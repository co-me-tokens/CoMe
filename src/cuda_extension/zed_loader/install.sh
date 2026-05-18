#!/bin/bash
# Build and install ZedLoader
#
# Usage:
#   ./install.sh          # Build everything (core + python + executable)
#   ./install.sh --no-exec  # Build without executable (no Rerun SDK needed)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
BUILD_EXEC="ON"
if [[ "$1" == "--no-exec" ]]; then
    BUILD_EXEC="OFF"
    echo "=== Building ZedLoader (without executable) ==="
else
    echo "=== Building ZedLoader ==="
fi

# Clean previous build
rm -rf build
mkdir build
cd build

# Configure with CMake
cmake -Wno-dev -DBUILD_ZEDLOADER_EXEC=${BUILD_EXEC} ..

# Build
make -j$(nproc)
