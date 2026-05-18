#!/usr/bin/env bash
set -euo pipefail

apt-get update && apt-get install -y --no-install-recommends \
    libboost-graph-dev libboost-program-options-dev libboost-random-dev \
    libboost-timer-dev libboost-chrono-dev \
    && rm -rf /var/lib/apt/lists/*

GTSAM_SRC=/tmp/gtsam
git clone --depth 1 https://github.com/borglab/gtsam.git -b develop "$GTSAM_SRC"

cmake -S "$GTSAM_SRC" -B "$GTSAM_SRC/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGTSAM_BUILD_PYTHON=ON \
    -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_BUILD_TESTS=OFF \
    -DGTSAM_BUILD_UNSTABLE=OFF

cmake --build "$GTSAM_SRC/build" -j"${MAX_JOBS:-4}"

cmake --install "$GTSAM_SRC/build"
ldconfig

pip install --no-deps "$GTSAM_SRC/build/python"

rm -rf "$GTSAM_SRC"
