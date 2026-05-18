#!/bin/bash
set -e

case "$CUDA_MAJOR_VERSION" in
  13)
    pip install --no-cache-dir opencv-python-headless -c /tmp/constraints.txt
    ;;
  *)
    # do nothing
    ;;
esac
