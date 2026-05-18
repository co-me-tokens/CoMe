#!/bin/bash
set -euo pipefail

python -c "import pybind11" >/dev/null 2>&1 || {
    echo "ERROR: Missing dependency 'pybind11'. Install dependencies first."
    exit 1
}

python -c "import pybind11_stubgen" >/dev/null 2>&1 || {
    echo "ERROR: Missing dependency 'pybind11-stubgen'. Install dependencies first."
    exit 1
}

python setup.py build_ext --inplace -f
