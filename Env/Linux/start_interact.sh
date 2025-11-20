#!/usr/bin/env bash
set -euo pipefail

# Optional override:
#   FORCE_CUDA=12 ./start_interact.sh
#   FORCE_CUDA=13 ./start_interact.sh
if [[ "${FORCE_CUDA:-}" == "12" ]]; then
    SERVICE="linux-cuda12-dev"
elif [[ "${FORCE_CUDA:-}" == "13" ]]; then
    SERVICE="linux-cuda13-dev"
else
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi not found. Make sure NVIDIA drivers are installed on the host."
        exit 1
    fi

    # Example line in nvidia-smi:
    # | NVIDIA-SMI 560.35.03    Driver Version: 560.35.03    CUDA Version: 13.0 |
    CUDA_MAJOR_STR=$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9]+' | head -n1 || true)

    if [[ -z "${CUDA_MAJOR_STR}" ]]; then
        echo "ERROR: Failed to detect CUDA version from nvidia-smi."
        exit 1
    fi

    CUDA_MAJOR=$CUDA_MAJOR_STR

    if (( CUDA_MAJOR >= 13 )); then
        SERVICE="linux-cuda13-dev"
    elif (( CUDA_MAJOR >= 12 )); then
        SERVICE="linux-cuda12-dev"
    else
        echo "ERROR: Detected CUDA ${CUDA_MAJOR}, but this compose file expects >= 12."
        exit 1
    fi
fi

echo "Starting service: ${SERVICE}"
docker compose run --rm --service-ports "${SERVICE}" bash
