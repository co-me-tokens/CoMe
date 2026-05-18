#!/bin/bash

ARCH=$(uname -m)
DETAIL_ARCH=$(uname -r)
CUDA_MAJOR_VER=$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9]+' | head -n1 || true)

if [[ "$ARCH" = "x86_64" ]]; then
    echo "Detected x86_64 Architecture."

    if [[ "$CUDA_MAJOR_VER" = "12" ]]; then
        SELECTED_IMAGE="yutianchen/co-me:linux-cu128-amd64"
        SELECTED_PLATFORM="linux/amd64"

    elif [[ "$CUDA_MAJOR_VER" = "13" ]]; then
        SELECTED_IMAGE="yutianchen/co-me:linux-cu130-amd64"
        SELECTED_PLATFORM="linux/amd64"

    else
        echo "!!! Error: Unsupported CUDA Major Version: $CUDA_MAJOR_VER , CUDA_VERSION = $CUDA_VERSION"
        exit 1
    fi

elif [[ "$ARCH" = "aarch64" ]]; then
    echo "Detected ARM64 Architecture."

    if [[ "$DETAIL_ARCH" =~ "tegra" && "$CUDA_MAJOR_VER" = "12" ]]; then
        SELECTED_IMAGE="yutianchen/co-me:linux-cu126-orin-arm64"
        SELECTED_PLATFORM="linux/arm64"

    elif [[ "$CUDA_MAJOR_VER" = "13" ]]; then
        SELECTED_IMAGE="yutianchen/co-me:linux-cu130-thor-arm64"
        SELECTED_PLATFORM="linux/arm64"

    else 
        echo "!!! Error: Unsupported CUDA Major Version: $CUDA_MAJOR_VER , CUDA_VERSION = $CUDA_VERSION"
        exit 1
    fi

else
    echo "Unknown architecture: $ARCH"
    exit 1
fi

# Write to .env file in the root
# Docker Compose automatically reads .env files in the same directory
echo "DEV_IMAGE=$SELECTED_IMAGE" > ./docker/.env
echo "DEV_PLATFORM=$SELECTED_PLATFORM" >> ./docker/.env
echo -e "Configured .env at ./docker/.env with\n\tImage: $SELECTED_IMAGE"
