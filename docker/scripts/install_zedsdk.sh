#!/bin/bash
set -e

# 1. Detect the version
# We rely on the $CUDA_MAJOR_VERSION environment variable provided by the base image.
# If it's not set, default to a fallback or exit error.
ARCH=$(uname -m)
DETAIL_ARCH=$(uname -r)


echo "Running installation script..."

# 2. Conditional Logic based on architecture.
if [[ "$ARCH" = "x86_64" ]]; then

    if [[ "$CUDA_MAJOR_VERSION" = "12" ]]; then
        echo ">>> Branch: CUDA 12 Detected."
        echo ">>> Installing ZedSDK for CUDA 12..."
        
        # commands for CUDA 12
        wget -O /tmp/zed_sdk.run https://download.stereolabs.com/zedsdk/4.2/cu12/ubuntu24 && \
            chmod +x /tmp/zed_sdk.run && \
            /tmp/zed_sdk.run -- silent runtime_only skip_tools skip_od_module

    elif [[ "$CUDA_MAJOR_VERSION" = "13" ]]; then
        echo ">>> Branch: CUDA 13 Detected."
        echo ">>> Installing ZedSDK for CUDA 13..."

        # commands for CUDA 13
        wget -O /tmp/zed_sdk.run https://download.stereolabs.com/zedsdk/4.2/cu12/ubuntu24 && \
            chmod +x /tmp/zed_sdk.run && \
            /tmp/zed_sdk.run -- silent runtime_only skip_tools skip_od_module

    else
        echo "!!! Error: Unsupported CUDA Major Version: $CUDA_MAJOR_VERSION"
        exit 1  
    fi

elif [[ "$DETAIL_ARCH" =~ "tegra" && "$ARCH" = "aarch64" ]]; then

    # command for CUDA 12.6, Orin AGX
    wget -O /tmp/zed_sdk.run https://download.stereolabs.com/zedsdk/4.2/l4t36.4/jetsons && \
        chmod +x /tmp/zed_sdk.run && \
        /tmp/zed_sdk.run -- silent runtime_only skip_tools skip_od_module

elif [ "$CUDA_MAJOR_VERSION" = "13" ]; then
    echo ">>> Branch: CUDA 13 Detected."
    echo ">>> Installing ZedSDK for CUDA 13..."

    # commands for CUDA 13
    wget -O /tmp/zed_sdk.run https://download.stereolabs.com/zedsdk/5.1/cu13/ubuntu24 && \
        chmod +x /tmp/zed_sdk.run && \
        /tmp/zed_sdk.run -- silent runtime_only skip_tools skip_od_module

else
    echo "!!! Error: Unsupported CUDA Major Version: $CUDA_MAJOR_VERSION"
    exit 1
fi

echo ">>> ZedSDK Installation completed successfully."
