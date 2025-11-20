FROM nvcr.io/nvidia/pytorch:25.10-py3
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_HOME=/workspace/.torch_cache

# 0)  Tell PyTorch which GPU architectures we want to compile for.
#     •  8.6  – Ampere (RTX 30, A4000, etc.)
#     •  8.9  – Ada (RTX 40 series)
#     •  9.0+PTX – Hopper preview (H100) + keep PTX for forward-compat.
#   If you only ever deploy to one class of GPUs, trim the list.
ARG TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# 1) Install OS packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential git \
      curl ca-certificates \
      vim pkg-config tmux zstd wget \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3) Copy in requirements.txt (which includes the PyTorch+cu128 lines)
COPY requirements.txt /workspace/requirements.txt

# 4) Use the venv’s pip to install all Python dependencies
RUN pip install --no-cache-dir -r /workspace/requirements.txt
RUN pip install uniception --no-deps

# Install Zed SDK For interactive demo
RUN wget -O ./zed_sdk.run https://download.stereolabs.com/zedsdk/5.1/l4t38.2/jetsons && \
    chmod +x ./zed_sdk.run && \
    ./zed_sdk.run -- silent runtime_only skip_tools skip_od_module

# 5) Put the venv’s bin directory first in PATH
ENV PATH="/opt/venv/bin:${PATH}"

# 8) Set working directory and switch to non-root
WORKDIR /workspace
RUN chown -R ubuntu:ubuntu /workspace
USER ubuntu

# 9) Expose port 8888 if you plan to run Jupyter
EXPOSE 8888

# 10) Default command: drop into bash
CMD ["/bin/bash"]
