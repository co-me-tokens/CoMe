ARG BASE_IMAGE="nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04"

FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

# 0)  Tell PyTorch which GPU architectures we want to compile for.
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX"

# 1) Install OS packages (build tools, python3, python3-venv, python3-pip, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential cmake \
      git curl ca-certificates vim \
      python3 python3-dev python3-venv python3-pip \
      pkg-config libopencv-dev tmux \
    && rm -rf /var/lib/apt/lists/*

# 2) Create a venv at /opt/venv and upgrade pip inside it
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip

# 3) Copy in requirements.txt (which includes the PyTorch+cu128 lines)
COPY requirements.txt /workspace/requirements.txt

# 4) Use the venv’s pip to install all Python dependencies
ARG TORCH_INDEX="https://download.pytorch.org/whl/cu128"
ARG TORCH_VERSION="2.7.0+cu128"
ARG TORCH_VISION_VERSION="0.22.0+cu128"

RUN /opt/venv/bin/pip install \
    --extra-index-url ${TORCH_INDEX} \
    torch==${TORCH_VERSION} torchvision==${TORCH_VISION_VERSION}
# Create a constraint file that locks these versions
RUN NUMPY_VERSION="$(/opt/venv/bin/python -c 'import numpy as np; print(np.__version__)')" \
 && printf "torch==%s\ntorchvision==%s\nnumpy==%s\n" \
    "${TORCH_VERSION}" "${TORCH_VISION_VERSION}" "${NUMPY_VERSION}" \
    > /tmp/pinned-torch-constraints.txt

RUN /opt/venv/bin/pip install --no-cache-dir \
    -r /workspace/requirements.txt \
    -c /tmp/pinned-torch-constraints.txt

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
