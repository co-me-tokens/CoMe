ARG BASE_IMAGE="nvcr.io/nvidia/pytorch:25.06-py3"
FROM ${BASE_IMAGE}

# Limit nvcc/clang/ninja parallelism to reduce OOM when several images build at once
# on the same host. Override per build: docker compose build --build-arg BUILD_MAX_JOBS=8
ARG BUILD_MAX_JOBS=4
ENV MAX_JOBS=${BUILD_MAX_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_MAX_JOBS}

ARG  CUDA_MAJOR_VERSION="12"
ENV  DEBIAN_FRONTEND=noninteractive
ENV  CUDA_MAJOR_VERSION="${CUDA_MAJOR_VERSION}"

# Install additional development tools
# Dev    requirement: {tmux, libeigen3-dev, libopencv-dev, clangd}
# ZedSDK requirement: {zstd}c
# NOTE:
# Installing Ubuntu OpenCV packages can pull system UCX variants that conflict with
# the HPCX UCX/UCC stack bundled in the NGC base image. Keep package install minimal,
# and force HPCX search order to avoid torch import failures.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux libeigen3-dev libopencv-dev clangd pkg-config libboost-dev libboost-serialization-dev \
    zstd \
    && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:${LD_LIBRARY_PATH}"

# Setup the working directory
WORKDIR /home/root/workspace

# Freeze some key packages from the current environment
RUN  pip freeze | grep -E '^(torch==|torchvision==|numpy==|scipy==|tensorrt==)' \
     > /tmp/constraints.txt

# Load and install Python Requirements
COPY ./requirements.txt /tmp/requirements.txt
RUN  pip install --no-cache-dir --upgrade-strategy only-if-needed\
     -r /tmp/requirements.txt -c /tmp/constraints.txt \
     && rm /tmp/requirements.txt

# Unofficial support for Rerun > 0.23 with Numpy < 2.0
RUN  pip install --no-cache-dir --upgrade-strategy only-if-needed rerun-sdk==0.30.0 --no-deps

# Configure Git
RUN  git config --global --add safe.directory '*'

# if CUDA_MAJOR_VERSION is 13, install cv2
COPY ./scripts/install_cv2.sh /tmp/install_cv2.sh
RUN  bash /tmp/install_cv2.sh

COPY ./scripts/install_gtsam.sh /tmp/install_gtsam.sh
RUN  bash /tmp/install_gtsam.sh

# no-deps to avoid flush existing env.
RUN pip install uniception==0.1.6 --no-deps
RUN FORCE_CUDA=1 pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git"

# Install Zed SDK
COPY ./scripts/install_zedsdk.sh /tmp/install_zedsdk.sh
RUN  bash /tmp/install_zedsdk.sh

# Install Cursor Agent CLI for self-managed long running jobs
# RUN curl https://cursor.com/install -fsS | bash && \
#     mkdir -p /root/.local/bin && \
#     ln -sf /root/.local/share/cursor-agent/versions/*/cursor-agent /root/.local/bin/cursor-agent && \
#     ln -sf /root/.local/bin/cursor-agent /root/.local/bin/agent

ENV PATH="/root/.local/bin:${PATH}"

CMD ["bash"]
