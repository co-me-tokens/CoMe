# Co-Me: Confidence Guided Token Merging for Visual Geometric Transformers


<p align="center">
  <a href="https://co-me-tokens.github.io/"><img src="https://img.shields.io/badge/Homepage-4385f4?style=flat&logoColor=white"></a>
  <a href="https://arxiv.org/abs/2511.14751"><img src="https://img.shields.io/badge/arXiv-b31b1b?style=flat&logo=arxiv&logoColor=white"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-yellow"></a>
</p>

<p align="center">
<img width="1503" height="664" alt="image" src="https://github.com/user-attachments/assets/934099c9-38a8-4344-9399-57c7a6c40b69" />
</p>

## 🔥 Updates
* [Feb 2026] Our work is accepted by the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026. We will present our work at CVPR 2026 in Denver.

## 📦 Environment & Installation

> [!NOTE]
>
> We do not provide pre-compiled kernels, you **must compile and install the CUDA kernels** manually.
>
> It is recommended to use the provided Docker images to simplify the workflow.


### Support Matrix

| Environment Setup | Platform               | Verified GPU Model(s) |
| ----------------- | ---------------------- | ------------ |
| Docker✅ / Conda✅| Linux + CUDA 12.8     | A100, RTX Ada6000     |
| Docker✅ / Conda✅| Linux + CUDA 13.0     | RTX 5090          |
| Docker✅ / Conda❓| L4T + CUDA 13.0 | NVIDIA Jetson Thor            |
| Docker❌ / Conda✅| Windows 11 + CUDA 13.0 + VS2022 | RTX 5070 |

> Legend: ✅: Verified support, ❓: Unverified but should support, ❌ Unsupported.

### Get the Source

The FlashAttention kernel is built against NVIDIA CUTLASS, which is vendored as a git submodule. Clone with submodules:

```bash
git clone --recursive <repository-url>
# or, for an existing clone:
git submodule update --init src/cuda_extension/cutlass
```

### Docker Environment Setup (Recommended 👍)

Pre-built images are published on [Docker Hub](https://hub.docker.com/r/yutianchen/co-me). The Docker configuration lives in `docker/`.

1. **Configure the toolchain** — installs git hooks and writes the Docker `.env`:

   ```bash
   bash tools/setup.sh
   ```

2. **Build (or pull) the image** — `docker/compose.sh` wraps `docker compose` and auto-selects the `amd64` / `arm64` profile for your host:

   ```bash
   cd docker && bash compose.sh build
   ```

   `docker/docker-compose.yaml` defines one dev service per CUDA version — `linux-cuda126-dev`, `linux-cuda128-dev`, `linux-cuda130-dev` (x86-64) and `jetson-thor-dev`, `jetson-orin-dev` (ARM64).

3. **Mount your datasets** — edit `tools/docker-compose.user.yaml` to bind-mount dataset directories into the container.

4. **Enter the dev container** — `tools/dev.sh` opens an interactive shell in the dev container, with the repository and your dataset mounts attached:

   ```bash
   bash tools/dev.sh
   ```

Build the CUDA kernels inside this container as described below.

### Conda / Virtual Environment Setup

If you are on Windows or prefer a virtual environment, install the Python dependencies with:

```bash
pip install -r docker/requirements.txt
```

A CUDA-enabled PyTorch build must already be present in the environment (the Docker images are based on the NVIDIA NGC PyTorch containers, which ship PyTorch).

### Building the CUDA Kernels

`tools/setup.sh` drives the kernel build. Run `bash tools/setup.sh --help` to list the available features:

```bash
bash tools/setup.sh cuext flash      # build the Co-Me token-merge + FlashAttention kernels
```

- `cuext` — Co-Me token-merge kernel (`src/cuda_extension/co_me`)
- `flash` — Co-Me FlashAttention kernel (`src/cuda_extension/flash_attn`)
- `zedloader` — optional Stereolabs ZED camera loader

Each extension can also be built directly via its own `install.sh`:

```bash
bash src/cuda_extension/co_me/install.sh
bash src/cuda_extension/flash_attn/install.sh
```

#### Building on Windows

On **Windows**, launch the *Developer PowerShell for VS 2022* (or equivalent), activate your environment, and build each extension in place:

```powershell
PS> cd src\cuda_extension\co_me
PS> python setup.py build_ext --inplace -f
PS> cd ..\flash_attn
PS> python setup.py build_ext --inplace -f
```

### Checkpoints

The confidence-predictor checkpoints ship with the repository under `output/confidence_distill/`. Each Co-Me model config (`config/model/co_me_*.yaml`) points at the appropriate checkpoint through its `ckpt_path` field — **no additional download is required.**

## 🚀 Quick Start: Runtime Benchmark

`python -m src.bench` measures the inference latency of a reference model and its Co-Me–accelerated counterpart back-to-back, on synthetic input (no dataset download required):

```bash
# Default: VGGT vs. Co-Me accelerated VGGT
$ python -m src.bench

 Name                Count   Median (ms)   P90 (ms)   P99 (ms) 
───────────────────────────────────────────────────────────────
 Reference Model        10       6671.10    6719.05    6958.61 
 Accelerated Model      10       1319.27    1337.53    1405.01
```

```
# Benchmark another backbone, e.g. MapAnything
python -m src.bench model@ref=ma model@acc=co_me_fused_ma_3x3
```

The reference model, accelerated model, input dataset, and sample count are configured in `config/bench.yaml`.

> [!NOTE]
> All metrics reported in the paper were measured on an NVIDIA H100 HBM3 80G. Benchmarking on a different GPU model may yield different results.

## 🛠️ Model Evaluation

We release the depth evaluation pipeline used in the paper. `python -m src.evaluate` runs a Co-Me accelerated model against its reference and reports depth metrics (L1 error, δ₁.₂₅ accuracy) alongside latency:

```bash
python -m src.evaluate model@acc=co_me_fused_vggt_3x3 model@ref=vggt evaluate.target=depth dataset=dtu_mvs
```

- `model@ref` — reference model: `vggt`, `vggt_star`, `fastvggt`, `to_me_direct_vggt_r05`, `pi3`, `pi3x`, `da3`, `ma`
- `model@acc` — Co-Me accelerated model: see `config/model/co_me_*.yaml`
- `dataset` — evaluation dataset, e.g. `dtu_mvs` (see `config/dataset/`)

To run a model on your own images, video, or live camera, use `python -m src.infer` together with one of the `custom_*` dataset configs (`config/dataset/custom_{image,video,camera}.yaml`).

## 📋Citation / BibTex

```bibtex
@misc{chen2025comeconfidenceguidedtokenmerging,
      title={Co-Me: Confidence-Guided Token Merging for Visual Geometric Transformers}, 
      author={Yutian Chen and Yuheng Qiu and Ruogu Li and Ali Agha and Shayegan Omidshafiei and Jay Patrikar and Sebastian Scherer},
      year={2025},
      eprint={2511.14751},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.14751}, 
}
```
