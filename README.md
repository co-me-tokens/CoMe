# Co-Me: Confidence Guided Token Merging for Visual Geometric Transformers


<p align="center">
  <a href="https://co-me-tokens.github.io/"><img src="https://img.shields.io/badge/Homepage-4385f4?style=flat&logoColor=white"></a>
  <a href="https://arxiv.org/abs/2511.14751"><img src="https://img.shields.io/badge/arXiv-b31b1b?style=flat&logo=arxiv&logoColor=white"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-yellow"></a>
</p>

<p align="center">
<image src="https://github.com/user-attachments/assets/1b6440eb-d0ca-4964-a81e-5363f96d82e7" alt="Co-Me Hero Image" width="75%" align="center"/>
</p>

## 📦 Environment & Installation

> [!NOTE]
> 
> We do not provide pre-compiled kernels, you **must compile and install the CUDA kernel** manually. 
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

### Docker Environment Setup (Recommended 👍)

We provide the pre-built docker image at [Docker Hub (Link)](https://hub.docker.com/r/yutianchen/co-me-tokens).  

```bash
# On Linux Platform (x86-64)
# This will automatically launch image with proper CUDA version.
cd Env/Linux && bash ./start_interact.sh
bash ./install_CUExt.sh
```

```bash
# On Jetson L4T Platform (ARM64)
cd Env/L4T && bash ./start_interact.sh
bash ./install_CUExt.sh
```

### Conda Environment Setup

If you are on the Windows platform or prefer using virtual environment instead you can also use the `requirements.txt` in `./Env/<Platform>`.

#### Installing CUDA Kernels on Linux

Simply run

```bash
bash install_CUExt.sh
```

#### Installing CUDA Kernels on Windows

For **Windows** platform user, to install the CUDA kernel you need to launch the *Developer Powershell for VS 2022* (or equivalent), activate the virtual environment, and run 

```powershell
PS> cd CUExt/CUDAExtension
PS> python setup.py build_ext --inplace -f
```

### Checkpoint

The checkpoint is already shipped with the git repository in `Model` directory. **No additional download is required.**

## 🚀 Quick Start: Measure Model Runtime

1. Measure runtime of VGGT @ 32 frames
    
    ```bash
    $ python -m Eval.vggt_accelerate_ratio vggt 32
    ```

2. Measure runtime of Co-Me Accelerated VGGT @ 32 frames
    
    ```bash
    $ python -m Eval.vggt_accelerate_ratio ours 32
    ```

3. Measure runtime of MapAnything @ 32 frames

    ```bash
    $ python -m Eval.mapanything_accelerate_ratio ma 32
    ```

4. Measure runtime of Co-Me Accelerated MapAnything @ 32 frames

    ```bash
    $ python -m Eval.mapanything_accelerate_ratio ours 32
    ```

>[!NOTE]
> All metrics reported on paper were measured on NVIDIA A100 PCIe 80G, benchmarking on different GPU models may yield different results.

## 🛠️ Model Evaluation

We released the depth evaluation pipeline for VGGT (`VGGT`, `VGGT*` and `ours`) used in Table 1 and 2 in the paper. To run this pipeline you can use

```bash
# VGGT evaluation and benchmarking
python -m Eval.vggt_depth --dataset dtu-mvs --seq_l 32 vggt

# VGGT* evaluation and benchmarking
python -m Eval.vggt_depth --dataset dtu-mvs --seq_l 32 vggt*

# Co-Me accelerated VGGT evaluation and benchmarking
python -m Eval.vggt_depth --dataset dtu-mvs --seq_l 32 ours \
    --grp_size 4 \
    --mask_setup bot-p 0.5 \
    --accelerator ./Model/VGGT_Accelerator/checkpoint.pth
```

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
