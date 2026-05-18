"""Build script for flash_attn_fwd_cuda — forward-only varlen flash attention."""

import os
import platform
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

EXTENSION_NAME = "flash_attn_fwd_cuda"

NVCC_THREADS = os.getenv("NVCC_THREADS", "4")


def get_cuda_gencode() -> list[str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build flash_attn_fwd_cuda.")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError(
            f"FlashAttention requires Ampere (sm80) or newer, got sm{major}{minor}."
        )
    arch = f"{major}{minor}"
    return ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]


def build_extension() -> CUDAExtension:
    root = Path(__file__).parent.resolve()
    cutlass_include = (root.parent / "cutlass" / "include").resolve()
    if not cutlass_include.exists():
        raise RuntimeError(
            f"CUTLASS include directory not found at {cutlass_include}. "
            "Run: git submodule update --init src/cuda_extension/cutlass"
        )

    sources = ["flash_api.cpp"] + sorted(
        str(p.relative_to(root)) for p in (root / "src").rglob("*.cu")
    )

    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-DFLASHATTENTION_DISABLE_BACKWARD",
        f"--threads={NVCC_THREADS}",
    ] + get_cuda_gencode()

    is_windows = platform.system() == "Windows"
    cxx_flags = ["/O2", "/std:c++17"] if is_windows else ["-O3", "-std=c++17"]

    include_dirs = [
        str(root),
        str(root / "src"),
        str(cutlass_include),
    ]

    print(f"[{EXTENSION_NAME}] sources={len(sources)} includes={include_dirs}")
    return CUDAExtension(
        name=EXTENSION_NAME,
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    )


def main() -> None:
    ext = build_extension()
    setup(
        name=EXTENSION_NAME,
        ext_modules=[ext],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
        install_requires=["torch"],
    )


if __name__ == "__main__":
    main()
