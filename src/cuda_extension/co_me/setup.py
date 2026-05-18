import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import typing as T

import pybind11
import torch
import setuptools
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

EXTENSION_NAME = "co_me_cuext"


@dataclass(frozen=True)
class BuildContext:
    root: Path
    is_windows: bool
    torch_version: str
    cuda_version: str
    capability: tuple[int, int]

    @property
    def sm_arch(self) -> str:
        return f"sm_{self.capability[0]}{self.capability[1]}"


def collect_context(root: Path) -> BuildContext:
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required to build {EXTENSION_NAME}.")
    torch_version = T.cast(str, getattr(torch, "__version__", "unknown"))
    torch_cuda_ns = getattr(torch, "version", None)
    cuda_version = getattr(torch_cuda_ns, "cuda", None)
    if not isinstance(cuda_version, str):
        raise RuntimeError("torch.version.cuda is None; CUDA SDK is not available.")
    return BuildContext(
        root=root,
        is_windows=platform.system() == "Windows",
        torch_version=torch_version,
        cuda_version=cuda_version,
        capability=torch.cuda.get_device_capability(),
    )


def build_extension(context: BuildContext) -> setuptools.Extension:
    capability = context.capability
    cxx_flags = ["/O2", "/std:c++17", "/permissive"] if context.is_windows else ["-O3", "-std=c++17"]
    nvcc_flags = [
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-DCUB_IGNORE_DEPRECATED_CPP_DIALECT",
        "-gencode",
        f"arch=compute_{capability[0]}{capability[1]},code=sm_{capability[0]}{capability[1]}",
    ]
    if context.is_windows:
        os.environ.setdefault("DISTUTILS_USE_SDK", "1")
        os.environ.setdefault("MSSdk", "1")
        nvcc_flags += ["-Xcompiler", "/O2", "-Xcompiler", "/std:c++17", "-Xcompiler", "/permissive"]
    else:
        nvcc_flags += ["-O3", "-Xcompiler", "-O3"]

    sources = ["src/binding.cpp"] + sorted(str(p) for p in Path("src").rglob("*.cu"))
    include_dirs = [str(context.root / "include"), pybind11.get_include()]
    print(f"[cuext] torch={context.torch_version} cuda={context.cuda_version} target={context.sm_arch}")
    return CUDAExtension(
        name=EXTENSION_NAME,
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    )


class CustomBuildExt(BuildExtension):
    def run(self) -> None:
        super().run()
        module_root = Path(__file__).parent.resolve()
        result = subprocess.run(
            [sys.executable, "-m", "pybind11_stubgen", EXTENSION_NAME, "--output-dir=."],
            check=False,
            capture_output=True,
            text=True,
            cwd=module_root,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pybind11_stubgen failed for {EXTENSION_NAME} with exit code {result.returncode}.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        generated_stub = module_root / f"{EXTENSION_NAME}.pyi"
        if not generated_stub.exists():
            raise RuntimeError(
                f"pybind11_stubgen did not generate expected stub file: {generated_stub}"
            )


def main() -> None:
    root = Path(__file__).parent.resolve()
    context = collect_context(root)
    extension = build_extension(context)
    setup(
        name=EXTENSION_NAME,
        ext_modules=[extension],
        cmdclass={"build_ext": CustomBuildExt},
        package_data={"": ["*.pyi"]},
        zip_safe=False,
        install_requires=["torch"],
    )


if __name__ == "__main__":
    main()
