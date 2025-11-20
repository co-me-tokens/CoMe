import os
import platform
import setuptools
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pybind11
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


class Console:
    """Lightweight helper to keep setup output consistent and readable."""

    divider = "=" * 60

    @staticmethod
    def section(title: str) -> None:
        print(f"\n{Console.divider}\n{title}\n{Console.divider}")

    @staticmethod
    def kv(label: str, value: str) -> None:
        print(f"{label:<28} : {value}")

    @staticmethod
    def bullet(label: str) -> None:
        print(f"  • {label}")


@dataclass
class BuildContext:
    root: Path
    is_windows: bool
    torch_version: str
    cuda_version: str
    capability: tuple[int, int]

    @property
    def sm_arch(self) -> str:
        return f"sm_{self.capability[0]}{self.capability[1]}"


class EnvironmentInspector:
    """Gathers environment information and ensures CUDA readiness."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.is_windows = platform.system() == "Windows"

    def collect(self) -> BuildContext:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Not Available")

        capability = torch.cuda.get_device_capability()
        context = BuildContext(
            root=self.root,
            is_windows=self.is_windows,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            capability=capability,
        )
        self._print_environment(context)
        return context

    def _print_environment(self, context: BuildContext) -> None:
        Console.section("CUDA Extension Build Environment")
        Console.kv("PyTorch", context.torch_version)
        Console.kv("CUDA available", str(torch.cuda.is_available()))
        Console.kv("CUDA toolkit", context.cuda_version)
        Console.kv("GPU compute capability", context.capability)
        Console.kv("Target SM architecture", context.sm_arch)
        Console.section("Host Platform")
        Console.kv("Platform", platform.system())
        Console.kv("Python", platform.python_version())
        Console.kv("Root", str(context.root))


class CompilerConfig:
    """Builds compiler flag sets for CXX and NVCC."""

    def __init__(self, context: BuildContext) -> None:
        self.context = context
        self.cxx_flags = self._make_cxx_flags()
        self.nvcc_flags = self._make_nvcc_flags()
        self._print_flags()

    def _make_cxx_flags(self) -> list[str]:
        if self.context.is_windows:
            return [
                "/O2",
                "/std:c++17",
                "/permissive",  # Fix ambiguous symbol errors on MSVC
            ]
        return ["-O3", "-std=c++17"]

    def _make_nvcc_flags(self) -> list[str]:
        flags = [
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-DCUB_IGNORE_DEPRECATED_CPP_DIALECT",
            "-gencode",
            f"arch=compute_{self.context.capability[0]}{self.context.capability[1]},"
            f"code=sm_{self.context.capability[0]}{self.context.capability[1]}",
        ]

        if self.context.is_windows:
            os.environ.setdefault("DISTUTILS_USE_SDK", "1")
            os.environ.setdefault("MSSdk", "1")
            flags += [
                "-Xcompiler",
                "/O2",
                "-Xcompiler",
                "/std:c++17",
                "-Xcompiler",
                "/permissive",
            ]
        else:
            flags += [
                "-O3",  # optimise generated device code
                "-Xcompiler",
                "-O3",  # forward the same flag to the host
            ]

        # Ensure CUB kernels are compiled for all architectures
        flags.append("-DCUB_IGNORE_DEPRECATED_CPP_DIALECT")
        return flags

    def _print_flags(self) -> None:
        Console.section("Compiler Flags")
        Console.kv("CXX flags", " ".join(self.cxx_flags))
        Console.kv("NVCC flags", " ".join(self.nvcc_flags))


class ExtensionConfig:
    """Encapsulates CUDAExtension construction."""

    def __init__(self, context: BuildContext, compiler: CompilerConfig) -> None:
        self.context = context
        self.compiler = compiler

    def build(self) -> setuptools.Extension:
        sources = ["src/binding.cpp"] + [str(p) for p in Path("src/").rglob("*.cu")]
        include_dirs = [
            str(self.context.root / "include"),
            pybind11.get_include(),
        ]
        Console.section("Extension Layout")
        for src in sources:
            Console.bullet(f"source: {src}")
        for inc in include_dirs:
            Console.bullet(f"include: {inc}")
        return CUDAExtension(
            name="pyramidinfer_cuext",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": self.compiler.cxx_flags,
                "nvcc": self.compiler.nvcc_flags,
            },
        )


class StubGenerator:
    """Generates .pyi stubs using pybind11-stubgen after build."""

    def __init__(self, module_name: str, expected_stub: str, workdir: Path) -> None:
        self.module_name = module_name
        self.expected_stub = expected_stub
        self.workdir = workdir

    def run(self) -> None:
        try:
            Console.section("Stub Generation")
            Console.bullet("running pybind11-stubgen")
            result = subprocess.run(
                ["python", "-m", "pybind11_stubgen", self.module_name, "--output-dir=."],
                capture_output=True,
                text=True,
                cwd=self.workdir,
            )

            if result.returncode == 0:
                Console.bullet("✓ Generated type stubs with pybind11-stubgen")
                self._report_stub()
            else:
                Console.bullet("⚠ pybind11-stubgen failed, no type stub generated")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            Console.bullet("⚠ pybind11-stubgen not available. No type stub generated.")
            raise exc

    def _report_stub(self) -> None:
        # Search widely to handle different pybind11-stubgen output layouts
        target_names = {f"{self.module_name}.pyi", f"{self.expected_stub}.pyi"}
        found = None
        for path in self.workdir.rglob("*.pyi"):
            if path.name in target_names:
                found = path
                break
        if found:
            Console.bullet(f"✓ Type stub generated: {found.relative_to(self.workdir)}")
        else:
            Console.bullet("⚠ Warning: Stub file not found after generation")


class CustomBuildExt(BuildExtension):
    """Hook build_ext to emit stubs after compilation."""

    def run(self) -> None:
        super().run()
        StubGenerator("CUDAExtension", "pyramidinfer_cuext", Path("..")).run()


def main() -> None:
    root = Path(__file__).parent.resolve()
    inspector = EnvironmentInspector(root)
    context = inspector.collect()
    compiler = CompilerConfig(context)
    extension = ExtensionConfig(context, compiler).build()

    setup(
        name="pyramidinfer_cuext",
        ext_modules=[extension],
        cmdclass={"build_ext": CustomBuildExt},
        package_data={"": ["*.pyi"]},
        zip_safe=False,
        setup_requires=[
            "pybind11-stubgen",  # For automatic stub generation
        ],
        install_requires=[
            "torch",
        ],
    )


if __name__ == "__main__":
    main()
