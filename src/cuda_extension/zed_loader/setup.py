#!/usr/bin/env python3
"""
Setup script for zed_loader Python bindings.

This uses CMake to build the C++ extension with proper ZED SDK integration.
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake-based extension"""
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    """Build extension using CMake"""
    
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        
        # CMake configuration
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE=Release",
            # Don't build the demo executable (requires Rerun SDK)
            "-DBUILD_ZEDLOADER_EXEC=OFF",
            # Suppress developer warnings from third-party CMake configs (e.g., ZED SDK)
            "-Wno-dev",
        ]
        
        build_args = []
        
        # Set parallel build
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            cpu_count = os.cpu_count() or 1
            build_args += [f"-j{cpu_count}"]
        
        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)
        
        # Configure
        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp,
            check=True
        )
        
        # Build
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp,
            check=True
        )


def get_version():
    """Extract version from CMakeLists.txt or use default"""
    return "0.1.0"


def get_long_description():
    """Read README if available"""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "ZED Camera loader with PyTorch tensor support and multi-threading"


setup(
    name="zed_loader",
    version=get_version(),
    author="MAC-VO Team",
    description="ZED Camera loader with PyTorch/ATen integration",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension("zed_loader")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
)
