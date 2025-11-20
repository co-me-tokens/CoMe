# CUDA-Torch Starter

<img width="640" alt="Repository Structure" src="https://github.com/user-attachments/assets/5e6d7c86-a083-4d6f-8bd7-d90e9604a6bc" />

This repository is a boilerplate for writing C++ / CUDA extension for PyTorch / libtorch.

## 1. Download the LibTorch and GoogleTest

1. LibTorch: `https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip`

2. GoogleTest: `https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip`

Place them / symbolic link them to `CUDAExtension/lib/libtorch` and `CUDAExtension/lib/googletest` respectively.

## 2. Use Dev Container for Correct PyTorch and CUDA ToolChain 

1. Run `docker compose build` to build the docker image for development environment on your local machine.

2. Use VSCode devcontainer feature to launch workspace and work.


## 3. Adding Custom CPP Extension / CUDA Kernel

See `./CUDAExtension/README.md`.


## 4. Running Python Scripts

1. After building the `CUDAExtension` package, run `python -m PythonSrc.<your_script_name>` and use `import CUDAExtension` in script to access the built extension.
