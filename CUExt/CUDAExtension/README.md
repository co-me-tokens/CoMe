# C++ / CUDA Extension Guide

## Adding a New Kernel

1. Declare the method(s) in `include/<your_kernel_name>.h`. A good practice will be declaring a CUDA kernel, a cpp wrapper for the CUDA kernel, a CPU fallback implementation using `libtorch`, and a *dispatch* function that switch between CUDA and CPU implementation based on input tensor device.

> [!NOTE]
> Adding a libtorch CPU implementation allows us to test the correctness on C++ unit test. This is very helpful for debugging since we have `gdb`, `cuda-gdb`, and so much more for C++ debug comparing to Python debugging tools.
> 

2. Create `src/<your_kernel_name>.cu` and add into `CMakeLists.txt`. This allow auto-complete and VSCode IntelliSense to work properly.

3. Implement the methods in `src/<your_kernel_name>.cu`.

4. Add `#include "<your_kernel_name>.h"` in `src/main.cpp` and run some simple test / debugging route.

5. Rebuild and run you executable at `./build/cuda_torch_app`

## Export New Kernel in Python Package

1. Include `"<your_kernel_name>.h"` in `src/binding.cpp` and add the PyBind accordingly to export your kernel to Python binding.

2. Add `src/<your_kernel_name>.cu` in `setup.py` to add your kernel in the module built by python setup tool.

3. Modify `__init__.py` to re-export your C++ methods in python package.

4. Run `python setup.py build_ext --inplace -f` to build the Python binding.

5. Now you can access your C++ extended methods in anyfile in `PythonSrc` using `import CUDAExtension`. To run scripts in `PythonSrc`, use `python -m PythonSrc.<your_python_script>`.

## Adding C++ Unit Test

We used GoogleTest framework for C++ unit test management.

1. Create `tests/test_<your_kernel_name>.cpp` and write unit tests.

2. Add `test_<your_kernel_name>.cpp` in `./tests/CMakeLists.txt` as `TEST_SOURCES`.

3. Re-build the UnitTest and run by executable `./build/tests/cuda_tests`.
