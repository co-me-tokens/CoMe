#!/bin/bash

# Install CUDA accelerated implementation for this repo.
(cd ./CUExt/CUDAExtension && python setup.py build_ext --inplace -f)
