# CUDAExtension/__init__.py
"""
CUDA Extension for PyTorch - Custom kernel operations
"""
import torch    # This is required otherwise import will fail.

# Import the compiled extension
try:
    from . import pyramidinfer_cuext as ext
    
    __all__ = ['ext']
    
except ImportError as e:
    print(f"Warning: Could not import CUDA extension: {e}")
    print("Make sure to build the extension first with: python setup.py build_ext --inplace")
    raise

__version__ = "0.1.0"
