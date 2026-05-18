"""
ZedLoader - ZED Camera loader with PyTorch integration

This module provides Python bindings for capturing data from ZED cameras
with direct conversion to PyTorch tensors and multi-threading support.
"""

# PyTorch must be imported before the native extension to initialize
# the Python type system that pybind11's torch type_caster depends on
import torch as _torch
from .zed_loader import (
    # Enums
    Resolution,
    # Classes
    ZedDataLoader,
    SensorParameters,
    # Exceptions
    ZedLoaderError,
    CameraNotFoundError, 
    CameraOpenError,
    InvalidParameterError,
    InternalError as ZedLoaderInternalError
)
