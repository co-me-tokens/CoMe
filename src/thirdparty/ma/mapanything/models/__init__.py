# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference-only model factory for MapAnything.
"""

import logging
import os
import warnings
from typing import List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Suppress DINOv2 warnings
logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available", category=UserWarning)
warnings.filterwarnings(
    "ignore", message="xFormers is not available", category=UserWarning
)


def resolve_special_float(value):
    if value == "inf":
        return np.inf
    elif value == "-inf":
        return -np.inf
    else:
        raise ValueError(f"Unknown special float value: {value}")


def _get_mapanything_class():
    from .mapanything import MapAnything

    return MapAnything


def init_model(
    model_str: str, model_config: DictConfig, torch_hub_force_reload: bool = False
):
    """
    Initialize a model using OmegaConf configuration.

    Args:
        model_str (str): Name of the model class to create.
        model_config (DictConfig): OmegaConf model configuration.
        torch_hub_force_reload (bool): Whether to force reload relevant parts of the model from torch hub.
    """
    if not OmegaConf.has_resolver("special_float"):
        OmegaConf.register_new_resolver("special_float", resolve_special_float)
    model_dict = OmegaConf.to_container(model_config, resolve=True)
    model = model_factory(
        model_str, torch_hub_force_reload=torch_hub_force_reload, **model_dict
    )

    return model


def _init_hydra_config(config_path: str, overrides: Optional[List[str]] = None):
    """
    Initialize Hydra config with proper composition and interpolation resolution.

    Args:
        config_path: Relative path to the config file (e.g., "configs/train.yaml")
        overrides: Optional list of Hydra overrides (e.g., ["model=vggt"])

    Returns:
        Composed OmegaConf config with all interpolations resolved
    """
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path).split(".")[0]

    # Get the project root (parent of mapanything package)
    package_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(package_dir))
    abs_config_dir = os.path.join(project_root, config_dir)

    # Get relative path from this module to the config directory
    relative_path = os.path.relpath(abs_config_dir, package_dir)

    # Clear any existing Hydra instance
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    # Initialize Hydra with the config directory
    hydra.initialize(version_base=None, config_path=relative_path)

    # Compose the config with overrides
    if overrides is not None:
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    else:
        cfg = hydra.compose(config_name=config_name)

    return cfg


def init_model_from_config(
    model_name: str,
    device: str = "cuda",
    machine: str = "default",
) -> torch.nn.Module:
    """
    Initialize a model using Hydra config composition.

    Models automatically load their pretrained weights via their config.

    Args:
        model_name: Name of the model config. Only "mapanything" is supported.
        device: Device to load model on (default: "cuda")

    Returns:
        Initialized model on the specified device in eval mode

    Example:
        >>> model = init_model_from_config("mapanything", device="cuda")
    """
    if model_name != "mapanything":
        raise ValueError(
            "Inference-only vendored MapAnything supports only model_name='mapanything', "
            f"got {model_name!r}."
        )

    # Use train.yaml as base config with model override for proper Hydra composition
    config_path = "configs/train.yaml"
    overrides = [f"model={model_name}", f"machine={machine}"]

    # Use Hydra to properly compose and resolve the config
    config = _init_hydra_config(config_path, overrides=overrides)

    # Initialize model using the factory
    model = init_model(
        model_str=config.model.model_str,
        model_config=config.model.model_config,
        torch_hub_force_reload=False,
    )

    model = model.to(device)

    return model


MODEL_CONFIGS = {
    "mapanything": {
        "class_getter": _get_mapanything_class,
    },
}


def model_factory(model_str: str, **kwargs):
    """
    Model factory for MapAnything.

    Args:
        model_str (str): Name of the model to create. Only "mapanything" is supported.
        **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
       nn.Module: An instance of the specified model.
    """
    if model_str not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_str}. Valid options are: {', '.join(MODEL_CONFIGS.keys())}"
        )

    model_config = MODEL_CONFIGS[model_str]
    model_class = model_config["class_getter"]()

    print(f"Initializing {model_class} with kwargs: {kwargs}")
    return model_class(**kwargs)


def get_available_models() -> list:
    """
    Get a list of available models in MapAnything.

    Returns:
        list: A list of available model names.
    """
    return ["mapanything"]


def __getattr__(name: str):
    if name == "MapAnything":
        return _get_mapanything_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "model_factory",
    "init_model",
    "init_model_from_config",
    "get_available_models",
    "MapAnything",
]
