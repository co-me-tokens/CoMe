"""
Inference-only public surface for the vendored MapAnything package.
"""

from __future__ import annotations

from .models import get_available_models, init_model, init_model_from_config, model_factory


def __getattr__(name: str):
    if name == "MapAnything":
        from .models import MapAnything

        return MapAnything
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MapAnything",
    "get_available_models",
    "init_model",
    "init_model_from_config",
    "model_factory",
]
