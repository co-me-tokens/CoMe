"""
Root loader helpers for the vendored MapAnything package.
"""
from __future__ import annotations
from .mapanything.models.mapanything.model import MapAnything

DEFAULT_MAPANYTHING_REPO_ID = "facebook/map-anything"


def get_MA() -> "MapAnything":
    return MapAnything.from_pretrained(DEFAULT_MAPANYTHING_REPO_ID)


__all__ = [
    "DEFAULT_MAPANYTHING_REPO_ID",
    "MapAnything",
    "get_MA",
]
