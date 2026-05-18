from pathlib import Path
from typing import Protocol
from .geometric_model import SceneGeometry, MultiViewInput


class RendererLike(Protocol):
    def render(self, scene: SceneGeometry, input: MultiViewInput, save_to: str | Path) -> None:
        ...
