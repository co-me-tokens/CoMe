from pathlib import Path

from ..interface.dataset import SceneGeometry, MultiViewInput


class EmptyRenderer:
    def __init__(self):
        ...
    def render(self, scene: SceneGeometry, input: MultiViewInput, save_to: str | Path):
        ...
