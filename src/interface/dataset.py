"""Dataset interfaces for multi-view geometry tasks."""

from dataclasses import dataclass
import typing as T

from torch.utils.data import Dataset

from .geometric_model import SceneGeometry, MultiViewInput


@dataclass(kw_only=True, slots=True)
class MultiViewTask:
    multiview_input: MultiViewInput
    scene_geometry: SceneGeometry | None

    @staticmethod
    def collate(batch: list["MultiViewTask"]) -> "MultiViewTask":
        if len(batch) == 0:
            raise ValueError("Cannot collate an empty MultiViewTask batch.")

        mv_input = MultiViewInput.collate([task.multiview_input for task in batch])

        if all(task.scene_geometry is None for task in batch):
            return MultiViewTask(multiview_input=mv_input, scene_geometry=None)
        if not all(task.scene_geometry is not None for task in batch):
            raise ValueError(
                "Non-uniform batch for scene_geometry: some samples provide SceneGeometry while others provide None."
            )

        geometries = T.cast(list[SceneGeometry], [task.scene_geometry for task in batch])
        return MultiViewTask(multiview_input=mv_input, scene_geometry=SceneGeometry.collate(geometries))


MultiViewGeometryDataset = Dataset[MultiViewTask]
