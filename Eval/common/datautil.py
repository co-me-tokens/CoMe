import torch
from typing import TypeVar
from torch.utils.data import Dataset, Subset


T_Co = TypeVar("T_Co", bound=Dataset)


def slice_dataset(dataset: Dataset[T_Co], slice_to: int | None) -> Dataset[T_Co]:
    if not slice_to: return dataset
    else:
        return Subset(dataset, indices=list(range(slice_to)))
