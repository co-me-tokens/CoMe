from typing import Any
from abc import ABC, abstractmethod
import torch


class StreamLike(ABC):
    @abstractmethod
    def succ(self, new_image: torch.Tensor) -> Any: ...
    
    @abstractmethod
    def reset(self) -> None: ...
