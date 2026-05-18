from omegaconf import DictConfig
from hydra.utils import instantiate

from ..interface.geometric_model import GeometricPredictorLike


def instantiate_accelerated_model(token_merger: DictConfig, accelerate_func: DictConfig, target_model: DictConfig) -> GeometricPredictorLike:
    model = instantiate(target_model).eval()
    merge = instantiate(token_merger, start_idx=model.patch_start_index)
    model = instantiate(accelerate_func, model=model, token_merger=merge)
    return model
