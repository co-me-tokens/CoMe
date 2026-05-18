import torch
import torch.cuda.nvtx as nvtx
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from .utility.diagnostic   import Diagnostics
from .utility.timer        import Timer


@hydra.main(version_base=None, config_path="../config", config_name="infer")
@torch.inference_mode()
# @Diagnostics.activate()
def main(config: DictConfig):
    Timer.reset()
    Timer.activate()
    
    data   = instantiate(config.dataset)
    render = instantiate(config.render)
    model  = instantiate(config.model).eval().cuda()
    
    for idx, sample in enumerate(data): # type: ignore
        if (config.first_n is not None) and (idx >= config.first_n): break
        with nvtx.range(f"sample_{idx}"):
            with nvtx.range("input_transfer"):
                input_mvs = sample.multiview_input.resize(504, 504).to(device="cuda")
            with nvtx.range("inference"), Timer.range("inference"):
                result = model(input_mvs)
            with nvtx.range("render"):
                render.render(result, input_mvs, save_to=f"{config.output_directory}/sample_{idx}")
    
    print(Timer.report())


if __name__ == "__main__":
    main()  # type: ignore
