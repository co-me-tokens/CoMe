import torch
import torch.cuda.nvtx as nvtx
import hydra
from   rich.progress import track
from   hydra.utils import instantiate
from   omegaconf   import DictConfig

from .utility.timer     import Timer


@hydra.main(version_base=None, config_path="../config", config_name="bench")
@torch.inference_mode()
def main(config: DictConfig):
    Timer.reset()
    Timer.activate()
    
    data       = instantiate(config.dataset)
    ref_model  = instantiate(config.ref).eval().cuda()

    for idx, sample in track(enumerate(data), description="Benchmarking Reference Model..."): # type: ignore
        if (config.first_n is not None) and (idx >= config.first_n): break
        with nvtx.range(f"Ref:sample_{idx}"):
            with nvtx.range("fetch_data"):
                input_mvs = sample.multiview_input.resize(504, 504).to(device="cuda")
            with nvtx.range("inference"), Timer.range("Reference Model"):
                ref_model(input_mvs)
    
    del data
    del ref_model
    
    data       = instantiate(config.dataset)
    acc_model  = instantiate(config.acc).eval().cuda()
    
    for idx, sample in track(enumerate(data), description="Benchmarking Accelerated Model..."): # type: ignore
        if (config.first_n is not None) and (idx >= config.first_n): break
        with nvtx.range(f"Acc:sample_{idx}"):
            with nvtx.range("fetch_data"):
                input_mvs = sample.multiview_input.resize(504, 504).to(device="cuda")
            with nvtx.range("inference"), Timer.range("Accelerated Model"):
                acc_model(input_mvs)

    print(Timer.report())


if __name__ == "__main__":
    main()
