import torch
import typing as T
from pathlib import Path
from rich.progress import track

from Network.vggt import VGGT, get_VGGT, get_VGGT_vanilla


VGGTLike: T.TypeAlias = VGGT


def get_model(method: str) -> VGGTLike:
    match method:
        case "vggt"      :
            return get_VGGT_vanilla()
        
        case "vggt*"     :
            return get_VGGT()
        
        case "ours"      :
            from Accelerate.vggt import accelerate_vggt, Accelerator_Config
            return accelerate_vggt(get_VGGT(), Accelerator_Config(
                Path("Model/VGGT_Accelerator/checkpoint.pth"),
                grp_size=4, mask_setup=('bot-p', 0.5), apply_attn_bias=True
            ))[0]

        case model_type:
            raise ValueError(f"Unrecognized model type: {model_type=}")


@torch.no_grad()
def profile_vggt_over_length(model_str: str, length: int):
    model  = get_model(model_str).cuda().eval()
    images = torch.randn((1, length, 3, 27 * 14, 36 * 14), device=torch.device("cuda"), dtype=torch.float32)
    
    assert model.depth_head is not None

    # Two rounds of warmup    
    for _ in track(range(2), description="warmup"): model.aggregator(images)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)
    
    start_event.record(torch.cuda.current_stream())
    result = model.aggregator(images)
    model.depth_head(result[0], images, result[1])
    end_event.record(torch.cuda.current_stream())
    
    torch.cuda.synchronize()
    
    time       = start_event.elapsed_time(end_event)
    result_str = f"{model_str} | Sequence length: {length}\t\t| Infer time {time:.3f}ms"
    print(result_str)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model" , type=str)
    parser.add_argument("length", type=int)
    args = parser.parse_args()
    
    profile_vggt_over_length(args.model, args.length)
