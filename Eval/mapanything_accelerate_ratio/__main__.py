import torch
import argparse
from pathlib import Path
from rich.progress import track
from torch.utils import benchmark
from uniception.models.info_sharing.base import MultiViewTransformerInput

from Network.map_anything import MapAnything, get_MapAnything
from Accelerate.map_anything import accelerate_mapanything, Accelerator_Config, install_nvtx_instrument


def get_model(method: str) -> MapAnything:
    match method:
        case "ma":
            model = get_MapAnything()
        
        case "ours":
            model = accelerate_mapanything(
                get_MapAnything(),
                Accelerator_Config(
                    accelerator=Path("Model/MapAnything_Accelerator/checkpoint.pth"),
                    grp_size=4, mask_setup=('bot-p', 0.5), attn_backend="torch-flex"
                )
            )[0]
        
        case model_type:
            raise ValueError(f"Unrecognized model type {model_type=}")
    
    return install_nvtx_instrument(model)


@torch.no_grad()
def profile_mapanything_over_length(method: str, length: int):
    model = get_model(method).cuda().eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    height, width = (504, 504)
    
    base_img = torch.zeros((1, 3, height, width), device=device, dtype=dtype)
    data_norm_type = getattr(model.encoder, "data_norm_type", None)
    
    pseudo_views = []
    for _ in range(length):
        view: dict = {"img": base_img.clone()}
        if data_norm_type is not None:
            view["data_norm_type"] = [data_norm_type]
        pseudo_views.append(view)
    
    @torch.inference_mode()
    def _backbone_only_inference() -> None:
        encoder_features = model._encode_n_views(pseudo_views)
        with torch.autocast("cuda", enabled=False):
            encoder_features = model._encode_and_fuse_optional_geometric_inputs(
                pseudo_views,
                encoder_features,
            )
        batch_size_per_view = pseudo_views[0]["img"].shape[0]
        input_scale_token = (
            model.scale_token.unsqueeze(0).unsqueeze(-1).repeat(
                batch_size_per_view,
                1,
                1,
            )
        )
        info_sharing_input = MultiViewTransformerInput(
            features=encoder_features,
            additional_input_tokens=input_scale_token,
        )
        info_sharing_output = model.info_sharing(info_sharing_input)
        if isinstance(info_sharing_output, tuple):
            info_sharing_output = info_sharing_output[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in track(range(2), description="warmup"):
        with torch.cuda.nvtx.range(f"warmup {_}"):
            _backbone_only_inference()
    torch.cuda.synchronize(device)

    timer = benchmark.Timer(
        stmt="with torch.cuda.nvtx.range('Inference'): _backbone_only_inference()",
        globals={"_backbone_only_inference": _backbone_only_inference, "torch": torch},
        num_threads=1,
    )
    measurement = timer.blocked_autorange(min_run_time=0.5)
    latency_ms = measurement.median * 1_000
    
    
    result_str = (
        f"{method} | Sequence length: {length}\t\t| Backbone infer time {latency_ms:.3f}ms"
    )
    print(result_str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("length", type=int)
    args = parser.parse_args()
    profile_mapanything_over_length(args.model, args.length)


if __name__ == "__main__":
    main()
