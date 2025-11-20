import json
import torch
from typing import Callable
from pathlib import Path
from rich.progress import track
from dataclasses import dataclass, asdict

from Data.Interface    import DepthDataset, DepthData
from Network.vggt      import VGGT

from ..metrics.depth  import evaluation_pipeline
from ..metrics.memory import memory_monitor

@dataclass
class EvalauteConfig:
    infer_shape    : tuple[int, int]

VGGTLike = VGGT

@torch.no_grad()
def depth_evaluate(
    model: VGGTLike, data: DepthDataset, config: EvalauteConfig,
    save_result: Path, save_mask: Path, get_mask: Callable[[], torch.Tensor | None] | None
):
    if save_result.exists():
        print(f"Skip the evaluation since result file {save_result} already exists!")
        return 
    
    save_result.parent.mkdir(parents=True, exist_ok=True)
    save_mask.mkdir(parents=True, exist_ok=True)
    is_mask_gen = (get_mask is not None)
    
    torch.cuda.empty_cache()
    model.eval().cuda()
    
    def infer_model(model: VGGTLike, images: torch.Tensor):
        assert model.depth_head is not None
        aggregated_tokens_list, ps_idx = model.aggregator(images)
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        return depth_map.permute(0, 1, 4, 2, 3), depth_conf

    def benchmark_model(model: VGGTLike, images: torch.Tensor) -> tuple[float, float]:
        assert model.depth_head is not None
        
        with memory_monitor() as mem_stat:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            
            start_event.record(torch.cuda.current_stream())
            result = model.aggregator(images)
            model.depth_head(result[0], images, result[1])
            end_event.record(torch.cuda.current_stream())
            
            torch.cuda.synchronize()

        mem_usage = mem_stat.peak_increased_gb
        assert mem_usage is not None
        
        return start_event.elapsed_time(end_event), mem_usage
    
    def infer_mask_to_eval_mask(infer_mask: torch.Tensor, infer_size: tuple[int, int], gt_depth: torch.Tensor):
        B, N, M = infer_mask.size(0), infer_size[0] // 14, infer_size[1] // 14
        infer_patch_mask = infer_mask.reshape(B, N, M)
        infer_pixel_mask = infer_patch_mask.repeat_interleave(14, -2).repeat_interleave(14, -1).unsqueeze(1)
        eval_pixel_mask  = torch.nn.functional.interpolate(infer_pixel_mask.float(), size=gt_depth.shape[2:], mode='nearest').bool()
        return eval_pixel_mask & (gt_depth < 100.) & (gt_depth > 0.)

    def reshape_to_output(depth: torch.Tensor, gt_depth: torch.Tensor):
        B, N = depth.size(0), depth.size(1)
        depth = torch.nn.functional.interpolate(depth.flatten(0, 1), size=gt_depth.shape[2:], mode='bicubic')
        return depth.view(B, N, *depth.shape[1:])
    
    def index_to_mask_path(idx: int) -> Path: return Path(save_mask, f"{idx:05d}.pth")
    
    result = []
    sample: DepthData
    for idx, sample in track(enumerate(data), description=f"{save_result}"):
        B, S         = sample.images.shape[:2]
        
        # gt_depth
        sample.images, sample.gt_depths = sample.images.to('cuda'), sample.gt_depths.to('cuda')
        infer_images = torch.nn.functional.interpolate(sample.images.flatten(0, 1), size=config.infer_shape)
        infer_images = infer_images.view(B, S, 3, config.infer_shape[0], config.infer_shape[1])
        
        depth, conf = infer_model(model, infer_images)
        time, mem   = benchmark_model(model, infer_images)
        
        depth       = reshape_to_output(depth, sample.gt_depths[0])[0]
        
        if is_mask_gen:
            inference_mask = get_mask()
            assert inference_mask is not None
            torch.save(inference_mask.cpu(), index_to_mask_path(idx))
        else:
            inference_mask = torch.load(index_to_mask_path(idx)).cuda()
        
        eval_mask = infer_mask_to_eval_mask(inference_mask, config.infer_shape, sample.gt_depths[0])
        
        result.append(
            {
                "runtime_ms": time, "memory_gb": mem
            } | asdict(
                evaluation_pipeline(depth, sample.gt_depths[0], eval_mask)
            )
        )
        print(f"[{idx:04d}/{len(data)}] | {time=:.3f}, {mem=:.2f} | L1={result[-1]['avg_l1']:.3f}")
    
    with open(save_result, "w") as f: json.dump(result, f)
