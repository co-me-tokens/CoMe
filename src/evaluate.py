import json
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn

import torch

from .interface.geometric_model import GeometricPredictorLike
from .utility.diagnostic        import Diagnostics
from .utility.timer             import Timer
from .metrics import metric_table


def _average_dicts(dicts: list[dict]) -> dict:
    """Average a list of uniformly-structured nested dicts of numeric values."""
    result = {}
    for key in dicts[0]:
        values = [d[key] for d in dicts]
        if isinstance(values[0], dict):
            result[key] = _average_dicts(values)
        elif isinstance(values[0], (int, float)):
            result[key] = sum(values) / len(values)
        else:
            raise TypeError(f"Cannot average values of type {type(values[0])!r} for key {key!r}")
    return result


@hydra.main(version_base=None, config_path="../config", config_name="evaluate")
@torch.inference_mode()
def main(config: DictConfig):
    Timer.reset()

    seed = config.evaluate.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device          = torch.device(config.device)
    
    Diagnostics.log("Loading datasets...")
    data            = instantiate(config.dataset)

    Diagnostics.log("Loading models...")
    ref_model: GeometricPredictorLike = instantiate(config.ref).eval()
    acc_model: GeometricPredictorLike = instantiate(config.acc).eval()

    Diagnostics.log("Starting evaluation pipeline...")
    acc_evals: list[dict] = []
    acc_infer_masks: list[torch.Tensor | None] = []

    _progress_columns = (
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    acc_model.to(device=device)
    with Progress(*_progress_columns) as progress:
        task = progress.add_task("Evaluating acc_model", total=len(data))
        for idx in range(len(data)):
            sample     = data[idx]
            task_input = sample.multiview_input.resize(*config.evaluate.infer_size)
            task_label = sample.scene_geometry
            assert task_label is not None

            with Timer.range(f"CoMe+{type(ref_model).__name__}"): acc_result = acc_model(task_input)

            acc_infer_masks.append(acc_result.infer_mask.cpu() if acc_result.infer_mask is not None else None)
            acc_score = metric_table[config.evaluate.target](task_label, acc_result, align_scale=config.evaluate.align_scale)
            acc_evals.append(acc_score.to_dict())
            del acc_result
            torch.cuda.empty_cache()
            progress.advance(task)

    acc_model.to(device="cpu")
    torch.cuda.empty_cache()

    ref_evals: list[dict] = []

    ref_model.to(device=device)
    with Progress(*_progress_columns) as progress:
        task = progress.add_task("Evaluating ref_model", total=len(data))
        for idx in range(len(data)):
            sample     = data[idx]
            task_input = sample.multiview_input.resize(*config.evaluate.infer_size)
            task_label = sample.scene_geometry
            assert task_label is not None

            with Timer.range(f"     {type(ref_model).__name__}"): ref_result = ref_model(task_input)

            saved_mask = acc_infer_masks[idx]
            
            if ref_result.depths is not None:
                ref_result.infer_mask = saved_mask.to(device=ref_result.depths.device) if saved_mask is not None else None
            else:
                ref_result.infer_mask = saved_mask
            
            ref_score = metric_table[config.evaluate.target](task_label, ref_result, align_scale=config.evaluate.align_scale)
            ref_evals.append(ref_score.to_dict())
            del ref_result
            torch.cuda.empty_cache()
            progress.advance(task)

    ref_model.to(device="cpu")
    torch.cuda.empty_cache()

    if len(data) == 0:
        raise RuntimeError("No evaluation samples were processed")

    torch.cuda.synchronize()

    _TIMER_KEY_MAP = {
        f"     {type(ref_model).__name__}": f"{type(ref_model).__name__}",
        f"CoMe+{type(ref_model).__name__}": f"Acc-{type(ref_model).__name__}",
    }
    timing = {}
    for entry in Timer.results():
        key = _TIMER_KEY_MAP.get(entry.name)
        if key is None:
            raise ValueError(f"Unexpected timer entry: {entry.name!r}")
        timing[key] = {"P50_ms": entry.P50, "P90_ms": entry.P90, "P99_ms": entry.P99}

    result = {
        "target": config.evaluate.target,
        "metrics": {
            f"{type(ref_model).__name__}"    : _average_dicts(ref_evals),
            f"Acc-{type(ref_model).__name__}": _average_dicts(acc_evals),
        },
        "timing": timing,
    }

    output_path = Path(config.evaluate.output_directory, "eval_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    Diagnostics.log(f"Results saved to {output_path}")


if __name__ == "__main__":
    Timer.activate()
    try:
        main()  #type: ignore
    finally:
        print(Timer.report())
