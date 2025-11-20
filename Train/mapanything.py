import ast
import json
import random
import datetime
import typing as T

import torch
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F

from pathlib import Path
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader

from Accelerate.core import Conv2DConfidence, GlobalAttentionConfidence
from Accelerate.map_anything import get_block, ProbeLocation
from Network.map_anything import get_MapAnything
from Network.map_anything.mapanything.utils.image import IMAGE_NORMALIZATION_DICT

from .common.sandbox import Sandbox
from .common.loss import pairwise_logistic_loss
from .common.routes import save_states


T_SupportData = T.Literal["tartanair"]
T_SupportLoss = T.Literal["mse", "pairwise-rank", "mse-raw"]


@dataclass
class MapAnythingTrainConfig:
    method: T.Literal["conv", "attn"]
    method_dim: T.Sequence[int]
    method_act: T.Literal["expp1", "none"]

    train_set: T_SupportData
    probe_loc: ProbeLocation

    train_batch_size: int
    train_sequence_l: int
    train_learn_rate: float
    train_infer_size: T.Sequence[tuple[int, int]]
    train_optim_step: int
    train_augment_π: bool
    train_augment_l: bool
    train_loss_func: T_SupportLoss
    train_save_freq: int
    train_scheduler: tuple[T.Literal["linear"], float, float, int, int] | T.Literal["none"]

    teacher_use_amp: bool
    teacher_amp_dtype: T.Literal["bf16", "fp16", "fp32"]
    teacher_memory_efficient: bool

    wandb_log_freq: int
    wandb_entity: str = "pyramid-infer"
    wandb_project: str = "MapAnything-ConfDistill"

    resume_run: Path | None = None
    resume_step: int | None = None

    input_norm_type: str = "dinov2"


@dataclass
class FeatureProbe:
    tokens: torch.Tensor | None = None


def install_probe(model, probe: FeatureProbe, loc: ProbeLocation):
    target_block = get_block(model, loc)

    def feature_recorder(_module, _args, output):
        probe.tokens = output

    target_block.register_forward_hook(feature_recorder)


def get_data(config: MapAnythingTrainConfig) -> torch.utils.data.DataLoader:
    match config.train_set:
        case "tartanair":
            from Data.TartanAir import TartanAir_DepthDataset, default_root

            dataset = TartanAir_DepthDataset(default_root, config.train_sequence_l)
            return DataLoader(
                dataset,
                batch_size=config.train_batch_size,
                shuffle=True,
                collate_fn=TartanAir_DepthDataset.collate,
                drop_last=True,
            )
        case _:
            raise ValueError("Unsupported dataset")


def get_student_model(config: MapAnythingTrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    match config.method:
        case "conv":
            return Conv2DConfidence(config.method_dim, config.method_act).to(device).train()
        case "attn":
            return GlobalAttentionConfidence(config.method_dim, config.method_act).to(device).train()
        case _:
            raise ValueError("Unsupported method")


class NoOpScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


def get_lr_scheduler(optim: torch.optim.Optimizer, config: MapAnythingTrainConfig):
    match config.train_scheduler:
        case "none":
            return NoOpScheduler(optim)
        case ("linear", start_lr, end_lr, decay_start_step, decay_end_step):
            def linear_decay(step):
                if step < decay_start_step:
                    return 1.0
                if step >= decay_end_step:
                    return end_lr / start_lr
                progress = (step - decay_start_step) / (decay_end_step - decay_start_step)
                return 1.0 - progress * (1.0 - end_lr / start_lr)

            return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=linear_decay)
        case _:
            raise ValueError("Unsupported scheduler")


def set_wandb(config: MapAnythingTrainConfig) -> Sandbox:
    import os

    slurm_task_id = os.getenv("SLURM_ARRAY_TASK_ID") or "0"
    run_name = f"{datetime.datetime.now():%Y%m%d_%H:%M:%S}_{slurm_task_id}"
    size_tag = "x".join(map(lambda s: f"{s[0]}x{s[1]}", config.train_infer_size))
    grp_name = f"{config.method}_{config.train_loss_func}_{config.probe_loc[1]}@{size_tag}"
    wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=run_name,
        group=grp_name,
        config=asdict(config),
    )

    space = Sandbox(Path("Result", "Models", "ConfDistill", "MapAnything", run_name))
    with space.open("config.json", "w") as f:
        json.dump(asdict(config), f)
    return space


@torch.no_grad()
def prepare_views(
    images: torch.Tensor,
    config: MapAnythingTrainConfig,
    device: torch.device,
) -> tuple[list[dict[str, T.Any]], torch.Tensor]:
    B, S = images.shape[:2]

    if config.train_augment_π:
        perm = torch.stack([torch.randperm(S, device=images.device) for _ in range(B)])
        batch_idx = torch.arange(B, device=images.device).unsqueeze(1)
        images = images[batch_idx, perm]

    if config.train_augment_l:
        seq_len = random.randint(1, S)
        images = images[:, :seq_len]
        S = seq_len

    target_shape = random.choice(config.train_infer_size)
    flat = images.reshape(B * S, *images.shape[2:])
    flat = F.interpolate(flat, size=target_shape, mode="bilinear", align_corners=False, antialias=True)
    flat = flat.clamp(0.0, 1.0)
    images = flat.view(B, S, *flat.shape[1:])

    norm_stats = IMAGE_NORMALIZATION_DICT[config.input_norm_type]
    mean = norm_stats.mean.view(1, 1, 3, 1, 1).to(images.device)
    std = norm_stats.std.view(1, 1, 3, 1, 1).to(images.device)
    normalized = (images - mean) / std
    normalized = normalized.to(device)

    views = [
        {"img": normalized[:, idx].contiguous(), "data_norm_type": [config.input_norm_type]}
        for idx in range(S)
    ]
    return views, normalized


@torch.no_grad()
def build_labels(
    predictions: list[dict[str, torch.Tensor]],
    device: torch.device,
    patch_size: int,
) -> torch.Tensor:
    if not predictions:
        raise ValueError("Empty predictions")

    pooled = []
    for pred in predictions:
        conf = pred["conf"].to(device)
        conf = F.avg_pool2d(conf.unsqueeze(1), kernel_size=patch_size, stride=patch_size)
        pooled.append(conf.flatten(2).transpose(1, 2))

    return torch.cat(pooled, dim=0)


@torch.no_grad()
def rank_tensor(tensor: torch.Tensor):
    original_shape = tensor.shape
    flat = tensor.flatten()
    sorted_indices = torch.argsort(flat)
    ranks = torch.empty_like(flat)
    ranks[sorted_indices] = torch.arange(len(flat), device=tensor.device, dtype=tensor.dtype)
    return ranks.view(original_shape)


@torch.no_grad()
def log_wandb(
    loss: torch.Tensor,
    label: torch.Tensor,
    output: torch.Tensor,
    predictions: list[dict[str, torch.Tensor]],
    config: MapAnythingTrainConfig,
    BPHWC: tuple[int, int, int, int, int],
    step: int,
):
    B, P, tH, tW, _ = BPHWC
    random_index = random.randint(0, B * P - 1)
    view_idx = random_index // B
    batch_idx = random_index % B

    pred = predictions[view_idx]
    image = pred["img_no_norm"][batch_idx].cpu().numpy()
    conf_shape = pred["conf"].shape[-2:]

    label_map = label[random_index].reshape(tH, tW).cpu()
    output_map = output[random_index].reshape(tH, tW).cpu()

    label_full = F.interpolate(label_map.unsqueeze(0).unsqueeze(0), size=conf_shape, mode="bilinear", align_corners=False).squeeze().cpu()
    output_full = F.interpolate(output_map.unsqueeze(0).unsqueeze(0), size=conf_shape, mode="bilinear", align_corners=False).squeeze().cpu()

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(image)
    axes[0].axis("off")

    match config.train_loss_func:
        case "pairwise-rank":
            axes[1].imshow(rank_tensor(output_map).cpu().numpy())
            axes[1].set_title("Predicted-Rank")
            axes[2].imshow(rank_tensor(label_map).cpu().numpy())
            axes[2].set_title("Label-Rank")
        case _:
            axes[1].imshow(output_full.numpy())
            axes[1].set_title("Predicted")
            axes[2].imshow(label_full.numpy())
            axes[2].set_title("Label")
    axes[1].axis("off")
    axes[2].axis("off")

    wandb.log(
        {
            f"loss/{config.train_loss_func}": loss.item(),
            "vis/sample": fig,
        },
        step=step,
    )
    plt.close(fig)


def get_loss_value(network_output: torch.Tensor, network_label: torch.Tensor, loss_method: T_SupportLoss):
    match loss_method:
        case "mse":
            pred = network_output.squeeze(-1)
            target = network_label.squeeze(-1)
            pred = (pred - pred.mean(dim=-1, keepdim=True)) / (pred.std(dim=-1, keepdim=True) + 1e-3)
            target = (target - target.mean(dim=-1, keepdim=True)) / (target.std(dim=-1, keepdim=True) + 1e-3)
            return torch.nn.functional.mse_loss(pred, target)
        case "mse-raw":
            pred = network_output.squeeze(-1)
            target = network_label.squeeze(-1)
            return torch.nn.functional.mse_loss(pred, target)
        case "pairwise-rank":
            return pairwise_logistic_loss(network_output.squeeze(-1), network_label.squeeze(-1))
        case _:
            raise ValueError("Unsupported loss")


def main_train(config: MapAnythingTrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = get_student_model(config).bfloat16()
    teacher = get_MapAnything().to(device).eval()
    probe = FeatureProbe()
    install_probe(teacher, probe, config.probe_loc)

    dataloader = get_data(config)
    space = set_wandb(config)

    optimizer = torch.optim.AdamW(student.parameters(), lr=config.train_learn_rate)
    scheduler = get_lr_scheduler(optimizer, config)
    optimizer.zero_grad()

    patch_size = teacher.encoder.patch_size
    optim_step = 0

    try:
        while True:
            for sample in dataloader:
                try:
                    with torch.inference_mode(), torch.no_grad():
                        views, normalized = prepare_views(sample.images, config, device)
                        teacher_amp_enabled = config.teacher_use_amp and device.type == "cuda"
                        amp_dtype = config.teacher_amp_dtype if teacher_amp_enabled else "fp32"

                        probe.tokens = None
                        predictions = teacher.infer(
                            views,
                            memory_efficient_inference=config.teacher_memory_efficient,
                            use_amp=teacher_amp_enabled,
                            amp_dtype=amp_dtype,
                            apply_mask=False,
                            mask_edges=False,
                            apply_confidence_mask=False,
                        )

                        if probe.tokens is None:
                            raise RuntimeError("Probe did not capture tokens")

                        tokens = probe.tokens
                        tokens = tokens[:, 1:, :]

                        B = normalized.size(0)
                        P = len(predictions)

                        conf_height, conf_width = predictions[0]["conf"].shape[-2:]
                        tH = conf_height // patch_size
                        tW = conf_width // patch_size

                        if conf_height % patch_size != 0 or conf_width % patch_size != 0:
                            raise ValueError("Confidence map is not aligned with encoder patch size")
                        if tokens.size(1) != tH * tW:
                            raise ValueError("Token count does not match spatial grid")

                        network_input = tokens.reshape(B * P, tH, tW, -1)
                        labels = build_labels(predictions, device, patch_size)

                    BPHWC = (B, P, tH, tW, network_input.size(-1))

                    preds = student(network_input.bfloat16(), BPHWC).float()
                    loss = get_loss_value(preds, labels, config.train_loss_func)

                    if not torch.isfinite(loss):
                        print(f"Warning! Train loss becomes non-finite value {loss}")
                        raise KeyboardInterrupt()

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    optim_step += 1
                    if optim_step % config.wandb_log_freq == 0:
                        log_wandb(loss, labels, preds, predictions, config, BPHWC, optim_step)
                    else:
                        print(f"loss={loss.item():.3f}")
                        wandb.log(
                            {
                                f"loss/{config.train_loss_func}": loss.item(),
                                "train/lr": scheduler.get_last_lr()[0],
                            },
                            step=optim_step,
                        )

                    if optim_step % config.train_save_freq == 0:
                        save_states(space, optim_step, student, optimizer, scheduler)

                    if optim_step > config.train_optim_step:
                        raise KeyboardInterrupt()
                except KeyboardInterrupt as e:
                    raise e from None
                except Exception as e:
                    raise e
                    # save_states(space, optim_step, student, optimizer, scheduler)
                    # print(f"Unstable training!! - saved checkpoint at optim_step={optim_step}. Continuing.")
                    # optimizer.zero_grad()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("Stopped upon user request")
    finally:
        save_states(space, optim_step, student, optimizer, scheduler)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-method", type=str, choices=["mse", "mse-raw", "pairwise-rank"])
    parser.add_argument("--final-activ", type=str, choices=["none", "expp1"])
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--train-batch", type=int)
    parser.add_argument("--train-set", type=str, default="tartanair")
    parser.add_argument("--train-lr", type=float)
    parser.add_argument("--train-step", type=int)
    parser.add_argument("--train-sched", type=lambda s: ast.literal_eval(s), default="'none'")
    parser.add_argument("--train-size", type=lambda s: ast.literal_eval(s), default=[(518, 518)])
    parser.add_argument("--probe-layer", type=int, default=15)
    parser.add_argument("--teacher-amp", action="store_true")
    parser.add_argument("--teacher-amp-dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--teacher-mem-efficient", action="store_true")

    args = parser.parse_args()

    config = MapAnythingTrainConfig(
        method="attn",
        method_dim=[1024, 32, 1],
        method_act=args.final_activ,
        probe_loc=("DINO", args.probe_layer),
        train_set=args.train_set,
        train_sequence_l=args.max_seq_len,
        train_learn_rate=args.train_lr,
        train_infer_size=args.train_size,
        train_optim_step=args.train_step,
        train_augment_π=True,
        train_augment_l=True,
        train_loss_func=args.loss_method,
        train_batch_size=args.train_batch,
        train_save_freq=250,
        train_scheduler=args.train_sched,
        teacher_use_amp=args.teacher_amp,
        teacher_amp_dtype=args.teacher_amp_dtype,
        teacher_memory_efficient=args.teacher_mem_efficient,
        wandb_log_freq=10,
    )

    main_train(config)
