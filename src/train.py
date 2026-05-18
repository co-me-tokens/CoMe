"""Confidence distillation training pipeline."""

import json
import random
import typing as T

from pathlib import Path
from dataclasses import dataclass, fields
from enum import Enum, auto

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .utility.diagnostic import Diagnostics
from .interface.geometric_model import GeometricPredictorLike, MultiViewInput, SceneGeometry, TensorProbe
from .interface.dataset import MultiViewGeometryDataset, MultiViewTask

from .accelerate.common.confidence_predictor import GlobalAttentionConfidence


_PATCH_SIZE = 14


class TrainTarget(Enum):
    Distill_DepthConf = auto()
    Distill_PointConf = auto()
    Distill_Intersect = auto()


@dataclass(slots=True, frozen=True)
class TrainConfiguration:
    train_target    : TrainTarget
    learning_rate   : float
    weight_decay    : float

    inject_block_idx: int
    predictor_dims  : list[int]
    predictor_active: T.Literal["expp1", "none"]
    predictor_num_register: int

    batch_size      : int
    mini_batch_size : int
    num_step        : int
    segment_length  : int
    infer_size      : tuple[int, int]
    loss_function   : T.Literal["mse", "pairwise-rank", "abs_mse"]
    permute_segment : bool
    crop_segment    : bool
    train_dtype     : torch.dtype

    output_directory: Path
    save_frequency  : int

    use_wandb       : bool
    wandb_group     : str
    wandb_entity    : str = "yutianch_cmu"
    wandb_project   : str = "CoMe-Next Confidence Distillation"

    resume_from     : Path | None = None

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "TrainConfiguration":
        dtype = getattr(torch, cfg.training.train_dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"'{cfg.training.train_dtype}' is not a valid torch.dtype")
        if cfg.predictor.num_register < 0:
            raise ValueError(f"predictor.num_register must be non-negative, got {cfg.predictor.num_register}")

        return cls(
            train_target     = TrainTarget[cfg.train_target],
            learning_rate    = cfg.optimizer.lr,
            weight_decay     = cfg.optimizer.weight_decay,
            inject_block_idx = cfg.predictor.inject_block_idx,
            predictor_dims   = list(cfg.predictor.dims),
            predictor_active = cfg.predictor.active,
            predictor_num_register = cfg.predictor.num_register,
            batch_size       = cfg.training.batch_size,
            mini_batch_size  = cfg.training.mini_batch_size,
            num_step         = cfg.training.num_step,
            segment_length   = cfg.training.segment_length,
            infer_size       = tuple(cfg.training.infer_size),
            loss_function    = cfg.training.loss_function,
            permute_segment  = cfg.training.permute_segment,
            crop_segment     = cfg.training.crop_segment,
            train_dtype      = dtype,
            output_directory = Path(cfg.checkpoint.output_directory),
            save_frequency   = cfg.checkpoint.save_frequency,
            use_wandb        = cfg.wandb.enabled,
            wandb_group      = cfg.wandb.group,
            wandb_entity     = cfg.wandb.entity,
            wandb_project    = cfg.wandb.project,
            resume_from      = Path(cfg.checkpoint.resume_from) if cfg.checkpoint.resume_from is not None else None,
        )

    def to_dict(self) -> dict[str, T.Any]:
        result: dict[str, T.Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            match value:
                case Enum():
                    result[f.name] = value.name
                case torch.dtype():
                    result[f.name] = str(value)
                case Path():
                    result[f.name] = str(value)
                case _:
                    result[f.name] = value
        return result


# ==== Checkpoint Management ====

def save_checkpoint(
    run_dir: Path, step: int, config: TrainConfiguration,
    predictor: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
):
    step_dir = run_dir / f"step_{step:05d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    with open(step_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    model_state = predictor.state_dict()
    train_state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}

    torch.save(model_state, step_dir / "checkpoint.pth")
    torch.save(train_state, step_dir / "training_state.pth")

    Diagnostics.log(f"Checkpoint saved at {step_dir}")


def resume_train(resume_path: Path, predictor: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler) -> int:
    if not resume_path.is_dir(): raise FileNotFoundError(f"Resume path does not exist: {resume_path}")

    model_state = torch.load(resume_path / "checkpoint.pth", weights_only=True)
    train_state = torch.load(resume_path / "training_state.pth", weights_only=True)
    predictor.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(train_state["optimizer"])
    scheduler.load_state_dict(train_state["scheduler"])

    dir_name = resume_path.name
    step     = int(dir_name.removeprefix("step_"))

    Diagnostics.log(f"Resumed from {resume_path} at step {step}")
    return step


# ==== Supervision Signal ====

def get_label(geometry: SceneGeometry, target: TrainTarget, patch_size: int) -> torch.Tensor:
    match target:
        case TrainTarget.Distill_DepthConf:
            assert geometry.depths_conf is not None, "Model did not produce depths_conf"
            signal = geometry.depths_conf
        case TrainTarget.Distill_PointConf:
            assert geometry.points_conf is not None, "Model did not produce points_conf"
            signal = geometry.points_conf
        case TrainTarget.Distill_Intersect:
            assert geometry.depths_conf is not None and geometry.points_conf is not None, \
                "Model did not produce both depths_conf and points_conf"
            signal = torch.min(geometry.depths_conf, geometry.points_conf)
        case _:
            raise ValueError(f"Unsupported train target: {target}")
        
    signal = signal.squeeze(2)                          # (B, P, H, W)
    B, P = signal.shape[:2]
    signal = signal.flatten(0, 1)                       # (B*P, H, W)
    signal = signal.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    signal = signal.mean(dim=[-1, -2])                  # (B*P, tH, tW)
    signal = signal.flatten(1, 2)                       # (B*P, N)
    return signal.reshape(B, P * signal.shape[1]).unsqueeze(-1)  # (B, P*N, 1)


# ==== Loss Functions ====

def loss_function(output: torch.Tensor, label: torch.Tensor, method: str) -> torch.Tensor:
    """Compute training loss between predicted and target confidence.

    Args:
        output: [N, T, 1] predicted confidence.
        label:  [N, T, 1] target confidence.
        method: "mse" (z-normalized) or "pairwise-rank" or "abs_mse".
    """
    output = output.squeeze(-1)
    label  = label.squeeze(-1)

    match method:
        case "mse":
            norm_out = (output - output.mean(dim=-1, keepdim=True)) / (output.std(dim=-1, keepdim=True) + 1e-3)
            norm_lbl = (label  - label.mean(dim=-1, keepdim=True))  / (label.std(dim=-1, keepdim=True)  + 1e-3)
            return F.mse_loss(norm_out, norm_lbl)
        case "pairwise-rank":
            N, T = output.shape
            n_pairs = min(8192 * 4, T)
            idx_a = torch.randint(0, T, (n_pairs,), device=output.device)
            idx_b = torch.randint(0, T, (n_pairs,), device=output.device)
            pred_diff   = output[:, idx_a] - output[:, idx_b]
            target_sign = (label[:, idx_a] - label[:, idx_b]).sign()
            return -F.logsigmoid(pred_diff * target_sign).mean()
        case "abs_mse":
            return F.mse_loss(output.log(), label.log().clone())
        case _:
            raise ValueError(f"Unsupported loss function: {method}")


# ==== Data Augmentation & Iteration ====

def augment(mv_input: MultiViewInput, config: TrainConfiguration) -> MultiViewInput:
    B, S = mv_input.images.shape[:2]
    images = mv_input.images
    intrinsics = mv_input.intrinsics

    if config.crop_segment and S > 2:
        s_crop = random.randint(2, S)
        images = images[:, :s_crop]
        if intrinsics is not None:
            intrinsics = intrinsics[:, :s_crop]
        S = s_crop

    if config.permute_segment:
        perms = torch.stack([torch.randperm(S) for _ in range(B)])
        b_idx = torch.arange(B).unsqueeze(1)
        images = images[b_idx, perms]
        if intrinsics is not None:
            intrinsics = intrinsics[b_idx, perms]

    return MultiViewInput(images=images, intrinsics=intrinsics)


# ==== Training Loop ====

class _ParallelGeometryInference(torch.nn.Module):
    def __init__(
        self,
        model: GeometricPredictorLike,
        inject_block_idx: int,
        train_target: TrainTarget,
        patch_size: int,
        token_height: int,
        token_width: int,
    ) -> None:
        super().__init__()
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected model to be a torch.nn.Module, got {type(model)!r}")

        self.model = model
        self.train_target = train_target
        self.patch_size = patch_size
        self.token_height = token_height
        self.token_width = token_width
        self.probe = TensorProbe()
        self.model.inject_probe(inject_block_idx, self.probe)

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mv_input = MultiViewInput(images=images, intrinsics=intrinsics)
        geometry = self.model(mv_input)
        label = get_label(geometry, self.train_target, self.patch_size)

        if self.probe.data is None:
            raise RuntimeError("Probe did not capture features")

        mini_batch_size, segment_length = images.shape[:2]
        features = self.probe.data.reshape(
            mini_batch_size * segment_length,
            self.token_height,
            self.token_width,
            -1,
        )
        return label, features


@Diagnostics.activate()
def confidence_distillation(
    dataset: MultiViewGeometryDataset,
    model: GeometricPredictorLike,
    config: TrainConfiguration,
    optimizer_cfg: DictConfig,
    scheduler_cfg: DictConfig,
) -> None:
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected model to be a torch.nn.Module, got {type(model)!r}")

    predictor = GlobalAttentionConfidence(
        config.predictor_dims,
        config.predictor_active,
        num_register=config.predictor_num_register,
    )
    predictor = predictor.to(device="cuda", dtype=config.train_dtype).train()

    if config.batch_size % config.mini_batch_size != 0:
        raise ValueError(f"batch_size ({config.batch_size}) must be divisible by mini_batch_size ({config.mini_batch_size})")

    n_mini_steps = config.batch_size // config.mini_batch_size

    optimizer: torch.optim.Optimizer  = instantiate(optimizer_cfg, params=predictor.parameters())
    scheduler: torch.optim.lr_scheduler.LRScheduler = instantiate(scheduler_cfg, optimizer=optimizer)

    run_dir = Path(config.output_directory)
    run_dir.mkdir(parents=True, exist_ok=True)
    Diagnostics.set_file_output(run_dir / "diagnostic.log")

    step = 0
    if config.resume_from is not None:
        step = resume_train(Path(config.resume_from), predictor, optimizer, scheduler)

    if config.use_wandb:
        import wandb
        if Path("./tools/wandb_key.secret").exists():
            with open("./tools/wandb_key.secret", "r") as f:
                wandb.login(key=f.read().strip())
        
        wandb.init(
            project=config.wandb_project,
            dir=config.output_directory,
            entity=config.wandb_entity,
            group=config.wandb_group,
            config=config.to_dict(),
        )
    else:
        wandb = None
        

    tH, tW = config.infer_size[0] // _PATCH_SIZE, config.infer_size[1] // _PATCH_SIZE
    visible_device_count = torch.cuda.device_count()
    if visible_device_count == 0:
        raise RuntimeError("confidence_distillation requires at least one visible CUDA device")

    geometry_inference: torch.nn.Module = _ParallelGeometryInference(
        model=model,
        inject_block_idx=config.inject_block_idx,
        train_target=config.train_target,
        patch_size=_PATCH_SIZE,
        token_height=tH,
        token_width=tW,
    ).cuda().eval()
    if visible_device_count > 1:
        geometry_inference = torch.nn.DataParallel(
            geometry_inference,
            device_ids=list(range(visible_device_count)),
        )

    data_loader = DataLoader(
        dataset, collate_fn=MultiViewTask.collate,
        batch_size=config.mini_batch_size, shuffle=True,
        num_workers=4, persistent_workers=True, prefetch_factor=2
    )

    try:
        accum_step       = 0
        accumulated_loss = 0.0
        optimizer.zero_grad()

        while step < config.num_step:
            task: MultiViewTask
            loader_iter = iter(data_loader)

            while step < config.num_step:
                try:
                    task = next(loader_iter)
                except StopIteration:
                    break
                except Exception as e:
                    Diagnostics.log(f"Skipping corrupted batch: {e}")
                    continue

                mv_input = augment(task.multiview_input, config)
                mv_input = mv_input.resize(*config.infer_size)
                mv_input = mv_input.to(device="cuda")
                mini_batch_size = mv_input.images.size(0)

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    label, features = geometry_inference(mv_input.images, mv_input.intrinsics)

                BPHWC  = (mini_batch_size, mv_input.images.size(1), tH, tW, features.size(-1))
                output = predictor(features.to(config.train_dtype), BPHWC).float()
                output = output.reshape(mini_batch_size, -1, 1)               # (B, P*N, 1)
                loss   = loss_function(output, label, config.loss_function) / n_mini_steps
                loss.backward()
                accumulated_loss += loss.item()
                accum_step += 1

                if accum_step < n_mini_steps:
                    continue

                if not torch.isfinite(torch.tensor(accumulated_loss)):
                    save_checkpoint(run_dir, step, config, predictor, optimizer, scheduler)
                    raise RuntimeError(f"Training loss became non-finite ({accumulated_loss}) at step {step}")

                optimizer.step()
                scheduler.step()
                step += 1

                lr = scheduler.get_last_lr()[0]
                Diagnostics.log(f"step={step:05d}  loss={accumulated_loss:.4f}  lr={lr:.2e}")

                if wandb is not None:
                    wandb.log({"loss": accumulated_loss, "lr": lr}, step=step)

                if step % config.save_frequency == 0:
                    save_checkpoint(run_dir, step, config, predictor, optimizer, scheduler)

                torch.cuda.empty_cache()

                accum_step       = 0
                accumulated_loss = 0.0
                optimizer.zero_grad()

    except KeyboardInterrupt:
        Diagnostics.log("Training interrupted by user.")

    finally:
        save_checkpoint(run_dir, step, config, predictor, optimizer, scheduler)
        if wandb is not None: wandb.finish()
        Diagnostics.clear_file_output()



@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    from .dataset.tartanair_v1 import AllTartanAirv1_Dataset
    
    config  = TrainConfiguration.from_hydra(cfg)
    dataset = AllTartanAirv1_Dataset(
        Path("/data/tartanair_v1"), segment_length=config.segment_length,
        step=1, device=torch.device("cpu")
    )
    model   = instantiate(cfg.model).eval().cuda()

    confidence_distillation(dataset, model, config, cfg.optimizer, cfg.scheduler)


def patch():
    """
    NOTE: Patching to run smoothly on various platforms / archs.
    """
    # NOTE: On CU126 there are some tricky problem w/ gradient stride when running training 
    #   w/ CUDNN accelerated attention operators under bf16.
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
      torch.backends.cuda.enable_cudnn_sdp(False)


if __name__ == "__main__":
    patch()
    main()  #type: ignore
