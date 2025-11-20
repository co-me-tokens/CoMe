import ast
import json
import torch
import wandb
import random
import datetime
import typing as T
import matplotlib.pyplot as plt

from pathlib import Path
from rich.progress import track
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader

from Accelerate.core import Conv2DConfidence, GlobalAttentionConfidence
from Accelerate.vggt import get_block, ProbeLocation
from Network.vggt import VGGT, get_VGGT

from .common.sandbox import Sandbox
from .common.loss    import pairwise_logistic_loss
from .common.routes  import save_states, load_states


T_SupportData = T.Literal["tartanair"]
T_SupportLoss = T.Literal["mse", "pairwise-rank", "mse-raw"]


@dataclass
class VGGTTrainConfig:
    method    : T.Literal["conv", "attn"]
    method_dim: T.Sequence[int]
    method_act: T.Literal["expp1", "none"]
    
    train_set: T_SupportData
    probe_loc: ProbeLocation
    
    train_batch_size: int
    train_sequence_l: int
    train_learn_rate: float
    train_infer_size: T.Sequence[tuple[int, int]]
    train_optim_step: int
    train_augment_π : bool
    train_augment_l : bool
    train_supervise : T.Literal["depth_conf", "point_conf"]
    train_loss_func : T_SupportLoss
    train_save_freq : int
    train_scheduler : tuple[T.Literal["linear"], float, float, int, int] | T.Literal["none"]
    
    wandb_log_freq  : int
    wandb_entity    : str = "pyramid-infer"
    wandb_project   : str = "VGGT-ConfDistill-Sep17"
    
    resume_run : Path | None = None
    resume_step: int  | None = None


@dataclass
class FeatureProbe:
    tokens: torch.Tensor | None = None

def install_probe(model: VGGT, probe: FeatureProbe, loc: ProbeLocation):
    target_block = get_block(model, loc)
    def feature_recorder(module, args, output):
        probe.tokens = output    
    target_block.register_forward_hook(feature_recorder)

def get_data(config: VGGTTrainConfig) -> torch.utils.data.DataLoader:
    match config.train_set:
        case "tartanair":
            from Data.TartanAir import TartanAir_DepthDataset, default_root
            dataset = TartanAir_DepthDataset(default_root, config.train_sequence_l)
            return DataLoader(
                dataset, batch_size=config.train_batch_size,
                shuffle=True, collate_fn=TartanAir_DepthDataset.collate,
                drop_last=True
            )
        case _:
            raise ValueError("Unsupported")

def get_model(config: VGGTTrainConfig):
    match config.method:
        case "conv":
            return Conv2DConfidence(config.method_dim, config.method_act).train().to("cuda")
        case "attn":
            return GlobalAttentionConfidence(config.method_dim, config.method_act).train().to("cuda")
        case _:
            raise ValueError("Unsupported")

class NoOpScheduler(torch.optim.lr_scheduler.LRScheduler):
    """No-op scheduler that does nothing but maintains the same interface."""
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def get_lr_scheduler(optim: torch.optim.Optimizer, config: VGGTTrainConfig) -> torch.optim.lr_scheduler.LRScheduler:
    match config.train_scheduler:
        case "none":
            return NoOpScheduler(optim)
        
        case ("linear", start_lr, end_lr, decay_start_step, decay_end_step):
            def linear_decay(step):
                if step < decay_start_step:
                    return 1.0
                elif step >= decay_end_step:
                    return end_lr / start_lr
                else:
                    # Linear interpolation
                    progress = (step - decay_start_step) / (decay_end_step - decay_start_step)
                    return 1.0 - progress * (1.0 - end_lr / start_lr)
            
            return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=linear_decay)
        
        case _:
            raise ValueError("Unsupported")

def set_wandb(config: VGGTTrainConfig) -> Sandbox:
    import os
    slurm_task_id = os.getenv("SLURM_ARRAY_TASK_ID") or "0"
    run_name = f"{datetime.datetime.now():%Y%m%d_%H:%M:%S}_{slurm_task_id}"
    grp_name = f"{config.method}_{config.train_loss_func}_{config.probe_loc}@{config.train_infer_size}"
    wandb.init(
        entity=config.wandb_entity, project=config.wandb_project,
        name=run_name, group=grp_name, config=asdict(config)
    )
    
    space = Sandbox(Path("Result", "Models", "ConfDistill", "VGGT", run_name))
    with space.open("config.json", "w") as f:
        json.dump(asdict(config), f)
    return space

@torch.no_grad()
def img_preprocess(images: torch.Tensor, config: VGGTTrainConfig):
    B, S = images.shape[:2]
    
    if config.train_augment_π:
        π  = torch.stack([torch.randperm(S) for _ in range(B)])
        bᵢ = torch.arange(B).unsqueeze(1)
        images = images[bᵢ, π]
    
    if config.train_augment_l:
        S_aug = random.randint(1, S)
        images = images[:, :S_aug]
    
    target_shape = random.choice(config.train_infer_size)
    images = torch.nn.functional.interpolate(
        images.flatten(start_dim=0, end_dim=1),
        size=target_shape,
        mode='bilinear', antialias=True
    ).view(*images.shape[:-2], *target_shape)
    
    return images.cuda()

@torch.no_grad()
def get_supervise_signal(images: torch.Tensor, vggt: VGGT, config: VGGTTrainConfig):
    aggregated_tokens_list, ps_idx = vggt.aggregator(images)
    
    match config.train_supervise:
        case "depth_conf":
            depth_map, depth_conf = vggt.depth_head(aggregated_tokens_list, images, ps_idx)
            return depth_conf
        case "point_conf":
            point_map, point_conf = vggt.point_head(aggregated_tokens_list, images, ps_idx)
            return point_conf
        case _:
            raise ValueError("Unsupported")

@torch.no_grad()
def rank_tensor(tensor):
    """
    Reassigns the values of a 2D tensor to their ranks.

    The rank is based on the flattened tensor, with 0 being the smallest
    value and (H*W - 1) being the largest.

    Args:
        tensor (torch.Tensor): A 2D tensor of shape (H, W).

    Returns:
        torch.Tensor: A tensor of the same shape with values replaced by their ranks.
    """
    # 1. Store the original shape
    original_shape = tensor.shape

    # 2. Flatten the tensor to 1D
    flattened_tensor = tensor.flatten()
    
    # 3. Get the indices that would sort the flattened tensor
    # This tells us the original position of each value in the sorted list.
    sorted_indices = torch.argsort(flattened_tensor)

    # 4. Create a new tensor to hold the ranks
    # We use empty_like to ensure the new tensor has the same properties (like device)
    ranks = torch.empty_like(flattened_tensor)
    
    # 5. Assign ranks based on the sorted indices
    # torch.arange creates the sequence of ranks [0, 1, 2, ...]
    # We place these ranks into the `ranks` tensor at the `sorted_indices`.
    ranks[sorted_indices] = torch.arange(len(flattened_tensor), 
                                         device=tensor.device, 
                                         dtype=tensor.dtype)
    
    # 6. Reshape the ranks back to the original shape
    return ranks.view(original_shape)

@torch.no_grad()
def log_wandb(loss: torch.Tensor, network_label: torch.Tensor, network_output: torch.Tensor, images: torch.Tensor, config: VGGTTrainConfig, BPHWC: tuple[int, ...]):
    B, P, H, W, C = BPHWC
    
    random_index = random.randint(0, config.train_batch_size * (images.size(1)-1))
    
    image  = images.flatten(0, 1)[random_index].permute(1, 2, 0).cpu()
    output = network_output[random_index].reshape(H, W).cpu()
    label  = network_label[random_index].reshape(H, W).cpu()
    
    
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(image); axes[0].axis('off')
    
    match config.train_loss_func:
        case "mse":
            axes[1].imshow(output); axes[1].axis('off')
            axes[1].set_title("Predicted-Value")
            
            axes[2].imshow(label); axes[2].axis('off')
            axes[2].set_title("Label-Value")
        
        case "pairwise-rank":
            # Recoloring based on rank since we only care about rank consistency
            axes[1].imshow(rank_tensor(output)); axes[1].axis('off')
            axes[1].set_title("Predicted-Rank")
            
            axes[2].imshow(rank_tensor(label)); axes[2].axis('off')
            axes[2].set_title("Label-Rank")
        
        case _: raise ValueError("Unsupported")

    wandb.log({
        f"loss/{config.train_loss_func}" : loss.item(),
        "vis/visualize_loss": fig
    })
    plt.close(fig)

def get_loss_value(network_output: torch.Tensor, network_label: torch.Tensor, loss_method: T_SupportLoss):
    match loss_method:
        case "mse":
            network_output = network_output.squeeze(-1)
            network_label  = network_label.squeeze(-1)
            
            normalized_output = (network_output - network_output.mean(dim=-1, keepdim=True)) / (network_output.std(dim=-1, keepdim=True) + 1e-3)
            normalized_label  = (network_label - network_label.mean(dim=-1, keepdim=True)) / (network_label.std(dim=-1, keepdim=True) + 1e-3)
            return torch.nn.functional.mse_loss(normalized_output, normalized_label)
        
        case "mse-raw":
            network_output = network_output.squeeze(-1)
            network_label  = network_label.squeeze(-1)
            return torch.nn.functional.mse_loss(network_output, network_label)
        
        case "pairwise-rank":
            return pairwise_logistic_loss(network_output.squeeze(-1), network_label.squeeze(-1))

        case _:
            raise ValueError("Unsupproted")

def main_train(config: VGGTTrainConfig):
    model      = get_model(config).bfloat16()
    probe      = FeatureProbe()
    space      = set_wandb(config)
    dataloader = get_data(config)
    
    vggt  = get_VGGT().cuda().eval()
    install_probe(vggt, probe, config.probe_loc)
    
    optim_step = 0
    optimizer  = torch.optim.AdamW(model.parameters(), lr=config.train_learn_rate)
    
    scheduler  = get_lr_scheduler(optimizer, config)
    optimizer.zero_grad()
    
    try:
        while True:
            for sample in track(dataloader):
                try:
                    with torch.no_grad():
                        images = img_preprocess(sample.images, config)
                        label  = get_supervise_signal(images, vggt, config)
                        tH, tW = images.shape[-2] // 14, images.shape[-1] // 14,
                        assert probe.tokens is not None
                        
                        network_label = label.flatten(0, 1).unfold(1, 14, 14).unfold(2, 14, 14).mean(dim=[-1, -2]).unsqueeze(-1)
                        network_label = network_label.flatten(1, 2)
                        
                        network_input = probe.tokens[:, vggt.aggregator.patch_start_idx:]
                        network_input = network_input.reshape(config.train_batch_size * images.size(1), tH, tW, -1)
                    
                    BPHWC = (config.train_batch_size, images.size(1), tH, tW, network_input.size(-1))
                    network_output = model(network_input.bfloat16(), BPHWC).float()
                    
                    loss  = get_loss_value(network_output, network_label, config.train_loss_func)
                    
                    if not torch.isfinite(loss):
                        print(f"Warning! Train loss becomes non-finite value {loss}")
                        raise KeyboardInterrupt()
                    
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    optim_step += 1
                    if optim_step % config.wandb_log_freq  == 0:
                        log_wandb(loss, network_label, network_output, images, config, BPHWC)
                    else:
                        print(f"{loss.item()=:.3f}")
                        wandb.log(
                            {f"loss/{config.train_loss_func}" : loss.item(), f"train/lr": scheduler.get_last_lr()[0]},
                            step=optim_step
                        )
                    
                    if optim_step % config.train_save_freq == 0:
                        save_states(space, optim_step, model, optimizer, scheduler)
                    
                    if optim_step > config.train_optim_step: raise KeyboardInterrupt()
                except KeyboardInterrupt as e:
                    raise e from None
                except:
                    save_states(space, optim_step, model, optimizer, scheduler)
                    print(f"Unstable training!! -  saved checkpoint at {optim_step=}. Continue w/ best effort recovery.")
                    optimizer.zero_grad()    
                
                torch.cuda.empty_cache()
                
    
    except KeyboardInterrupt:
        print("Stopped upon user request")
        
    finally:
        save_states(space, optim_step, model, optimizer, scheduler)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-method", type=str, choices=["mse", "mse-raw", "pairwise-rank"])
    parser.add_argument("--final-activ", type=str, choices=["none", "expp1"])
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--train-batch", type=int)
    parser.add_argument("--train-set"  , type=str, default="tartanair")
    parser.add_argument("--train-lr"   , type=float)
    parser.add_argument("--train-step" , type=int)
    parser.add_argument("--train-sched", type=lambda s: ast.literal_eval(s), default="'none'")
    parser.add_argument("--train-target", type=str, choices=["depth_conf", "point_conf"])
    parser.add_argument("--inject-layer", type=int, default=15)
    args   = parser.parse_args()
    
    default_config = VGGTTrainConfig(
        method='attn',
        method_dim=[1024, 32, 1],
        method_act=args.final_activ,
        probe_loc =("DINO", args.inject_layer),
        
        train_set       =args.train_set,
        train_sequence_l=args.max_seq_len,
        train_learn_rate=args.train_lr,
        train_infer_size=[(27 * 14, 36 * 14)],
        train_optim_step=args.train_step,
        train_augment_π =True,
        train_augment_l =True,
        train_supervise =args.train_target,
        train_batch_size=args.train_batch,
        train_loss_func =args.loss_method,
        train_save_freq =250,
        train_scheduler =args.train_sched,
        
        wandb_log_freq  =10
    )
    
    main_train(default_config)
