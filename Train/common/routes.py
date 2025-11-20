import torch
from .sandbox import Sandbox

_Const_TrainState_FileTemplate = "train_state_{step:05d}.pth"
_Const_ModelState_FileTemplate = "checkpoint_{step:05d}.pth"


def save_states(space: Sandbox, step: int, model: torch.nn.Module, optim: torch.optim.Optimizer | None=None, sched: torch.optim.lr_scheduler.LRScheduler | None=None):
    torch.save(model.state_dict(), space.path(f"checkpoint_{step:05d}.pth"))
    
    if (not optim) and (not sched): return
    
    train_states = {
        "torch_optimizer"   : optim.state_dict() if optim else None,
        "torch_lr_scheduler": sched.state_dict() if sched else None
    }
    torch.save(train_states, space.path(f"train_state_{step:05d}.pth"))


def load_states(space: Sandbox, step: int, model: torch.nn.Module, optim: torch.optim.Optimizer | None, sched: torch.optim.lr_scheduler.LRScheduler | None = None):
    model_ckpt = torch.load(space.path(f"checkpoint_{step:05d}.pth"))
    model.load_state_dict(model_ckpt)
    
    if not space.path(f"train_state_{step:05d}.pth").exists(): return

    train_ckpt = torch.load(space.path(f"train_state_{step:05d}.pth"))
    if optim:
        assert train_ckpt["torch_optimizer"] is not None
        optim.load_state_dict(train_ckpt["torch_optimizer"])
    
    if sched:
        assert train_ckpt["torch_lr_scheduler"] is not None
        sched.load_state_dict(train_ckpt["torch_lr_scheduler"])
