import torch
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    avg_l1    : float
    avg_rmse  : float
    avg_absrel: float
    avg_δ1_25 : float


def _align_monocular_depth(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    f_pred = pred.flatten() if mask is None else pred[mask]
    f_gt   = gt.flatten() if mask is None else gt[mask]
    
    scale = f_gt.mean() / f_pred.mean().clamp(min=1e-6)
    return scale * pred


def _l1_error(aligned_pred: torch.Tensor, gt_depth: torch.Tensor) -> float:
    return (aligned_pred - gt_depth).abs().nanmean().item()


def _rmse_error(aligned_pred: torch.Tensor, gt_depth: torch.Tensor) -> float:
    return (aligned_pred - gt_depth).square().nanmean().sqrt().item()


def _absrel_error(aligned_pred: torch.Tensor, gt_depth: torch.Tensor) -> float:
    return ((aligned_pred - gt_depth).abs() / gt_depth.clamp_min(min=1e-6)).nanmean().item()


def _δ1_25_error(aligned_pred: torch.Tensor, gt_depth: torch.Tensor) -> float:
    ratio_pred_gt = aligned_pred / gt_depth.clamp_min(min=1e-6)
    ratio_gt_pred = gt_depth / aligned_pred.clamp_min(min=1e-6)
    
    delta_ratio   = torch.maximum(ratio_pred_gt, ratio_gt_pred) < 1.25
    return delta_ratio.float().nanmean().item()



def evaluation_pipeline(pred_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor | None) -> EvaluationResult:
    gt_depth     = gt_depth.to(pred_depth)
    aligned_pred = _align_monocular_depth(pred_depth, gt_depth, mask)
    
    if mask is not None:
        aligned_pred = aligned_pred[mask]
        gt_depth     = gt_depth[mask]
    else:
        aligned_pred = aligned_pred.flatten()
        gt_depth     = gt_depth.flatten()
    
    return EvaluationResult(
        avg_l1      =_l1_error(aligned_pred, gt_depth),
        avg_rmse    =_rmse_error(aligned_pred, gt_depth),
        avg_absrel  =_absrel_error(aligned_pred, gt_depth),
        avg_δ1_25   =_δ1_25_error(aligned_pred, gt_depth)
    )
