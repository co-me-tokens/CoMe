"""Depth evaluation: L1 error and delta-1.25 accuracy across good/bad/all pixel categories."""

import torch

from ..interface.geometric_model import SceneGeometry
from ..interface.metric import DepthEvaluation, Metric


def _masked_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """Compute avg L1 and delta-1.25 over masked pixels. Returns (nan, nan) when mask is empty."""
    if mask.sum() == 0:
        return float("nan"), float("nan")

    p, g = pred[mask], gt[mask]
    avg_l1 = (p - g).abs().mean().item()
    ratio  = torch.max(p / g, g / p)
    avg_delta_1_25 = (ratio < 1.25).float().mean().item()
    return avg_l1, avg_delta_1_25


def evaluate_depth(label: SceneGeometry, predict: SceneGeometry, *, align_scale: bool, max_depth: float=100.) -> DepthEvaluation:
    """
    Given the predicted and ground truth scene geometry, evaluate the depth quality.

    - If align_scale = True
      Apply scale alignment before depth evaluation.
      If there is infer_mask in the prediction, apply scale alignment only on the 'good' parts (pixel where infer_mask=False).

    All evaluation results will be categorized into three types:
        - good: metrics solely on places where inference_mask = False (if present), = all if no mask provided
        - bad : metrics solely on places where inference_mask = True  (if present), = all if no mask provided
        - all : metrics on all pixels
    """
    assert label.depths is not None, "label.depths is None — cannot evaluate depth without ground truth."
    assert predict.depths is not None, "predict.depths is None — cannot evaluate depth without prediction."

    _, _, _, H, W = label.depths.shape
    predict = predict.resize(H, W)

    # ==== Masks (B S 1 H W, bool) ====
    valid_mask = (label.depths > 0) & (label.depths < max_depth)

    if predict.infer_mask is not None:
        good_mask = valid_mask & ~predict.infer_mask
        bad_mask  = valid_mask &  predict.infer_mask
    else:
        good_mask = valid_mask
        bad_mask  = valid_mask

    all_mask = valid_mask

    # ==== Scale alignment ====
    assert predict.depths is not None, "predict.depths is None — cannot evaluate depth without prediction."
    pred_depths = predict.depths

    if align_scale:
        if good_mask.sum() == 0:
            raise ValueError("No valid good pixels available for scale alignment.")
        scale = label.depths[good_mask].mean() / pred_depths[good_mask].mean().clamp(min=1e-6)
        pred_depths = pred_depths * scale

    # ==== Metrics per category ====
    l1_good, d125_good = _masked_metrics(pred_depths, label.depths, good_mask)
    l1_bad,  d125_bad  = _masked_metrics(pred_depths, label.depths, bad_mask)
    l1_all,  d125_all  = _masked_metrics(pred_depths, label.depths, all_mask)

    mask_ratio = (bad_mask.sum() / all_mask.sum()).item() if (predict.infer_mask is not None and all_mask.sum() > 0) else 0.0

    return DepthEvaluation(
        metric_scale   = align_scale,
        mask_ratio     = mask_ratio,
        avg_l1         = Metric(good=l1_good, bad=l1_bad, all=l1_all),
        avg_delta_1_25 = Metric(good=d125_good, bad=d125_bad, all=d125_all),
    )
