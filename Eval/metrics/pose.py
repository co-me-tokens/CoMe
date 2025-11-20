import pypose as pp
from dataclasses import dataclass
from evo.core.trajectory import PosePath3D
from evo.core.metrics    import PoseRelation, Unit, Result
from evo.main_ape import ape
from evo.main_rpe import rpe


@dataclass
class PoseEvaluationResult:
    ate: float  # Unit: meter
    aoe: float  # Unit: deg
    
    rta: float  # Unit: meter
    rra: float  # Unit: deg

    per_pose_ate: list[float]
    per_pose_aoe: list[float]
    per_pair_rra: list[float]
    per_pair_rta: list[float]

def evaluate_pipeline(pred: pp.LieTensor, gt: pp.LieTensor) -> PoseEvaluationResult:
    pred_path = PosePath3D(
        positions_xyz=pred.translation().cpu().numpy(), orientations_quat_wxyz=pred.rotation().cpu().numpy()
    )
    
    gt_path   = PosePath3D(
        positions_xyz=gt.translation().cpu().numpy(), orientations_quat_wxyz=gt.rotation().cpu().numpy()
    )

    ate = ape(
        gt_path, pred_path,
        pose_relation=PoseRelation.translation_part,
        correct_scale=True, align=True
    )
    aoe = ape(
        gt_path, pred_path,
        pose_relation=PoseRelation.rotation_angle_deg,
        correct_scale=False, align=False, align_origin=True
    )
    
    rra = rpe(
        gt_path, pred_path,
        pose_relation=PoseRelation.rotation_angle_deg,
        delta=1, delta_unit=Unit.frames,
        correct_scale=True, align=False, align_origin=True, all_pairs=True
    )
    
    rta = rpe(
        gt_path, pred_path,
        pose_relation=PoseRelation.translation_part,
        delta=1, delta_unit=Unit.frames,
        correct_scale=True, align=False, align_origin=True, all_pairs=True
    )
    
    return PoseEvaluationResult(
        ate=ate.stats['mean'], aoe=aoe.stats['mean'],
        rra=rra.stats['mean'], rta=rta.stats['mean'],
        
        per_pose_ate=ate.np_arrays['error_array'].tolist(),
        per_pose_aoe=aoe.np_arrays['error_array'].tolist(),
        per_pair_rra=rra.np_arrays['error_array'].tolist(),
        per_pair_rta=rta.np_arrays['error_array'].tolist()
    )
