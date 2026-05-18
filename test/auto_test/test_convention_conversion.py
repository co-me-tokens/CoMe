"""Correctness tests for SceneGeometry coordinate convention conversion.

Covers validation, point/pose/intrinsics transformation, round-trip
consistency (same- and cross-handedness), and edge cases.
"""

import math

import pytest
import torch

from src.interface.geometric_model import (
    SceneGeometry,
    PoseConvention as P,
    CoordinateConvention,
    _validate_convention,
    _conversion_matrix,
    _axis_pair,
    _axis_sign,
)


ATOL = 1e-5


# ==== Helpers ====

def _make_geo(
    points: torch.Tensor,
    poses: torch.Tensor,
    K: torch.Tensor,
    conv: CoordinateConvention = (P.R, P.D, P.F),
) -> SceneGeometry:
    B, S = poses.shape[:2]
    _, _, _, H, W = points.shape
    return SceneGeometry(
        pose_convention=conv,
        depths=torch.ones(B, S, 1, H, W),
        depths_conf=None,
        points=points.clone(),
        points_conf=None,
        poses=poses.clone(),
        intrinsics=K.clone(),
        infer_mask=None,
    )


def _identity_quat() -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 0.0, 1.0])


def _z_rotation_quat(angle: float) -> torch.Tensor:
    """Quaternion (xyzw) for a rotation of *angle* radians about the Z axis."""
    return torch.tensor([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])


B, S, H, W = 1, 2, 4, 4
_POINTS = torch.randn(B, S, 3, H, W)
_K = torch.tensor([[[320.0, 0.0, 160.0],
                     [0.0, 240.0, 120.0],
                     [0.0,   0.0,   1.0]]]).unsqueeze(0).expand(B, S, -1, -1).clone()


# ==== Validation ====

@pytest.mark.parametrize("conv", [
    (P.R, P.D, P.F),
    (P.F, P.R, P.U),
    (P.L, P.U, P.B),
    (P.D, P.B, P.L),
])
def test_valid_conventions_accepted(conv: CoordinateConvention):
    _validate_convention(conv)


@pytest.mark.parametrize("conv", [
    (P.F, P.F, P.R),
    (P.R, P.R, P.U),
    (P.R, P.L, P.U),
    (P.U, P.D, P.F),
    (P.F, P.B, P.R),
])
def test_invalid_conventions_rejected(conv: CoordinateConvention):
    with pytest.raises(ValueError, match="axis pairs"):
        _validate_convention(conv)


# ==== Axis helpers ====

def test_axis_pair_groups():
    assert _axis_pair(P.F) == _axis_pair(P.B) == 0
    assert _axis_pair(P.R) == _axis_pair(P.L) == 1
    assert _axis_pair(P.U) == _axis_pair(P.D) == 2


def test_axis_sign_polarity():
    for pos in (P.F, P.R, P.U):
        assert _axis_sign(pos) == 1
    for neg in (P.B, P.L, P.D):
        assert _axis_sign(neg) == -1


# ==== Conversion matrix ====

def test_conversion_matrix_identity():
    C = _conversion_matrix((P.R, P.D, P.F), (P.R, P.D, P.F), device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(C, torch.eye(3))


def test_conversion_matrix_sign_flip():
    C = _conversion_matrix((P.R, P.D, P.F), (P.R, P.U, P.B), device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(C, torch.diag(torch.tensor([1.0, -1.0, -1.0])))


def test_conversion_matrix_permutation():
    C = _conversion_matrix((P.R, P.D, P.F), (P.F, P.R, P.D), device=torch.device("cpu"), dtype=torch.float32)
    pt = C @ torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(pt, torch.tensor([3.0, 1.0, 2.0]))


def test_conversion_matrix_is_orthogonal():
    C = _conversion_matrix((P.R, P.D, P.F), (P.U, P.F, P.L), device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(C @ C.T, torch.eye(3), atol=ATOL, rtol=0.0)


# ==== SceneGeometry.to() — point transformation ====

def test_point_transform_sign_flip():
    poses = torch.cat([torch.zeros(B, S, 3), _identity_quat().expand(B, S, -1)], dim=-1)
    geo = _make_geo(_POINTS, poses, _K)
    geo.to(convention=(P.R, P.U, P.B))
    expected = _POINTS.clone()
    expected[:, :, 1] *= -1
    expected[:, :, 2] *= -1
    torch.testing.assert_close(geo.points, expected, atol=ATOL, rtol=0.0)


def test_point_transform_permutation():
    poses = torch.cat([torch.zeros(B, S, 3), _identity_quat().expand(B, S, -1)], dim=-1)
    geo = _make_geo(_POINTS, poses, _K)
    geo.to(convention=(P.F, P.R, P.D))
    assert geo.points is not None
    torch.testing.assert_close(geo.points[:, :, 0], _POINTS[:, :, 2], atol=ATOL, rtol=0.0)
    torch.testing.assert_close(geo.points[:, :, 1], _POINTS[:, :, 0], atol=ATOL, rtol=0.0)
    torch.testing.assert_close(geo.points[:, :, 2], _POINTS[:, :, 1], atol=ATOL, rtol=0.0)


# ==== SceneGeometry.to() — pose transformation ====

def test_translation_sign_flip():
    t = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    poses = torch.cat([t, _identity_quat().expand(B, S, -1)], dim=-1)
    geo = _make_geo(_POINTS, poses, _K)
    geo.to(convention=(P.R, P.U, P.B))
    assert geo.poses is not None
    torch.testing.assert_close(geo.poses[..., :3], torch.tensor([[[1.0, -2.0, -3.0], [4.0, -5.0, -6.0]]]), atol=ATOL, rtol=0.0)


def test_identity_rotation_preserved():
    poses = torch.cat([torch.randn(B, S, 3), _identity_quat().expand(B, S, -1)], dim=-1)
    geo = _make_geo(_POINTS, poses, _K)
    geo.to(convention=(P.L, P.U, P.B))
    assert geo.poses is not None
    import pypose as pp
    R = pp.SO3(geo.poses[..., 3:]).matrix()
    torch.testing.assert_close(R, torch.eye(3).expand_as(R), atol=ATOL, rtol=0.0)


def test_nontrivial_rotation_round_trip():
    q = _z_rotation_quat(math.pi / 4)
    poses = torch.cat([torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]), q.expand(B, S, -1)], dim=-1)
    geo = _make_geo(_POINTS, poses, _K)
    assert geo.poses is not None
    poses_save = geo.poses.clone()
    geo.to(convention=(P.F, P.U, P.R))
    geo.to(convention=(P.R, P.D, P.F))
    torch.testing.assert_close(geo.poses, poses_save, atol=ATOL, rtol=0.0)


# ==== SceneGeometry.to() — intrinsics transformation ====

def test_intrinsics_round_trip():
    poses = torch.cat([torch.zeros(B, S, 3), _identity_quat().expand(B, S, -1)], dim=-1)
    geo = _make_geo(_POINTS, poses, _K)
    assert geo.intrinsics is not None
    K_save = geo.intrinsics.clone()
    geo.to(convention=(P.R, P.U, P.B))
    geo.to(convention=(P.R, P.D, P.F))
    torch.testing.assert_close(geo.intrinsics, K_save, atol=ATOL, rtol=0.0)


# ==== Round-trip consistency ====

@pytest.mark.parametrize("target", [
    (P.R, P.U, P.B),
    (P.F, P.R, P.D),
    (P.U, P.F, P.L),
    (P.L, P.U, P.B),
    (P.D, P.B, P.R),
])
def test_round_trip(target: CoordinateConvention):
    q = _z_rotation_quat(math.pi / 3)
    poses = torch.cat([torch.randn(B, S, 3), q.expand(B, S, -1)], dim=-1)
    geo = _make_geo(_POINTS, poses, _K)
    assert geo.points is not None and geo.poses is not None and geo.intrinsics is not None
    pts_save, poses_save, K_save = geo.points.clone(), geo.poses.clone(), geo.intrinsics.clone()

    geo.to(convention=target)
    geo.to(convention=(P.R, P.D, P.F))

    torch.testing.assert_close(geo.points, pts_save, atol=ATOL, rtol=0.0)
    torch.testing.assert_close(geo.poses, poses_save, atol=ATOL, rtol=0.0)
    torch.testing.assert_close(geo.intrinsics, K_save, atol=ATOL, rtol=0.0)


# ==== Edge cases ====

def test_same_convention_is_noop():
    poses = torch.cat([torch.zeros(B, S, 3), _identity_quat().expand(B, S, -1)], dim=-1)
    geo = _make_geo(_POINTS, poses, _K)
    result = geo.to(convention=(P.R, P.D, P.F))
    assert result is geo


def test_points_none():
    geo = SceneGeometry(
        pose_convention=(P.R, P.D, P.F),
        depths=torch.ones(1, 1, 1, 2, 2),
        depths_conf=None,
        points=None,
        points_conf=None,
        poses=torch.tensor([[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]]),
        intrinsics=_K[:, :1].clone(),
        infer_mask=None,
    )
    geo.to(convention=(P.R, P.U, P.B))
    assert geo.points is None
    assert geo.poses is not None
    torch.testing.assert_close(geo.poses[..., :3], torch.tensor([[[1.0, -2.0, -3.0]]]), atol=ATOL, rtol=0.0)


def test_invalid_target_in_to():
    poses = torch.cat([torch.zeros(B, S, 3), _identity_quat().expand(B, S, -1)], dim=-1)
    geo = _make_geo(_POINTS, poses, _K)
    with pytest.raises(ValueError, match="axis pairs"):
        geo.to(convention=(P.F, P.F, P.R))


def test_convention_updated_after_conversion():
    poses = torch.cat([torch.zeros(B, S, 3), _identity_quat().expand(B, S, -1)], dim=-1)
    geo = _make_geo(_POINTS, poses, _K)
    geo.to(convention=(P.L, P.U, P.B))
    assert geo.pose_convention == (P.L, P.U, P.B)
