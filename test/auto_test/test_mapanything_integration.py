"""MapAnything package integration tests."""

from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn

from src.interface.geometric_model import MultiViewInput, PoseConvention, TensorProbe
from src.thirdparty import ma
from src.thirdparty.ma import MapAnything, get_MA
from src.thirdparty.ma.mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_model


def _build_bare_mapanything() -> MapAnything:
    model = MapAnything.__new__(MapAnything)
    nn.Module.__init__(model)
    return model


def test_mapanything_import_surface():
    assert callable(get_MA)
    assert MapAnything.__name__ == "MapAnything"
    assert callable(initialize_mapanything_model)


def test_mapanything_package_uses_relative_self_imports_only():
    package_root = Path("/workspace/src/thirdparty/ma/mapanything")
    offenders: list[str] = []

    for path in sorted(package_root.rglob("*.py")):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("from mapanything.") or stripped.startswith("import mapanything"):
                offenders.append(f"{path}:{line_number}:{stripped}")

    assert not offenders, "\n".join(offenders)


def test_get_ma_uses_default_hf_repo(monkeypatch):
    sentinel = object()

    class DummyMapAnything:
        called_repo_id: str | None = None

        @classmethod
        def from_pretrained(cls, repo_id: str):
            cls.called_repo_id = repo_id
            return sentinel

    monkeypatch.setattr(ma, "_get_mapanything_class", lambda: DummyMapAnything)

    model = get_MA()

    assert model is sentinel
    assert DummyMapAnything.called_repo_id == ma.DEFAULT_MAPANYTHING_REPO_ID


def test_mapanything_forward_multiview_input_returns_scene_geometry(monkeypatch):
    batch_size = 1
    num_views = 2
    height = 2
    width = 2

    model = _build_bare_mapanything()
    model.encoder = SimpleNamespace(data_norm_type="dinov2", patch_size=1)

    images = torch.rand(batch_size, num_views, 3, height, width)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(batch_size, num_views, 1, 1)
    normalized = images + 0.25
    mv_input = MultiViewInput(images=images, intrinsics=intrinsics)

    pose_identity = torch.eye(4).unsqueeze(0)
    pose_translated = torch.eye(4).unsqueeze(0)
    pose_translated[:, :3, 3] = torch.tensor([[1.0, 2.0, 3.0]])

    infer_outputs = [
        {
            "depth_z": torch.full((batch_size, height, width, 1), 1.5),
            "pts3d": torch.full((batch_size, height, width, 3), 2.0),
            "camera_poses": pose_identity,
            "intrinsics": intrinsics[:, 0],
            "conf": torch.full((batch_size, height, width), 0.7),
            "mask": torch.tensor([[[[True], [False]], [[True], [True]]]], dtype=torch.bool),
        },
        {
            "depth_z": torch.full((batch_size, height, width, 1), 3.5),
            "pts3d": torch.full((batch_size, height, width, 3), 4.0),
            "camera_poses": pose_translated,
            "intrinsics": intrinsics[:, 1],
            "conf": torch.full((batch_size, height, width), 0.9),
            "mask": torch.tensor([[[[False], [False]], [[True], [True]]]], dtype=torch.bool),
        },
    ]

    def fake_normalize(self, input: MultiViewInput) -> torch.Tensor:
        assert input is mv_input
        return normalized

    def fake_infer(self, views, memory_efficient_inference=True, minibatch_size=None, **kwargs):
        assert memory_efficient_inference is False
        assert minibatch_size is None
        assert len(views) == num_views
        assert torch.equal(views[0]["img"], normalized[:, 0])
        assert torch.equal(views[1]["img"], normalized[:, 1])
        assert views[0]["data_norm_type"] == ["dinov2"]
        assert views[1]["data_norm_type"] == ["dinov2"]
        assert torch.equal(views[0]["intrinsics"], intrinsics[:, 0])
        assert torch.equal(views[1]["intrinsics"], intrinsics[:, 1])
        return infer_outputs

    monkeypatch.setattr(model, "_normalize_multiview_input", MethodType(fake_normalize, model))
    monkeypatch.setattr(model, "infer", MethodType(fake_infer, model))

    scene = model(mv_input)

    assert scene.pose_convention == (PoseConvention.R, PoseConvention.D, PoseConvention.F)
    assert scene.depths is not None
    assert scene.points is not None
    assert scene.poses is not None
    assert scene.intrinsics is not None
    assert scene.depths_conf is not None
    assert scene.points_conf is not None
    assert scene.infer_mask is not None

    assert scene.depths.shape == (batch_size, num_views, 1, height, width)
    assert scene.points.shape == (batch_size, num_views, 3, height, width)
    assert scene.poses.shape == (batch_size, num_views, 7)
    assert scene.intrinsics.shape == (batch_size, num_views, 3, 3)
    assert scene.infer_mask.shape == (batch_size, num_views, 1, height, width)

    torch.testing.assert_close(scene.depths[0, 0, 0], torch.full((height, width), 1.5))
    torch.testing.assert_close(scene.depths[0, 1, 0], torch.full((height, width), 3.5))
    torch.testing.assert_close(scene.points[0, 0], torch.full((3, height, width), 2.0))
    torch.testing.assert_close(scene.points[0, 1], torch.full((3, height, width), 4.0))
    torch.testing.assert_close(scene.depths_conf[0, 0, 0], torch.full((height, width), 0.7))
    torch.testing.assert_close(scene.depths_conf[0, 1, 0], torch.full((height, width), 0.9))
    torch.testing.assert_close(scene.points_conf, scene.depths_conf)
    torch.testing.assert_close(scene.poses[0, 0, :3], torch.zeros(3))
    torch.testing.assert_close(scene.poses[0, 0, 3:], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(scene.poses[0, 1, :3], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(scene.poses[0, 1, 3:], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(scene.intrinsics, intrinsics)
    torch.testing.assert_close(
        scene.infer_mask[0, 0, 0],
        torch.tensor([[False, True], [False, False]], dtype=torch.bool),
    )
    torch.testing.assert_close(
        scene.infer_mask[0, 1, 0],
        torch.tensor([[True, True], [False, False]], dtype=torch.bool),
    )


def test_mapanything_probe_support_uses_encoder_prefix_tokens():
    model = _build_bare_mapanything()
    blocks = nn.ModuleList([nn.Identity()])
    model.encoder = SimpleNamespace(
        pretrained=SimpleNamespace(
            num_register_tokens=2,
            blocks=blocks,
        )
    )

    probe = TensorProbe()
    model.inject_probe(0, probe)
    tokens = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    _ = blocks[0](tokens)

    assert model.patch_start_index == 3
    assert probe.data is not None
    torch.testing.assert_close(probe.data, tokens[:, 3:])


def test_mapanything_hydra_model_config_targets_root_loader():
    config_text = Path("/workspace/config/model/ma.yaml").read_text().strip()
    assert config_text == "_target_: src.thirdparty.ma.get_MA"
