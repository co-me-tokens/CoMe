# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Depth Anything 3 API module.

Provides model construction via config and forward inference.
Supports checkpoint loading from Hugging Face Hub via ``from_pretrained``.
"""

from __future__ import annotations

from typing import cast

import torch
import pypose as pp
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from ...interface.geometric_model import MultiViewInput, SceneGeometry, TensorProbe, PoseConvention
from .cfg import create_object, load_config
from .model.da3 import affine_inverse
from .registry import MODEL_REGISTRY
from .utils.geometry import as_homogeneous, unproject_depth

torch.backends.cudnn.benchmark = False

SAFETENSORS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


def _raw_dict_to_scene_geometry(raw: dict[str, torch.Tensor]) -> SceneGeometry:
    """Convert a raw DA3 output dict to a SceneGeometry instance."""
    depths = raw["depth"].unsqueeze(2)
    depths_conf = raw["depth_conf"].unsqueeze(2) if "depth_conf" in raw else None

    poses: torch.Tensor | None = None
    points: torch.Tensor | None = None
    intrinsics: torch.Tensor | None = raw.get("intrinsics", None)

    if "extrinsics" in raw:
        w2c = raw["extrinsics"]
        c2w = affine_inverse(w2c)
        R = c2w[..., :3, :3]
        t = c2w[..., :3, 3]
        with torch.autocast("cuda", enabled=False):
            q = pp.from_matrix(R, ltype=pp.SO3_type, check=False)
        poses = torch.cat([t, q.tensor()], dim=-1)

        if intrinsics is not None:
            c2w_homo = as_homogeneous(c2w)
            world_pts = unproject_depth(raw["depth"].unsqueeze(-1), intrinsics, c2w_homo)
            points = world_pts.permute(0, 1, 4, 2, 3).contiguous()

    return SceneGeometry(
        pose_convention=(PoseConvention.R, PoseConvention.D, PoseConvention.F),
        depths=depths,
        depths_conf=depths_conf,
        points=points,
        points_conf=depths_conf,
        poses=poses,
        intrinsics=intrinsics,
        infer_mask=None,
    )
    

class DepthAnything3(nn.Module, PyTorchModelHubMixin):
    """
    Depth Anything 3 main API class.

    Wraps the underlying ``DepthAnything3Net`` (or ``NestedDepthAnything3Net``)
    and provides mixed-precision forward inference.

    Usage::

        model = DepthAnything3.from_pretrained("depth-anything/DA3-Large")
        model = model.to("cuda").eval()
        output = model(images)  # (B, N, 3, H, W) -> dict
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    PATCH_SIZE = 14

    _commit_hash: str | None = None  # Set by mixin when loading from Hub

    def __init__(self, model_name: str = "da3-large", **kwargs):
        super().__init__()
        self.model_name = model_name

        self.config = load_config(MODEL_REGISTRY[self.model_name])
        self.model = create_object(self.config)
        self.model.eval()

    @property
    def patch_start_index(self) -> int:
        return cast(int, self.model.patch_start_index)

    def inject_probe(self, block_index: int, probe: TensorProbe) -> None:
        self.model.inject_probe(block_index, probe)

    @torch.inference_mode()
    def forward(
        self,
        image: torch.Tensor | MultiViewInput,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "first",
    ) -> dict[str, torch.Tensor] | SceneGeometry:
        """
        Forward pass through the model.

        Args:
            image: Either a tensor batch with shape ``(B, N, 3, H, W)`` that is already
                   ImageNet-normalised, or a ``MultiViewInput`` with images in ``[0, 1]``.
            extrinsics: Optional camera extrinsics with shape ``(B, N, 4, 4)``.
            intrinsics: Optional camera intrinsics with shape ``(B, N, 3, 3)``.
            export_feat_layers: Layer indices to return intermediate features for.
            infer_gs: Enable Gaussian Splatting branch.
            use_ray_pose: Use ray-based pose estimation instead of camera decoder.
            ref_view_strategy: Strategy for selecting reference view from multiple views.

        Returns:
            Dictionary containing model predictions. Read fields with bracket notation,
            e.g. ``result["depth"]``, ``result["extrinsics"]`` (not attribute access).
        """
        if isinstance(image, MultiViewInput):
            normalized_images = self._normalize_multiview_input(image)
            raw = self._forward_raw(
                normalized_images,
                extrinsics=extrinsics,
                intrinsics=image.intrinsics,
                export_feat_layers=export_feat_layers,
                infer_gs=infer_gs,
                use_ray_pose=use_ray_pose,
                ref_view_strategy=ref_view_strategy,
            )
            return _raw_dict_to_scene_geometry(raw)
        else:
            return self._forward_raw(
                image,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                export_feat_layers=export_feat_layers,
                infer_gs=infer_gs,
                use_ray_pose=use_ray_pose,
                ref_view_strategy=ref_view_strategy,
            )

    def _normalize_multiview_input(self, input: MultiViewInput) -> torch.Tensor:
        images = input.images
        if images.ndim != 5:
            raise ValueError(f"Expected MultiViewInput.images to have shape [B, S, 3, H, W], got {tuple(images.shape)}")
        if images.shape[2] != 3:
            raise ValueError(f"Expected RGB inputs with 3 channels, got {images.shape[2]}")
        if not torch.is_floating_point(images):
            raise TypeError(f"Expected floating-point images in [0, 1], got dtype {images.dtype}")
        if images.shape[-2] % self.PATCH_SIZE != 0 or images.shape[-1] % self.PATCH_SIZE != 0:
            raise ValueError(
                f"Expected image height/width divisible by {self.PATCH_SIZE}, got {tuple(images.shape[-2:])}"
            )

        image_min = float(images.min().item())
        image_max = float(images.max().item())
        if image_min < 0.0 or image_max > 1.0:
            raise ValueError(
                f"Expected unnormalized image values in [0, 1] before DA3 normalization, got range [{image_min}, {image_max}]"
            )

        if input.intrinsics is not None and input.intrinsics.shape[:2] != images.shape[:2]:
            raise ValueError(
                "Expected MultiViewInput.intrinsics to match image batch/view dimensions, "
                f"got images {tuple(images.shape[:2])} and intrinsics {tuple(input.intrinsics.shape[:2])}"
            )

        mean = images.new_tensor(self.IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = images.new_tensor(self.IMAGENET_STD).view(1, 1, 3, 1, 1)
        return (images - mean) / std

    def _forward_raw(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> dict[str, torch.Tensor]:
        feat_layers = [] if export_feat_layers is None else export_feat_layers
        autocast_enabled = image.device.type == "cuda"
        autocast_dtype = torch.bfloat16 if autocast_enabled and torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad():
            with torch.autocast(
                device_type=image.device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                return self.model(
                    image, extrinsics, intrinsics, feat_layers, infer_gs, use_ray_pose, ref_view_strategy
                )
