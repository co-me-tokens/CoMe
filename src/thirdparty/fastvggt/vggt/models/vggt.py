# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import pypose as pp
from typing import cast
from huggingface_hub import PyTorchModelHubMixin

from .....interface.geometric_model import MultiViewInput, SceneGeometry, PoseConvention, TensorProbe

from .aggregator import Aggregator
from ..heads.camera_head import CameraHead
from ..heads.dpt_head import DPTHead
from ..utils.pose_enc import pose_encoding_to_extri_intri


class FastVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        merging: int = 0,
        merge_ratio: float = 0.9,
    ):
        super().__init__()
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            merging=merging,
            merge_ratio=merge_ratio,
        )

    @property
    def patch_start_index(self) -> int:
        return self.aggregator.patch_start_idx

    def update_patch_dimensions(self, patch_width: int, patch_height: int) -> None:
        def update_attention_in_module(module: nn.Module) -> None:
            for _name, child in module.named_children():
                update_attention_in_module(child)
                if hasattr(child, "patch_width") and hasattr(child, "patch_height"):
                    child.patch_width = patch_width
                    child.patch_height = patch_height

        update_attention_in_module(self.aggregator)

    def forward(self, input: "MultiViewInput") -> "SceneGeometry":
        images = input.images
        _, _, _, H, W = images.shape
        self.update_patch_dimensions(
            W // self.aggregator.patch_size,
            H // self.aggregator.patch_size,
        )
        aggregated_tokens, patch_start_idx = self.aggregator(images)

        with torch.autocast(device_type="cuda", enabled=False):
            pose_enc_list = self.camera_head(aggregated_tokens)
            pose_enc = pose_enc_list[-1]
            _, intrinsics = pose_encoding_to_extri_intri(
                pose_enc, image_size_hw=images.shape[-2:], build_intrinsics=True
            )
            assert intrinsics is not None

            depth, depth_conf = self.depth_head(
                aggregated_tokens, images=images, patch_start_idx=patch_start_idx
            )

            pts3d, pts3d_conf = self.point_head(
                aggregated_tokens, images=images, patch_start_idx=patch_start_idx
            )
            points = pts3d.permute(0, 1, 4, 2, 3).contiguous()
            points_conf = pts3d_conf.unsqueeze(2).contiguous()

        return SceneGeometry(
            pose_convention=(PoseConvention.R, PoseConvention.D, PoseConvention.F),
            depths=depth.permute(0, 1, 4, 2, 3).contiguous(),
            depths_conf=depth_conf.unsqueeze(2).contiguous(),
            points=points,
            points_conf=points_conf,
            poses=pp.SE3(pose_enc[..., :7]).Inv().tensor(),  # w2c -> c2w
            intrinsics=intrinsics,
            infer_mask=None,
        )

    def inject_probe(self, block_index: int, probe: TensorProbe) -> None:
        def hook_feature_tensor(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
            probe.data = output[:, self.aggregator.patch_start_idx:]

        blocks = cast(torch.nn.ModuleList, self.aggregator.patch_embed.blocks)
        blocks[block_index].register_forward_hook(hook_feature_tensor)
