"""Visual Place Recognition model (inference-only, no PyTorch Lightning dependency)."""

import torch.nn as nn

from .models_salad import helper


class VPRModel(nn.Module):
    """Visual Place Recognition model using a backbone encoder and an aggregator."""

    def __init__(
        self,
        backbone_arch='resnet50',
        backbone_config={},
        agg_arch='ConvAP',
        agg_config={},
    ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
