import torch
import typing as tp
import torch.nn as nn
import jaxtyping as jt
from typing import cast, Any, overload

from ...thirdparty.da3 import DepthAnything3
from ...thirdparty.da3.model.da3 import DepthAnything3Net, NestedDepthAnything3Net
from ...thirdparty.da3.model.dinov2.dinov2 import DinoV2
from ...thirdparty.da3.model.dinov2.vision_transformer  import DinoVisionTransformer
from ...thirdparty.da3.model.dinov2.layers.block        import Block

from ...interface.token_merger      import ITokenMerger
from ...interface.geometric_model   import MultiViewInput, SceneGeometry
from ...utility.diagnostic          import Diagnostics

from ..common.patch import patch_torch_method
from .module        import JaggedBlock, JaggedBlock_w_Bias


def fused_accelerate(model: DepthAnything3, token_merger: ITokenMerger, encoder_layer: int, use_attn_bias_correction: bool) -> DepthAnything3:
    block_fn = JaggedBlock_w_Bias if use_attn_bias_correction else JaggedBlock
    
    network  = tp.cast(DepthAnything3Net | NestedDepthAnything3Net, model.model)
    backbone = tp.cast(DinoV2, network.backbone)
    encoder  = tp.cast(DinoVisionTransformer, backbone.pretrained)
    
    accelerated_blks: list[tuple[int, Block]] = [
        (i, cast(Block, encoder.blocks[i]))
        for i in range(encoder_layer + 1, len(encoder.blocks))
    ]
    
    # Internal state maintained by the accelerate closure (mutable internally)
    state_inference_shape_bschw: tuple[int, int, int, int, int] | None = None
    state_context: tp.Any | None = None
    state_revidx : torch.Tensor | None = None
    
    # NOTE: 1) Register a hook such that after start_point run the token_merger.build_ctx(...)
    #       with the resulting token features (since it's DINO encoder it should have correct shape)
    def hook_on_da3_inference_start(module, args):
        nonlocal state_inference_shape_bschw
        
        image_input = args[0]
        B, S, C, H, W = image_input.shape
        
        tH, tW = H // 14, W // 14
        tC     = encoder.num_features
        state_inference_shape_bschw = (B, S, tC, tH, tW)
        
        Diagnostics.log(f"start inference on shape {state_inference_shape_bschw}")

    network.register_forward_pre_hook(hook_on_da3_inference_start)

    # NOTE: 2) Systematically replace the blocks in the encoder.
    for idx, block in accelerated_blks:
        encoder.blocks[idx] = block_fn(block, "bf16")
    
    # NOTE: 3) rewrite the forward of Backbone
    def patch_backbone_forward(original_forward):
        def patch_backbone_foward_impl(
            self: DinoVisionTransformer,
            x   : jt.Float[torch.Tensor, "B S 3 H W"],
            n   : list[int] = [1],
            export_feat_layers: list[int] = [],
            **kwargs
        ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[torch.Tensor]]:
            nonlocal state_context
            nonlocal state_revidx
            
            B, S, _, H, W = x.shape
            x             = self.prepare_tokens_with_masks(x)
            
            output: list[tuple[torch.Tensor, torch.Tensor]] = []
            pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)
            
            assert self.rope is not None
            assert pos is not None
            assert pos_nodiff is not None
            assert self.alt_start != -1
            assert self.cat_token
            assert kwargs.get("cam_token", None) is None
            assert self.rope_start == self.alt_start
            assert len(export_feat_layers) == 0
            assert encoder_layer < self.alt_start
            assert state_inference_shape_bschw is not None
            
            enum_blocks  = [(i, blk) for i, blk in enumerate(self.blocks)]
            B, S, C, H, W = state_inference_shape_bschw
            
            # Run first half of the encoder.
            x = x.reshape(B*S, H*W + 1, C)
            for idx, block in enum_blocks[0:encoder_layer + 1]:
                Diagnostics.log(f"Inference iter {idx:02d}, Block type={type(block).__name__}, x type={type(x).__name__}")
                x     = block(x, pos=None)
            
            
            # Run token merging program
            start_index = model.patch_start_index
            image_input: jt.Float[torch.Tensor, "B*S 1+H*W C"] = x
            patch_input: jt.Float[torch.Tensor, "B S C H W"] = image_input[:, start_index:].permute(0, 2, 1).reshape(B, S, C, H, W)
            state_context = token_merger.build_ctx(patch_input)
            Diagnostics.log(f"token_merger ({type(token_merger).__name__}) context built.")
            
            # Token Merging Start right now!
            Diagnostics.log("fused acceleration mode: starting token merging immediately.")
            merged_x    , state_revidx = token_merger.merge(state_context, image_input.unsqueeze(0))
            merged_lpos , _            = token_merger.merge(state_context, pos.bfloat16(), stub=state_revidx)
            merged_gpos , _            = token_merger.merge(state_context, pos_nodiff.bfloat16(), stub=state_revidx)
            merged_lpos , merged_gpos  = merged_lpos.tokens.long(), merged_gpos.tokens.long()
            
            # Run second half of the encoder
            for idx, block in enum_blocks[encoder_layer + 1:self.alt_start]:
                Diagnostics.log(f"Inference iter {idx:02d}, Block type={type(block).__name__}, x type={type(x).__name__}")
                merged_x = block(merged_x, pos=None)
            
            # Start running alternative attention (w/ jagged tokens)
            ref_token = self.camera_token[:, :1].expand(B, -1, -1)
            src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
            cam_token = torch.cat([ref_token, src_token], dim=1)
            merged_x.tokens[:, merged_x.offset[:-1]] = cam_token
            
            frame_offset  = merged_x.offset
            global_offset = merged_x.offset[torch.arange(start=0, end=B*S+1, step=S, device=merged_x.tokens.device, dtype=torch.long)]

            local_x = None
            for idx, block in enum_blocks[self.alt_start:]:
                if idx % 2 == 1:
                    merged_x.offset = global_offset
                    merged_x = block(merged_x, pos=merged_gpos)
                    merged_x.offset = frame_offset
                else:
                    merged_x = block(merged_x, pos=merged_lpos)
                    local_x  = merged_x
            
                if idx in n:
                    assert local_x is not None
                    sp_local_x = token_merger.split(state_context, state_revidx, local_x)
                    sp_x       = token_merger.split(state_context, state_revidx, merged_x)
                    out_x      = torch.cat([sp_local_x, sp_x], dim=-1)
                    
                    output.append((out_x[:, :, 0], out_x))
            return output, []
        return patch_backbone_foward_impl
    patch_torch_method(encoder, "_get_intermediate_layers_not_chunked", patch_backbone_forward)

    # NOTE: 4) write proper inference mask at the end of inference
    def hook_write_inference_mask(module, args: tuple[MultiViewInput], output: SceneGeometry) -> SceneGeometry:
        infer_mask = token_merger.build_infer_mask(state_context)
        infer_mask = infer_mask.unsqueeze(dim=2).repeat_interleave(repeats=14, dim=-1).repeat_interleave(repeats=14, dim=-2)
        
        # Set inference_mask - we are not responsible for merged regions' output quality!
        output.infer_mask = infer_mask
        
        return output
    model.register_forward_hook(hook_write_inference_mask)

    return model
