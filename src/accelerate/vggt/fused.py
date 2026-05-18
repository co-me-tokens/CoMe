import torch
import jaxtyping as jt
from typing import cast, Any


from ...thirdparty.vggt                           import VGGT
from ...thirdparty.vggt.layers.block              import Block
from ...thirdparty.vggt.models.aggregator         import Aggregator, slice_expand_and_flatten

from ...interface.token_merger    import ITokenMerger
from ...interface.geometric_model import MultiViewInput, SceneGeometry
from ...utility.diagnostic        import Diagnostics

from ..common.patch import patch_torch_forward
from .module import JaggedBlock, JaggedBlock_w_Bias


def fused_accelerate(model: VGGT, token_merger: ITokenMerger, encoder_layer: int, use_attn_bias_correction: bool = True) -> VGGT:
    """
    Given a VGGT model and the corresponding token merger, inject the token merging acceleration.
    """
    block_fn     = JaggedBlock_w_Bias if use_attn_bias_correction else JaggedBlock

    encode_blocks = cast(torch.nn.ModuleList, model.aggregator.patch_embed.blocks)
    frame_blocks  = model.aggregator.frame_blocks
    globe_blocks  = model.aggregator.global_blocks
    
    start_point   = cast(Block, encode_blocks[encoder_layer])
    accelerated_encode_blks: list[Block] = [cast(Block, encode_blocks[i]) for i in range(encoder_layer + 1, len(encode_blocks))]
    accelerated_frames_blks: list[Block] = [cast(Block, frame_blocks[i]) for i in range(len(frame_blocks))]
    accelerated_global_blks: list[Block] = [cast(Block, globe_blocks[i]) for i in range(len(globe_blocks))]
    
    # Internal States maintained by the accelerate closure (mutable internally)
    state_inference_shape_bschw: tuple[int, int, int, int, int] | None = None
    state_context: Any | None = None
    state_revidx : torch.Tensor | None = None

    # NOTE: 1) Register a hook such that after the start_point run the token_merger.build_ctx(...)
    #       with the resulting token features (since it's DINO encoder it should have correct shape)
    def hook_on_vggt_inference_start(module, args):
        nonlocal state_inference_shape_bschw
        
        image_input   = args[0]
        B, S, C, H, W = image_input.shape
        tH, tW = H // 14, W // 14
        tC     = module.patch_embed.num_features
        state_inference_shape_bschw = (B, S, tC, tH, tW)
        
        Diagnostics.log(f"start inference on shape {state_inference_shape_bschw}")
    
    model.aggregator.register_forward_pre_hook(hook_on_vggt_inference_start)

    # NOTE: 2) Register hook at the 'start_point' so that the network perdicts coarse confidence estimation
    #       and build the token merge mask. In *fused* version we will merge the token immediately.
    def hook_on_vggt_accel_start(module, args, output) -> ITokenMerger.JaggedTokens:
        assert state_inference_shape_bschw is not None
        nonlocal state_context
        nonlocal state_revidx
        
        start_index = model.aggregator.patch_start_idx
        image_input: jt.Float[torch.Tensor, "B*S 5+tH*tW C"] = output
        patch_input: jt.Float[torch.Tensor, "B S C H W"] = image_input[:, start_index:].permute(0, 2, 1).reshape(
            *state_inference_shape_bschw
        )
        state_context = token_merger.build_ctx(patch_input)
        
        Diagnostics.log(f"token_merger ({type(token_merger).__name__}) context built.")
        
        # Token Merging Start right now!
        Diagnostics.log("fused acceleration mode: starting token merging immediately.")
        merged_input, state_revidx = token_merger.merge(state_context, image_input.unsqueeze(0))
        return merged_input
    
    start_point.register_forward_hook(hook_on_vggt_accel_start)

    # WARNING: Things start to be different from direct_accelerate from here...
    # NOTE: 3) Systematically replace the blocks in the encoder.
    for idx, block in enumerate(accelerated_encode_blks):
        encode_blocks[encoder_layer + 1 + idx] = block_fn(block, "bf16")

    # NOTE: 4) split the tokens at the end of encoder
    def hook_after_last_encoder_block(module, args, output: ITokenMerger.JaggedTokens) -> torch.Tensor:
        assert state_context is not None
        assert state_revidx  is not None
        return token_merger.split(state_context, state_revidx, output).squeeze(0)
    
    encode_blocks[-1].register_forward_hook(hook_after_last_encoder_block)
    
    # NOTE: 5) rewrite the forward of Aggregator
    for idx, block in enumerate(accelerated_frames_blks):
        frame_blocks[idx] = block_fn(block, "bf16")
    
    for idx, block in enumerate(accelerated_global_blks):
        globe_blocks[idx] = block_fn(block, "bf16")
    
    def patch_aggregator_forward(original_forward):
        def patch_aggregator_forward_impl(self: Aggregator, images: torch.Tensor):
            B, S, C_in, H, W = images.shape
            assert C_in == 3, f"Expect 3 input channels, got {C_in}"
            
            # Normalize images and reshape for patch embed
            images = (images - self._resnet_mean) / self._resnet_std
            
            # Reshape to [B*S, C, H, W] for patch embedding
            images = images.view(B * S, C_in, H, W)
            patch_tokens = self.patch_embed(images)["x_norm_patchtokens"]
                        
            # Expand camera and register tokens to match batch size and sequence length
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
            register_token = slice_expand_and_flatten(self.register_token, B, S)

            # Concatenate special tokens with patch tokens
            tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

            if self.patch_start_idx > 0:
                # do not use position embedding for special tokens (camera and register tokens)
                # so set pos to 0 for the special tokens
                pos = pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
                pos = torch.cat([pos_special, pos], dim=1)

            # update P because we added special tokens
            _, P, C = tokens.shape

            tokens = tokens.reshape(B, S, -1, C)
            pos    = pos.reshape(B, S, -1, 2).to(dtype=torch.bfloat16)
            merged_tokens, revidx = token_merger.merge(state_context, tokens)
            merged_pos            = token_merger.merge(state_context, pos, stub=revidx)[0].tokens.long()
            original_offset       = merged_tokens.offset


            block_idx, frame_idx, globe_idx = 0, 0, 0
            output: dict[int, torch.Tensor] = dict()
            frame_intermediate, globe_intermediate = [], []
            global_selector = torch.arange(start=0, end=B*S+1, step=S, device=tokens.device, dtype=torch.long)
            
            for _ in range(self.aa_block_num):
                for attn_type in self.aa_order:
                    match attn_type:
                        case "frame":
                            merged_tokens.offset = original_offset
                            for _ in range(self.aa_block_size):
                                merged_tokens = self.frame_blocks[frame_idx](merged_tokens, pos=merged_pos)
                                frame_intermediate.append(merged_tokens)
                                frame_idx += 1
                        
                        case "global":
                            merged_tokens.offset = original_offset[global_selector]
                            for _ in range(self.aa_block_size):
                                merged_tokens = self.global_blocks[globe_idx](merged_tokens, pos=merged_pos)
                                globe_intermediate.append(merged_tokens)
                                globe_idx += 1
                        
                        case _: raise ValueError(f"Unknown attention type: {attn_type}")
                
                for i in range(self.aa_block_size):
                    if self.retain_layers is None or block_idx in self.retain_layers:
                        frame_intermediate[i].offset = original_offset
                        globe_intermediate[i].offset = original_offset
                        
                        orig_frame_tokens = token_merger.split(state_context, revidx, frame_intermediate[i])
                        orig_globe_tokens = token_merger.split(state_context, revidx, globe_intermediate[i])
                        
                        output[block_idx] = torch.cat((orig_frame_tokens, orig_globe_tokens), dim=-1)
                    
                    block_idx += 1
                
                frame_intermediate.clear()
                globe_intermediate.clear()

            return output, self.patch_start_idx
        return patch_aggregator_forward_impl

    patch_torch_forward(model.aggregator, patch_aggregator_forward)
    
    # NOTE: 6) After the entire inference process, we need to write the mask into result
    #       so that the caller can receive reliable confidence masking on output
    def hook_write_inference_mask(module: VGGT, args: tuple[MultiViewInput], output: SceneGeometry) -> SceneGeometry:
        infer_mask = token_merger.build_infer_mask(state_context)
        infer_mask = infer_mask.unsqueeze(dim=2).repeat_interleave(repeats=14, dim=-1).repeat_interleave(repeats=14, dim=-2)
        
        # Set inference_mask - we are not responsible for merged regions' output quality!
        output.infer_mask = infer_mask
        
        return output
        
    model.register_forward_hook(hook_write_inference_mask)
    return model
