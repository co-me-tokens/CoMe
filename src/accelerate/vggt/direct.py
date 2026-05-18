import torch
import jaxtyping as jt
from typing import cast, Any
from beartype import beartype

from ...thirdparty.vggt import VGGT
from ...thirdparty.vggt.layers.block import Block
from ...thirdparty.vggt.layers.attention import Attention
from ...thirdparty.vggt.layers.mlp import Mlp
from ...interface.token_merger import ITokenMerger
from ...interface.geometric_model import SceneGeometry, MultiViewInput
from ...utility.diagnostic import Diagnostics

from ..common.patch import patch_torch_forward

from .module import JaggedAttention_w_Bias, JaggedAttention


def direct_accelerate(model: VGGT, token_merger: ITokenMerger, encoder_layer: int, use_attn_bias_correction: bool = True) -> VGGT:
    """
    Given a VGGT model and the corresponding token merger, inject the token merging acceleration.
    """
    blocks       = cast(torch.nn.ModuleList, model.aggregator.patch_embed.blocks)
    frame_blocks = model.aggregator.frame_blocks
    globe_blocks = model.aggregator.global_blocks
    
    start_point = cast(Block, blocks[encoder_layer])
    accelerated_encode_blks: list[Block] = [cast(Block, blocks[i]) for i in range(encoder_layer + 1, len(blocks))] 
    accelerated_frames_blks: list[Block] = [cast(Block, frame_blocks[i]) for i in range(len(frame_blocks))]
    accelerated_global_blks: list[Block] = [cast(Block, globe_blocks[i]) for i in range(len(globe_blocks))]
    
    # Internal States maintained by the accelerate closure (mutable internally)
    state_inference_shape_bschw: tuple[int, int, int, int, int] | None = None
    state_context: Any | None = None

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
    #       and build the token merge mask.
    def hook_on_vggt_accel_start(module, args, output):
        assert state_inference_shape_bschw is not None
        nonlocal state_context
        
        start_index = model.aggregator.patch_start_idx
        image_input: jt.Float[torch.Tensor, "B*S 5+tH*tW C"] = output
        patch_input: jt.Float[torch.Tensor, "B S C H W"] = image_input[:, start_index:].permute(0, 2, 1).reshape(
            *state_inference_shape_bschw
        )
        state_context = token_merger.build_ctx(patch_input)
        
        Diagnostics.log(f"token_merger ({type(token_merger).__name__}) context built.")
    
    start_point.register_forward_hook(hook_on_vggt_accel_start)
    
    # NOTE: 3) For every remaining blocks in the encoder & frame/global block...
    #       run token merging before the attention block, run JaggedAttention, and then split back 
    #       to normal tokens.
    def replace_vggt_block_attention_inference(module: Attention, *, is_global: bool, run_dtype: JaggedAttention.SupportType, name: str):
        attn_dtype: torch.dtype
        match run_dtype:
            case "bf16": attn_dtype = torch.bfloat16
            case "fp16": attn_dtype = torch.float16
            case _: raise ValueError(f"Unsupported dtype for the attention patch - expect fp16/bf16, get {run_dtype}")
        
        if use_attn_bias_correction:
            attn_op = JaggedAttention_w_Bias(module, run_dtype=run_dtype)
        else:
            attn_op = JaggedAttention(module, run_dtype=run_dtype)
        
        def impl_factory(original_forward):
            # Higher order function can be dizzy to track ... but here is the function
            # we used to replace the block.attn.forward(...):
            @jt.jaxtyped(typechecker=beartype)
            def impl_forward(
                self: Attention,
                x   : jt.Float[torch.Tensor, "B N C"],
                pos : jt.Int[torch.Tensor, "B N 2"] | None = None
            ) -> jt.Float[torch.Tensor, "B N C"]:
                assert state_context is not None
                assert state_inference_shape_bschw is not None
                
                B, S, C, _, _ = state_inference_shape_bschw
                input_shape   = x.shape
                input_dtype   = x.dtype
                
                x = x.reshape(B, S, -1, C)
                merged_x, reverse_idx = token_merger.merge(state_context, x.to(dtype=attn_dtype))
                
                if pos is not None:
                    pos = pos.reshape(B, S, -1, 2)
                    merged_pos, _ = token_merger.merge(state_context, pos.to(dtype=attn_dtype), stub=reverse_idx)
                    pos           = merged_pos.tokens.long()
                
                # NOTE: In the global case, we need to coalese the samples in same batch index
                #       into a single sample. We did this via manipulating the offset field of
                #       tokens.
                if is_global:
                    global_index_selector = torch.arange(start=0, end=B*S+1, step=S, device=merged_x.offset.device, dtype=torch.long)
                    merged_x.offset = merged_x.offset[global_index_selector]
                
                if Diagnostics.is_active:
                    sample_sizes = (merged_x.offset[1:] - merged_x.offset[:-1]).tolist()
                    Diagnostics.log(f"{name} attn - infer w/ {sample_sizes=}")
                
                merged_x = attn_op.forward(merged_x, pos=pos)
                
                x = token_merger.split(state_context, reverse_idx, merged_x)
                x = x.view(*input_shape).to(dtype=input_dtype)
                return x
            return impl_forward
        return impl_factory
    
    for idx, block in enumerate(accelerated_encode_blks):
        idx  = encoder_layer + idx
        attn = cast(Attention, block.attn)
        patch_torch_forward(
            attn,
            replace_vggt_block_attention_inference(attn, is_global=False, run_dtype="bf16", name=f"Encode Blk {idx:02d}")
        )
    
    for idx, block in enumerate(accelerated_frames_blks):
        attn = cast(Attention, block.attn)
        patch_torch_forward(
            attn,
            replace_vggt_block_attention_inference(attn, is_global=False, run_dtype="bf16", name=f"Frame  Blk {idx:02d}")
        )
    
    for idx, block in enumerate(accelerated_global_blks):
        attn = cast(Attention, block.attn)
        patch_torch_forward(
            attn,
            replace_vggt_block_attention_inference(attn, is_global=True , run_dtype="bf16", name=f"Global Blk {idx:02d}")
        )

    # NOTE: 4) For every remaining blocks in the encoder & frame/global block...
    #       run token merging before the MLP layer, and then split back to normal tokens
    def replace_vggt_block_mlp_inference(original_forward):
        
        @jt.jaxtyped(typechecker=beartype)
        def impl_forward(self: Mlp, x: jt.Float[torch.Tensor, "B N C"]) -> jt.Float[torch.Tensor, "B N C"]:
            assert state_context is not None
            assert state_inference_shape_bschw is not None
            
            B, S, C, _, _ = state_inference_shape_bschw
            input_shape   = x.shape
            
            x = x.reshape(B, S, -1, C)
            merged_x, reverse_idx = token_merger.merge(state_context, x)
            
            merged_x.tokens = original_forward(merged_x.tokens)
            
            x = token_merger.split(state_context, reverse_idx, merged_x)
            x = x.view(*input_shape)
            return x
        return impl_forward
    
    for block in accelerated_encode_blks:
        mlp = cast(Mlp, block.mlp)
        patch_torch_forward(mlp, replace_vggt_block_mlp_inference)
    
    for block in accelerated_frames_blks:
        mlp = cast(Mlp, block.mlp)
        patch_torch_forward(mlp, replace_vggt_block_mlp_inference)
    
    for block in accelerated_global_blks:
        mlp = cast(Mlp, block.mlp)
        patch_torch_forward(mlp, replace_vggt_block_mlp_inference)
    
    # NOTE: 5) After the entire inference process, we need to write the mask into result
    #       so that the caller can receive reliable confidence masking on output
    def hook_write_inference_mask(module: VGGT, args: tuple[MultiViewInput], output: SceneGeometry) -> SceneGeometry:
        infer_mask = token_merger.build_infer_mask(state_context)
        infer_mask = infer_mask.unsqueeze(dim=2).repeat_interleave(repeats=14, dim=-1).repeat_interleave(repeats=14, dim=-2)
        
        # Set inference_mask - we are not responsible for merged regions' output quality!
        output.infer_mask = infer_mask
        
        return output
        
    model.register_forward_hook(hook_write_inference_mask)
    return model
