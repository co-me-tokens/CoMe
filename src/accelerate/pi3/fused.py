import torch
import torch.nn as nn
import jaxtyping as jt
from typing import cast, Any, overload

from ...thirdparty.pi3 import Pi3, Pi3X
from ...thirdparty.pi3.models.layers.block        import BlockRope
from ...thirdparty.pi3.models.dinov2.layers.block import Block

from ...interface.token_merger    import ITokenMerger
from ...interface.geometric_model import MultiViewInput, SceneGeometry
from ...utility.diagnostic        import Diagnostics

from ..common.patch import patch_torch_method
from .module        import JaggedBlock, JaggedBlock_w_Bias


def _patch_encoder_attn_for_jagged(block: Block) -> None:
    """Patch missing AttentionRope-compatible attributes onto DINOv2 encoder attention."""
    attn: Any = block.attn
    if not hasattr(attn, 'head_dim'):
        attn.head_dim = attn.qkv.in_features // attn.num_heads
    if not hasattr(attn, 'q_norm'):
        attn.q_norm = nn.Identity()
    if not hasattr(attn, 'k_norm'):
        attn.k_norm = nn.Identity()
    if not hasattr(attn, 'rope'):
        attn.rope = None


@overload
def fused_accelerate(model: Pi3, token_merger: ITokenMerger, encoder_layer: int, use_attn_bias_correction: bool) -> Pi3:
    ...
@overload
def fused_accelerate(model: Pi3X, token_merger: ITokenMerger, encoder_layer: int, use_attn_bias_correction: bool) -> Pi3X:
    ...


def fused_accelerate(model: Pi3 | Pi3X, token_merger: ITokenMerger, encoder_layer: int, use_attn_bias_correction: bool) -> Pi3 | Pi3X:
    """
    Given a Pi3 | Pi3X model and the corresponding token merger, inject the token merging acceleration.
    """
    if isinstance(model, Pi3X) and model.use_multimodal:
        raise NotImplementedError(
            "Fused acceleration for Pi3X with multimodal enabled is not supported. "
            "Call model.disable_multimodal() before fused_accelerate()."
        )

    block_fn      = JaggedBlock_w_Bias if use_attn_bias_correction else JaggedBlock
    encode_blocks = cast(torch.nn.ModuleList, model.encoder.blocks)
    decode_blocks = model.decoder

    start_point   = cast(Block, encode_blocks[encoder_layer])
    accelerated_encode_blks: list[Block] = [
        cast(Block, encode_blocks[i]) for i in range(encoder_layer + 1, len(encode_blocks))
    ]
    accelerated_decode_blks: list[BlockRope] = [
        cast(BlockRope, decode_blocks[i]) for i in range(len(decode_blocks))
    ]

    # Internal States maintained by the accelerate closure (mutable internally)
    state_inference_shape_bschw: tuple[int, int, int, int, int] | None = None
    state_context              : Any | None                            = None
    state_revidx               : torch.Tensor | None                   = None

    # ==== Step 1: Capture input shape ====
    def hook_on_pi3_inference_start(module, args):
        nonlocal state_inference_shape_bschw

        image_input   = args[0].images
        B, N, C, H, W = image_input.shape
        tH, tW = H // 14, W // 14
        tC     = model.encoder.num_features
        state_inference_shape_bschw = (B, N, tC, tH, tW)

        Diagnostics.log(f"start inference on shape {state_inference_shape_bschw}")

    model.register_forward_pre_hook(hook_on_pi3_inference_start)

    # ==== Step 2: Build merge context and merge at start_point ====
    def hook_on_pi3_accel_start(module, args, output) -> ITokenMerger.JaggedTokens:
        assert state_inference_shape_bschw is not None
        nonlocal state_context
        nonlocal state_revidx

        num_prefix = 1 + model.encoder.num_register_tokens
        image_input: jt.Float[torch.Tensor, "BN prefix_plus_hw C"] = output
        patch_input: jt.Float[torch.Tensor, "B N C tH tW"] = image_input[:, num_prefix:].permute(0, 2, 1).reshape(
            *state_inference_shape_bschw
        )
        state_context = token_merger.build_ctx(patch_input)

        Diagnostics.log(f"token_merger ({type(token_merger).__name__}) context built.")
        Diagnostics.log("fused acceleration mode: starting token merging immediately.")

        merged_input, state_revidx = token_merger.merge(state_context, image_input.unsqueeze(0))
        return merged_input

    start_point.register_forward_hook(hook_on_pi3_accel_start)

    # ==== Step 3: Patch encoder attention and replace encoder blocks ====
    for idx, block in enumerate(accelerated_encode_blks):
        _patch_encoder_attn_for_jagged(block)
        encode_blocks[encoder_layer + 1 + idx] = block_fn(block, "bf16")  # type: ignore[arg-type]

    # ==== Step 4: Split tokens at the end of encoder ====
    def hook_after_last_encoder_block(module, args, output: ITokenMerger.JaggedTokens) -> torch.Tensor:
        assert state_context is not None
        assert state_revidx  is not None
        return token_merger.split(state_context, state_revidx, output).squeeze(0)

    encode_blocks[-1].register_forward_hook(hook_after_last_encoder_block)

    # ==== Step 5: Replace decoder blocks ====
    for idx, block in enumerate(accelerated_decode_blks):
        decode_blocks[idx] = block_fn(block, "bf16")  # type: ignore[arg-type]

    # ==== Step 6: Patch decode to run jagged decoder with alternating offsets ====
    def patch_decode(original_decode):
        def patched_decode_impl(self: Pi3 | Pi3X, hidden, N, H, W, *args, **kwargs):
            assert state_context is not None

            if len(hidden.shape) == 4:
                B = hidden.shape[0]
            else:
                B = hidden.shape[0] // N

            hidden = hidden.reshape(B * N, -1, hidden.shape[-1])

            register_token = self.register_token.repeat(B, N, 1, 1).reshape(
                B * N, *self.register_token.shape[-2:]
            )
            hidden = torch.cat([register_token, hidden], dim=1)
            hw = hidden.shape[1]

            pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, hidden.device)
            if self.patch_start_idx > 0:
                pos = pos + 1
                pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
                pos = torch.cat([pos_special, pos], dim=1)

            original_pos = pos.reshape(B * N, hw, -1)

            merged_tokens, revidx = token_merger.merge(
                state_context, hidden.reshape(B, N, hw, -1)
            )
            merged_pos = token_merger.merge(
                state_context, pos.reshape(B, N, hw, -1).to(dtype=torch.bfloat16), stub=revidx
            )[0].tokens.long()
            original_offset = merged_tokens.offset

            cross_view_selector = torch.arange(
                start=0, end=B * N + 1, step=N, device=hidden.device, dtype=torch.long
            )

            final_output: list[torch.Tensor] = []
            num_decoder_blocks = len(self.decoder)

            for i in range(num_decoder_blocks):
                if i % 2 == 0:
                    merged_tokens.offset = original_offset
                else:
                    merged_tokens.offset = original_offset[cross_view_selector]

                merged_tokens = self.decoder[i](merged_tokens, pos=merged_pos)

                if i + 1 in [num_decoder_blocks - 1, num_decoder_blocks]:
                    merged_tokens.offset = original_offset
                    dense = token_merger.split(state_context, revidx, merged_tokens)
                    final_output.append(dense.reshape(B * N, hw, -1))

            return torch.cat([final_output[0], final_output[1]], dim=-1), original_pos

        return patched_decode_impl

    patch_torch_method(model, "decode", patch_decode)

    # ==== Step 7: Write inference mask into output ====
    def hook_write_inference_mask(module: Pi3 | Pi3X, args: tuple[MultiViewInput], output: SceneGeometry) -> SceneGeometry:
        infer_mask = token_merger.build_infer_mask(state_context)
        infer_mask = infer_mask.unsqueeze(dim=2).repeat_interleave(repeats=14, dim=-1).repeat_interleave(repeats=14, dim=-2)

        output.infer_mask = infer_mask
        return output

    model.register_forward_hook(hook_write_inference_mask)
    return model
