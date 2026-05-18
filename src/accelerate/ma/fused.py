import torch
import torch.cuda.nvtx as nvtx
import jaxtyping as jt
from typing import cast, Any

from ...thirdparty.ma import MapAnything
from ...thirdparty.ma.mapanything.utils.device import get_amp_dtype, get_autocast_device_type
from uniception.models.utils.transformer_blocks import SelfAttentionBlock
from uniception.models.info_sharing.alternating_attention_transformer import MultiViewAlternatingAttentionTransformerIFR
from uniception.models.info_sharing.base                              import MultiViewTransformerInput, MultiViewTransformerOutput
from uniception.models.utils.intermediate_feature_return              import feature_take_indices

from ...interface.token_merger      import ITokenMerger
from ...interface.geometric_model   import MultiViewInput, SceneGeometry
from ...utility.diagnostic          import Diagnostics

from ..common.patch import patch_torch_forward, patch_torch_method
from .module        import (
    DINO_JaggedBlock, DINO_JaggedBlock_w_Bias, 
    Self_JaggedBlock, Self_JaggedBlock_w_Bias
)


def fused_accelerate(model: MapAnything, token_merger: ITokenMerger, encoder_layer: int, use_attn_bias_correction: bool = True) -> MapAnything:
    """
    Given a MapAnything model and the corresponding token merger, inject the token merging acceleration.
    """
    dino_block_fn = DINO_JaggedBlock_w_Bias if use_attn_bias_correction else DINO_JaggedBlock
    self_block_fn = Self_JaggedBlock_w_Bias if use_attn_bias_correction else Self_JaggedBlock
    
    encode_blocks = cast(torch.nn.ModuleList, model.encoder.model.blocks)                   #type: ignore
    selfat_blocks = cast(torch.nn.ModuleList, model.info_sharing.self_attention_blocks)    #type: ignore
    
    start_point   = encode_blocks[encoder_layer]
    accelerated_encode_blocks = encode_blocks[encoder_layer+1:]
    accelerated_selfat_blocks = selfat_blocks[:]

    # Internal States maintained by the accelerate closure (mutable internally)
    state_inference_shape_bschw: tuple[int, int, int, int, int] | None = None
    state_context              : Any | None = None
    state_revidx               : torch.Tensor | None = None
    
    # NOTE: 1) Register a hook to collect the metadata of input.
    def hook_on_ma_inference_start(module, args):
        nonlocal state_inference_shape_bschw
        
        image_input = args[0]
        B, S, C, H, W = image_input.images.shape
        tH, tW = H // 14, W // 14
        tC     = module.encoder.model.num_features
        state_inference_shape_bschw = (B, S, tC, tH, tW)
        
        Diagnostics.log(f"start inference on shape {state_inference_shape_bschw}")
    
    model.register_forward_pre_hook(hook_on_ma_inference_start)
    
    
    # NOTE: 2) Register hook at the 'start_point' so that the network predicts coarse confidence estimation
    #       and build the token merge mask. In *fused* version we will merge the token immediately.
    def hook_on_ma_accel_start(module, args, output) -> ITokenMerger.JaggedTokens:
        assert state_inference_shape_bschw is not None
        nonlocal state_context
        nonlocal state_revidx
        
        start_index = model.patch_start_index
        image_input: jt.Float[torch.Tensor, "B*S 5+tH*tW C"] = output
        patch_input: jt.Float[torch.Tensor, "B S C H W"]     = image_input[:, start_index:].permute(0, 2, 1).reshape(
            *state_inference_shape_bschw
        )
        with nvtx.range("accel_build_ctx"):
            state_context = token_merger.build_ctx(patch_input)
        Diagnostics.log(f"token_merger ({type(token_merger).__name__}) context built.")

        with nvtx.range("accel_encoder_merge"):
            Diagnostics.log("fused acceleration mode: starting token merging immediately.")
            merged_input, state_revidx = token_merger.merge(state_context, image_input.unsqueeze(0))
        return merged_input

    start_point.register_forward_hook(hook_on_ma_accel_start)
    
    # NOTE: 3) Systematically replace the blocks in the encoder.
    for idx, block in enumerate(accelerated_encode_blocks):
        encode_blocks[encoder_layer + 1 + idx] = dino_block_fn(block, "bf16")
    
    # NOTE: 4) split tokens at the end of encoder.
    def hook_after_last_encoder_block(module, args, output: ITokenMerger.JaggedTokens) -> torch.Tensor:
        assert state_context is not None
        assert state_revidx  is not None
        with nvtx.range("accel_encoder_split"):
            return token_merger.split(state_context, state_revidx, output).squeeze(0)
    encode_blocks[-1].register_forward_hook(hook_after_last_encoder_block)
    
    # NOTE: 5) rewrite the forward of MultiViewAlternatingAttentionTransformerIFR class.
    info_sharing = cast(MultiViewAlternatingAttentionTransformerIFR, model.info_sharing)

    for idx, block in enumerate(accelerated_selfat_blocks):
        selfat_blocks[idx] = self_block_fn(cast(SelfAttentionBlock, block), "bf16")
    
    def patch_info_sharing_forward(original_forward):
        def impl(self: MultiViewAlternatingAttentionTransformerIFR, model_input: MultiViewTransformerInput):

            # NOTE: For simplicity we only implement what's needed in MA.
            assert model_input.additional_input_tokens_per_view is not None
            assert model_input.additional_input_tokens          is not None
            assert self.custom_positional_encoding              is None
            assert self.distinguish_ref_and_non_ref_views       == True
            assert self.use_pe_for_non_reference_views          == False
            assert self.norm_intermediate                       == True
            assert self.intermediates_only                      == False
            
            # Get the indices of the intermediate features to return
            intermediate_multi_view_features = []
            take_indices, _ = feature_take_indices(self.depth, self.indices)
            
            # Initialize the multi-view features from the model input and number of views for current input
            multi_view_features = model_input.features
            num_of_views = len(multi_view_features)
            batch_size, _, height, width = multi_view_features[0].shape
            
            additional_tokens_per_view = model_input.additional_input_tokens_per_view
            num_of_additional_tokens_per_view = additional_tokens_per_view[0].shape[2]
            multi_view_features_with_tokens = []
            for view_features, view_tokens in zip(multi_view_features, additional_tokens_per_view):
                # view_features: (N, C, H, W)
                # view_tokens: (N, C, T)
                # Flatten spatial dimensions: (N, C, H, W) -> (N, C, H*W)
                view_features_flat = view_features.reshape(batch_size, self.input_embed_dim, height * width)
                # Concatenate tokens: (N, C, H*W + T)
                view_with_tokens = torch.cat([view_features_flat, view_tokens], dim=2)
                multi_view_features_with_tokens.append(view_with_tokens)

            # Stack all views: (N, V, C, H*W + T)
            multi_view_features = torch.stack(multi_view_features_with_tokens, dim=1)
            # Permute to (N, V, H*W + T, C)
            multi_view_features = multi_view_features.permute(0, 1, 3, 2)
            # Reshape to (N, V * (H*W + T), C)
            multi_view_features = multi_view_features.reshape(
                batch_size,
                num_of_views * (height * width + num_of_additional_tokens_per_view),
                self.input_embed_dim,
            ).contiguous()
            
            num_of_tokens_per_view = height * width + num_of_additional_tokens_per_view
            
            
            # Process additional input tokens if provided
            additional_tokens = model_input.additional_input_tokens

            # Reshape to channel-last format for transformer processing
            additional_tokens = additional_tokens.permute(0, 2, 1).contiguous()  # (N, C, T) -> (N, T, C)

            # Concatenate the additional tokens to the multi-view features
            multi_view_features = torch.cat([multi_view_features, additional_tokens], dim=1)

            # Project input features to the transformer dimension
            multi_view_features = self.proj_embed(multi_view_features)

            # Create patch positions for each view if custom positional encoding is used
            multi_view_positions = [None] * (num_of_views + model_input.additional_input_tokens.shape[1])

            # Add positional encoding for reference view (idx 0)
            ref_view_pe = cast(torch.Tensor, self.view_pos_table)[0].clone().detach()
            ref_view_pe = ref_view_pe.reshape((1, 1, self.dim))
            ref_view_pe = ref_view_pe.repeat(batch_size, num_of_tokens_per_view, 1)
            ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]
            ref_view_features = ref_view_features + ref_view_pe
            
            non_ref_view_features = multi_view_features[
                :, num_of_tokens_per_view : num_of_views * num_of_tokens_per_view, :
            ]
            
            additional_features = multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
            view_features       = torch.cat([ref_view_features, non_ref_view_features], dim=1)

            # ==== Merge spatial tokens into JaggedTokens ====
            with nvtx.range("accel_is_merge"):
                spatial_per_view = view_features.reshape(
                    batch_size, num_of_views, num_of_tokens_per_view, self.dim
                )[:, :, :height * width, :]                                             # (N, V, H*W, C)
                pv_additional = view_features.reshape(
                    batch_size, num_of_views, num_of_tokens_per_view, self.dim
                )[:, :, height * width:, :]                                             # (N, V, T_pv, C)
                T_pv = num_of_additional_tokens_per_view
                T_g  = additional_features.shape[1]

                spatial_jagged, new_stub = token_merger.merge(
                    state_context, spatial_per_view, start_index=0
                )
                view_jagged = spatial_jagged.add_suffix(
                    pv_additional.reshape(batch_size * num_of_views, T_pv, self.dim)
                )

                frame_offset = view_jagged.offset.clone()
                global_selector = torch.arange(
                    0, batch_size * num_of_views + 1, step=num_of_views,
                    device=frame_offset.device, dtype=frame_offset.dtype,
                )
                global_offset = frame_offset[global_selector]

            Diagnostics.log(f"info_sharing jagged merge done: {view_jagged.tokens.shape}, offset len={frame_offset.shape[0]}")

            # ==== Alternating attention loop (fully jagged) ====
            intermediate_jagged: list[tuple[ITokenMerger.JaggedTokens, torch.Tensor]] = []

            with nvtx.range("accel_is_attn_loop"):
                for depth_idx in range(self.depth):
                    with nvtx.range(f"accel_is_block_{depth_idx}"):
                        if depth_idx % 2 == 0:
                            view_jagged.offset = global_offset
                            combined = view_jagged.add_suffix(additional_features)
                            combined = self.self_attention_blocks[depth_idx](combined)
                            view_jagged, additional_features = combined.split_suffix(T_g)
                            view_jagged.offset = frame_offset
                        else:
                            view_jagged = self.self_attention_blocks[depth_idx](view_jagged)

                        if depth_idx in take_indices:
                            normed = view_jagged.apply(self.norm)
                            intermediate_jagged.append((
                                ITokenMerger.JaggedTokens(normed.tokens, normed.weight, frame_offset.clone()),
                                self.norm(additional_features.clone()),
                            ))

            # ==== Helper: convert jagged snapshot → MultiViewTransformerOutput ====
            def _jagged_to_mv_output(
                vj: ITokenMerger.JaggedTokens,
                add_feat: torch.Tensor,
                apply_norm: bool,
            ) -> MultiViewTransformerOutput:
                if apply_norm:
                    vj = vj.apply(self.norm)
                    add_feat = self.norm(add_feat)

                stripped, pv_dense = vj.split_suffix(T_pv)
                spatial_dense = token_merger.split(state_context, new_stub, stripped)
                spatial_dense = spatial_dense.reshape(batch_size, num_of_views, height * width, self.dim)
                spatial_dense = spatial_dense.reshape(
                    batch_size, num_of_views, height, width, self.dim
                ).permute(0, 1, 4, 2, 3).contiguous()                              # (N, V, C, H, W)

                feat_list = [spatial_dense[:, v].contiguous() for v in range(num_of_views)]

                pv_dense = pv_dense.reshape(batch_size, num_of_views, T_pv, self.dim)
                pv_list = [
                    pv_dense[:, v].permute(0, 2, 1).contiguous()                    # (N, C, T_pv)
                    for v in range(num_of_views)
                ]

                add_token_feat = add_feat.permute(0, 2, 1).contiguous()             # (N, C, T_g)

                return MultiViewTransformerOutput(
                    features=feat_list,
                    additional_token_features=add_token_feat,
                    additional_token_features_per_view=pv_list,
                )

            # ==== Build intermediate outputs ====
            with nvtx.range("accel_is_intermediate_split"):
                intermediate_multi_view_features_out: list[MultiViewTransformerOutput] = []
                for snap_vj, snap_add in intermediate_jagged:
                    intermediate_multi_view_features_out.append(
                        _jagged_to_mv_output(snap_vj, snap_add, apply_norm=False)
                    )

            # ==== Build final output ====
            with nvtx.range("accel_is_final_split"):
                output_mv = _jagged_to_mv_output(view_jagged, additional_features, apply_norm=True)

            Diagnostics.log(f"info_sharing split done: {len(intermediate_multi_view_features_out)} intermediates + final")
            return output_mv, intermediate_multi_view_features_out
        return impl
    patch_torch_forward(info_sharing, patch_info_sharing_forward)
    
    # NOTE: 6) write proper inference mask at the end of inference
    def hook_write_inference_mask(module, args: tuple[MultiViewInput], output: SceneGeometry) -> SceneGeometry:
        infer_mask = token_merger.build_infer_mask(state_context)
        infer_mask = infer_mask.unsqueeze(dim=2).repeat_interleave(repeats=14, dim=-1).repeat_interleave(repeats=14, dim=-2)
        
        # Set inference_mask - we are not responsible for merged regions' output quality!
        output.infer_mask = infer_mask if output.infer_mask is None else infer_mask | output.infer_mask
        
        return output
    model.register_forward_hook(hook_write_inference_mask)


    return model
