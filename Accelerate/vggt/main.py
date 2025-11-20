import torch
import json
from pathlib import Path
from Network.vggt import VGGT
from dataclasses import dataclass
from .. import core as accel_core
from ..common import sdpa_dispatch, T_Backend

from .probe import get_block, set_block
import typing as T



@dataclass
class Accelerator_Config:
    accelerator      : Path
    grp_size         : int
    mask_setup       : tuple[T.Literal["z-score"], float] | tuple[T.Literal["bot-k"], int] | tuple[T.Literal["bot-p"], float]
    accel_dino_attn  : bool | list[int] = True
    accel_dino_mlp   : bool | list[int] = True
    accel_frame_attn : bool | list[int] = True
    accel_frame_mlp  : bool | list[int] = True
    accel_global_attn: bool | list[int] = True
    accel_global_mlp : bool | list[int] = True
    apply_attn_bias  : bool = True
    attn_backend     : T_Backend = "torch-flex"
    

@dataclass
class Accelerator_State:
    prev_mask: torch.Tensor | None = None
    mask     : torch.Tensor | None = None
    dst_lidx : torch.Tensor | None = None
    att_bias : torch.Tensor | None = None
    BSCHW    : tuple[int, ...] | None = None
    pos_merge_cache: torch.Tensor | None = None
    
    frame_tok_shape: tuple[int, ...] | None = None
    globe_tok_shape: tuple[int, ...] | None = None


def accelerate_vggt(model: VGGT, cfg: Accelerator_Config):
    # Internal States of Acceleration Structure
    state    : Accelerator_State = Accelerator_State()
    # End

    merger     = accel_core.TokenMerger(start_idx=model.aggregator.patch_start_idx, grp_size=cfg.grp_size)

    with open(Path(cfg.accelerator.parent, "config.json"), "r") as f: config = json.load(f)
    probe_loc = config["probe_loc"]
    
    match config["method"]:
        case "conv":
            predictor = accel_core.Conv2DConfidence(config["method_dim"], config["method_act"])
        case "attn":
            predictor = accel_core.GlobalAttentionConfidence(config['method_dim'], config['method_act'])
        case _:
            raise NotImplementedError()
    
    predictor.load_state_dict(torch.load(cfg.accelerator))
    predictor = predictor.bfloat16().eval().to("cuda")
    
    # Methods
    def context_management(original_forward):
        def implement(self, images: torch.Tensor):
            nonlocal state
            B, S, C, H, W = images.shape
            state.BSCHW = tuple(images.shape)
            state.globe_tok_shape = (B , S * ((W // 14) * (H // 14) + model.aggregator.patch_start_idx), 1024)
            state.frame_tok_shape = (B * S , ((W // 14) * (H // 14) + model.aggregator.patch_start_idx), 1024)
            
            result = original_forward(images)
            
            # Clear context variable.
            prev_mask = state.mask
            state   = Accelerator_State()
            state.prev_mask = prev_mask
            return result
        return implement
    
    def create_infer_mask(original_forward):
        def implement(self, *args, **kwargs):
            assert state.BSCHW is not None
            tok_H, tok_W = state.BSCHW[-2] // 14, state.BSCHW[-1] // 14
            
            tokens = original_forward(*args, **kwargs)
            
            if isinstance(tokens, dict):
                lean_tokens = tokens["x_norm_patchtokens"]
            
            elif probe_loc[0] == 'DINO':
                lean_tokens = tokens[:, model.aggregator.patch_embed.num_register_tokens + 1:]
                lean_tokens = model.aggregator.patch_embed.norm(lean_tokens)
            
            else:
                if tokens.size(0) != state.BSCHW[0] * state.BSCHW[1]:
                    # Convert to frame format (B*P, N, C)
                    tokens = tokens.reshape(state.BSCHW[0] * state.BSCHW[1], -1, tokens.size(-1))
                
                lean_tokens = tokens[:, model.aggregator.patch_start_idx:]
            
            with torch.no_grad():
                with torch.cuda.nvtx.range("Confidence Predictor"):
                    confidence = predictor(lean_tokens.bfloat16(), (state.BSCHW[0], state.BSCHW[1], tok_H, tok_W, lean_tokens.size(-1)))
                
                confidence = confidence.squeeze(-1).float()
                # torch.save(confidence, "confidence.pth")
                
                confidence = confidence.reshape(lean_tokens.size(0), lean_tokens.size(1)//cfg.grp_size, cfg.grp_size).mean(dim=-1)
                
                # NOTE: I intentiaonally put this here since this is part of the acceleration structure overhead
                #       this part should be included in runtime stat even if user override the mask.
                match cfg.mask_setup:
                    case ("z-score", z_score_thresh):
                        assert config["train_loss_func"] == 'mse'
                        z_scores = (confidence - confidence.mean(dim=(-1), keepdim=True)) / confidence.std(dim=(-1), keepdim=True)
                        mask     = z_scores < z_score_thresh
                        mask     = merger.align_boolean_mask(mask).bool()
                        
                    case ("bot-k"  , bot_k_thresh):
                        # assert config["train_loss_func"] == 'pairwise-rank'
                        bot_k = torch.topk(confidence, k=int(bot_k_thresh), dim=-1, largest=False, sorted=False)
                        mask          = torch.zeros_like(confidence, dtype=torch.bool)
                        batch_indices = torch.arange(mask.size(0)).unsqueeze(1).expand_as(bot_k.indices)
                        mask[batch_indices, bot_k.indices] = True
                        
                    case ("bot-p"  , bot_p_thresh):
                        # assert config["train_loss_func"] == 'pairwise-rank'
                        bot_k_thresh = int(confidence.size(-1) * bot_p_thresh)
                        
                        bot_k = torch.topk(confidence, k=int(bot_k_thresh), dim=-1, largest=False, sorted=False)
                        mask          = torch.zeros_like(confidence, dtype=torch.bool)
                        batch_indices = torch.arange(mask.size(0)).unsqueeze(1).expand_as(bot_k.indices)
                        mask[batch_indices, bot_k.indices] = True
                    
                    case _: raise ValueError("Unsupport masking setup")
                
                if state.mask is None: state.mask = mask
                assert state.mask is not None
                
                state.dst_lidx = merger.precompute_contract_idx(state.mask)
                
                if cfg.apply_attn_bias:
                    state.att_bias = merger.precompute_attn_mask(state.mask, state.dst_lidx)
            
            return tokens
        return implement
    
    def patched_attn_forward(self, x: torch.Tensor, pos: torch.Tensor | None, frame_attn: bool):
        if cfg.apply_attn_bias:
            assert state.att_bias is not None
            assert state.BSCHW is not None
            if frame_attn:
                attn_bias = state.att_bias.view(state.BSCHW[0] * state.BSCHW[1], 1, 1, -1)
            else:
                attn_bias = state.att_bias.view(state.BSCHW[0], 1, 1, -1)
        else:
            attn_bias = None
        
        B, N, C = x.shape
        with torch.cuda.nvtx.range("QKV Projection"):
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        with torch.cuda.nvtx.range("RoPE"):
            if self.rope is not None:
                q = self.rope(q, pos)
                k = self.rope(k, pos)

        with torch.cuda.nvtx.range("SDPA"):
            if self.fused_attn:
                q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
                x = sdpa_dispatch(
                    q, k, v,
                    attn_bias=attn_bias.bfloat16() if (attn_bias is not None) else None,
                    backend=cfg.attn_backend
                )
                
                x = x.to(qkv)
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = (attn + attn_bias).softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v
        
        with torch.cuda.nvtx.range("Projection"):
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
        
        x = self.proj_drop(x)
        return x
    
    def apply_frame_attn_token_merge(original_forward):
        def implement(self, *args, **kwargs):
            assert (state.dst_lidx is not None) and (state.frame_tok_shape is not None)
            tokens: torch.Tensor
            tokens, pos   = args[0], kwargs.get('pos', None)
            
            merged_toks, ctx = merger.merge(tokens, state.dst_lidx)
            
            if pos is None:
                merged_pos = None
            elif state.pos_merge_cache is not None:
                merged_pos = state.pos_merge_cache
            else:
                merged_pos = (merger.merge(pos.float(), state.dst_lidx)[0]).long()
                state.pos_merge_cache = merged_pos
            
            # Original Attention Method, with attention bias correction.
            merged_toks    = patched_attn_forward(self, merged_toks, merged_pos, frame_attn=True)
            # End
            
            tokens         = merger.split(merged_toks, ctx, state.dst_lidx).view(state.frame_tok_shape)
            return tokens
        return implement
     
    def apply_globe_attn_token_merge(original_forward):
        def implement(self, *args, **kwargs):
            assert (state.dst_lidx is not None)
            assert (state.frame_tok_shape) and (state.globe_tok_shape)
            tokens: torch.Tensor
            tokens, pos   = args[0], kwargs.get('pos', None)
            
            tB, tP, tC    = tokens.shape
            
            tokens           = tokens.view(*state.frame_tok_shape)
            merged_toks, ctx = merger.merge(tokens, state.dst_lidx)
            merged_toks      = merged_toks.view(state.globe_tok_shape[0], -1, tC)
            
            
            if pos is None:
                merged_pos = None
            elif state.pos_merge_cache is not None:
                merged_pos = state.pos_merge_cache.view(state.globe_tok_shape[0], -1, 2)
            else:
                pos              = pos.view(state.frame_tok_shape[0], state.frame_tok_shape[1], 2)
                merged_pos       = (merger.merge(pos.float(), state.dst_lidx)[0]).long()
                state.pos_merge_cache = merged_pos
                merged_pos = state.pos_merge_cache.view(state.globe_tok_shape[0], -1, 2)
            
            # (Almost) Original Attention Method, with attention bias correction.
            merged_toks = patched_attn_forward(self, merged_toks, merged_pos, frame_attn=False)
            merged_toks = merged_toks.view(state.frame_tok_shape[0], -1, tC)
            # End
            
            tokens         = merger.split(merged_toks, ctx, state.dst_lidx).view(tB, tP, tC)
            return tokens
        return implement
      
    def apply_frame_mlp_token_merge(original_forward):
        def implement(self, *args, **kwargs):
            assert (state.frame_tok_shape) and (state.dst_lidx is not None)
            tokens: torch.Tensor
            tokens, pos   = args[0], kwargs.get('pos', None)
            assert pos is None
            
            tokens           = tokens.view(*state.frame_tok_shape)
            merged_toks, ctx = merger.merge(tokens, state.dst_lidx)
            
            merged_toks      = original_forward(merged_toks)
            
            tokens           = merger.split(merged_toks, ctx, state.dst_lidx)
            return tokens
        return implement

    def apply_globe_mlp_token_merge(original_forward):
        def implement(self, *args, **kwargs):
            assert (state.frame_tok_shape) and (state.globe_tok_shape) and (state.dst_lidx is not None)
            tokens: torch.Tensor
            
            tokens, pos   = args[0], kwargs.get('pos', None)
            assert pos is None
            tB, tP, tC       = tokens.shape
            
            tokens           = tokens.view(*state.frame_tok_shape)
            merged_toks, ctx = merger.merge(tokens, state.dst_lidx)
            merged_toks      = merged_toks.view(state.globe_tok_shape[0], -1, tC)
            
            merged_toks      = original_forward(merged_toks)
            
            merged_toks      = merged_toks.view(state.frame_tok_shape[0], -1, tC)
            tokens           = merger.split(merged_toks, ctx, state.dst_lidx).view(tB, tP, tC)
            return tokens
        return implement
    
    # Monkey patch the acceleration structure on VGGT.
    model.aggregator = accel_core.patch_torch_forward(model.aggregator, context_management)
    injected_block   = accel_core.patch_torch_forward(get_block(model, probe_loc), create_infer_mask)
    set_block(model, probe_loc, injected_block)
    
    if cfg.accel_dino_attn and probe_loc[0] == 'DINO':
        idx_arr = cfg.accel_dino_attn if isinstance(cfg.accel_dino_attn, list) else \
                  list(range(probe_loc[1] + 1, len(model.aggregator.patch_embed.blocks)))
        
        for idx in idx_arr:
            model.aggregator.patch_embed.blocks[idx].attn = accel_core.patch_torch_forward(
                model.aggregator.patch_embed.blocks[idx].attn, apply_frame_attn_token_merge
            )
    else:
        print(f"{cfg.accel_dino_attn=}")
    
    if cfg.accel_dino_mlp and probe_loc[0] == 'DINO':
        idx_arr = cfg.accel_dino_mlp if isinstance(cfg.accel_dino_mlp, list) else \
                  list(range(probe_loc[1] + 1, len(model.aggregator.patch_embed.blocks)))
        
        for idx in idx_arr:
            model.aggregator.patch_embed.blocks[idx].mlp = accel_core.patch_torch_forward(
                model.aggregator.patch_embed.blocks[idx].mlp, apply_frame_mlp_token_merge
            )
    else:
        print(f"{cfg.accel_dino_mlp=}")
    
    if cfg.accel_frame_attn:
        if probe_loc[0] == 'DINO':
            idx_arr = cfg.accel_frame_attn if isinstance(cfg.accel_frame_attn, list) else \
                      list(range(len(model.aggregator.frame_blocks)))
        else:
            idx_arr = cfg.accel_frame_attn if isinstance(cfg.accel_frame_attn, list) else \
                      list(range(probe_loc[1] + 1, len(model.aggregator.frame_blocks)))
        
        for idx in idx_arr:
            model.aggregator.frame_blocks[idx].attn = accel_core.patch_torch_forward(
                model.aggregator.frame_blocks[idx].attn, apply_frame_attn_token_merge
            )
    else:
        print(f"{cfg.accel_frame_attn=}")
    
    if cfg.accel_frame_mlp:
        if probe_loc[0] == 'DINO':
            idx_arr = cfg.accel_frame_mlp if isinstance(cfg.accel_frame_mlp, list) else \
                      list(range(len(model.aggregator.frame_blocks)))
        else:
            idx_arr = cfg.accel_frame_mlp if isinstance(cfg.accel_frame_mlp, list) else \
                      list(range(probe_loc[1] + 1, len(model.aggregator.frame_blocks)))
        
        for idx in idx_arr:
            model.aggregator.frame_blocks[idx].mlp = accel_core.patch_torch_forward(
                model.aggregator.frame_blocks[idx].mlp, apply_frame_mlp_token_merge
            )
    else:
        print(f"{cfg.accel_frame_mlp=}")
    
    if cfg.accel_global_attn:
        if probe_loc[0] == 'DINO':
            idx_arr = cfg.accel_global_attn if isinstance(cfg.accel_global_attn, list) else \
                      list(range(len(model.aggregator.global_blocks)))
        else:
            idx_arr = cfg.accel_global_attn if isinstance(cfg.accel_global_attn, list) else \
                      list(range(probe_loc[1] + 1, len(model.aggregator.global_blocks)))
        
        
        for idx in idx_arr:
            model.aggregator.global_blocks[idx].attn = accel_core.patch_torch_forward(
                model.aggregator.global_blocks[idx].attn, apply_globe_attn_token_merge
            )
    else:
        print(f"{cfg.accel_global_attn=}")
    
    if cfg.accel_global_mlp:
        if probe_loc[0] == 'DINO':
            idx_arr = cfg.accel_global_mlp if isinstance(cfg.accel_global_mlp, list) else \
                      list(range(len(model.aggregator.global_blocks)))
        else:
            idx_arr = cfg.accel_global_mlp if isinstance(cfg.accel_global_mlp, list) else \
                      list(range(probe_loc[1] + 1, len(model.aggregator.global_blocks)))
        
        for idx in idx_arr:
            model.aggregator.global_blocks[idx].mlp = accel_core.patch_torch_forward(
                model.aggregator.global_blocks[idx].mlp, apply_globe_mlp_token_merge
            )
    else:
        print(f"{cfg.accel_global_mlp=}")

    # User Interface
    def get_mask() -> torch.Tensor | None:
        if state.prev_mask is None: return None
        
        mask = state.prev_mask
        return ~mask.unsqueeze(-1).repeat(1, 1, cfg.grp_size).flatten(start_dim=-2, end_dim=-1)
    
    def set_mask(mask: torch.Tensor): state.mask = mask
    # End

    return model, get_mask, set_mask
