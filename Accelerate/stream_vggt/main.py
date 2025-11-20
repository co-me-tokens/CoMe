import torch
import json
from pathlib import Path
from Network.stream_vggt import StreamVGGT
from dataclasses import dataclass

from .. import core as accel_core
from ..common import sdpa_dispatch
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


@dataclass
class Accelerator_State:
    mask     : torch.Tensor | None = None
    prev_mask: torch.Tensor | None = None
    
    attn_bias: torch.Tensor | None = None
    dst_lidx : torch.Tensor | None = None
    
    frame_tok_shape: tuple[int, ...] | None = None
    globe_tok_shape: tuple[int, ...] | None = None
    BSCHW          : tuple[int, ...] | None = None
    
    pos_merge_cache: torch.Tensor | None = None


def accelerate_stream_vggt(model: StreamVGGT, cfg: Accelerator_Config):
    # Internal States of Acceleration Structure
    state = Accelerator_State()
    
    merger = accel_core.TokenMerger(start_idx=model.aggregator.patch_start_idx, grp_size=cfg.grp_size)
    
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
    is_first_frame = False
    
    def aggregator_reset(original_method):
        def implement(self) -> None:
            nonlocal state
            original_method()
            prev_mask       = state.mask
            state           = Accelerator_State()
            state.prev_mask = prev_mask
            
            if isinstance(predictor, accel_core.GlobalAttentionConfidence):
                predictor.reset()
            
        return implement
    
    def aggregator_succ(original_method):
        def implement(self, new_image: torch.Tensor):
            nonlocal is_first_frame
            is_first_frame = not self._stream_initialized
            
            B, S, C, H, W = new_image.shape
            state.globe_tok_shape = (B , S * ((W // 14) * (H // 14) + model.aggregator.patch_start_idx), 1024)
            state.frame_tok_shape = (B * S , ((W // 14) * (H // 14) + model.aggregator.patch_start_idx), 1024)
            state.BSCHW   = (B, S, C, H, W)
            
            result = original_method(new_image)
            
            state.prev_mask = state.mask
            state.mask = None
            state.pos_merge_cache = None
            return result
        return implement

    def succ_infer_mask(original_method):
        def implement(self, *args, **kwargs):
            assert state.BSCHW is not None
            tok_H, tok_W = state.BSCHW[-2] // 14, state.BSCHW[-1] // 14
            
            tokens = original_method(*args, **kwargs)
            
            if isinstance(tokens, dict):
                lean_tokens = tokens["x_norm_patchtokens"]
            elif probe_loc[0] == "DINO":
                lean_tokens = tokens[:, model.aggregator.patch_embed.num_register_tokens + 1:]
                lean_tokens = model.aggregator.patch_embed.norm(lean_tokens)
            else:
                if tokens.size(0) != state.BSCHW[0] * state.BSCHW[1]:
                    # Convert to frame format (B*P, N, C)
                    tokens = tokens.reshape(state.BSCHW[0] * state.BSCHW[1], -1 , tokens.size(-1))
                
                lean_tokens = tokens[:, model.aggregator.patch_start_idx:]
            
            with torch.no_grad():
                if isinstance(predictor, accel_core.Conv2DConfidence):
                    confidence = predictor(lean_tokens.bfloat16(), (state.BSCHW[0], state.BSCHW[1], tok_H, tok_W, lean_tokens.size(-1)))
                else:
                    confidence = predictor.succ(lean_tokens.bfloat16(), (state.BSCHW[0], state.BSCHW[1], tok_H, tok_W, lean_tokens.size(-1)))
                confidence = confidence.squeeze(-1).float()

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
                        assert config["train_loss_func"] == 'pairwise-rank'
                        bot_k = torch.topk(confidence, k=int(bot_k_thresh), dim=-1, largest=False, sorted=False)
                        mask          = torch.zeros_like(confidence, dtype=torch.bool)
                        batch_indices = torch.arange(mask.size(0)).unsqueeze(1).expand_as(bot_k.indices)
                        mask[batch_indices, bot_k.indices] = True
                        
                    case ("bot-p"  , bot_p_thresh):
                        assert config["train_loss_func"] == 'pairwise-rank'
                        bot_k_thresh = int(confidence.size(-1) * bot_p_thresh)
                        
                        bot_k = torch.topk(confidence, k=int(bot_k_thresh), dim=-1, largest=False, sorted=False)
                        mask          = torch.zeros_like(confidence, dtype=torch.bool)
                        batch_indices = torch.arange(mask.size(0)).unsqueeze(1).expand_as(bot_k.indices)
                        mask[batch_indices, bot_k.indices] = True
                    
                    case _: raise ValueError("Unsupport masking setup")
                
                if state.mask is None:
                    # NOTE: This is intentional, since VGGT use first frame as reference.
                    if is_first_frame: state.mask = torch.zeros_like(mask)
                    else: state.mask = mask
                assert state.mask is not None
                
                state.dst_lidx = merger.precompute_contract_idx(state.mask)
                
                if cfg.apply_attn_bias:
                    state.attn_bias = merger.precompute_attn_mask(state.mask, state.dst_lidx)
            
            return tokens
        return implement
    
    def patched_attn_forward(self, x: torch.Tensor, pos: torch.Tensor | None, frame_attn: bool):
        if cfg.apply_attn_bias:
            assert state.attn_bias is not None
            assert state.BSCHW is not None
            if frame_attn:
                attn_bias = state.attn_bias.view(state.BSCHW[0] * state.BSCHW[1], 1, 1, -1)
            else:
                attn_bias = state.attn_bias.view(state.BSCHW[0], 1, 1, -1)
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
                    backend="torch-memeff"
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
    
    def patched_stream_attn_succ(self, x: torch.Tensor, pos: torch.Tensor | None, frame_attn: bool):
        if cfg.apply_attn_bias:
            assert state.attn_bias is not None
            assert state.BSCHW is not None
            
            if frame_attn:
                attn_bias = state.attn_bias.view(state.BSCHW[0] * state.BSCHW[1], 1, 1, -1)
            else:
                attn_bias = state.attn_bias.view(state.BSCHW[0], 1, 1, -1)
        else:
            attn_bias = None
            
        B, new_N, C = x.shape
        with torch.cuda.nvtx.range("QKV Projection"):
            qkv = self.qkv(x).reshape(B, new_N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        new_q, new_k, new_v = qkv.unbind(0)
        new_q, new_k = self.q_norm(new_q), self.k_norm(new_k)
        
        with torch.cuda.nvtx.range("RoPE"):
            if self.rope is not None:
                new_q = self.rope(new_q, pos)
                new_k = self.rope(new_k, pos)
        
        new_q, new_k, new_v = new_q.bfloat16(), new_k.bfloat16(), new_v.bfloat16()
        
        with torch.cuda.nvtx.range("KV Cache"):
            # KV Cache Management
            if self._kv_cache is None:
                self._init_stream(new_k, new_v)
            
            assert self._kv_cache is not None
            self._kv_cache[0].push(new_k)
            self._kv_cache[1].push(new_v)
            
            if cfg.apply_attn_bias and (attn_bias is not None):
                self._kv_cache[2].push(attn_bias)
        
            q, k, v = new_q, self._kv_cache[0].data, self._kv_cache[1].data

        
        with torch.cuda.nvtx.range("SDPA"):
            if self.fused_attn:
                q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
                
                if cfg.apply_attn_bias:
                    bias = self._kv_cache[2].data
                    bias = bias.bfloat16()
                else:
                    bias = None
                
                x = sdpa_dispatch(
                    q, k, v,
                    attn_bias=bias,
                    backend="torch-memeff"
                )
                
                x = x.to(qkv)
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v
        
        with torch.cuda.nvtx.range("Projection"):
            x = x.transpose(1, 2).reshape(B, new_N, C)
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
            elif state.pos_merge_cache is None:
                merged_pos = (merger.merge(pos.float(), state.dst_lidx)[0]).long()
                state.pos_merge_cache = merged_pos
            else:
                merged_pos = state.pos_merge_cache
            
            # Original Attention Method, with attention bias correction.
            merged_toks    = patched_attn_forward(self, merged_toks, merged_pos, frame_attn=True)
            # End
            
            tokens         = merger.split(merged_toks, ctx, state.dst_lidx).view(state.frame_tok_shape)
            return tokens
        return implement

    def apply_globe_attn_succ_token_merge(original_forward):
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
            elif state.pos_merge_cache is None:
                pos              = pos.view(state.frame_tok_shape[0], state.frame_tok_shape[1], 2)
                merged_pos       = (merger.merge(pos.float(), state.dst_lidx)[0]).long()
                state.pos_merge_cache = merged_pos
                
                merged_pos       = merged_pos.view(state.globe_tok_shape[0], -1, 2)
            else:
                merged_pos       = state.pos_merge_cache
                merged_pos       = merged_pos.view(state.globe_tok_shape[0], -1, 2)
            
            # (Almost) Original Attention Method, with attention bias correction.
            merged_toks = patched_stream_attn_succ(self, merged_toks, merged_pos, frame_attn=False)
            # merged_toks = original_forward(merged_toks, pos=merged_pos)
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
    
    # Monkey patch the acceleration structure on StreamVGGT
    accel_core.patch_torch_method(model.aggregator, "succ" , aggregator_succ)
    accel_core.patch_torch_method(model.aggregator, "reset", aggregator_reset)
    set_block(model, probe_loc, accel_core.patch_torch_forward(get_block(model, probe_loc), succ_infer_mask))
    
    # User Interface
    def get_mask() -> torch.Tensor | None: return state.prev_mask
    def set_mask(mask: torch.Tensor): state.mask = mask
    # End
    
    if cfg.accel_dino_attn and probe_loc[0] == 'DINO':
        idx_arr = cfg.accel_dino_attn if isinstance(cfg.accel_dino_attn, list) else \
                  list(range(probe_loc[1] + 1, len(model.aggregator.patch_embed.blocks)))
        
        for idx in idx_arr:
            model.aggregator.patch_embed.blocks[idx].attn = accel_core.patch_torch_forward(
                model.aggregator.patch_embed.blocks[idx].attn, apply_frame_attn_token_merge
            )
    
    if cfg.accel_dino_mlp and probe_loc[0] == 'DINO':
        idx_arr = cfg.accel_dino_mlp if isinstance(cfg.accel_dino_mlp, list) else \
                  list(range(probe_loc[1] + 1, len(model.aggregator.patch_embed.blocks)))
        
        for idx in idx_arr:
            model.aggregator.patch_embed.blocks[idx].mlp = accel_core.patch_torch_forward(
                model.aggregator.patch_embed.blocks[idx].mlp, apply_frame_mlp_token_merge
            )
    
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
    
    if cfg.accel_global_attn:
        if probe_loc[0] == 'DINO':
            idx_arr = cfg.accel_global_attn if isinstance(cfg.accel_global_attn, list) else \
                      list(range(len(model.aggregator.global_blocks)))
        else:
            idx_arr = cfg.accel_global_attn if isinstance(cfg.accel_global_attn, list) else \
                      list(range(probe_loc[1] + 1, len(model.aggregator.global_blocks)))
        
        
        for idx in idx_arr:
            model.aggregator.global_blocks[idx].attn = accel_core.patch_torch_method(
                model.aggregator.global_blocks[idx].attn, "succ", apply_globe_attn_succ_token_merge
            )
    
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
    
    return model, get_mask, set_mask
