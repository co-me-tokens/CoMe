import torch
import json
from pathlib import Path
from dataclasses import dataclass

from Network.map_anything import MapAnything
from .. import core as accel_core
from ..common import sdpa_dispatch, T_Backend
from .probe import get_block
import typing as T


@dataclass
class Accelerator_Config:
    accelerator: Path
    grp_size   : int
    mask_setup : tuple[T.Literal["z-score", "bot-p"], float]
    attn_backend: T_Backend = "torch-flex"
    
    accel_enc_attn : bool = True
    accel_dec_attn : bool = True
    accel_enc_mlp  : bool = True
    accel_dec_mlp  : bool = True
    apply_attn_bias: bool = True


@dataclass
class Accelerator_State:
    pHpW: tuple[int, int] | None   = None
    curr_mask: torch.Tensor | None = None
    dino_dst_lidx : torch.Tensor | None = None
    dino_attn_bias: torch.Tensor | None = None
    info_dst_lidx : torch.Tensor | None = None
    info_attn_bias: torch.Tensor | None = None

T_GetMask_Fn = T.Callable[[], torch.Tensor | None]


def accelerate_mapanything(model: MapAnything, cfg: Accelerator_Config) \
    -> tuple[MapAnything, T_GetMask_Fn]:
    # Preconditions
    assert not model.encoder.with_registers
    # End   
    
    # Internal States of Acceleration Structure
    state: Accelerator_State = Accelerator_State()
    dino_merger = accel_core.TokenMerger(start_idx=1, grp_size=cfg.grp_size)
    info_merger = accel_core.TokenMerger(start_idx=0, grp_size=cfg.grp_size)
    # End
    
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
    
    
    # Define Hooked Routines
    def context_management(original_forward):
        def implement(self, *args, **kwargs):
            nonlocal state
            _, _, imH, imW = args[0][0]["img"].shape
            state  = Accelerator_State(pHpW=(imH//14, imW//14))
            result = original_forward(*args, **kwargs)
            return result
        return implement
    
    def mask_predictor(original_forward):
        def implement(self, x: torch.Tensor):
            # assert state.pHpW is not None
            if state.pHpW is None:
                print(f"Warning! I don't see state.pHpW, are you running inference bypassing the MapAnything model? I assume input image is (36*14 x 36*14) sized")
                state.pHpW = (36, 36)
            tokens = original_forward(x)
            
            # Injected Logic for estimate mask
            lean_tokens = tokens[:, 1:]
            with torch.no_grad(), torch.cuda.nvtx.range("Confidence Predictor"):
                B, S, H, W, C = 1, tokens.size(0), state.pHpW[0], state.pHpW[1], tokens.size(2)
                assert lean_tokens.size(1) == H * W
                
                confidence = predictor(lean_tokens.bfloat16(), (B, S, H, W, C))
                confidence = confidence.squeeze(-1).float()
                confidence = confidence.reshape(lean_tokens.size(0), lean_tokens.size(1)//cfg.grp_size, cfg.grp_size).mean(dim=-1)
                
                match cfg.mask_setup:
                    case ("z-score", z_score_thresh):
                        z_scores = (confidence - confidence.mean(dim=(-1), keepdim=True)) / confidence.std(dim=(-1), keepdim=True)
                        mask     = z_scores < z_score_thresh
                        mask     = dino_merger.align_boolean_mask(mask).bool()
                        
                    case ("bot-p"  , bot_p_thresh):
                        bot_k_thresh = int(confidence.size(-1) * bot_p_thresh)
                        bot_k         = torch.topk(confidence, k=bot_k_thresh, dim=-1, largest=False, sorted=False)
                        mask          = torch.zeros_like(confidence, dtype=torch.bool)
                        batch_indices = torch.arange(mask.size(0)).unsqueeze(1).expand_as(bot_k.indices)
                        mask[batch_indices, bot_k.indices] = True
                    
                    case _: raise ValueError("Unsupport masking setup")
                
                # assert state.curr_mask is None
                state.curr_mask = mask
                state.dino_dst_lidx = dino_merger.precompute_contract_idx(state.curr_mask)
                state.info_dst_lidx = info_merger.precompute_contract_idx(state.curr_mask)
                
                if cfg.apply_attn_bias:
                    state.dino_attn_bias = dino_merger.precompute_attn_mask(state.curr_mask, state.dino_dst_lidx)
                    state.info_attn_bias = info_merger.precompute_attn_mask(state.curr_mask, state.info_dst_lidx)
            # Injected Logic End
            
            return tokens
        return implement
    
    def dino_mlp_acceleration(original_forward):
        def implement(self, x: torch.Tensor):
            assert state.dino_dst_lidx is not None
            m_x, m_ctx = dino_merger.merge(x, state.dino_dst_lidx)
            m_x = original_forward(m_x)
            x = dino_merger.split(m_x, m_ctx, state.dino_dst_lidx)
            return x
        return implement
    
    def frame_mlp_acceleration(original_forward):
        def implement(self, x: torch.Tensor):
            assert state.dino_dst_lidx is not None
            m_x, m_ctx = info_merger.merge(x, state.dino_dst_lidx)
            m_x = original_forward(m_x)
            x = info_merger.split(m_x, m_ctx, state.dino_dst_lidx)
            return x
        return implement
    
    def global_mlp_acceleration(original_forward):
        def implement(self, x: torch.Tensor):
            assert state.dino_dst_lidx is not None
            m_x, m_ctx = info_merger.merge(x[:, :-1].flatten(0, 1).reshape(state.dino_dst_lidx.size(0), -1, 768), state.dino_dst_lidx)
            m_xs = torch.cat([m_x.reshape(x.size(0), -1, 768), x[:, -1:]], dim=1)
            m_xs = original_forward(m_xs)
            x_s = info_merger.split(m_xs[:, :-1].flatten(0, 1).reshape(state.dino_dst_lidx.size(0), -1, 768), m_ctx, state.dino_dst_lidx)
            x   = torch.cat([x_s.reshape(x.size(0), -1, 768), m_xs[:, -1:]], dim=1)
            return x
        return implement

    def dino_attn_acceleration(original_forward):
        def implement(self, x: torch.Tensor):
            assert state.dino_dst_lidx is not None
            m_x, m_ctx = dino_merger.merge(x, state.dino_dst_lidx)
            
            # Original Attention Method
            B, N, C = m_x.shape
            qkv = (
                self.qkv(m_x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            )
            q, k, v   = torch.unbind(qkv, 0)
            if state.dino_attn_bias is not None:
                attn_bias = state.dino_attn_bias.view(x.size(0), 1, 1, -1)
            else:
                attn_bias = None
            m_x     = sdpa_dispatch(q, k, v, attn_bias, cfg.attn_backend)
            m_x     = m_x.permute(0, 2, 1, 3).reshape(B, N, C)
            m_x     = self.proj(m_x)
            m_x     = self.proj_drop(m_x)
            # End
            
            x = dino_merger.split(m_x, m_ctx, state.dino_dst_lidx)
            return x
        return implement
    
    def frame_attn_acceleration(original_forward):
        def implement(self, x: torch.Tensor, pos: torch.Tensor):
            assert state.info_dst_lidx is not None
            assert not self.use_scalable_softmax
            assert not self.use_entropy_scaling
            
            m_x, m_ctx = info_merger.merge(x, state.info_dst_lidx)
            
            # Original Attention Method
            B, N, C = m_x.shape
            qkv = (
                self.qkv(m_x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            )
            q, k, v   = torch.unbind(qkv, 0)
            q, k      = self.q_norm(q), self.k_norm(k)
            if state.info_attn_bias is not None:
                attn_bias = state.info_attn_bias.view(x.size(0), 1, 1, -1)
            else:
                attn_bias = None

            m_x     = sdpa_dispatch(q, k, v, attn_bias, cfg.attn_backend)
            m_x     = m_x.permute(0, 2, 1, 3).reshape(B, N, C)
            m_x     = self.proj(m_x)
            m_x     = self.proj_drop(m_x)
            # End
            
            x = info_merger.split(m_x, m_ctx, state.info_dst_lidx)
            return x
        return implement
    
    def global_attn_acceleration(original_forward):
        def implement(self, x: torch.Tensor, pos):
            assert state.info_dst_lidx is not None
            assert not self.use_scalable_softmax
            assert not self.use_entropy_scaling
            
            m_x, m_ctx = info_merger.merge(x[:, :-1].flatten(0, 1).reshape(state.info_dst_lidx.size(0), -1, 768), state.info_dst_lidx)
            m_xs = torch.cat([m_x.reshape(x.size(0), -1, 768), x[:, -1:]], dim=1)
            
            # Original Attention Method
            B, N, C = m_xs.shape
            qkv = (
                self.qkv(m_xs).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            )
            q, k, v   = torch.unbind(qkv, 0)
            q, k      = self.q_norm(q), self.k_norm(k)

            if state.info_attn_bias is not None:
                attn_bias = torch.cat([
                    state.info_attn_bias.reshape(x.size(0),-1 ),
                    torch.zeros(x.size(0), 1, dtype=state.info_attn_bias.dtype, device=state.info_attn_bias.device)
                ], dim=1).view(x.size(0), 1, 1, -1)
            else:
                attn_bias = None
            m_xs     = sdpa_dispatch(q, k, v, attn_bias, cfg.attn_backend)
            m_xs     = m_xs.permute(0, 2, 1, 3).reshape(B, N, C)
            m_xs     = self.proj(m_xs)
            m_xs     = self.proj_drop(m_xs)
            # End

            x_s = info_merger.split(m_xs[:, :-1].flatten(0, 1).reshape(state.info_dst_lidx.size(0), -1, 768), m_ctx, state.info_dst_lidx)
            x   = torch.cat([x_s.reshape(x.size(0), -1, 768), m_xs[:, -1:]], dim=1)
            return x
        return implement
    # 
    
    
    # Register Hooks
    sa_blks = model.info_sharing.self_attention_blocks
    ga_blks = [blk for i, blk in enumerate(sa_blks) if i % 2 == 0]
    fa_blks = [blk for i, blk in enumerate(sa_blks) if i % 2 == 1]
    accel_core.patch_torch_forward(model, context_management)
    accel_core.patch_torch_forward(get_block(model, probe_loc), mask_predictor)
    
    if cfg.accel_enc_mlp:
        for i in range(probe_loc[1] + 1, len(model.encoder.model.blocks)):
            accel_core.patch_torch_forward(
                model.encoder.model.blocks[i].mlp , dino_mlp_acceleration
            )
    
    if cfg.accel_enc_attn:
        for i in range(probe_loc[1] + 1, len(model.encoder.model.blocks)):
            accel_core.patch_torch_forward(
                model.encoder.model.blocks[i].attn, dino_attn_acceleration
            )
    
    if cfg.accel_dec_mlp:
        for i in range(len(fa_blks)):
            accel_core.patch_torch_forward(fa_blks[i].mlp , frame_mlp_acceleration )
        
        for i in range(len(ga_blks)):
            accel_core.patch_torch_forward(ga_blks[i].mlp , global_mlp_acceleration )
    
    if cfg.accel_dec_attn:
        for i in range(len(fa_blks)):
            accel_core.patch_torch_forward(fa_blks[i].attn, frame_attn_acceleration)

        for i in range(len(ga_blks)):
            accel_core.patch_torch_forward(ga_blks[i].attn, global_attn_acceleration)
    
    # Expose Getter and Setter for User
    def get_mask():
        if state.curr_mask is None: return None
        return ~state.curr_mask.repeat_interleave(cfg.grp_size, dim=1)
    
    return model, get_mask
