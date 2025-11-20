import torch
from Network.map_anything import MapAnything
from ..core import patch_torch_forward

def install_nvtx_instrument(model: MapAnything) -> MapAnything:
    """Install NVTX markers on the MapAnything model"""
    
    def install_markers_of(label: str):
        def nvtx_hook_factory(original_forward):
            def hooked_nvtx_forward(self, *args, **kwargs):
                with torch.cuda.nvtx.range(label):
                    return original_forward(*args, **kwargs)
            return hooked_nvtx_forward
        return nvtx_hook_factory

    for blk_idx in range(len(model.encoder.model.blocks)):
        blk = model.encoder.model.blocks[blk_idx]
        patch_torch_forward(blk, install_markers_of(f"Encoder Blk[{blk_idx}]"))
        patch_torch_forward(blk.mlp , install_markers_of(f"MLP"))
        patch_torch_forward(blk.attn, install_markers_of(f"Attn"))
    
    sa_blks = model.info_sharing.self_attention_blocks
    ga_blks = [blk for i, blk in enumerate(sa_blks) if i % 2 == 0]
    fa_blks = [blk for i, blk in enumerate(sa_blks) if i % 2 == 1]
    
    for fa_idx, fa_blk in enumerate(fa_blks):
        patch_torch_forward(fa_blk, install_markers_of(f"Frame Blk[{fa_idx}]"))
        patch_torch_forward(fa_blk.mlp , install_markers_of(f"MLP"))
        patch_torch_forward(fa_blk.attn, install_markers_of(f"Attn"))
    
    for ga_idx, ga_blk in enumerate(ga_blks):
        patch_torch_forward(ga_blk, install_markers_of(f"Global Blk[{ga_idx}]"))
        patch_torch_forward(ga_blk.mlp , install_markers_of(f"MLP"))
        patch_torch_forward(ga_blk.attn, install_markers_of(f"Attn"))

    return model
