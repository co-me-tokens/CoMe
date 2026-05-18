"""
Block that can process jagged tokens
"""
from typing import cast

import torch
import jaxtyping as jt

from ....thirdparty.vggt.layers.block       import Block
from ....thirdparty.vggt.layers.attention   import Attention
from ....interface.token_merger             import ITokenMerger
from ....utility.diagnostic                 import Diagnostics

from .bias_attention  import JaggedAttention_w_Bias
from .plain_attention import JaggedAttention


class JaggedBlock(torch.nn.Module):
    def __init__(self, original_block: Block, attn_dtype: JaggedAttention.SupportType):
        super().__init__()
        self.block = original_block
        self.attn  = JaggedAttention(cast(Attention, original_block.attn), run_dtype=attn_dtype)
        
        # TODO: Maybe support the training (i.e. backward pass) in future?
        assert not original_block.training, "JaggedBlock is invalid in trining mode!"
    
    def forward(self, x: ITokenMerger.JaggedTokens, pos: jt.Int[torch.Tensor, "M 2"] | None = None) -> ITokenMerger.JaggedTokens:
        if Diagnostics.is_active():
            Diagnostics.log(f"{type(self).__name__} get {x.tokens.shape}, with sizes={(x.offset[1:] - x.offset[:-1]).tolist()}")
        
        original_dtype = x.tokens.dtype
        attn_dtype     = self.attn.run_dtype
        
        # Attention sub-block
        attn_x = self.attn(
                x.apply(self.block.norm1).apply(lambda y: y.to(dtype=attn_dtype)), pos=pos
            ).apply(
                lambda y: y.to(dtype=original_dtype)
            ).apply(
                self.block.ls1
            )
        x = x.apply(lambda residual_x: residual_x + attn_x.tokens)

        # MLP sub-block
        mlp_x  = x.apply(self.block.norm2).apply(self.block.mlp).apply(self.block.ls2)
        x = x.apply(lambda residual_x: residual_x + mlp_x.tokens)
        
        return x


class JaggedBlock_w_Bias(torch.nn.Module):
    def __init__(self, original_block: Block, attn_dtype: JaggedAttention.SupportType):
        super().__init__()
        self.block = original_block
        self.attn  = JaggedAttention_w_Bias(cast(Attention, original_block.attn), run_dtype=attn_dtype)
        
        # TODO: Maybe support the training (i.e. backward pass) in future?
        assert not original_block.training, "JaggedBlock is invalid in trining mode!"
    
    def forward(self, x: ITokenMerger.JaggedTokens, pos: jt.Int[torch.Tensor, "M 2"] | None = None) -> ITokenMerger.JaggedTokens:
        if Diagnostics.is_active():
            Diagnostics.log(f"{type(self).__name__} get {x.tokens.shape}, with sizes={(x.offset[1:] - x.offset[:-1]).tolist()}")
        
        original_dtype = x.tokens.dtype
        attn_dtype     = self.attn.run_dtype
        
        # Attention sub-block
        attn_x = self.attn(
                x.apply(self.block.norm1).apply(lambda y: y.to(dtype=attn_dtype)), pos=pos
            ).apply(
                lambda y: y.to(dtype=original_dtype)
            ).apply(
                self.block.ls1
            )
        x = x.apply(lambda residual_x: residual_x + attn_x.tokens)

        # MLP sub-block
        mlp_x  = x.apply(self.block.norm2).apply(self.block.mlp).apply(self.block.ls2)
        x = x.apply(lambda residual_x: residual_x + mlp_x.tokens)
        
        return x
