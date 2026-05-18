# type: ignore
# DINO block is from torch hub (no importable type stubs), uniception types propagate from attention module.
import torch
from uniception.models.utils.transformer_blocks import SelfAttentionBlock

from ....interface.token_merger import ITokenMerger
from ....utility.diagnostic     import Diagnostics
from .attention import (
    DINO_JaggedAttention, DINO_JaggedAttention_w_Bias,
    Self_JaggedAttention, Self_JaggedAttention_w_Bias
)


class DINO_JaggedBlock_w_Bias(torch.nn.Module):
    def __init__(self, block: torch.nn.Module, run_dtype: DINO_JaggedAttention_w_Bias.SupportType):
        super().__init__()
        self.block = block
        self.attn  = DINO_JaggedAttention_w_Bias(block.attn, run_dtype=run_dtype)

        assert not block.training, "DINO_JaggedBlock_w_Bias is invalid in training mode!"

    def forward(self, x: ITokenMerger.JaggedTokens, xpos: ITokenMerger.JaggedTokens | None = None) -> ITokenMerger.JaggedTokens:
        if Diagnostics.is_active():
            Diagnostics.log(f"{type(self).__name__} get {x.tokens.shape}, with sizes={(x.offset[1:] - x.offset[:-1]).tolist()}")

        original_dtype = x.tokens.dtype
        attn_dtype     = self.attn.run_dtype

        attn_x = self.attn(
                x.apply(self.block.norm1).apply(lambda y: y.to(dtype=attn_dtype))
            ).apply(
                lambda y: y.to(dtype=original_dtype)
            ).apply(
                self.block.ls1
            )
        x = x.apply(lambda residual_x: residual_x + attn_x.tokens)

        mlp_x = x.apply(self.block.norm2).apply(self.block.mlp).apply(self.block.ls2)
        x = x.apply(lambda residual_x: residual_x + mlp_x.tokens)

        return x


class DINO_JaggedBlock(torch.nn.Module):
    def __init__(self, block: torch.nn.Module, run_dtype: DINO_JaggedAttention.SupportType):
        super().__init__()
        self.block = block
        self.attn  = DINO_JaggedAttention(block.attn, run_dtype=run_dtype)

        assert not block.training, "DINO_JaggedBlock is invalid in training mode!"

    def forward(self, x: ITokenMerger.JaggedTokens, xpos: ITokenMerger.JaggedTokens | None = None) -> ITokenMerger.JaggedTokens:
        if Diagnostics.is_active():
            Diagnostics.log(f"{type(self).__name__} get {x.tokens.shape}, with sizes={(x.offset[1:] - x.offset[:-1]).tolist()}")

        original_dtype = x.tokens.dtype
        attn_dtype     = self.attn.run_dtype

        attn_x = self.attn(
                x.apply(self.block.norm1).apply(lambda y: y.to(dtype=attn_dtype))
            ).apply(
                lambda y: y.to(dtype=original_dtype)
            ).apply(
                self.block.ls1
            )
        x = x.apply(lambda residual_x: residual_x + attn_x.tokens)

        mlp_x = x.apply(self.block.norm2).apply(self.block.mlp).apply(self.block.ls2)
        x = x.apply(lambda residual_x: residual_x + mlp_x.tokens)

        return x


class Self_JaggedBlock_w_Bias(torch.nn.Module):
    def __init__(self, block: SelfAttentionBlock, run_dtype: Self_JaggedAttention_w_Bias.SupportType):
        super().__init__()
        self.block = block
        self.attn  = Self_JaggedAttention_w_Bias(block.attn, run_dtype=run_dtype)

        assert not block.training, "SA_JaggedBlock_w_Bias is invalid in training mode!"

    def forward(self, x: ITokenMerger.JaggedTokens, xpos: ITokenMerger.JaggedTokens | None = None) -> ITokenMerger.JaggedTokens:
        if Diagnostics.is_active():
            Diagnostics.log(f"{type(self).__name__} get {x.tokens.shape}, with sizes={(x.offset[1:] - x.offset[:-1]).tolist()}")

        original_dtype = x.tokens.dtype
        attn_dtype     = self.attn.run_dtype

        attn_x = self.attn(
                x.apply(self.block.norm1).apply(lambda y: y.to(dtype=attn_dtype))
            ).apply(
                lambda y: y.to(dtype=original_dtype)
            ).apply(
                self.block.ls1
            )
        x = x.apply(lambda residual_x: residual_x + attn_x.tokens)

        mlp_x = x.apply(self.block.norm2).apply(self.block.mlp).apply(self.block.ls2)
        x = x.apply(lambda residual_x: residual_x + mlp_x.tokens)

        return x


class Self_JaggedBlock(torch.nn.Module):
    def __init__(self, block: SelfAttentionBlock, run_dtype: Self_JaggedAttention.SupportType):
        super().__init__()
        self.block = block
        self.attn  = Self_JaggedAttention(block.attn, run_dtype=run_dtype)

        assert not block.training, "SA_JaggedBlock is invalid in training mode!"

    def forward(self, x: ITokenMerger.JaggedTokens, xpos: ITokenMerger.JaggedTokens | None = None) -> ITokenMerger.JaggedTokens:
        if Diagnostics.is_active():
            Diagnostics.log(f"{type(self).__name__} get {x.tokens.shape}, with sizes={(x.offset[1:] - x.offset[:-1]).tolist()}")

        original_dtype = x.tokens.dtype
        attn_dtype     = self.attn.run_dtype

        attn_x = self.attn(
                x.apply(self.block.norm1).apply(lambda y: y.to(dtype=attn_dtype))
            ).apply(
                lambda y: y.to(dtype=original_dtype)
            ).apply(
                self.block.ls1
            )
        x = x.apply(lambda residual_x: residual_x + attn_x.tokens)

        mlp_x = x.apply(self.block.norm2).apply(self.block.mlp).apply(self.block.ls2)
        x = x.apply(lambda residual_x: residual_x + mlp_x.tokens)

        return x
