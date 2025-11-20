import torch
import typing as T

from Network.stream_vggt.layers.streaming import StreamBlock, StreamLike


class Conv2DConfidence(torch.nn.Module):
    def __init__(self, hidden_dims: T.Sequence[int], activation: str):
        super().__init__()
        layers = []
        for in_feature, out_feature in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.extend([
                torch.nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=3, stride=1, padding=1),
                torch.nn.GELU()
            ])
        
        self.module     = torch.nn.Sequential(*layers)
        self.activation = activation
    
    def forward(self, tokens: torch.Tensor, BPHWC: tuple[int, ...]) -> torch.Tensor:
        B, P, H, W, C = BPHWC
        tokens = tokens.reshape(B * P, H * W, C).permute(0, 3, 1, 2)
        result = self.module(tokens).flatten(2).permute(0, 2, 1)
        
        match self.activation:
            case "expp1": return torch.exp(result) + 1.
            case "none" : return result
            case _      : raise ValueError(f"Unrecognized activation: {self.activation}") 


class GlobalAttentionConfidence(torch.nn.Module, StreamLike):
    def __init__(self, hidden_dims: T.Sequence[int], activation: str):
        super().__init__()

        self.proj1      = torch.nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[1])
        self.block      = StreamBlock(dim=hidden_dims[1], num_heads=1, mlp_ratio=2)
        self.proj2      = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=hidden_dims[1], out_channels=hidden_dims[1], kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            torch.nn.GELU(),
            torch.nn.Conv2d(in_channels=hidden_dims[1], out_channels=hidden_dims[2], kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        )
        self.activation = activation
    
    def forward(self, tokens: torch.Tensor, BPHWC: tuple[int, ...]) -> torch.Tensor:
        B, P, H, W, C = BPHWC
        
        tokens = tokens.reshape(B, P * H * W, C)
        tokens = self.proj1(tokens)
        tokens = self.block(tokens)
        tokens = tokens.reshape(B * P, H, W, -1).permute(0, 3, 1, 2)
        tokens = self.proj2(tokens).permute(0, 2, 3, 1).flatten(1, 2)
        
        match self.activation:
            case "expp1": return torch.exp(tokens) + 1.
            case "none" : return tokens
            case _      : raise ValueError(f"Unrecognized activation: {self.activation}") 

    def succ(self, tokens: torch.Tensor, BPHWC: tuple[int, ...]) -> torch.Tensor:
        B, P, H, W, C = BPHWC
        tokens = tokens.reshape(B, P * H * W, C)
        tokens = self.proj1(tokens)
        tokens = self.block.succ(tokens)
        tokens = tokens.reshape(B * P, H, W, -1).permute(0, 3, 1, 2)
        tokens = self.proj2(tokens).permute(0, 2, 3, 1).flatten(1, 2)
        
        match self.activation:
            case "expp1": return torch.exp(tokens) + 1.
            case "none" : return tokens
            case _      : raise ValueError(f"Unrecognized activation: {self.activation}") 

    def reset(self) -> None:
        self.block.reset()
