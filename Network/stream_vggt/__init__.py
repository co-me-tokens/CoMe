import torch
from .models.vggt import StreamVGGT
from .models.interface import StreamLike


def get_StreamVGGT():
    model = StreamVGGT()
    model.load_state_dict(torch.load("./Result/Models/StreamVGGT.pth"))
    return model
