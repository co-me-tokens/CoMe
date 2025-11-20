import torch
from .models.vggt import VGGT


def get_VGGT():
    URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(URL))
    return model

def get_VGGT_vanilla():
    URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model = VGGT(vanilla=True)
    model.load_state_dict(torch.hub.load_state_dict_from_url(URL))
    return model
