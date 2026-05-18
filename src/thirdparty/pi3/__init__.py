from .models.pi3 import Pi3
from .models.pi3x import Pi3X


def get_Pi3() -> Pi3:
    return Pi3.from_pretrained("yyfz233/Pi3")


def get_Pi3X() -> Pi3X:
    model = Pi3X.from_pretrained("yyfz233/Pi3X")
    
    # TODO: Support multi-modal Pi3X acceleration.
    model.disable_multimodal()
    
    return model
