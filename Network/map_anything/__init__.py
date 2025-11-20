from pathlib import Path
from .mapanything.models import MapAnything

def get_MapAnything():
    if (pretrained_path := Path("./.torch_cache/mapanything.pth")).exists():
        return MapAnything.from_pretrained(pretrained_path)
    return MapAnything.from_pretrained("facebook/map-anything")
