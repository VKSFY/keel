"""Built-in asset loaders for the AssetRegistry."""
from .json_loader import json_loader
from .texture_loader import make_texture_loader

__all__ = ["json_loader", "make_texture_loader"]
