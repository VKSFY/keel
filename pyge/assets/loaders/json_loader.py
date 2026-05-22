"""JSON loader: registered for `.json` paths, returns the parsed value."""
from __future__ import annotations

import json
from typing import Any


def json_loader(path: str) -> Any:
    """Read and parse a JSON file at `path`. Returns whatever json.load returns."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
