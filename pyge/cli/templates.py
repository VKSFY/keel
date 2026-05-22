"""Inline templates for `pyge new`. No file I/O — pure Python strings."""
from __future__ import annotations


MAIN_PY_TEMPLATE: str = '''\
import pyge
from pyge.renderer import setup_renderer_2d
from pyge.assets import setup_assets

app = pyge.App(title="{project_name}", width=800, height=600)
assets = setup_assets(app, watch_dirs=["assets/"])
renderer = setup_renderer_2d(app)


@app.system(pyge.Phase.UPDATE)
def update(world, dt):
    pass


@app.system(pyge.Phase.RENDER)
def render(world, dt):
    pass


if __name__ == "__main__":
    app.run()
'''


PYPROJECT_TEMPLATE: str = '''\
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name}"
version = "0.1.0"
description = "A PyGE project"
requires-python = ">=3.10"
dependencies = [
    "pyge",
]

[project.scripts]
{project_name} = "main:app.run"
'''


README_TEMPLATE: str = '''\
# {project_name}

A new PyGE project.

## Run

```
pyge run
```

The dev loop watches every `.py` file in this directory and restarts the
process whenever you save a change. Use `pyge build` (eventually) to
package for distribution.

## Layout

- `main.py` — entry point.
- `assets/` — textures, JSON data, scenes; hot-reloaded by AssetRegistry.
- `scenes/` — saved Scene JSON files.
'''
