"""asset_hot_reload.py — watch an image on disk, re-upload it to the GPU on save.

Demonstrates the full asset pipeline end-to-end. The example bootstraps a
64x64 PNG at examples/assets/hero.png if it's missing. After load:

  - Press 1 / 2 / 3 / 4 to rewrite the PNG to red / green / blue / yellow
    from inside the app.
  - OR open the file in any image editor, modify it, save.

Either way the on-screen sprite swaps to the new pixels on the next frame:
file change → watchdog (background thread) → poll() (PRE_UPDATE main thread)
→ registry.reload(handle) → TextureAtlas.reload → tex.write to the GPU.

The handle and its texture_id are stable across reloads — only the pixels
behind them change. The Sprite component is never touched.

Input note: this uses app.input.is_key_down with edge-detection rather than
KeyEvent. Edge-detected polling is robust regardless of how many sim ticks
run per visual frame (KeyEvents only reach Phase.UPDATE systems, which can
miss presses at higher refresh rates).

Run with:
    python examples/asset_hot_reload.py
"""
import time
from pathlib import Path

from PIL import Image

import pyge
from pyge.renderer import setup_renderer_2d


HERO_PATH = Path(__file__).resolve().parent / "assets" / "hero.png"
HERO_SIZE = 64

COLORS = {
    pyge.KEY_1: (235,  60,  60, 255),  # red
    pyge.KEY_2: ( 60, 200,  80, 255),  # green
    pyge.KEY_3: ( 80, 130, 235, 255),  # blue
    pyge.KEY_4: (240, 220,  70, 255),  # yellow
}

# Don't write the PNG more than once per 200ms per key — saves cost when a
# key is held and prevents pathological watchdog/IO bursts on Windows.
WRITE_THROTTLE_SECONDS = 0.2


def write_solid_png(path: Path, rgba: tuple[int, int, int, int]) -> None:
    """Overwrite `path` with a HERO_SIZE x HERO_SIZE solid-color RGBA PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (HERO_SIZE, HERO_SIZE), rgba).save(str(path))


if not HERO_PATH.exists():
    write_solid_png(HERO_PATH, (220, 220, 220, 255))


# --- App ----------------------------------------------------------------

app = pyge.App(title="Asset Hot Reload", width=600, height=600)
setup_renderer_2d(app)
registry = app.setup_assets(watch_dirs=[str(HERO_PATH.parent)])

hero_handle = registry.load(str(HERO_PATH))
app.world.spawn(
    pyge.Transform2D(x=300.0, y=300.0),
    pyge.Sprite(texture_id=registry.get(hero_handle), width=256.0, height=256.0),
)


# --- Systems ------------------------------------------------------------

# Per-key edge-detect state and per-key write throttle.
_was_down: set[int] = set()
_last_write: dict[int, float] = {}


@app.system(pyge.Phase.UPDATE)
def repaint_on_keypress(world, dt):
    """Detect a fresh keypress (down this tick, not down last tick) and rewrite
    the PNG. The FileWatcher's PRE_UPDATE poll picks up the file change next
    tick and runs registry.reload → atlas.reload → tex.write."""
    if app.input.is_key_down(pyge.KEY_ESCAPE):
        app.window.close()
        return

    now = time.perf_counter()
    for key, rgba in COLORS.items():
        is_down = app.input.is_key_down(key)
        was_down = key in _was_down
        if is_down and not was_down and (now - _last_write.get(key, 0.0)) >= WRITE_THROTTLE_SECONDS:
            write_solid_png(HERO_PATH, rgba)
            _last_write[key] = now
            print(f"[hot-reload] wrote {rgba} → {HERO_PATH.name}")
        if is_down:
            _was_down.add(key)
        else:
            _was_down.discard(key)


print(f"[hot-reload] watching {HERO_PATH.parent}")
print("[hot-reload] press 1/2/3/4 to repaint, or edit the PNG directly")
app.run()
