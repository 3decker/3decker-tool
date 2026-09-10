"""Named preset and session persistence for the web GUI.

Deliberately NOT the wx GUI's wx.lib.agw.persist.PersistenceManager format
(that's tied to real wx widget objects, which this GUI has none of) --
plain JSON files instead, one per named preset plus one fixed "session"
file for the auto-save/restore-on-launch behavior gui.py's own CONFIG_PATH
provides. Presets live in their own subfolder so they don't collide with
the wx GUI's tmp/presets/*.cfg files even though both share the same
parent tmp/ directory (see gui_web/__init__.py's own CONFIG_DIR comment).
"""
import json
import os
from os import path

from . import CONFIG_DIR
from nunif.utils.filename import sanitize_filename

PRESET_DIR = path.join(CONFIG_DIR, "gui_web_presets")
SESSION_PATH = path.join(CONFIG_DIR, "iw3-gui-web-session.json")
# Separate small files, same convention as gui.py's own LAYOUT_CONFIG_PATH/
# ZOOM_CONFIG_PATH (ADR-092) -- live-applied UI preferences, not part of
# the conversion settings session above.
LAYOUT_PATH = path.join(CONFIG_DIR, "iw3-gui-web-layout.txt")
ZOOM_PATH = path.join(CONFIG_DIR, "iw3-gui-web-zoom.txt")


def _ensure_preset_dir():
    os.makedirs(PRESET_DIR, exist_ok=True)


def list_presets():
    _ensure_preset_dir()
    return sorted(path.splitext(fn)[0] for fn in os.listdir(PRESET_DIR) if fn.endswith(".json"))


def save_preset(name, settings):
    _ensure_preset_dir()
    name = sanitize_filename(name)
    if not name:
        raise ValueError("preset name cannot be empty")
    with open(path.join(PRESET_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    return name


def load_preset(name):
    name = sanitize_filename(name)
    fp = path.join(PRESET_DIR, f"{name}.json")
    if not path.exists(fp):
        return None
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_preset(name):
    name = sanitize_filename(name)
    fp = path.join(PRESET_DIR, f"{name}.json")
    if path.exists(fp):
        os.remove(fp)
        return True
    return False


def save_session(settings):
    try:
        with open(SESSION_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f)
    except OSError:
        pass


def load_session():
    try:
        with open(SESSION_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def get_layout():
    try:
        with open(LAYOUT_PATH, "r", encoding="utf-8") as f:
            v = f.read().strip()
            return v if v in ("tabs", "single_page") else "tabs"
    except OSError:
        return "tabs"


def set_layout(mode):
    if mode not in ("tabs", "single_page"):
        return
    try:
        with open(LAYOUT_PATH, "w", encoding="utf-8") as f:
            f.write(mode)
    except OSError:
        pass


def get_zoom():
    try:
        with open(ZOOM_PATH, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 100


def set_zoom(pct):
    try:
        with open(ZOOM_PATH, "w", encoding="utf-8") as f:
            f.write(str(int(pct)))
    except OSError:
        pass
