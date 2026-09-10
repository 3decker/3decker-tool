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
