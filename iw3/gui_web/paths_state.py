"""Remembers the last folder used for Input/Output file pickers, across
GUI restarts -- a small separate JSON file (not the wx GUI's
PersistenceManager .cfg format, no shared state between the two GUIs)."""
import json
from os import path

from . import CONFIG_DIR

STATE_PATH = path.join(CONFIG_DIR, "iw3-gui-web-paths.json")


def _load():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save(state):
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except OSError:
        pass


def get_last_input_dir():
    return _load().get("last_input_dir") or ""


def get_last_output_dir():
    return _load().get("last_output_dir") or ""


def set_last_input_dir(directory):
    if not directory:
        return
    state = _load()
    state["last_input_dir"] = directory
    _save(state)


def set_last_output_dir(directory):
    if not directory:
        return
    state = _load()
    state["last_output_dir"] = directory
    _save(state)
