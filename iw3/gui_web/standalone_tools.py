"""Standalone Tools for the web GUI (ADR-087): the 6 mini sub-apps that
shell out to a separate CLI module as a subprocess, structurally different
from the main iw3_main() conversion pipeline (worker.py) -- HDR/DV
Reinjection, Add Subtitle Track, Add Audio Track, Retroactive Stereo Tag,
Sharpen, and RIFE Frame Interpolation (standalone). Search Subtitles
(OpenSubtitles) is NOT one of these -- gui.py's own version calls
subtitle_search_cli.search() directly in-process (a network call, not a
subprocess), a genuinely different integration shape; deferred separately.

Every one of these 6 CLI modules already exposes its own create_parser(),
exactly like iw3.utils does for the main pipeline -- build_tool_command()
below reuses that directly (same diff-free construction command_line.py's
build_cli_command() uses for the main command) instead of hand-mapping each
tool's flags a second time, so this can never drift from what each module's
own real argparse definition says.
"""
import argparse
import importlib
import subprocess
import sys
import threading

from .worker import _push
from .crash_log import log_exception

TOOLS = [
    {"key": "reinject", "module": "iw3.reinject_hdr_cli", "label": "Retroactive HDR/DV Reinjection"},
    {"key": "submux", "module": "iw3.subtitle_mux_cli", "label": "Add Subtitle Track"},
    {"key": "audiomux", "module": "iw3.audio_mux_cli", "label": "Add Audio Track"},
    {"key": "stereotag", "module": "iw3.stereo_mode_tag_cli", "label": "Retroactively Tag MKV as 3D"},
    {"key": "sharpen", "module": "iw3.sharpen_cli", "label": "Sharpen"},
    {"key": "rife_standalone", "module": "iw3.rife_cli", "label": "RIFE Frame Interpolation"},
]
_TOOLS_BY_KEY = {t["key"]: t for t in TOOLS}


def tool_schema(tool_key):
    """Real field list for one tool, read live from its own create_parser()
    -- name/type/choices/default, the same shape the main FIELDS schema
    uses, so the frontend's existing field-rendering conventions apply here
    too (just a much shorter, per-tool list, not tab/group-organized)."""
    tool = _TOOLS_BY_KEY[tool_key]
    module = importlib.import_module(tool["module"])
    parser = module.create_parser()
    fields = []
    for action in parser._actions:
        if not action.option_strings or action.dest in ("help",):
            continue
        if isinstance(action, argparse._StoreTrueAction):
            widget, value_type = "checkbox", "bool"
        elif isinstance(action, argparse._StoreFalseAction):
            widget, value_type = "checkbox", "bool"
        elif action.choices:
            widget, value_type = "select", "str"
        else:
            widget = "combo_editable"
            value_type = "float" if action.type is float else "int" if action.type is int else "str"
        fields.append({
            "name": action.dest,
            "label": action.dest.replace("_", " ").title(),
            "widget": widget,
            "value_type": value_type,
            "choices": [str(c) for c in action.choices] if action.choices else [],
            "default": action.default if not isinstance(action.default, bool) else None,
            "required": bool(action.required),
            "help": action.help or "",
        })
    return {"key": tool["key"], "label": tool["label"], "fields": fields}


def build_tool_command(tool_key, values):
    """Builds a real `python -m iw3.<tool>_cli ...` argv list from a
    {dest: value} dict, reading each flag's real shape (store_true vs.
    store_false vs. a real value) from the module's own create_parser() --
    never a hand-maintained per-tool mapping."""
    tool = _TOOLS_BY_KEY[tool_key]
    module = importlib.import_module(tool["module"])
    parser = module.create_parser()

    argv = []
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        dest = action.dest
        if dest not in values:
            continue
        value = values[dest]
        if value is None or value == "":
            continue
        flag = next((o for o in action.option_strings if o.startswith("--")), action.option_strings[0])
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(flag)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                argv.append(flag)
        else:
            argv.append(flag)
            argv.append(str(value))

    return [sys.executable, "-m", tool["module"]] + argv


class StandaloneToolJob:
    """Runs one standalone tool's real CLI module as a subprocess, streaming
    its output to the frontend -- same shape as update_manager.UpdateJob
    (itself modeled on worker.ConversionJob), just for a shorter-lived job
    with no progress-percentage channel, only a log."""

    def __init__(self, window, tool_key):
        self.window = window
        self.tool_key = tool_key
        self.running = False

    def start(self, values):
        if self.running:
            raise RuntimeError(f"{self.tool_key} is already running")
        cmd = build_tool_command(self.tool_key, values)
        self.running = True
        threading.Thread(target=self._run, args=(cmd,), daemon=True).start()

    def _run(self, cmd):
        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            for line in proc.stdout:
                _push(self.window, "tool_log", {"tool": self.tool_key, "line": line})
            proc.wait()
            _push(self.window, "tool_done", {"tool": self.tool_key, "ok": proc.returncode == 0})
        except Exception:
            log_path = log_exception(f"StandaloneToolJob[{self.tool_key}]._run")
            _push(self.window, "tool_log", {"tool": self.tool_key, "line": f"\nInternal error -- see {log_path}\n"})
            _push(self.window, "tool_done", {"tool": self.tool_key, "ok": False})
        finally:
            self.running = False
