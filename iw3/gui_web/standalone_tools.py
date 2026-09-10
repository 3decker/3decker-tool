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
from .gpu_query import query_nvidia_smi_gpu_names, device_choice_to_gpu_id

# Real suggested values for tool fields create_parser() alone can't tell us
# (a free-typed str/int field with no argparse choices= restriction, but a
# real EditableComboBox in gui.py still offers real suggestions) -- found by
# reading each tool's actual wx control (ADR-089, a second, more targeted
# pass after ADR-088's main-schema-focused one). Keyed by (tool_key, field
# name); deliberately small and inline here rather than a separate file
# like field_tooltips.py, given its size.
TOOL_FIELD_CHOICES = {
    ("audiomux", "language"): ["en", "es", "fr", "de", "it", "pt", "ru", "ja",
                                "ko", "zh", "nl", "sv", "no", "da", "pl", "tr", "ar", "hi"],
    ("sharpen", "sharpen_strength"): ["0.25", "0.5", "0.75", "1.0"],
    # gui.py presents this as label->value pairs ("H.264 (default)"->None,
    # "H.265/HEVC -- libx265 (CPU)"->"libx265", "...hevc_nvenc (GPU)"->
    # "hevc_nvenc") via wx.ComboBox.Append(label, clientData) -- this
    # schema's select widget only supports same-string value/label pairs, so
    # "" (meaning "don't pass --video-codec at all", i.e. rife_cli's own
    # default) stands in for the "H.264 (default)" entry.
    ("rife_standalone", "video_codec"): ["", "libx265", "hevc_nvenc"],
}


def _rife_standalone_gpu_choices():
    # Real wx control (gui.py's RIFE standalone GPU dropdown) offers each
    # real device plus CPU, but deliberately NOT "All CUDA Device" -- this
    # tool runs as a single subprocess against rife_cli.py's own --gpu,
    # which takes one plain int (no nargs="+" list, unlike the main iw3
    # CLI's --gpu), so there's no multi-GPU split to offer here. Reuses the
    # same nvidia-smi-subprocess query the main Device field's dynamic
    # choices already use (never torch.cuda.*, ADR-034/071/075).
    names = query_nvidia_smi_gpu_names() or []
    return [f"{i}:{name}" for i, name in enumerate(names)] + ["CPU"]


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
        choices = [str(c) for c in action.choices] if action.choices else []
        if tool_key == "rife_standalone" and action.dest == "gpu":
            widget = "select"
            choices = _rife_standalone_gpu_choices()
        else:
            override = TOOL_FIELD_CHOICES.get((tool_key, action.dest))
            if override:
                choices = override
        fields.append({
            "name": action.dest,
            "label": action.dest.replace("_", " ").title(),
            "widget": widget,
            "value_type": value_type,
            "choices": choices,
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
        if tool_key == "rife_standalone" and dest == "gpu" and isinstance(value, str) and (":" in value or value == "CPU"):
            # Convert the dropdown's "0:NVIDIA GeForce RTX 5090"/"CPU" value
            # back into the plain int rife_cli.py's own --gpu expects (no
            # nargs="+" list, unlike the main iw3 CLI) -- device_choice_to_
            # gpu_id() returns a single-element list ([0], [-1]) since this
            # tool's own choices never include "All CUDA Device".
            value = device_choice_to_gpu_id(value)[0]
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
