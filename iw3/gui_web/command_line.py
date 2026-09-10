"""Copy Command / Import Command -- builds/parses a real `python -m iw3 ...`
string from GUI settings, the same idea as iw3/gui.py's ADR-074 feature and
built the same way: diff the real args.Namespace against create_parser()'s
own defaults, so the printed command can never drift from what create_parser()
actually defines. Reuses worker.build_args() directly rather than
re-implementing Namespace construction a second time.
"""
import argparse
import shlex
import subprocess

from iw3.utils import create_parser
from .schema import FIELDS
from .worker import build_args

_SPECIAL_CASED = {"stereo_format", "anaglyph_method", "metadata", "exif_transpose", "fp16", "device"}


def build_cli_command(settings):
    """settings: the same dict worker.build_args() takes. Returns a real,
    minimal `python -m iw3 ...` command line string -- only flags that
    differ from create_parser()'s own defaults are included."""
    args = build_args(settings)
    parser = create_parser(required_true=False)
    defaults = parser.parse_args([])

    parts = []
    for action in parser._actions:
        if not action.option_strings or action.dest in ("help", "input", "output"):
            continue
        dest = action.dest
        value = getattr(args, dest, None)
        default_value = getattr(defaults, dest, None)
        if value == default_value:
            continue
        flag = next((o for o in action.option_strings if o.startswith("--")), action.option_strings[0])
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                parts.append(flag)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                parts.append(flag)
        elif isinstance(value, (list, tuple)):
            if value:
                parts.append(flag)
                parts.extend(str(v) for v in value)
        elif value is None:
            continue
        else:
            parts.append(flag)
            parts.append(str(value))

    argv = ["-i", settings["input"], "-o", settings["output"]] + parts
    return subprocess.list2cmdline(["python", "-m", "iw3"] + argv)


def parse_cli_command(text):
    """Reverse of build_cli_command(). Returns (settings_dict, None) on
    success or (None, error_message) on failure -- mirrors gui.py's
    parse_cli_command_text()'s own (None, error) contract rather than
    raising, so a bad paste shows a message instead of crashing.

    Known simplification vs. gui.py's own _split_windows_command_line():
    uses Python's shlex in non-POSIX mode, a reasonable but not
    byte-identical approximation of cmd.exe quoting rules."""
    text = text.strip()
    for prefix in ("python -m iw3", "python.exe -m iw3", "python3 -m iw3"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    try:
        tokens = shlex.split(text, posix=False)
        tokens = [t.strip('"') for t in tokens]
    except ValueError as e:
        return None, f"Could not parse the command line: {e}"

    parser = create_parser(required_true=False)
    try:
        args = parser.parse_args(tokens)
    except SystemExit:
        return None, "Could not parse the command line -- check it's a real `python -m iw3 ...` command."

    if not args.input or not args.output:
        return None, "The command must include both -i/--input and -o/--output."

    settings = {"input": args.input, "output": args.output}
    for f in FIELDS:
        if f.name in _SPECIAL_CASED or f.cli_arg is None:
            continue
        dest = f.cli_arg.lstrip("-").replace("-", "_")
        value = getattr(args, dest, None)
        if value is None:
            continue
        if f.value_type in ("int_list", "str_list") and isinstance(value, (list, tuple)):
            settings[f.name] = " ".join(str(v) for v in value)
        else:
            settings[f.name] = value

    settings["exif_transpose"] = not getattr(args, "disable_exif_transpose", False)
    settings["fp16"] = not getattr(args, "disable_amp", False)
    settings["metadata"] = getattr(args, "metadata", None) == "filename"

    if getattr(args, "anaglyph", None):
        settings["stereo_format"] = "anaglyph"
        settings["anaglyph_method"] = args.anaglyph
    elif getattr(args, "half_sbs", False):
        settings["stereo_format"] = "half_sbs"
    elif getattr(args, "tb", False):
        settings["stereo_format"] = "full_tb"
    elif getattr(args, "half_tb", False):
        settings["stereo_format"] = "half_tb"
    elif getattr(args, "vr180", False):
        settings["stereo_format"] = "vr180"
    elif getattr(args, "cross_eyed", False):
        settings["stereo_format"] = "cross_eyed"
    elif getattr(args, "rgbd", False):
        settings["stereo_format"] = "rgbd"
    elif getattr(args, "half_rgbd", False):
        settings["stereo_format"] = "half_rgbd"
    # else: leave stereo_format unset -- the frontend keeps whatever is
    # currently selected, same as Device below.

    # Device is deliberately NOT reconstructed here: args.gpu is just a list
    # of ids ([-1], [-2], [0], ...) with no display name attached, and the
    # dropdown's real choices (GPU names) are only known client-side from
    # its own earlier nvidia-smi query -- reconstructing "0:<name>" here
    # would require re-querying nvidia-smi a second time for a cosmetic
    # label. Left unset; the frontend keeps its current Device selection.

    return settings, None
