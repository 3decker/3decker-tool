"""Background conversion job orchestration for the web GUI (ADR-081).

Builds a real argparse.Namespace the exact way iw3/gui.py's wx GUI already
does -- create_parser(required_true=False), then override only the fields
this UI manages, leaving every other flag at its real parser default -- and
runs iw3_main() on a background thread, exactly like the wx GUI (never a
subprocess for the main pipeline).

Progress reaches the frontend via WebTQDM/_stage_fn, the evaluate_js analog
of nunif.gui.common.TQDMGUI/gui.py's StageChangeEvent -- same three-method
shape and same two channels (fine-grained tqdm updates, coarse stage-name
events), just pushed with window.evaluate_js(...) from a background thread
instead of wx.PostEvent. Confirmed in the Phase 0 spike that evaluate_js is
safe to call from a non-pywebview-event thread.

Cancel/Suspend use the same threading.Event pair and the same (initially
counterintuitive) inverted Suspend semantics as the wx GUI: suspend_event
SET means running, CLEAR means paused; Cancel sets both (so a paused job's
suspend_event.wait() unblocks and it can then see stop_event and exit).
"""
import functools
import json
import threading

from iw3.utils import create_parser, set_state_args, iw3_main
from nunif.utils.video import pyav_init_cuda_primary_context

from .schema import FIELDS
from .crash_log import log_exception
from .gpu_query import device_choice_to_gpu_id

_cuda_context_initialized = False
_cuda_context_lock = threading.Lock()


def ensure_cuda_context():
    # Deferred from import time -- only called right before real GPU/video
    # work begins (ConversionJob.start()), matching iw3/__main__.py's own
    # timing and the wx GUI's ensure_cuda_context(). Must run before any
    # torch.cuda.* call that actually claims a CUDA context, or hwaccel
    # decode + torch.compile crashes 100% of the time (ADR-034/071/075).
    global _cuda_context_initialized
    with _cuda_context_lock:
        if not _cuda_context_initialized:
            pyav_init_cuda_primary_context()
            _cuda_context_initialized = True


def _push(window, channel, payload):
    try:
        window.evaluate_js(
            f"window.__iw3PushEvent && window.__iw3PushEvent("
            f"{json.dumps(channel)}, {json.dumps(payload)});"
        )
    except Exception:
        # A push failing (e.g. the window was already closed) must never
        # take down the conversion thread itself.
        log_exception("worker._push evaluate_js failed")


class WebTQDM:
    """evaluate_js analog of nunif.gui.common.TQDMGUI. Constructed exactly
    the same way (tqdm_fn(desc=, total=, ncols=) -> an object with
    .update(n=1)/.close())."""

    def __init__(self, window, **kwargs):
        self.window = window
        self.desc = kwargs.get("desc") or ""
        _push(window, "progress", {"type": "init", "total": kwargs.get("total"), "desc": self.desc})

    def update(self, n=1):
        _push(self.window, "progress", {"type": "update", "n": n, "desc": self.desc})

    def close(self):
        _push(self.window, "progress", {"type": "close", "desc": self.desc})


def _stage_fn(window, name):
    _push(window, "stage", {"name": name})


# stereo_format doesn't map 1:1 to a CLI arg -- see schema.py's own
# docstring. Exactly one of these mutually-exclusive iw3 output flags (or
# none, for iw3's own default Full SBS) is ever set True. "full_sbs" is
# deliberately absent from this dict -- it's the real default that results
# from leaving every flag False, not a flag of its own; _apply_stereo_format
# below falls through to that for it (and for any other unrecognized value).
_FORMAT_TO_FLAG = {
    "half_sbs": "half_sbs",
    "full_tb": "tb",
    "half_tb": "half_tb",
    "vr180": "vr180",
    "cross_eyed": "cross_eyed",
    "rgbd": "rgbd",
    "half_rgbd": "half_rgbd",
}


def _apply_stereo_format(args, stereo_format, anaglyph_method):
    for flag in ("half_sbs", "tb", "half_tb", "vr180", "cross_eyed", "rgbd", "half_rgbd"):
        setattr(args, flag, False)
    args.anaglyph = None

    if stereo_format == "anaglyph":
        args.anaglyph = anaglyph_method or "dubois"
    elif stereo_format in _FORMAT_TO_FLAG:
        setattr(args, _FORMAT_TO_FLAG[stereo_format], True)
    # else (unrecognized/empty): leave every flag False -- iw3's own default.


def _coerce(value_type, raw):
    if value_type == "float":
        return float(raw)
    if value_type == "int":
        return int(float(raw))  # tolerates a select value arriving as "2.0"-ish text
    if value_type == "bool":
        return bool(raw)
    if value_type == "int_list":
        # "2 1" or "2,1" -> [2, 1] -- the real --edge-dilation/--inpaint-
        # overlap-frames style args (nargs="+") take 1+ ints this way.
        parts = str(raw).replace(",", " ").split()
        return [int(p) for p in parts]
    if value_type == "str_list":
        # "film, grain" or "film grain" -> ["film", "grain"] -- --tune's
        # nargs="+" shape.
        parts = str(raw).replace(",", " ").split()
        return parts
    if value_type == "gpu_id":
        # "0:NVIDIA GeForce RTX 5090" -> [0], "CPU" -> [-1], "All CUDA
        # Device" -> [0, 1, ...]. Always a list -- set_state_args() does
        # args.gpu[0] and `for gpu_id in args.gpu` unconditionally, so a
        # bare scalar breaks it (confirmed by direct test). Matches
        # gui.py's own real convention exactly (gui.py:5586-5591).
        return device_choice_to_gpu_id(raw)
    return raw


def build_args(settings):
    """settings: dict from JS (schema field name -> raw JSON value), plus
    "input"/"output" path strings. Real coercion per field's value_type,
    since e.g. divergence is a free-typed field on the JS side, not a
    native numeric widget -- mirrors how the wx GUI's own EditableComboBox
    fields get parsed before being placed on the Namespace."""
    parser = create_parser(required_true=False)
    args = parser.parse_args([])

    args.input = settings["input"]
    args.output = settings["output"]

    for f in FIELDS:
        if f.cli_arg is None:
            continue  # special-cased fields, applied below
        dest = f.cli_arg.lstrip("-").replace("-", "_")
        raw = settings.get(f.name, f.default)
        if raw is None or raw == "":
            continue
        setattr(args, dest, _coerce(f.value_type, raw))

    _apply_stereo_format(
        args, settings.get("stereo_format", "half_sbs"), settings.get("anaglyph_method"))
    # --metadata is an optional-value flag (nargs="?", const="filename"),
    # not a plain boolean -- the schema models it as a checkbox and this
    # turns that back into the real flag's shape.
    args.metadata = "filename" if settings.get("metadata") else None
    # exif_transpose/fp16 are both modeled as their natural on/off sense in
    # the schema, but the real flags are their inverses (matches gui.py's
    # own disable_exif_transpose=not chk_exif_transpose.GetValue() and
    # disable_amp=not chk_fp16.GetValue()). Default True if the field is
    # somehow absent from settings, matching this GUI's own field defaults.
    args.disable_exif_transpose = not settings.get("exif_transpose", True)
    args.disable_amp = not settings.get("fp16", True)
    return args


class ConversionJob:
    """Owns the one conversion that can be in flight at a time for this
    window -- exactly one, matching the wx GUI's own single-job model."""

    def __init__(self, window):
        self.window = window
        self.stop_event = threading.Event()
        self.suspend_event = threading.Event()
        self.suspend_event.set()  # set = running (inverted; see module docstring)
        self.thread = None
        self.running = False

    def start(self, settings):
        if self.running:
            raise RuntimeError("a conversion is already running")
        ensure_cuda_context()
        args = build_args(settings)
        self.stop_event.clear()
        self.suspend_event.set()
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(args,), daemon=True)
        self.thread.start()

    def _run(self, args):
        try:
            set_state_args(
                args,
                stop_event=self.stop_event,
                suspend_event=self.suspend_event,
                tqdm_fn=functools.partial(WebTQDM, self.window),
                stage_fn=functools.partial(_stage_fn, self.window),
            )
            iw3_main(args)
            _push(self.window, "done", {"ok": True})
        except Exception as e:
            log_path = log_exception("ConversionJob._run")
            _push(self.window, "done", {"ok": False, "error": str(e), "log_path": log_path})
        finally:
            self.running = False

    def cancel(self):
        self.stop_event.set()
        self.suspend_event.set()  # unblock a paused wait so it can see the stop

    def suspend(self):
        self.suspend_event.clear()

    def resume(self):
        self.suspend_event.set()
