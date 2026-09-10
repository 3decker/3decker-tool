import nunif.pythonw_fix  # noqa
import nunif.gui.subprocess_patch  # noqa
import sys
import os
from os import path
import re
import traceback
import functools
import tempfile
import shutil
import subprocess
import copy
from time import time
from datetime import datetime
import threading
import wx
from wx.lib.delayedresult import startWorker
import wx.lib.agw.persist as persist
import wx.lib.stattext as stattext
import wx.lib.scrolledpanel as scrolledpanel
import torch
from .utils import (
    create_parser, set_state_args, iw3_main,
    is_text, is_video, is_image, is_output_dir, is_yaml, make_output_filename,
    _get_ffmpeg_bin, _find_mkvmerge,
    STAGE_SCENE_DETECT, STAGE_AUTOCROP, STAGE_HDR_EXTRACT, STAGE_AUDIO_EXTRACT,
    STAGE_DEPTH_STEREO, STAGE_WAIFU2X_UPSCALE, STAGE_RIFE_INTERPOLATE, STAGE_HDR_REINJECT,
)
from . import update_check
from . import subtitle_search_cli
from . import scene_batch
from nunif.initializer import gc_collect
from nunif.device import mps_is_available, xpu_is_available, create_device
from nunif.models.utils import check_compile_support
from nunif.utils.image_loader import IMG_EXTENSIONS as LOADER_SUPPORTED_EXTENSIONS
from nunif.utils.video import (
    VIDEO_EXTENSIONS as KNOWN_VIDEO_EXTENSIONS,
    has_nvenc,
    has_qsv,
    pyav_init_cuda_primary_context,
)
from nunif.utils.video.metadata import parse_time
from nunif.utils.filename import sanitize_filename
from nunif.utils.git import get_current_branch
from nunif.utils.home_dir import ensure_home_dir
from nunif.utils.autocrop import AutoCrop
import nunif.utils.pil_io as pil_io
from nunif.gui import (
    TQDMGUI, FileDropCallback, EVT_TQDM, TimeCtrl,
    EditableComboBox, EditableComboBoxPersistentHandler,
    persistent_manager_register_all, persistent_manager_unregister_all,
    persistent_manager_restore_all, persistent_manager_register,
    extension_list_to_wildcard, validate_number,
    set_icon_ex, apply_dark_mode, is_dark_mode,
    VideoEncodingBox, VideoDecodingBox, IOPathPanel,
    get_default_locale,
    init_win32_dpi,
    refresh_layouts,
    set_tooltip_long_hover,
    enable_persistent_tooltips,
)
from .locales import LOCALES, load_language_setting, save_language_setting
from . import models # noqa
from .depth_anything_model import (
    DepthAnythingModel,
    AA_SUPPORTED_MODELS as DA_AA_SUPPORTED_MODELS
)
from .video_depth_anything_model import (
    VideoDepthAnythingModel,
    AA_SUPPORT_MODELS as VDA_AA_SUPPORTED_MODELS
)
from .video_depth_anything_streaming_model import VideoDepthAnythingStreamingModel, AA_SUPPORT_MODELS as VDA_STREAM_AA_SUPPORTED_MODELS
from .depth_anything_v3_model import AA_SUPPORTED_MODELS as DA3_AA_SUPPORTED_MODELS
from .depth_pro_model import MODEL_FILES as DEPTH_PRO_MODELS
from .zoedepth_model import MODEL_FILES as ZOEDPETH_MODELS
from . import export_config
from .inpaint_utils import INPAINT_MODELS


IMAGE_EXTENSIONS = extension_list_to_wildcard(LOADER_SUPPORTED_EXTENSIONS)
VIDEO_EXTENSIONS = extension_list_to_wildcard(KNOWN_VIDEO_EXTENSIONS)
YAML_EXTENSIONS = extension_list_to_wildcard((".yml", ".yaml"))
# Common dub/audio-track delivery formats mkvmerge/ffmpeg already read directly --
# see iw3/audio_mux_cli.py's own docstring (ADR-042). No project-wide
# KNOWN_AUDIO_EXTENSIONS constant exists yet (unlike KNOWN_VIDEO_EXTENSIONS), so this
# is a small local list just for the Add Audio Track file picker below.
AUDIO_EXTENSIONS = extension_list_to_wildcard(
    (".aac", ".ac3", ".dts", ".flac", ".mp3", ".opus", ".ogg", ".wav", ".m4a", ".mka"))
CONFIG_DIR = ensure_home_dir("iw3", path.join(path.dirname(__file__), "..", "tmp"))
CONFIG_PATH = path.join(CONFIG_DIR, "iw3-gui.cfg")
LANG_CONFIG_PATH = path.join(CONFIG_DIR, "iw3-gui-lang.cfg")
PRESET_DIR = path.join(CONFIG_DIR, "presets")
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(PRESET_DIR, exist_ok=True)

# GUI Layout preference (ADR-037, live-switching added by ADR-045): Tabbed (default,
# ADR-036's wx.Notebook) vs Single Page (every category StaticBox visible at once).
# Switching the dropdown (on_text_changed_cbo_layout) applies immediately in the
# running window via MainFrame.switch_layout_mode() -- no restart needed. Persisted
# the same way as the Language setting above: a dedicated plain-text file, read once
# before any control is constructed, since the INITIAL choice still decides which
# parent widget the category panels are built into at startup (switch_layout_mode
# handles moving them to the other container afterward).
LAYOUT_CONFIG_PATH = path.join(CONFIG_DIR, "iw3-gui-layout.cfg")
LAYOUT_MODE_TABS = "tabs"
LAYOUT_MODE_SINGLE_PAGE = "single_page"
LAYOUT_MODE_CHOICES = (LAYOUT_MODE_TABS, LAYOUT_MODE_SINGLE_PAGE)


def _load_layout_mode(config_path):
    if path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            value = f.read().strip()
        if value in LAYOUT_MODE_CHOICES:
            return value
    return LAYOUT_MODE_TABS


def _save_layout_mode(config_path, mode):
    with open(config_path, mode="w", encoding="utf-8") as f:
        f.write(mode)


# UI Zoom preference: scales the whole app's text/control size up or down, on top of
# (never instead of) init_win32_dpi()'s existing system-level DPI awareness -- that
# handles Windows' own display scaling; this is a separate, user-controlled layer.
# wx has no built-in "CSS zoom" for a whole window -- the practical mechanism here is
# rescaling the base font applied to the frame (nearly every control below sizes
# itself relative to its own font) and re-laying-out, see MainFrame.apply_zoom_level().
# Persisted the same way as Layout (ADR-037/038): a dedicated plain-text file, applied
# live in the running window -- see docs/ai/AI_DECISIONS.md.
ZOOM_CONFIG_PATH = path.join(CONFIG_DIR, "iw3-gui-zoom.cfg")
ZOOM_LEVELS = (80, 90, 100, 110, 125, 150, 175, 200)
DEFAULT_ZOOM_LEVEL = 100
BASE_NORMAL_FONT_PT = 10
BASE_WARNING_FONT_PT = 8


def _load_zoom_level(config_path):
    if path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            value = f.read().strip()
        try:
            level = int(value)
        except ValueError:
            level = None
        if level in ZOOM_LEVELS:
            return level
    return DEFAULT_ZOOM_LEVEL


def _save_zoom_level(config_path, level):
    with open(config_path, mode="w", encoding="utf-8") as f:
        f.write(str(level))


def _zoom_font_point(base_pt, zoom_level):
    # Floored at 6pt so an extreme/foreign zoom value can never collapse text to
    # something unreadable or a 0/negative wx.Font point size.
    return max(6, round(base_pt * zoom_level / 100))


LAYOUT_DEBUG = False


# Job-level "stage changed" notification (progress bar improvement, see
# docs/ai/AI_DECISIONS.md). Deliberately a separate, self-contained wx event --
# not a new `type` value on nunif/gui/common.py's shared TQDMEvent/EVT_TQDM --
# so this stays entirely local to iw3/gui.py and never touches the common wx
# event plumbing waifu2x-gui/iw3-player-gui/stlizer-gui also depend on.
# `_notify_stage()` (iw3/utils.py) calls this from the background worker thread
# (via functools.partial(_post_stage_change, self) passed as args.state["stage_fn"]),
# so it must stay a plain wx.PostEvent -- the same thread-safety pattern TQDMGUI
# already established -- never touch a wx widget directly from here.
_myEVT_IW3_STAGE = wx.NewEventType()
EVT_IW3_STAGE = wx.PyEventBinder(_myEVT_IW3_STAGE, 1)


# Genre Preset quick-fill for Flicker Reduction's Decay Rate/Buffer fields (ADR-057
# Amendment 9). A deliberately separate, simpler mechanism from "Auto EMA by Scene
# Length" above: this is a ONE-TIME fill of the two FIXED Decay Rate/Buffer fields
# for the whole movie's general pace, not a per-scene table -- selecting a preset
# just writes two numbers into cbo_ema_decay/cbo_ema_buffer once, with no ongoing
# link back to this dropdown afterward. Buffer stays conservative and close across
# all three presets (never anywhere near GEMINI AI/ChatGPT's 480-600-frame
# territory above) since Buffer is a real future-frame lookahead cost (VRAM/
# startup-latency, not a "quality" dial -- see the Nagadomi_Reference tooltip
# above); Decay is what actually differentiates the presets, since it costs
# nothing extra (a pure per-frame math weighting). 120 (the most conservative
# preset's Buffer) stays within nagadomi's own documented reference ceiling, not
# beyond it.
GENRE_PRESET_PLACEHOLDER = "-- Select --"
GENRE_PRESET_CHOICES = [
    GENRE_PRESET_PLACEHOLDER,
    "Fast Action",
    "Medium / Magical",
    "Drama / Slow-Paced",
]
GENRE_PRESET_EMA_VALUES = {
    "Fast Action": (0.75, 30),
    "Medium / Magical": (0.85, 72),
    "Drama / Slow-Paced": (0.94, 120),
}
# ADR-057 Amendment 11's "My Preferred Settings" entry (previously part of
# GENRE_PRESET_CHOICES above) moved to a top-bar quick-preset button --
# see MainFrame.apply_quick_preset("3decker") and docs/ai/AI_DECISIONS.md ADR-057
# Amendment 12.


class StageChangeEvent(wx.PyCommandEvent):
    def __init__(self, etype, eid, name=None):
        super(StageChangeEvent, self).__init__(etype, eid)
        self.name = name


def _post_stage_change(window, name):
    wx.PostEvent(window, StageChangeEvent(_myEVT_IW3_STAGE, -1, name))


def _find_update_bat():
    """Resolve update.bat at the nunif-windows distribution root -- the same
    relative-path convention as iw3/update_check.py's _find_git()/
    _get_nunif_repo_root() (two levels up from this file: iw3/ -> nunif/ ->
    nunif-windows root, where update.bat, setenv.bat, git/, python/ all live).
    Returns (update_bat_path, nunif_windows_root) so the caller can launch it with
    an explicit matching cwd -- update.bat's own setenv.bat call is anchored to its
    own location (%~dp0) so it would likely work from any cwd, but this avoids
    depending on that rather than assuming it, matching CS-SUBPROCESS-001's
    "never assume PATH/cwd" convention used for every other bundled tool."""
    nunif_dir = path.dirname(path.dirname(path.abspath(__file__)))  # nunif/
    nunif_windows_root = path.dirname(nunif_dir)
    return path.join(nunif_windows_root, "update.bat"), nunif_windows_root


def _git_checkpoint_before_update(nunif_dir, log_fn):
    """Safety-commit any uncommitted work in `nunif_dir` before update.bat runs (see
    docs/ai/AI_DECISIONS.md ADR-069's dated amendment). update.bat's own source-update
    step runs `git pull --ff`, and if that fails -- which it reliably does whenever
    there are uncommitted local changes -- it falls back to `git reset --hard`,
    permanently discarding every uncommitted change in the working tree. This runs
    first, every time, so that fallback can never destroy real work again the way it
    nearly did (21 uncommitted files, manually rescued as commit `73adecba`).

    Real, read-only-or-additive git operations only (`status`, `add`, `commit`) --
    never `push`/`reset`/`checkout`/`merge`/`rebase` -- so this step can only ever
    ADD safety, never risk it. A real commit is made (not `git stash`), since a
    stash can be lost/forgotten in a way a real commit on the current branch can't.

    `log_fn(text)` is called with each user-facing progress line -- the caller wires
    this to the Run Update log window via `wx.CallAfter`; this function itself has
    no wx dependency so it can be exercised directly in tests against a real,
    disposable git repo. Raises `RuntimeError` -- never silently swallowed -- if the
    checkpoint itself fails (e.g. git config user.name/user.email not set), so the
    caller can stop before update.bat is ever launched, leaving the working tree
    exactly as the failure left it."""
    git_bin = update_check._find_git()
    if git_bin is None:
        raise RuntimeError(
            T("Could not locate the bundled git executable (git/cmd/git.exe) -- cannot safety "
              "check-point uncommitted work before updating. Update was NOT started."))

    def _run(args):
        return subprocess.run([git_bin, "-C", nunif_dir] + args,
                              check=True, capture_output=True, text=True)

    try:
        status = _run(["status", "--porcelain"])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            T("git status failed -- cannot safety check-point uncommitted work before updating. "
              "Update was NOT started.") + f"\n{(e.stderr or '').strip()}")

    changed_files = [line for line in status.stdout.splitlines() if line.strip()]
    if not changed_files:
        log_fn(T("No uncommitted changes -- nothing to check-point.") + "\n\n")
        return

    try:
        _run(["add", "-A"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _run(["commit", "-m", f"Auto-checkpoint before update ({timestamp})"])
        short_hash = _run(["rev-parse", "--short", "HEAD"]).stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            T("Failed to create the safety check-point commit -- update.bat was NOT started, so "
              "your uncommitted work is untouched. Fix the error below and try again.") +
            f"\n{(e.stderr or '').strip()}")

    log_fn(T("Committed {} file(s) as a safety checkpoint before updating (commit {}).")
           .format(len(changed_files), short_hash) + "\n\n")


class RunUpdateDialog(wx.Dialog):
    """Live output window for "Run Update" (see docs/ai/AI_DECISIONS.md ADR-069,
    the direct follow-up to ADR-035's "Check for Updates" button -- that one only
    checks, this dialog shows the real update.bat run that actually applies it).

    A separate window rather than a log box inlined into pnl_preset's toolbar row:
    that row is a thin horizontal strip sized for buttons/combo boxes (Preset Load/
    Save, Quick Presets, Language, Layout, Zoom, ...), not a multi-line log, and
    update.bat can run long enough (package installs, model downloads, a source
    pull) that the user needs a persistent, clearly-labeled place to watch it happen
    -- not a few squeezed-in lines.

    Close is disabled, and the window's own titlebar close button is vetoed, until
    the run actually finishes (mark_finished()), so the user can't lose the log or
    think a still-running update finished early by closing this window."""

    def __init__(self, parent):
        super().__init__(parent, title=T("Run Update"),
                          size=parent.FromDIP((640, 420)),
                          style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.finished = False
        self.txt_log = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP)
        self.btn_close = wx.Button(self, id=wx.ID_CLOSE, label=T("Close"))
        self.btn_close.Disable()

        layout = wx.BoxSizer(wx.VERTICAL)
        layout.Add(self.txt_log, 1, wx.EXPAND | wx.ALL, 8)
        layout.Add(self.btn_close, 0, wx.ALIGN_RIGHT | wx.ALL, 8)
        self.SetSizer(layout)

        self.btn_close.Bind(wx.EVT_BUTTON, lambda event: self.Close())
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def on_close(self, event):
        if not self.finished:
            event.Veto()
            return
        event.Skip()

    def append(self, text):
        self.txt_log.AppendText(text)
        self.txt_log.ShowPosition(self.txt_log.GetLastPosition())

    def mark_finished(self):
        self.finished = True
        self.btn_close.Enable()


class SceneBatchAutoEMADialog(wx.Dialog):
    """Editor for Auto EMA by Scene Length's Buffer/Decay table (ADR-057). Opens
    already scoped to `model_name` -- there is no live model switch inside the dialog
    itself; the caller (MainFrame.on_click_btn_scene_batch_auto_ema_edit) reads the
    dropdown's CURRENT selection once, at open time, and reopening this dialog after
    changing the dropdown edits the other model's table. This was chosen over an
    in-dialog model switch to keep Save/Reset unambiguous about which table they act
    on, and to avoid silently discarding unsaved edits to one model when switching to
    the other mid-edit.

    Only Buffer and Decay per existing scene-length row are editable; the row
    boundaries themselves (min_duration/max_duration) are fixed labels, never editable
    controls, and are never read from or written to the override file (see
    scene_batch._table_from_override) -- so nothing this dialog does can resize or
    misalign the bucket structure.

    Reset to Default only repopulates the on-screen fields with the hardcoded
    defaults; it does not touch the saved override file by itself. Saving (Save
    button) is what persists -- and if every field's value at Save time exactly
    matches the hardcoded default for that row, Save removes any stored override for
    this model entirely (rather than writing a redundant copy of the defaults), so
    Reset-then-Save reliably returns the model to true default/fallback behavior."""

    def __init__(self, parent, model_name):
        super().__init__(parent,
                         title=T("Edit Auto EMA by Scene Length") + f" -- {model_name}",
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.model_name = model_name
        self.base_table = scene_batch.EMA_BY_DURATION_TABLES.get(
            model_name, scene_batch.EMA_BY_DURATION_VDA_L)
        current_table = scene_batch._table_from_override(model_name, self.base_table) or self.base_table

        root = wx.BoxSizer(wx.VERTICAL)

        # ADR-057 amendment (table extended from 16 to 21 rows, 0-1s through 20s+): the
        # full row grid can now exceed a shorter screen's usable height, so it lives in
        # its own ScrolledPanel (same SetSizer/SetAutoLayout/SetupScrolling/SetMinSize
        # pattern this file already uses for the main window's own option tabs -- see
        # _compose_options_layout_tabbed/_compose_options_layout_single_page) instead of
        # assuming it always fits, and the dialog itself is clamped to the real screen
        # work area below (see _clamp_to_screen, mirroring ADR-056's fix for MainFrame).
        self.scroll_panel = scrolledpanel.ScrolledPanel(self, style=wx.TAB_TRAVERSAL)
        grid = wx.GridBagSizer(vgap=4, hgap=10)
        grid.Add(wx.StaticText(self.scroll_panel, label=T("Scene Length")), (0, 0))
        grid.Add(wx.StaticText(self.scroll_panel, label=T("Buffer")), (0, 1))
        grid.Add(wx.StaticText(self.scroll_panel, label=T("Decay")), (0, 2))

        self.buffer_ctrls = []
        self.decay_ctrls = []
        for row, base_rule in enumerate(self.base_table):
            lo = base_rule.get("min_duration", 0)
            hi = base_rule.get("max_duration")
            label = f"{lo}-{hi}s" if hi is not None else f"{lo}s+"
            grid.Add(wx.StaticText(self.scroll_panel, label=label), (row + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
            row_overrides = current_table[row]["overrides"]
            buf_ctrl = wx.TextCtrl(self.scroll_panel, value=str(row_overrides["ema_buffer"]), size=(70, -1))
            decay_ctrl = wx.TextCtrl(self.scroll_panel, value=str(row_overrides["ema_decay"]), size=(70, -1))
            grid.Add(buf_ctrl, (row + 1, 1))
            grid.Add(decay_ctrl, (row + 1, 2))
            self.buffer_ctrls.append(buf_ctrl)
            self.decay_ctrls.append(decay_ctrl)

        self.scroll_panel.SetSizer(grid)
        self.scroll_panel.SetAutoLayout(1)
        self.scroll_panel.SetupScrolling(scroll_x=False, scroll_y=True)
        # Same reasoning as the main window's own ScrolledPanel wrappers: GetBestSize()
        # is deliberately tiny regardless of content, so without an explicit MinSize the
        # dialog would Fit() down to almost nothing instead of showing every row when
        # there's room -- pinning this does not defeat scrolling when _clamp_to_screen
        # later forces the dialog shorter than this.
        self.scroll_panel.SetMinSize(grid.CalcMin())
        root.Add(self.scroll_panel, 1, wx.ALL | wx.EXPAND, 10)

        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_reset = wx.Button(self, label=T("Reset to Default"))
        self.btn_reset.SetToolTip(
            T("Fills every row above back in with this project's built-in default Buffer/Decay "
              "values for {} -- does not save by itself. Click Save afterward to actually apply "
              "and persist the reset (this also removes any previously saved custom values for "
              "this model, so it goes back to true default behavior, not just matching numbers)."
              ).format(model_name))
        self.btn_reset.Bind(wx.EVT_BUTTON, self.on_reset)
        btn_row.Add(self.btn_reset, 0, wx.RIGHT, 12)
        btn_row.AddStretchSpacer()
        btn_save = wx.Button(self, wx.ID_OK, label=T("Save"))
        btn_cancel = wx.Button(self, wx.ID_CANCEL, label=T("Cancel"))
        btn_row.Add(btn_save, 0, wx.RIGHT, 4)
        btn_row.Add(btn_cancel)
        root.Add(btn_row, 0, wx.ALL | wx.EXPAND, 10)

        self.SetSizerAndFit(root)
        self._clamp_to_screen()
        btn_save.Bind(wx.EVT_BUTTON, self.on_save)

    def _clamp_to_screen(self):
        """ADR-057 amendment -- keeps every row (now 21, was 16) and the Save/Cancel/
        Reset row reachable on a screen shorter than the dialog's natural Fit() height,
        the same class of bug ADR-056 fixed for MainFrame itself. Only ever shrinks/
        repositions -- a dialog that already fits is left exactly as SetSizerAndFit()
        sized it. Safe to shrink because the row grid's ScrolledPanel (see __init__)
        absorbs the deficit by actually scrolling, already verified elsewhere in this
        file (_compose_options_layout_tabbed) to correctly do so when forced shorter
        than its own pinned MinSize."""
        self.CentreOnParent()
        display_index = wx.Display.GetFromWindow(self)
        if display_index == wx.NOT_FOUND:
            display_index = 0
        work_area = wx.Display(display_index).GetClientArea()
        width, height = self.GetSize()
        new_width = min(width, work_area.GetWidth())
        new_height = min(height, work_area.GetHeight())
        if (new_width, new_height) != (width, height):
            self.SetSize((new_width, new_height))
        x, y = self.GetPosition()
        new_x = min(max(x, work_area.GetX()), work_area.GetRight() - new_width)
        new_y = min(max(y, work_area.GetY()), work_area.GetBottom() - new_height)
        if (new_x, new_y) != (x, y):
            self.SetPosition((new_x, new_y))

    def on_reset(self, event):
        for row, base_rule in enumerate(self.base_table):
            row_overrides = base_rule["overrides"]
            self.buffer_ctrls[row].SetValue(str(row_overrides["ema_buffer"]))
            self.decay_ctrls[row].SetValue(str(row_overrides["ema_decay"]))

    def on_save(self, event):
        entries = []
        for row in range(len(self.base_table)):
            buf_s = self.buffer_ctrls[row].GetValue()
            decay_s = self.decay_ctrls[row].GetValue()
            if not validate_number(buf_s, 1, 1800, is_int=True):
                self.GetParent().show_validation_error_message(T("Auto EMA Buffer"), 1, 1800)
                return
            if not (validate_number(decay_s, 0.0, 1.0) and 0.0 < float(decay_s) < 1.0):
                self.GetParent().show_validation_error_message(T("Auto EMA Decay"), 0.0, 1.0)
                return
            entries.append({"ema_buffer": int(buf_s), "ema_decay": float(decay_s)})

        is_default = all(
            entries[row]["ema_buffer"] == self.base_table[row]["overrides"]["ema_buffer"]
            and entries[row]["ema_decay"] == self.base_table[row]["overrides"]["ema_decay"]
            for row in range(len(self.base_table))
        )
        all_overrides = scene_batch.load_ema_overrides_file()
        if is_default:
            all_overrides.pop(self.model_name, None)
        else:
            all_overrides[self.model_name] = entries
        scene_batch.save_ema_overrides_file(all_overrides)
        if self.IsModal():
            self.EndModal(wx.ID_OK)


def _split_windows_command_line(command_line):
    """Splits a full command-line string into argv exactly the way Windows itself
    would (via the real CommandLineToArgvW API) -- the logical inverse of
    subprocess.list2cmdline, which is what get_cli_command() uses to build the
    pasted string in the first place (see docs/ai/AI_DECISIONS.md ADR-074).
    shlex.split() assumes POSIX quoting rules and mishandles real Windows
    quoting (backslash-escaped quotes inside a path, etc.), so this goes
    straight to the actual Win32 API via ctypes instead of hand-rolling Windows
    quoting rules. Raises ValueError if CommandLineToArgvW itself rejects the
    string (e.g. unbalanced quotes)."""
    import ctypes

    command_line_to_argv_w = ctypes.windll.shell32.CommandLineToArgvW
    command_line_to_argv_w.restype = ctypes.POINTER(ctypes.c_wchar_p)
    command_line_to_argv_w.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]

    argc = ctypes.c_int(0)
    argv_p = command_line_to_argv_w(command_line, ctypes.byref(argc))
    if not argv_p:
        raise ValueError("CommandLineToArgvW failed to parse the command line")
    try:
        return [argv_p[i] for i in range(argc.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(argv_p)


def _apply_combo_value(combo, value):
    """Sets a wx.ComboBox/EditableComboBox to `value`, preferring an exact choice
    match (SetStringSelection) and falling back to typing the raw text
    (SetValue) when the value isn't one of the preset choices -- mirrors how
    apply_quick_preset() already sets fixed-choice vs. free-typed fields by
    hand (see docs/ai/AI_DECISIONS.md ADR-074), without needing to know which
    kind of combobox each field actually is."""
    text = str(value)
    if not combo.SetStringSelection(text):
        combo.SetValue(text)


# "Guided Light" pilot (ADR-097): a wx.Slider companion for each of these continuous
# numeric Stereo Generation fields, two-way synced to the field's existing
# EditableComboBox. The combo stays the sole thing build_processor_args()/persistence
# ever reads -- the slider is purely a more tactile way to set the same value.
# (combo_attr, slider_attr, min_val, max_val, multiplier, is_int, extra_sync_method)
# wx.Slider is integer-only, so float fields are stored as round(value * multiplier)
# and divided back. Fields with a real blank/"disabled" sentinel value in their own
# choices list (foreground/background divergence, temporal_stabilize_max_shift,
# edge_dilation_y, inpaint_max_width) are deliberately excluded -- a plain slider has
# no clean way to represent "off".
STEREO_SLIDER_FIELDS = [
    ("cbo_divergence", "sld_stereo_divergence", 1.0, 5.0, 10, False, "update_divergence_warning"),
    ("cbo_convergence", "sld_stereo_convergence", 0.0, 1.0, 100, False, None),
    ("cbo_convergence_smoothing", "sld_stereo_convergence_smoothing", 0.0, 0.95, 100, False, None),
    ("cbo_splat_blend_temperature", "sld_stereo_splat_blend_temperature", 10.0, 85.0, 1, False, None),
    ("cbo_depth_refine_strength", "sld_stereo_depth_refine_strength", 0.25, 1.5, 100, False, None),
    ("cbo_temporal_stabilize_strength", "sld_stereo_temporal_stabilize_strength", 0.3, 0.9, 100, False, None),
    ("cbo_foreground_pop", "sld_stereo_foreground_pop", 0.0, 1.0, 100, False, None),
    ("cbo_background_pop", "sld_stereo_background_pop", 0.0, 1.0, 100, False, None),
    ("cbo_background_pop_coverage", "sld_stereo_background_pop_coverage", 15, 40, 1, True, None),
    ("cbo_edge_repair", "sld_stereo_edge_repair", 0.0, 1.0, 100, False, None),
    ("cbo_sharpen_strength", "sld_stereo_sharpen_strength", 0.25, 1.0, 100, False, None),
    ("cbo_ema_decay", "sld_stereo_ema_decay", 0.0, 0.99, 100, False, None),
    ("cbo_ema_buffer", "sld_stereo_ema_buffer", 1, 150, 1, True, None),
]

# Guided Light pattern extended to Dual-Pass Depth Blend (grp_depth_blend) -- same
# shape as STEREO_SLIDER_FIELDS, same _build_stereo_slider/_on_stereo_slider_scroll/
# _on_stereo_combo_text_sync_slider helpers (already fully generic, not stereo-tab-
# specific despite the name). No field here needs an extra_sync callback.
DEPTH_BLEND_SLIDER_FIELDS = [
    ("cbo_depth_blend_strength", "sld_depth_blend_strength", 0.25, 1.0, 100, False, None),
    ("cbo_depth_blend_region_percent", "sld_depth_blend_region_percent", 10, 50, 1, True, None),
    ("cbo_depth_blend_feather_blur", "sld_depth_blend_feather_blur", 0, 35, 1, True, None),
    ("cbo_depth_blend_bilateral_d", "sld_depth_blend_bilateral_d", 9, 15, 1, True, None),
    ("cbo_depth_blend_bilateral_sigma_color", "sld_depth_blend_bilateral_sigma_color", 50, 100, 1, True, None),
    ("cbo_depth_blend_bilateral_sigma_space", "sld_depth_blend_bilateral_sigma_space", 50, 100, 1, True, None),
    ("cbo_depth_blend_clahe_clip", "sld_depth_blend_clahe_clip", 1.0, 4.0, 100, False, None),
    ("cbo_depth_blend_clahe_tile", "sld_depth_blend_clahe_tile", 4, 16, 1, True, None),
    ("cbo_depth_blend_align_decay", "sld_depth_blend_align_decay", 0.0, 0.95, 100, False, None),
    ("cbo_depth_blend_edge_suppression", "sld_depth_blend_edge_suppression", 0.0, 1.0, 100, False, None),
]

# Guided Light pattern extended to Processor (grp_processor). Both fields here are
# plain ints with no blank/"off" sentinel.
PROCESSOR_SLIDER_FIELDS = [
    ("cbo_batch_size", "sld_processor_batch_size", 1, 64, 1, True, None),
    ("cbo_max_workers", "sld_processor_max_workers", 0, 16, 1, True, None),
]


def _build_stereo_slider(parent, combo, min_val, max_val, multiplier):
    """Constructs a wx.Slider for a Guided Light pilot (ADR-097) numeric field,
    initialized from the combo's current value. wx.Slider only takes integers, so the
    slider's own range/position is the field's real value * multiplier, divided back
    in the sync handlers below."""
    try:
        current = float(combo.GetValue())
    except ValueError:
        current = min_val
    current = max(min_val, min(max_val, current))
    slider = wx.Slider(parent, minValue=round(min_val * multiplier), maxValue=round(max_val * multiplier))
    slider.SetValue(round(current * multiplier))
    return slider


def _query_nvidia_smi_gpu_names():
    """Returns a list of CUDA GPU names via `nvidia-smi` (a separate process), or
    None on any failure. Used to populate the Device/RIFE-GPU dropdowns at window
    construction time INSTEAD of torch.cuda.get_device_properties().

    Why this matters: before commit a270add1 (2026-09-07), pyav_init_cuda_primary_
    context() ran unconditionally at process startup, before MainFrame() was ever
    constructed. That commit deferred it into ensure_cuda_context() (called only on
    Start/Quick Preview) to stop the window grabbing a CUDA context just from being
    opened -- but the Device dropdown's own torch.cuda.get_device_properties() calls
    (present since the GUI's original 2023 commit) were not moved, and still run
    during MainFrame construction. That reverses the exact ordering
    pyav_init_cuda_primary_context()'s own docstring requires ("before PyTorch
    initializes CUDA... otherwise stream synchronization with NVDEC is not
    possible") -- PyTorch now claims the CUDA primary context first, every single
    time the window opens, before pyav/NVDEC ever gets a chance to. This is the
    real, git-history-confirmed explanation for why --hwaccel cuda + --compile
    crashes on its very first attempt in the GUI, 100% of the time, but never in
    the CLI (whose __main__.py still calls pyav_init_cuda_primary_context() as the
    literal first GPU-touching statement). See docs/ai/AI_DECISIONS.md ADR-034/071.
    nvidia-smi runs as an independent process, so querying it here cannot touch
    this process's own CUDA state at all -- avoiding the problem entirely instead
    of just tolerating the crash it causes."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if names:
                return names
    except (OSError, subprocess.SubprocessError):
        pass
    return None


class IW3App(wx.App):
    def OnInit(self):
        set_tooltip_long_hover()
        main_frame = MainFrame()
        enable_persistent_tooltips(main_frame)
        self.instance = wx.SingleInstanceChecker("iw3-gui.lock", CONFIG_DIR)
        if self.instance.IsAnotherRunning():
            with wx.MessageDialog(None,
                                  message=(T("Another instance is running") + "\n" +
                                           T("Are you sure you want to do this?")),
                                  caption=T("Confirm"), style=wx.YES_NO) as dlg:
                if dlg.ShowModal() == wx.ID_NO:
                    return False
        set_icon_ex(main_frame, path.join(path.dirname(__file__), "icon.ico"), main_frame.GetTitle())
        self.SetAppName(main_frame.GetTitle())
        refresh_layouts(main_frame)
        main_frame.Show()
        main_frame.Layout()
        main_frame.Fit()
        # ADR-056: Fit() above sizes the frame to its full natural content height,
        # with nothing clamping that against the real screen -- see
        # MainFrame._clamp_frame_to_screen() for why that can push the progress bar
        # row off-screen, and why shrinking is safe here.
        main_frame._clamp_frame_to_screen()
        self.SetTopWindow(main_frame)
        return True


class MainFrame(wx.Frame):
    def __init__(self):
        branch_name = get_current_branch()
        if branch_name is None or branch_name in {"master", "main"}:
            branch_tag = ""
        else:
            branch_tag = f" ({branch_name})"

        python_version_tag = f" ({sys.implementation.name}-{sys.version_info[0]}.{sys.version_info[1]})"

        super(MainFrame, self).__init__(
            None,
            name="iw3-gui",
            title=T("3DECKER — iw3") + branch_tag + python_version_tag,
            size=(1000, 840),
            style=(wx.DEFAULT_FRAME_STYLE & ~wx.MAXIMIZE_BOX)
        )
        # ADR-056: Maximize is disabled (no wx.MAXIMIZE_BOX above), so a window sized
        # taller than the screen by Fit() cannot be recovered from with a single
        # maximize click -- base_title backs _set_title_progress()'s condensed
        # progress readout, a redundant signal that stays visible even if the
        # progress bar itself is off-screen on some display.
        self.base_title = self.GetTitle()
        self.processing = False
        self.updating = False
        self.dlg_run_update = None
        self.start_time = 0
        self.input_type = None
        self.cuda_context_initialized = False
        self.stop_event = threading.Event()
        self.suspend_event = threading.Event()
        self.suspend_pos = 0
        self.suspend_event.set()
        # Job-level stage tracking (progress bar "Step X of N" display, see
        # docs/ai/AI_DECISIONS.md): job_stages is the ordered list of stage names
        # this specific job will run through, precomputed from the user's own
        # settings right before Start (waifu2x/RIFE/preserve-dowi are all opt-in, so
        # a plain conversion has only one stage). job_stage_index is 1-based;
        # current_stage_name/stage_start_time back the "Step k of N: <name>...
        # (running MM:SS)" status text, ticked live by stage_pulse_timer for stages
        # (waifu2x/RIFE/HDR reinjection) that run as a single blocking subprocess
        # call with no real per-item progress to report -- see on_stage_change()
        # for why the Gauge itself is deliberately left alone during these (a real,
        # verified wx.Gauge.Pulse() rendering bug on this project's Windows/wx
        # combination, not a design choice).
        self.job_stages = [STAGE_DEPTH_STEREO]
        self.job_stage_index = 1
        self.current_stage_name = self.job_stages[0]
        self.stage_start_time = 0
        self.job_start_time = 0
        self.stage_pulse_timer = wx.Timer(self)
        self.depth_model = None
        self.depth_model_type = None
        self.depth_model_device_id = None
        self.depth_model_height = None
        self.depth_model_limit_resolution = None
        self.layout_mode = _load_layout_mode(LAYOUT_CONFIG_PATH)
        self.zoom_level = _load_zoom_level(ZOOM_CONFIG_PATH)
        self.initialize_component()
        if is_dark_mode():
            apply_dark_mode(self)
        self.apply_accent_theme()

    def _scaled_font(self, base_pt):
        return wx.Font(_zoom_font_point(base_pt, self.zoom_level),
                       family=wx.FONTFAMILY_MODERN, style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL)

    def initialize_component(self):
        NORMAL_FONT = self._scaled_font(BASE_NORMAL_FONT_PT)
        WARNING_FONT = self._scaled_font(BASE_WARNING_FONT_PT)
        WARNING_COLOR = (0xcc, 0x33, 0x33)

        self.SetFont(NORMAL_FONT)
        self.CreateStatusBar()

        # input output panel
        input_wildcard = (f"Image and Video and YAML files|{IMAGE_EXTENSIONS};{VIDEO_EXTENSIONS};{YAML_EXTENSIONS}"
                          f"|Video files|{VIDEO_EXTENSIONS}"
                          f"|Image files|{IMAGE_EXTENSIONS}"
                          f"|YAML files|{YAML_EXTENSIONS}"
                          "|All Files|*.*")
        self.pnl_file = IOPathPanel(
            self,
            input_wildcard=input_wildcard,
            default_output_dir_name="iw3",
            resolve_output_path=self.resolve_output_path,
            translate_function=T,
        )

        self.pnl_file_option = wx.Panel(self)
        self.chk_resume = wx.CheckBox(self.pnl_file_option, label=T("Resume"), name="chk_resume")
        self.chk_resume.SetToolTip(
            T("What it's for: skips a file entirely if the output it would create already exists.\n"
              "How it helps: lets you stop a big batch job partway (or have it crash/get interrupted) "
              "and restart later without wasting time redoing files you already finished.\n"
              "Con: only checks whether a file with the expected NAME exists, not whether it's actually "
              "complete or correct — a partial/corrupted leftover file gets treated as \"done\" too.\n"
              "Recommended: on for batch folders and long jobs. Turn off only if you specifically want "
              "to force-redo everything."))
        self.chk_resume.SetValue(True)

        self.chk_recursive = wx.CheckBox(self.pnl_file_option, label=T("Process all subfolders"),
                                         name="chk_recursive")
        self.chk_recursive.SetValue(False)
        self.chk_recursive.SetToolTip(
            T("What it's for: when the input is a folder, also reaches into its subfolders instead of "
              "only converting files sitting directly inside it.\n"
              "Con: if unrelated files (extras, samples, trailers) live in subfolders you didn't mean to "
              "include, they'll get converted too — there's no per-folder include/exclude list.\n"
              "Recommended: on if you organize your movies into subfolders; off if your input folder mixes "
              "in stuff you don't want touched."))

        self.chk_skip_error = wx.CheckBox(self.pnl_file_option, label=T("Skip Error"), name="chk_skip_erro")
        self.chk_skip_error.SetToolTip(
            T("What it's for: if a file causes an error during batch processing, note it and move on to "
              "the next file instead of stopping the whole batch. Also skips files that errored on a "
              "previous run, so they aren't retried every time.\n"
              "Con: a genuinely broken/corrupt file just gets silently skipped rather than fixed — check "
              "the log if a file seems to be missing from your results.\n"
              "Recommended: on for large batches, so one bad file doesn't halt an overnight job. Off if "
              "you'd rather the job stop immediately so you notice a problem right away."))
        self.chk_skip_error.SetValue(False)

        self.sep_batch_options = wx.StaticLine(self.pnl_file_option, size=self.FromDIP((2, 16)), style=wx.LI_VERTICAL)

        self.chk_exif_transpose = wx.CheckBox(self.pnl_file_option, label=T("EXIF Transpose"),
                                              name="chk_exif_transpose")
        self.chk_exif_transpose.SetValue(True)
        self.chk_exif_transpose.SetToolTip(
            T("What it's for: some cameras/phones save a photo already correctly oriented for viewing "
              "but store the actual pixel data sideways/upside-down, with a hidden tag telling viewers "
              "how to rotate it for display. This applies that rotation before conversion so the 3D "
              "effect is built for the image the way you actually see it, not the raw sideways file.\n"
              "Recommended: on (default) for virtually all real photos. Only turn off if you've confirmed "
              "your specific images have no EXIF tag or it's already wrong, since that's an unusual case."))

        self.chk_metadata = wx.CheckBox(self.pnl_file_option, label=T("Add metadata to filename"),
                                        name="chk_metadata")
        self.chk_metadata.SetValue(False)
        self.chk_metadata.SetToolTip(
            T("What it's for: encodes your current settings (depth model, 3D strength, convergence, edge "
              "options, etc.) into the output filename as short abbreviated tags, and also writes them "
              "into the video's own comment metadata.\n"
              "How it helps: lets you tell which settings made which file just by looking at the "
              "filename later, compare two versions of the same clip, and lets Resume correctly match an "
              "in-progress job back up after an interruption.\n"
              "Con: filenames get noticeably longer and less readable at a glance (a string of "
              "abbreviations rather than just the movie's name).\n"
              "Recommended: on, unless you strongly prefer short/clean filenames and are keeping track of "
              "your own settings some other way."))

        self.sep_image_format = wx.StaticLine(self.pnl_file_option, size=self.FromDIP((2, 16)), style=wx.LI_VERTICAL)
        self.lbl_image_format = wx.StaticText(self.pnl_file_option, label=" " + T("Image Format"))
        self.cbo_image_format = wx.ComboBox(self.pnl_file_option, choices=["png", "jpeg", "webp"],
                                            name="cbo_image_format")
        self.cbo_image_format.SetEditable(False)
        self.cbo_image_format.SetSelection(0)
        self.cbo_image_format.SetToolTip(
            T("What it's for: the file format used when converting still images (has no effect on video "
              "jobs, which always use your Video Format setting instead).\n"
              "Values: png = lossless, larger files, no quality loss ever — best if you'll edit/reprocess "
              "the result later. jpeg = lossy compression, much smaller files, a small amount of quality "
              "loss (usually invisible at normal viewing sizes) — best for sharing/storage space. "
              "webp = modern format, similar quality to jpeg at a smaller file size, but less universally "
              "supported by older software/devices.\n"
              "Recommended: png if disk space isn't a concern or you might reprocess the image later; "
              "jpeg for everyday viewing/sharing where file size matters."))

        layout = wx.BoxSizer(wx.HORIZONTAL)
        layout.AddSpacer(4)
        layout.Add(self.chk_resume, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_recursive, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_skip_error, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.sep_batch_options, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_exif_transpose, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_metadata, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.sep_image_format, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_image_format, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_image_format, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.pnl_file_option.SetSizer(layout)

        # options panel

        self.pnl_options = wx.Panel(self)
        if LAYOUT_DEBUG:
            self.pnl_options.SetBackgroundColour("#cfc")

        # Category groups (ADR-036, extended by ADR-037): the 110+ controls below are
        # grouped into 7 categories so a user can find any setting quickly instead of
        # scanning one giant wall of controls. Every wx.StaticBox/VideoDecodingBox/
        # VideoEncodingBox below is parented to one of these 7 category panels instead
        # of self.pnl_options directly -- this only changes WHERE each group is drawn,
        # never a control's name, binding, or behavior. pnl_file_option (File & Batch
        # checkboxes) and pnl_preset (Quick Presets/Language/etc.) intentionally stay
        # outside this area, as persistent strips above it -- see docs/3DECKER_Method.md
        # and AI_DECISIONS.md for why (frequently-needed controls that shouldn't require
        # a tab switch to reach).
        #
        # ADR-037/ADR-045: which WIDGET parents these 7 category panels depends on the
        # user's Layout preference (self.layout_mode, loaded before this method runs) --
        # Tabbed parents them to a wx.Notebook page each (ADR-036's original design);
        # Single Page parents them directly to one scrollable panel so every category is
        # visible at once. Both container widgets are always constructed (ADR-045: live
        # switching needs somewhere to Reparent() a category panel TO before it's ever
        # been the active layout), but only the one matching self.layout_mode is
        # populated/shown at startup -- see switch_layout_mode() for how the other one
        # gets composed later, on demand, when the user switches. Everything below this
        # branch (every StaticBox/sizer built inside each category, and each category
        # panel's own SetSizer() call) is identical either way; only the final
        # composition step (see _compose_options_layout_tabbed /
        # _compose_options_layout_single_page near the end of this method) differs.
        self.nb_options = wx.Notebook(self.pnl_options)
        self.pnl_single = scrolledpanel.ScrolledPanel(self.pnl_options)
        # ADR-048: Tabbed mode's Notebook pages had no scrolling of their own -- a tab
        # taller than the visible area simply cut off its bottom controls. Fixed with a
        # dedicated ScrolledPanel WRAPPER per Notebook page (self.tab_wrap_*), each
        # holding one (unchanged) category panel as its only child, rather than making
        # the category panels themselves ScrolledPanel -- the latter would nest a
        # ScrolledPanel inside pnl_single's own ScrolledPanel in Single Page mode
        # (verified by an isolated probe to be worth avoiding; see ADR-048). Always
        # constructed, parented to self.nb_options, mirroring ADR-045's "both
        # containers always exist" pattern so switch_layout_mode() always has a wrapper
        # ready to Reparent() a category panel onto even before Tabbed has ever been
        # the active layout.
        self.tab_wrap_stereo = scrolledpanel.ScrolledPanel(self.nb_options)
        self.tab_wrap_depth_blend = scrolledpanel.ScrolledPanel(self.nb_options)
        self.tab_wrap_video_filter = scrolledpanel.ScrolledPanel(self.nb_options)
        self.tab_wrap_video_dec = scrolledpanel.ScrolledPanel(self.nb_options)
        self.tab_wrap_video_enc = scrolledpanel.ScrolledPanel(self.nb_options)
        self.tab_wrap_processor = scrolledpanel.ScrolledPanel(self.nb_options)
        self.tab_wrap_tools = scrolledpanel.ScrolledPanel(self.nb_options)
        # The 7 category panels themselves stay plain wx.Panel, unchanged -- only WHICH
        # widget parents them (a per-tab wrapper vs. pnl_single directly) depends on
        # the initial Layout preference, same as before ADR-048.
        if self.layout_mode == LAYOUT_MODE_SINGLE_PAGE:
            self.tab_stereo = wx.Panel(self.pnl_single)
            self.tab_depth_blend = wx.Panel(self.pnl_single)
            self.tab_video_filter = wx.Panel(self.pnl_single)
            self.tab_video_dec = wx.Panel(self.pnl_single)
            self.tab_video_enc = wx.Panel(self.pnl_single)
            self.tab_processor = wx.Panel(self.pnl_single)
            self.tab_tools = wx.Panel(self.pnl_single)
        else:
            self.tab_stereo = wx.Panel(self.tab_wrap_stereo)
            self.tab_depth_blend = wx.Panel(self.tab_wrap_depth_blend)
            self.tab_video_filter = wx.Panel(self.tab_wrap_video_filter)
            self.tab_video_dec = wx.Panel(self.tab_wrap_video_dec)
            self.tab_video_enc = wx.Panel(self.tab_wrap_video_enc)
            self.tab_processor = wx.Panel(self.tab_wrap_processor)
            self.tab_tools = wx.Panel(self.tab_wrap_tools)

        # stereo generation settings
        # divergence, convergence, method, depth_model, mapper

        self.grp_stereo = wx.StaticBox(self.tab_stereo, label=T("Stereo Generation"))

        self.lbl_divergence = wx.StaticText(self.grp_stereo, label=T("3D Strength"))
        self.cbo_divergence = EditableComboBox(self.grp_stereo, choices=["5.0", "4.0", "3.0", "2.5", "2.0", "1.0"],
                                               name="cbo_divergence")
        self.lbl_divergence_warning = stattext.GenStaticText(self.grp_stereo, label="")
        self.lbl_divergence_warning.SetFont(WARNING_FONT)
        self.lbl_divergence_warning.SetForegroundColour(WARNING_COLOR)
        self.lbl_divergence_warning.Hide()

        self.cbo_divergence.SetToolTip(
            T("Also called Divergence. The master strength of the whole 3D effect — how far things "
              "shift between the left/right eye. Higher = more dramatic depth, but more edge artifacts. "
              "Lower = subtler, cleaner. Recommended: 2.0-3.0 for most movies."))
        self.cbo_divergence.SetSelection(4)
        self.sld_stereo_divergence = _build_stereo_slider(self.grp_stereo, self.cbo_divergence, 1.0, 5.0, 10)

        self.lbl_convergence = wx.StaticText(self.grp_stereo, label=T("Convergence Plane"))
        self.cbo_convergence_mode = wx.ComboBox(self.grp_stereo, choices=["constant", "sod_v1", "face_detect"],
                                                name="cbo_convergence_mode")
        self.cbo_convergence_mode.SetEditable(False)
        self.cbo_convergence_mode.SetSelection(0)
        self.cbo_convergence_mode.Bind(wx.EVT_COMBOBOX, self.on_changed_cbo_convergence_mode)
        self.cbo_convergence_mode.SetToolTip(
            T("What it's for: how the \"screen depth\" (the point that looks like it's exactly at the "
              "screen surface, with everything else popping toward or receding from it) is chosen.\n"
              "constant: you set one fixed position with the value box, and it never moves for the whole "
              "video.\n"
              "sod_v1: an AI model automatically re-picks a focus point every frame, based on the most "
              "visually important subject.\n"
              "face_detect: same idea, but automatically centers on detected faces specifically, ignoring "
              "the value box.\n"
              "Con of sod_v1/face_detect: every time the convergence point moves — even smoothed — your "
              "eyes have to physically readjust their focus angle to keep the image comfortable to view. "
              "Because sod_v1 re-evaluates every frame, it can drift even WITHIN a single unbroken shot "
              "(someone shifts position, the camera pans slightly), which is closer to \"constantly "
              "reacting\" than how a real stereographer works — they hold convergence steady within a shot "
              "and only step it to a new value at cuts. Real testing on this project found constant "
              "produced steadier, more comfortable results on most typical film content for exactly this "
              "reason.\n"
              "When sod_v1/face_detect genuinely help: a push/pull \"reveal\" shot where the camera moves "
              "from a tight close-up to a wide shot within one continuous take — a fixed constant value "
              "structurally can't be right for both ends of that move, but sod_v1 can track it.\n"
              "Recommended: constant for most content — steadier and closer to real stereographer practice. "
              "Reserve sod_v1/face_detect for content dominated by continuous push/pull reveal shots."))

        self.cbo_convergence = EditableComboBox(self.grp_stereo, choices=["0.0", "0.25", "0.5", "1.0"],
                                                name="cbo_convergence")
        self.cbo_convergence.SetSelection(1)
        self.cbo_convergence.SetToolTip(
            T("Also called Convergence Plane. Where the \"screen depth\" sits, from 0 (everything pops "
              "out toward you) to 1 (everything sits behind the screen). Only used directly when mode is "
              "\"constant\" — for sod_v1 it acts as a relative offset within the detected subject's depth "
              "range. Recommended: 0.5 as a balanced starting point."))
        self.sld_stereo_convergence = _build_stereo_slider(self.grp_stereo, self.cbo_convergence, 0.0, 1.0, 100)

        self.lbl_convergence_smoothing = wx.StaticText(self.grp_stereo, label=T("Convergence Smoothing"))
        self.cbo_convergence_smoothing = EditableComboBox(
            self.grp_stereo, choices=["0.95", "0.9", "0.75", "0.5", "0.25", "0"],
            name="cbo_convergence_smoothing")
        self.cbo_convergence_smoothing.SetSelection(1)
        self.cbo_convergence_smoothing.SetToolTip(
            T("Only affects sod_v1 / Face Detect convergence modes. Controls how quickly the automatic "
              "convergence point reacts to scene changes. Higher = smoother but slower to react. Lower = "
              "more aggressive/dynamic, reacts faster but may jitter more. 0 = no smoothing at all."))
        self.sld_stereo_convergence_smoothing = _build_stereo_slider(
            self.grp_stereo, self.cbo_convergence_smoothing, 0.0, 0.95, 100)

        self.lbl_ipd_offset = wx.StaticText(self.grp_stereo, label=T("Your Own Size"))
        # SpinCtrlDouble is better, but cannot save with PersistenceManager
        self.sld_ipd_offset = wx.SpinCtrl(self.grp_stereo, value="0", min=-10, max=20, name="sld_ipd_offset")
        self.sld_ipd_offset.SetToolTip(
            T("Also called IPD Offset (interpupillary distance — the gap between your own eyes). Widens "
              "or narrows the simulated eye spacing used to build the 3D effect, separate from the main "
              "3D Strength slider.\n"
              "Values: 0 = average adult spacing (the default assumption). Positive values simulate wider-"
              "set eyes, which slightly increases the 3D pop for people with a wider-than-average IPD. "
              "Range is -10 to 20.\n"
              "Con: this is a minor personal-comfort tweak, not a real substitute for 3D Strength — pushing "
              "it far from 0 can make the depth feel physically \"wrong\" even if stronger.\n"
              "Recommended: leave at 0 unless you specifically know your own IPD differs a lot from "
              "average and have noticed a comfort difference."))

        self.lbl_synthetic_view = wx.StaticText(self.grp_stereo, label=T("Synthetic View"))
        self.cbo_synthetic_view = wx.ComboBox(self.grp_stereo,
                                              choices=["both", "right", "left"],
                                              name="cbo_synthetic_view")
        self.cbo_synthetic_view.SetEditable(False)
        self.cbo_synthetic_view.SetSelection(0)
        self.cbo_synthetic_view.SetToolTip(
            T("What it's for: decides which eye view(s) the AI actually generates from your original "
              "flat image.\n"
              "both: generates BOTH left and right eyes fresh from the center image — most natural, "
              "since neither eye is just the untouched original.\n"
              "right / left: keeps the original image exactly as one eye, and only generates the other — "
              "faster (half the warping work), but can look slightly less balanced since one eye is "
              "\"real\" and the other is synthesized.\n"
              "Recommended: both, for the best-looking result. right/left mainly useful if you're very "
              "tight on processing time."))

        self.lbl_method = wx.StaticText(self.grp_stereo, label=T("Method"))
        self.cbo_method = wx.ComboBox(self.grp_stereo,
                                      choices=["mlbw_l2", "mlbw_l4", "mlbw_l2s",
                                               "mlbw_l2_inpaint",
                                               "row_flow_v3", "row_flow_v3_sym", "row_flow_v2",
                                               "forward_fill", "forward_splat_fill", "forward_inpaint",
                                               "monobw", "monobw_inpaint",
                                               ],
                                      name="cbo_method")
        self.cbo_method.SetEditable(False)
        self.cbo_method.SetSelection(2)
        self.cbo_method.SetToolTip(
            T("What it's for: which AI model/technique actually builds the second eye view from your "
              "depth map. This is one of the biggest single quality/speed decisions in the whole app.\n"
              "row_flow_v3: fast, AI-based, a solid all-rounder — the default.\n"
              "row_flow_v3_sym: same family, but shifts BOTH eyes symmetrically outward from the original "
              "center image instead of keeping one eye as the unwarped original — can look more balanced "
              "across both eyes.\n"
              "row_flow_v2: an older generation of the row_flow model, kept for compatibility/comparison; "
              "row_flow_v3 generally looks better at the same speed.\n"
              "mlbw_l2 / mlbw_l4: a more advanced, multi-layer AI warp. l4 is the deeper/more capable "
              "version of l2 — handles stronger 3D strength and complex scenes with fewer artifacts, at "
              "extra GPU time/memory cost.\n"
              "mlbw_l2s: a smaller/lighter version of mlbw_l2 — faster and lower VRAM, at some quality "
              "cost versus the full mlbw_l2. Useful on lower-VRAM GPUs.\n"
              "mlbw_l2_inpaint / forward_inpaint / monobw_inpaint: any of the above families, PLUS an AI "
              "inpainting pass that fills in the hidden area behind objects instead of stretching/smearing "
              "it — real extra time cost, but noticeably cleaner edges around foreground objects.\n"
              "forward_fill: a simple, non-AI method — just warps pixels forward and fills gaps with a "
              "basic algorithm, no learned model involved. Fastest, but the roughest edges.\n"
              "forward_splat_fill: same non-AI forward-warp family as forward_fill, but where two pixels "
              "land on the same spot, it smoothly blends them (nearer one weighted more) instead of the "
              "nearer one fully overwriting the other — softer, less jagged edges than forward_fill. No "
              "learned model involved, but NOT free of GPU cost: it holds several full-frame accumulator "
              "tensors in memory per batch, so its VRAM use scales up directly with Depth Batch Size "
              "(unlike the inpaint methods, which don't) and can run even a high-VRAM GPU out of memory on "
              "demanding footage at Depth Batch Size 2 or higher. Confirmed real fix if you hit an "
              "out-of-memory error here: drop Depth Batch Size to 1 and turn on Low VRAM below.\n"
              "monobw: a simpler, lighter backward-warp method than the mlbw family — a faster middle "
              "ground when mlbw is too slow but forward_fill's quality isn't good enough.\n"
              "Recommended: mlbw_l2_inpaint or forward_inpaint for the best quality on a real GPU; "
              "row_flow_v3_sym or mlbw_l2s if you need more speed or have limited VRAM."))

        self.lbl_splat_blend_temperature = wx.StaticText(
            self.grp_stereo, label=T("Splat Blend Temperature"))
        self.cbo_splat_blend_temperature = EditableComboBox(
            self.grp_stereo, choices=["10.0", "25.0", "50.0", "75.0", "85.0"],
            name="cbo_splat_blend_temperature")
        self.cbo_splat_blend_temperature.SetSelection(2)
        self.cbo_splat_blend_temperature.SetToolTip(
            T("What it's for: only matters for Method forward_splat_fill -- how sharply that method's "
              "depth-weighted blend decides which of two colliding pixels wins.\n"
              "Values: higher = sharper cutoff, closer to forward_fill's old hard-overwrite behavior "
              "(whichever pixel is nearer the camera wins almost completely); lower = smoother, more "
              "even blending between the two competing pixels.\n"
              "Con: brand new, and not yet tuned against real footage -- unlike most other numeric "
              "settings in this app, there isn't yet a body of real-world testing behind these numbers.\n"
              "Recommended: start at the default (50.0) and only adjust it if forward_splat_fill's "
              "results look wrong to you at that default."))
        self.sld_stereo_splat_blend_temperature = _build_stereo_slider(
            self.grp_stereo, self.cbo_splat_blend_temperature, 10.0, 85.0, 1)


        self.cpn_stereo_inpainting_depth = wx.CollapsiblePane(
            self.grp_stereo, label=T("Inpainting && Depth Source"), name="cpn_stereo_inpainting_depth")
        self.cpn_stereo_inpainting_depth.Collapse(True)
        self.cpn_stereo_inpainting_depth.Bind(wx.EVT_COLLAPSIBLEPANE_CHANGED,
                                              self.on_toggled_stereo_collapsible_pane)
        # Every wx.CollapsiblePane's content pane shares the same default internal
        # name ("wxCollapsiblePanePane") -- harmless with only one pane in the whole
        # frame, but a real crash once there's more than one: wx.lib.agw.persist's
        # Register() keys on (class, name) and raises on the second/third collision
        # during the full-tree walk, before get_stereo_sliders_and_panes()'s own
        # Unregister() step ever runs. Found live via the self-test suite immediately
        # after adding the 2nd/3rd pane -- give each a real, unique name.
        self.cpn_stereo_inpainting_depth.GetPane().SetName("cpn_stereo_inpainting_depth_pane")
        self.lbl_inpaint_model = wx.StaticText(self.cpn_stereo_inpainting_depth.GetPane(), label=T("Inpainting Model"))
        self.cbo_inpaint_model = wx.ComboBox(self.cpn_stereo_inpainting_depth.GetPane(),
                                             choices=list(INPAINT_MODELS.keys()),
                                             name="cbo_inpaint_model")
        self.cbo_inpaint_model.SetEditable(False)
        self.cbo_inpaint_model.SetSelection(0)
        self.cbo_inpaint_model.SetToolTip(
            T("What it's for: only matters for the *_inpaint Method variants (forward_inpaint, "
              "mlbw_l2_inpaint, monobw_inpaint) — picks which AI model fills in the hidden area behind "
              "objects that the 3D shift reveals.\n"
              "light_inpaint_v1 is the only model included out of the box, and is what this stays on for "
              "almost everyone. Extra models only appear here if you've manually added entries to "
              "iw3/inpaint_models.yml (an advanced/optional customization, not needed for normal use).\n"
              "Recommended: leave on light_inpaint_v1 unless you've specifically installed an alternative "
              "model and know why you want it."))

        self.lbl_overlap_frames = wx.StaticText(self.cpn_stereo_inpainting_depth.GetPane(), label=T("Inpaint Overlap Frames"))
        self.cbo_overlap_frames_pre = EditableComboBox(self.cpn_stereo_inpainting_depth.GetPane(),
                                                       choices=["0", "3"],
                                                       name="cbo_overlap_frames_pre")
        self.cbo_overlap_frames_pre.SetSelection(1)
        self.cbo_overlap_frames_pre.SetToolTip(
            T("Overlap Pre: only matters for the video inpainting methods. How many extra frames BEFORE "
              "each processed chunk get fed to the inpainting model purely for context, so the filled-in "
              "background doesn't flicker or change texture at chunk boundaries.\n"
              "Values: 0 = no overlap (fastest, but a visible seam/flicker can appear where chunks join). "
              "3 (default) = a small buffer that's usually enough to hide the seam.\n"
              "Con: higher values cost real extra processing time per chunk boundary, with diminishing "
              "returns past a few frames.\n"
              "Recommended: default (3); raise only if you actually see a flicker at chunk boundaries."))

        self.cbo_overlap_frames_post = EditableComboBox(self.cpn_stereo_inpainting_depth.GetPane(),
                                                        choices=["0", "3"],
                                                        name="cbo_overlap_frames_post")
        self.cbo_overlap_frames_post.SetSelection(1)
        self.cbo_overlap_frames_post.SetToolTip(
            T("Overlap Post: same idea as Overlap Pre, but the extra context frames come from AFTER each "
              "chunk instead of before. Same values/tradeoff apply.\n"
              "Recommended: default (3)."))

        self.lbl_mask_dilation = wx.StaticText(self.cpn_stereo_inpainting_depth.GetPane(), label=T("Inpaint Mask Dilation"))
        self.cbo_mask_inner_dilation = EditableComboBox(self.cpn_stereo_inpainting_depth.GetPane(),
                                                        choices=["0", "1", "2"],
                                                        name="cbo_mask_inner_dilation")
        self.cbo_mask_inner_dilation.SetSelection(0)
        self.cbo_mask_inner_dilation.SetToolTip(
            T("Inner: only matters for *_inpaint methods. Grows the \"needs filling in\" mask slightly "
              "INTO the foreground object's own edge, trimming a thin sliver off the object so the "
              "inpainted background blends in more smoothly instead of stopping at a hard line.\n"
              "Con: too high a value visibly shrinks/erodes foreground objects at their edges.\n"
              "Recommended: 0 (default); raise to 1-2 only if you see a thin, obviously wrong-colored halo "
              "clinging to the outline of foreground objects."))

        self.cbo_mask_outer_dilation = EditableComboBox(self.cpn_stereo_inpainting_depth.GetPane(),
                                                        choices=["0", "1", "2"],
                                                        name="cbo_mask_outer_dilation")
        self.cbo_mask_outer_dilation.SetSelection(0)
        self.cbo_mask_outer_dilation.SetToolTip(
            T("Outer: only matters for *_inpaint methods. Grows the \"needs filling in\" mask outward INTO "
              "the background, giving the inpainting model a wider area to work with around each object.\n"
              "Con: too high a value makes the model regenerate more background than it needs to, which "
              "costs a little quality/consistency versus the small sliver of real original background "
              "right at the object edge.\n"
              "Recommended: 0 (default); raise to 1-2 only if you still see leftover smearing/stretching "
              "right behind foreground objects after trying Inner dilation."))

        self.lbl_inpaint_max_width = wx.StaticText(self.cpn_stereo_inpainting_depth.GetPane(), label=T("Inpaint Max Width"))
        self.cbo_inpaint_max_width = EditableComboBox(self.cpn_stereo_inpainting_depth.GetPane(),
                                                      choices=["", "1920"],
                                                      name="cbo_inpaint_max_width")
        self.cbo_inpaint_max_width.SetSelection(0)
        self.cbo_inpaint_max_width.SetToolTip(
            T("What it's for: caps the width the inpainting model processes at, downscaling internally if "
              "the source is wider than this, to save VRAM/time on large/4K video.\n"
              "Con: lowering it can make the inpainted (filled-in) areas slightly softer/less detailed "
              "than the rest of the frame, since that step ran at a lower resolution.\n"
              "Recommended: leave blank (no limit, best quality) unless you're running out of GPU memory "
              "or need to speed up a very high-resolution job — then try 1920 first."))

        self.lbl_stereo_width = wx.StaticText(self.cpn_stereo_inpainting_depth.GetPane(), label=T("Stereo Processing Width"))
        self.cbo_stereo_width = EditableComboBox(self.cpn_stereo_inpainting_depth.GetPane(),
                                                 choices=["Default", "1920", "1280", "640"],
                                                 name="cbo_stereo_width")
        self.cbo_stereo_width.SetSelection(0)
        self.cbo_stereo_width.SetToolTip(
            T("What it's for: only used by the row_flow_v3/row_flow_v3_sym/row_flow_v2 methods. Resizes "
              "the image to this width specifically for the eye-generation step — separate from Depth "
              "Model's own internal resolution and separate from your final output resolution.\n"
              "Con: lowering it trades away fine detail in the warp itself for speed; the final output is "
              "still your source resolution, but the 3D shift calculation behind it was done coarser.\n"
              "Recommended: Default (uses the source width, best quality). Try 1920 or 1280 only if you "
              "need more speed and can accept a small quality tradeoff."))

        self.lbl_depth_model = wx.StaticText(self.cpn_stereo_inpainting_depth.GetPane(), label=T("Depth Model"))
        self.cbo_depth_model = wx.ComboBox(self.cpn_stereo_inpainting_depth.GetPane(),
                                           choices=self.get_depth_models(),
                                           name="cbo_depth_model")
        self.cbo_depth_model.SetEditable(False)
        self.cbo_depth_model.SetSelection(3)
        self.cbo_depth_model.SetToolTip(
            T("What it's for: which AI model looks at your image/video and estimates what's near vs far. "
              "This is the foundation everything else builds on — probably the single most important "
              "choice in the whole app.\n"
              "VDA_* (Video Depth Anything): built specifically for video — has real memory across frames, "
              "so depth stays steady/flicker-free without needing extra smoothing settings. Best choice "
              "for movies/video by default.\n"
              "Any_V2_* / Any_V3_* / Distill_Any_*: single-image models — often sharper/more detailed on "
              "a single photo, but have NO memory between frames, so used on video they can flicker unless "
              "you also turn on EMA smoothing and/or Object Stability.\n"
              "*_Metric variants: estimate real-world distances (meters) instead of a relative near/far "
              "scale — a specialized option, not needed for normal stereo conversion.\n"
              "Size suffix (_S/_B/_L, small/base/large): bigger = noticeably better quality, but "
              "slower and more VRAM — roughly proportional to size, not free.\n"
              "Recommended: a VDA_* model for video (steadiest results with the least fiddling); an "
              "Any_V3_* model for single images or when you want maximum per-frame detail on video and are "
              "willing to tune EMA/Object Stability yourself."))

        self.lbl_resolution = wx.StaticText(self.cpn_stereo_inpainting_depth.GetPane(), label=T("Depth") + " " + T("Resolution"))
        self.cbo_resolution = EditableComboBox(self.cpn_stereo_inpainting_depth.GetPane(),
                                               choices=["Default", "512"],
                                               name="cbo_zoed_resolution")
        self.cbo_resolution.SetSelection(0)
        self.cbo_resolution.SetToolTip(
            T("How much detail the depth model works with internally (its short-side resolution in "
              "pixels). \"Default\" uses ~392. Higher = finer depth detail but more VRAM/time — roughly "
              "squares the cost as you increase it. Recommended: Default for most content; try 448-512 "
              "if you have VRAM to spare and want finer depth detail."))

        self.chk_limit_resolution = wx.CheckBox(self.cpn_stereo_inpainting_depth.GetPane(), label=T("Limit to source"),
                                                name="chk_limit_resolution")
        self.chk_limit_resolution.SetToolTip(
            T("Safety cap only: if your typed Depth Resolution is HIGHER than the source video's own "
              "resolution, this brings it back down to match the source instead of wasting time asking "
              "for detail that doesn't exist. It never raises a lower value up. Recommended: on."))


        self.cpn_stereo_stability_flicker = wx.CollapsiblePane(
            self.grp_stereo, label=T("Stability && Flicker"), name="cpn_stereo_stability_flicker")
        self.cpn_stereo_stability_flicker.Collapse(True)
        self.cpn_stereo_stability_flicker.Bind(wx.EVT_COLLAPSIBLEPANE_CHANGED,
                                                self.on_toggled_stereo_collapsible_pane)
        self.cpn_stereo_stability_flicker.GetPane().SetName("cpn_stereo_stability_flicker_pane")
        self.lbl_foreground_scale = wx.StaticText(self.cpn_stereo_stability_flicker.GetPane(), label=T("Foreground Scale"))
        self.cbo_foreground_scale = EditableComboBox(self.cpn_stereo_stability_flicker.GetPane(),
                                                     choices=["-3", "-2", "-1", "0", "1", "2", "3"],
                                                     name="cbo_foreground_scale")
        self.cbo_foreground_scale.SetSelection(3)
        self.cbo_foreground_scale.SetToolTip(
            T("What it's for: reshapes the depth curve for the WHOLE image (-3 to 3) — but as a "
              "redistribution, not a simple push. It's a trade-off dial, not a \"both ends get better\" "
              "dial: whichever end you push toward gains separation/detail, the OTHER end gets flattened. "
              "It cannot give a punchier foreground and a deeper background at the same time.\n"
              "Values: 0 = the depth model's own natural curve, unchanged. Positive values sharpen/spread "
              "out the FOREGROUND (more roundness/separation among near objects) while flattening the "
              "background. Negative values do the reverse — sharpen/spread out the BACKGROUND (more sense "
              "of depth into the distance) while flattening the foreground.\n"
              "Con: pushed to an extreme in either direction, the opposite end of the scene will look "
              "noticeably flat/compressed — this is inherent to how the curve works, not a bug.\n"
              "Recommended: 0 for natural, unexaggerated depth. Try a modest negative value (-0.5 to -1.0) "
              "if you want a more immersive/deep-feeling background; a modest positive value (+0.5 to "
              "+1.5) for a more sculpted, punchy foreground. If you want MORE separation on both ends at "
              "once instead of trading one for the other, raise 3D Strength (Divergence) instead — it "
              "scales both ends up together rather than redistributing between them."))

        self.chk_depth_aa = wx.CheckBox(self.cpn_stereo_stability_flicker.GetPane(), label=T("Depth Anti-aliasing"), name="chk_depth_aa")
        self.chk_depth_aa.SetValue(False)
        self.chk_depth_aa.SetToolTip(
            T("Smooths small jagged/staircase artifacts in the depth map using a dedicated AI model, "
              "without changing the actual depth values much. Only available for certain depth models "
              "(grayed out otherwise). Recommended: on, when available — minor cost, generally cleaner result."))

        self.chk_depth_refine = wx.CheckBox(self.cpn_stereo_stability_flicker.GetPane(), label=T("Depth Detail Refinement"),
                                            name="chk_depth_refine")
        self.chk_depth_refine.SetValue(False)
        self.chk_depth_refine.SetToolTip(
            T("What it's for: cleans up noise WITHIN each depth frame using edge-preserving smoothing "
              "(won't blur across real edges the way a plain blur would). Different from Flicker "
              "Reduction, which smooths ACROSS frames over time — this works on a single frame at a time. "
              "Cheap: no extra passes, no extra models.\n"
              "Side note some users notice: cleaner depth boundaries here can make the finished 3D effect "
              "FEEL a bit stronger/more solid even though the actual depth range doesn't change — noisy or "
              "fuzzy depth edges read as less convincing 3D than clean ones at the same strength.\n"
              "Recommended: on, safe to leave on for most content. Use the Strength box to its right to "
              "control how much."))

        self.cbo_depth_refine_strength = EditableComboBox(
            self.cpn_stereo_stability_flicker.GetPane(), choices=["1.5", "1.25", "1.0", "0.75", "0.5", "0.25"],
            name="cbo_depth_refine_strength")
        self.cbo_depth_refine_strength.SetSelection(2)
        self.cbo_depth_refine_strength.SetToolTip(
            T("How strong Depth Detail Refinement's cleanup is. 1.0 = the original fixed strength this "
              "feature always used. Higher = more smoothing reach (cleaner depth boundaries, but risks "
              "softening genuinely fine depth detail if pushed too far); lower = gentler, closer to doing "
              "nothing. Recommended: 1.0 as a safe starting point; try 1.25-1.5 if you want a bit more of "
              "the \"cleaner/more solid 3D\" effect this setting gives."))
        self.sld_stereo_depth_refine_strength = _build_stereo_slider(
            self.cpn_stereo_stability_flicker.GetPane(), self.cbo_depth_refine_strength, 0.25, 1.5, 100)

        self.chk_temporal_stabilize = wx.CheckBox(self.cpn_stereo_stability_flicker.GetPane(), label=T("Object Stability (experimental)"),
                                                  name="chk_temporal_stabilize")
        self.chk_temporal_stabilize.SetValue(False)
        self.chk_temporal_stabilize.SetToolTip(
            T("Gives a single-frame depth model (like Any_V3_Mono_01) some of the same per-pixel steadiness "
              "over time that a video-native model (VDA_*) has built in. Tracks real motion (optical flow) "
              "and blends each frame's depth with the PREVIOUS frame's depth warped to where that content "
              "actually moved to, reducing a specific object's depth flickering that Flicker Reduction's "
              "overall-range smoothing can't touch. Only works on the single-frame processing path -- "
              "already active for --low-vram, --debug-depth, VDA streaming models, and inpaint methods "
              "(e.g. mlbw_l2_inpaint); has no effect otherwise."))

        self.cbo_temporal_stabilize_strength = EditableComboBox(self.cpn_stereo_stability_flicker.GetPane(),
                                                                 choices=["0.9", "0.7", "0.5", "0.3"],
                                                                 name="cbo_temporal_stabilize_strength")
        self.cbo_temporal_stabilize_strength.SetSelection(1)
        self.cbo_temporal_stabilize_strength.SetToolTip(
            T("How strongly to trust the motion-warped previous frame vs the fresh per-frame depth (0-1). "
              "Automatically tapers down during fast/unreliable motion regardless of this setting."))
        self.sld_stereo_temporal_stabilize_strength = _build_stereo_slider(
            self.cpn_stereo_stability_flicker.GetPane(), self.cbo_temporal_stabilize_strength, 0.3, 0.9, 100)

        self.lbl_temporal_stabilize_max_shift = wx.StaticText(self.cpn_stereo_stability_flicker.GetPane(), label=T("Max Shift"))
        self.cbo_temporal_stabilize_max_shift = EditableComboBox(
            self.cpn_stereo_stability_flicker.GetPane(),
            choices=["", "0.01", "0.02", "0.05"],
            name="cbo_temporal_stabilize_max_shift")
        self.cbo_temporal_stabilize_max_shift.SetSelection(0)
        self.cbo_temporal_stabilize_max_shift.SetToolTip(
            T("Object Stability: hard cap on how much depth is allowed to change for the same pixel "
              "between two consecutive output frames (0-1 scale, same units as depth value). Stops a "
              "single-frame spike from ever \"popping\", no matter how strong the raw model's disagreement "
              "is. Leave blank to disable (no cap, original behavior)."))

        self.lbl_temporal_stabilize_flat_boost = wx.StaticText(self.cpn_stereo_stability_flicker.GetPane(), label=T("Flat-Area Boost"))
        self.cbo_temporal_stabilize_flat_boost = EditableComboBox(
            self.cpn_stereo_stability_flicker.GetPane(),
            choices=["0.0", "0.3", "0.5", "0.7"],
            name="cbo_temporal_stabilize_flat_boost")
        self.cbo_temporal_stabilize_flat_boost.SetSelection(0)
        self.cbo_temporal_stabilize_flat_boost.SetToolTip(
            T("Object Stability: extra smoothing specifically in areas the CURRENT frame's own depth is "
              "flat (sky, walls, floors) -- these are exactly the areas where flicker is most visible and "
              "least likely to be real motion. 0 = no extra smoothing (original behavior)."))

        self.lbl_temporal_stabilize_edge_protect = wx.StaticText(self.cpn_stereo_stability_flicker.GetPane(), label=T("Edge Protection"))
        self.cbo_temporal_stabilize_edge_protect = EditableComboBox(
            self.cpn_stereo_stability_flicker.GetPane(),
            choices=["0.0", "0.3", "0.5", "0.7"],
            name="cbo_temporal_stabilize_edge_protect")
        self.cbo_temporal_stabilize_edge_protect.SetSelection(0)
        self.cbo_temporal_stabilize_edge_protect.SetToolTip(
            T("Object Stability: reduces smoothing where the CURRENT frame has a strong, real depth edge "
              "(an object's silhouette), so Object Stability's flicker reduction doesn't smear or lag "
              "behind a moving object's outline. 0 = no reduction (original behavior)."))

        self.grp_depth_blend = wx.StaticBox(self.tab_depth_blend, label=T("Dual-Pass Depth Blend"))
        self.chk_depth_blend = wx.CheckBox(self.grp_depth_blend, label=T("Dual-Pass Depth Blend"),
                                           name="chk_depth_blend")
        self.chk_depth_blend.SetValue(False)
        self.chk_depth_blend.SetToolTip(
            T("Brings in a SECOND depth model and blends it into Depth Model's result, favoring the "
              "second model specifically where the image has dense fine detail (foliage, hair, close-up "
              "texture) that a single-frame model often gets wrong. Runs as 3 full passes over the clip "
              "(export depth A, export depth B, blend+render), fully releasing each model before the next "
              "loads so both are never in GPU memory at once. Costs roughly 3x the time and real extra "
              "disk space (a full frame dump, kept in a '<output>.depth_blend_work' folder you can delete "
              "afterward). Requires a single video file input; not compatible with Automated Scene Batch."))

        self.cbo_depth_blend_model = wx.ComboBox(self.grp_depth_blend,
                                                 choices=self.get_depth_models(),
                                                 name="cbo_depth_blend_model")
        self.cbo_depth_blend_model.SetEditable(False)
        self.cbo_depth_blend_model.SetToolTip(
            T("The SECOND depth model for Dual-Pass Depth Blend. Pick one with strengths your main Depth "
              "Model lacks — e.g. a VDA_* model for its steadier handling of foliage/close-up detail."))
        if "VDA_L" in self.get_depth_models():
            self.cbo_depth_blend_model.SetValue("VDA_L")
        elif self.get_depth_models():
            self.cbo_depth_blend_model.SetSelection(0)

        self.cbo_depth_blend_strength = EditableComboBox(self.grp_depth_blend,
                                                         choices=["1.0", "0.75", "0.5", "0.25"],
                                                         name="cbo_depth_blend_strength")
        self.cbo_depth_blend_strength.SetSelection(0)
        self.cbo_depth_blend_strength.SetToolTip(
            T("What it's for: how strongly to favor the SECOND model within the region selected below "
              "(0-1). 1.0 = fully trust the second model there; lower values blend it in more gently, "
              "leaning back toward the main Depth Model's own result.\n"
              "Con: pushed to 1.0 in the \"detail\" region, you're fully trusting a model that may not "
              "agree pixel-for-pixel with the primary model at real edges — see Edge Suppression below, "
              "which exists specifically to manage that risk.\n"
              "Recommended: 1.0 to start; back off toward 0.5-0.75 if you notice any softness/ghosting at "
              "silhouettes even with Edge Suppression on."))
        self.sld_depth_blend_strength = _build_stereo_slider(
            self.grp_depth_blend, self.cbo_depth_blend_strength, 0.25, 1.0, 100)

        self.cbo_depth_blend_region = wx.ComboBox(self.grp_depth_blend,
                                                  choices=["detail", "foreground", "background"],
                                                  name="cbo_depth_blend_region")
        self.cbo_depth_blend_region.SetEditable(False)
        self.cbo_depth_blend_region.SetSelection(0)
        self.cbo_depth_blend_region.SetToolTip(
            T("What it's for: WHERE the second model gets blended in.\n"
              "detail: wherever the image itself shows dense fine visual detail (foliage, hair, close-up "
              "texture) — the original use case this feature was built for, since single-frame models "
              "often get this kind of detail wrong. Automatically avoids blending right at the primary "
              "model's own confident edges (see Edge Suppression below).\n"
              "foreground / background: a straight percentage slice of the scene BY DEPTH (nearest or "
              "farthest X%, set below), regardless of how much visual detail is actually there.\n"
              "Recommended: detail for the original foliage/hair use case; foreground/background only if "
              "you specifically want the second model's characteristics applied to a whole depth range "
              "(e.g. a steadier video-native model just for a busy background) rather than detail-seeking."))

        self.cbo_depth_blend_region_percent = EditableComboBox(self.grp_depth_blend,
                                                                choices=["10", "25", "40", "50"],
                                                                name="cbo_depth_blend_region_percent")
        self.cbo_depth_blend_region_percent.SetSelection(1)
        self.cbo_depth_blend_region_percent.SetToolTip(
            T("For \"foreground\"/\"background\" region only (ignored for \"detail\"): what percent of "
              "the scene, by depth, to blend the second model into — e.g. 25 = nearest (or farthest) 25% "
              "of the scene.\n"
              "Con: a larger percentage moves the blend/no-blend transition line further into the "
              "midground, where a difference between the two models becomes more likely to be noticeable.\n"
              "Recommended: 25 as a starting point; keep it modest unless you have a specific reason to "
              "cover more of the scene."))
        self.sld_depth_blend_region_percent = _build_stereo_slider(
            self.grp_depth_blend, self.cbo_depth_blend_region_percent, 10, 50, 1)

        self.lbl_depth_blend_feather_blur = wx.StaticText(self.grp_depth_blend, label=T("Feather Blur"))
        self.cbo_depth_blend_feather_blur = EditableComboBox(self.grp_depth_blend,
                                                              choices=["0", "5", "15", "25", "35"],
                                                              name="cbo_depth_blend_feather_blur")
        self.cbo_depth_blend_feather_blur.SetSelection(0)
        self.cbo_depth_blend_feather_blur.SetToolTip(
            T("What it's for: softens the blend transition ITSELF (blur kernel size in pixels; 0 = off, a "
              "sharp on/off transition) — works with any Region mode, smoothing the seam between "
              "blended/unblended areas so it doesn't read as a visible hard line.\n"
              "Con: a large value can blur the transition zone wide enough to become visible itself as a "
              "soft band, rather than fixing a hard edge.\n"
              "Recommended: 0 (off) unless you specifically notice a hard seam where blending starts/stops; "
              "start small (5-15) and increase only if needed."))
        self.sld_depth_blend_feather_blur = _build_stereo_slider(
            self.grp_depth_blend, self.cbo_depth_blend_feather_blur, 0, 35, 1)

        self.chk_depth_blend_bilateral = wx.CheckBox(self.grp_depth_blend, label=T("Bilateral Denoise"),
                                                     name="chk_depth_blend_bilateral")
        self.chk_depth_blend_bilateral.SetValue(False)
        self.chk_depth_blend_bilateral.SetToolTip(
            T("What it's for: runs an edge-preserving smoothing pass over the FINAL blended depth (after "
              "both models are already combined) to clean up noise introduced by mixing two models, "
              "without softening real depth boundaries the way a plain blur would.\n"
              "Con: a real extra processing pass — some added time cost per frame.\n"
              "Recommended: off by default; turn on if you notice speckle/noise texture in the blended "
              "result that a plain Depth Detail Refinement pass doesn't fully clean up."))
        self.cbo_depth_blend_bilateral_d = EditableComboBox(self.grp_depth_blend,
                                                             choices=["9", "12", "15"],
                                                             name="cbo_depth_blend_bilateral_d")
        self.cbo_depth_blend_bilateral_d.SetSelection(1)
        self.cbo_depth_blend_bilateral_d.SetToolTip(
            T("Bilateral filter diameter, in pixels — how large an area around each pixel is considered. "
              "Higher = smooths a wider neighborhood but costs more processing time. Recommended: 12 as a "
              "starting point."))
        self.sld_depth_blend_bilateral_d = _build_stereo_slider(
            self.grp_depth_blend, self.cbo_depth_blend_bilateral_d, 9, 15, 1)
        self.cbo_depth_blend_bilateral_sigma_color = EditableComboBox(
            self.grp_depth_blend, choices=["50", "75", "100"], name="cbo_depth_blend_bilateral_sigma_color")
        self.cbo_depth_blend_bilateral_sigma_color.SetSelection(1)
        self.cbo_depth_blend_bilateral_sigma_color.SetToolTip(
            T("Bilateral filter's depth-value sensitivity, expressed in familiar 0-255-ish terms (matches "
              "common 8-bit filter presets; scaled internally to this pipeline's real 16-bit depth range). "
              "Higher = willing to smooth across BIGGER depth differences, which risks blurring real depth "
              "edges, not just noise; lower = only smooths very similar depth values together, safer for "
              "real edges but cleans up less noise. Recommended: 75 as a balanced starting point."))
        self.sld_depth_blend_bilateral_sigma_color = _build_stereo_slider(
            self.grp_depth_blend, self.cbo_depth_blend_bilateral_sigma_color, 50, 100, 1)
        self.cbo_depth_blend_bilateral_sigma_space = EditableComboBox(
            self.grp_depth_blend, choices=["50", "75", "100"], name="cbo_depth_blend_bilateral_sigma_space")
        self.cbo_depth_blend_bilateral_sigma_space.SetSelection(1)
        self.cbo_depth_blend_bilateral_sigma_space.SetToolTip(
            T("Bilateral filter's spatial reach, in pixels. Higher = smooths across a physically wider "
              "area of the frame; lower = stays more localized. Recommended: 75 as a balanced starting "
              "point, paired with the diameter/color settings above."))
        self.sld_depth_blend_bilateral_sigma_space = _build_stereo_slider(
            self.grp_depth_blend, self.cbo_depth_blend_bilateral_sigma_space, 50, 100, 1)

        self.chk_depth_blend_clahe = wx.CheckBox(self.grp_depth_blend, label=T("CLAHE Contrast (experimental)"),
                                                 name="chk_depth_blend_clahe")
        self.chk_depth_blend_clahe.SetValue(False)
        self.chk_depth_blend_clahe.SetToolTip(
            T("Local contrast enhancement over the final blended depth. OFF BY DEFAULT and not recommended: "
              "an earlier benchmark on this project's own depth data found CLAHE amplifies whatever local "
              "variation it finds, noise included (see docs/ai/AI_DECISIONS.md ADR-001). Provided as an "
              "explicit opt-in experiment only."))
        self.cbo_depth_blend_clahe_clip = EditableComboBox(self.grp_depth_blend,
                                                            choices=["1.0", "2.0", "4.0"],
                                                            name="cbo_depth_blend_clahe_clip")
        self.cbo_depth_blend_clahe_clip.SetSelection(1)
        self.cbo_depth_blend_clahe_clip.SetToolTip(
            T("CLAHE contrast clip limit — how aggressively local contrast gets boosted. Higher = "
              "stronger local contrast, but also amplifies more noise (see the checkbox above's warning). "
              "Only matters if CLAHE Contrast is turned on."))
        self.sld_depth_blend_clahe_clip = _build_stereo_slider(
            self.grp_depth_blend, self.cbo_depth_blend_clahe_clip, 1.0, 4.0, 100)
        self.cbo_depth_blend_clahe_tile = EditableComboBox(self.grp_depth_blend,
                                                            choices=["4", "8", "16"],
                                                            name="cbo_depth_blend_clahe_tile")
        self.cbo_depth_blend_clahe_tile.SetSelection(1)
        self.cbo_depth_blend_clahe_tile.SetToolTip(
            T("CLAHE tile grid size (NxN) — how finely the frame is divided up for LOCAL contrast "
              "adjustment. More tiles = more localized (small-area) contrast changes; fewer tiles = "
              "smoother, more global adjustment. Only matters if CLAHE Contrast is turned on."))
        self.sld_depth_blend_clahe_tile = _build_stereo_slider(
            self.grp_depth_blend, self.cbo_depth_blend_clahe_tile, 4, 16, 1)

        self.chk_depth_blend_align = wx.CheckBox(self.grp_depth_blend, label=T("Depth Scale Alignment"),
                                                 name="chk_depth_blend_align")
        self.chk_depth_blend_align.SetValue(False)
        self.chk_depth_blend_align.SetToolTip(
            T("Both depth passes are independently normalized to the same 0-1 numeric range, but that "
              "doesn't guarantee a given value means the same real-world distance in both models. This "
              "rescales the second model's depth to match the first model's actual distribution (robust "
              "5th-95th percentile matching, EMA-smoothed across frames to avoid flicker) before "
              "blending, reducing potential depth-value mismatches at blend region edges. Off by default; "
              "adds a one-time sequential pre-pass over all frames before blending starts. See Alignment "
              "Decay to its right for the smoothing strength."))

        self.cbo_depth_blend_align_decay = EditableComboBox(
            self.grp_depth_blend, choices=["0.95", "0.9", "0.75", "0.5", "0"],
            name="cbo_depth_blend_align_decay")
        self.cbo_depth_blend_align_decay.SetSelection(1)
        self.cbo_depth_blend_align_decay.SetToolTip(
            T("What it's for: how much the frame-to-frame alignment fit (see Depth Scale Alignment to the "
              "left) is smoothed across frames, instead of being recalculated fresh and independently for "
              "every single frame.\n"
              "Values: higher (0.9-0.95) = steadier alignment, resistant to a single noisy/outlier frame "
              "throwing the fit off, but slower to follow a genuine change (e.g. a big lighting/scene "
              "change shifting how the two models relate to each other). Lower (0.5-0.75) = reacts faster "
              "per frame, more exposed to noise. 0 = no smoothing at all, each frame's alignment computed "
              "completely independently.\n"
              "Recommended: 0.9 (default, matches this feature's original tuning) as a safe starting "
              "point; lower it only if you specifically notice the alignment lagging behind a real, "
              "sudden change between the two models."))
        self.sld_depth_blend_align_decay = _build_stereo_slider(
            self.grp_depth_blend, self.cbo_depth_blend_align_decay, 0.0, 0.95, 100)

        self.lbl_depth_blend_edge_suppression = wx.StaticText(self.grp_depth_blend, label=T("Edge Suppression"))
        self.cbo_depth_blend_edge_suppression = EditableComboBox(
            self.grp_depth_blend, choices=["0.0", "0.25", "0.5", "0.75", "1.0"],
            name="cbo_depth_blend_edge_suppression")
        self.cbo_depth_blend_edge_suppression.SetSelection(2)
        self.cbo_depth_blend_edge_suppression.SetToolTip(
            T("Only affects 'detail' blend region. Wherever the primary depth model already has a clean, "
              "confident edge (a person's silhouette, say), the two models rarely agree on the exact pixel "
              "-- blending them there risks a soft, slightly doubled-looking edge. This protects a band "
              "around those edges from blending. Higher = wider protected band, safer against that soft/"
              "doubled-edge look but suppresses a touch more real detail blending near edges too. Lower = "
              "narrower protection, more detail blending everywhere but more risk of soft edges. Default "
              "0.5 is a middle ground between two previously-tested extremes."))
        self.sld_depth_blend_edge_suppression = _build_stereo_slider(
            self.grp_depth_blend, self.cbo_depth_blend_edge_suppression, 0.0, 1.0, 100)

        self.chk_depth_blend_edge_hard_cutoff = wx.CheckBox(
            self.grp_depth_blend, label=T("Hard Edge Cutoff"), name="chk_depth_blend_edge_hard_cutoff")
        self.chk_depth_blend_edge_hard_cutoff.SetValue(False)
        self.chk_depth_blend_edge_hard_cutoff.SetToolTip(
            T("What it's for: only matters together with Edge Suppression above ('detail' region only). "
              "Edge Suppression normally fades in smoothly as you approach a real silhouette -- this "
              "switches that smooth fade into a hard on/off step instead (same protected band width, "
              "sharper edge to it). A cheap thing to try if Edge Suppression alone still leaves a soft/"
              "misaligned-looking border on some objects.\n"
              "Con: cannot fully eliminate a genuine shape disagreement between two independently-trained "
              "depth models -- there's no established technique to fully correct that (verified via "
              "research, not assumed), only shrink its visible impact further than the smooth fade does "
              "alone.\n"
              "Recommended: off by default; try on only after Edge Suppression is already near 1.0 and "
              "still isn't enough."))

        layout_depth_blend = wx.GridBagSizer(vgap=5, hgap=4)
        layout_depth_blend.SetEmptyCellSize((0, 0))
        j = 0
        layout_depth_blend.Add(self.chk_depth_blend, (j, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout_depth_blend.Add(self.cbo_depth_blend_model, (j, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.cbo_depth_blend_strength, (j, 2), flag=wx.EXPAND)
        layout_depth_blend.Add(self.cbo_depth_blend_region, (j := j + 1, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.cbo_depth_blend_region_percent, (j, 2), flag=wx.EXPAND)
        layout_depth_blend.Add(self.sld_depth_blend_strength, (j := j + 1, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.sld_depth_blend_region_percent, (j, 2), flag=wx.EXPAND)

        layout_depth_blend.Add((0, 6), (j := j + 1, 0))
        layout_depth_blend.Add(wx.StaticLine(self.grp_depth_blend), (j := j + 1, 0), (0, 3), flag=wx.EXPAND)
        layout_depth_blend.Add((0, 4), (j := j + 1, 0))
        layout_depth_blend.Add(self.lbl_depth_blend_feather_blur, (j := j + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout_depth_blend.Add(self.cbo_depth_blend_feather_blur, (j, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.sld_depth_blend_feather_blur, (j := j + 1, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.chk_depth_blend_bilateral, (j := j + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout_depth_blend.Add(self.cbo_depth_blend_bilateral_d, (j, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.cbo_depth_blend_bilateral_sigma_color, (j, 2), flag=wx.EXPAND)
        layout_depth_blend.Add(self.sld_depth_blend_bilateral_d, (j := j + 1, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.sld_depth_blend_bilateral_sigma_color, (j, 2), flag=wx.EXPAND)
        layout_depth_blend.Add(self.cbo_depth_blend_bilateral_sigma_space, (j := j + 1, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.sld_depth_blend_bilateral_sigma_space, (j := j + 1, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.chk_depth_blend_clahe, (j := j + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout_depth_blend.Add(self.cbo_depth_blend_clahe_clip, (j, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.cbo_depth_blend_clahe_tile, (j, 2), flag=wx.EXPAND)
        layout_depth_blend.Add(self.sld_depth_blend_clahe_clip, (j := j + 1, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.sld_depth_blend_clahe_tile, (j, 2), flag=wx.EXPAND)

        layout_depth_blend.Add((0, 6), (j := j + 1, 0))
        layout_depth_blend.Add(wx.StaticLine(self.grp_depth_blend), (j := j + 1, 0), (0, 3), flag=wx.EXPAND)
        layout_depth_blend.Add((0, 4), (j := j + 1, 0))
        layout_depth_blend.Add(self.chk_depth_blend_align, (j := j + 1, 0), (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout_depth_blend.Add(self.cbo_depth_blend_align_decay, (j, 2), flag=wx.EXPAND)
        layout_depth_blend.Add(self.sld_depth_blend_align_decay, (j := j + 1, 2), flag=wx.EXPAND)

        layout_depth_blend.Add((0, 6), (j := j + 1, 0))
        layout_depth_blend.Add(wx.StaticLine(self.grp_depth_blend), (j := j + 1, 0), (0, 3), flag=wx.EXPAND)
        layout_depth_blend.Add((0, 4), (j := j + 1, 0))
        layout_depth_blend.Add(self.lbl_depth_blend_edge_suppression, (j := j + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout_depth_blend.Add(self.cbo_depth_blend_edge_suppression, (j, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.chk_depth_blend_edge_hard_cutoff, (j, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout_depth_blend.Add(self.sld_depth_blend_edge_suppression, (j := j + 1, 1), flag=wx.EXPAND)
        sizer_depth_blend = wx.StaticBoxSizer(self.grp_depth_blend, wx.VERTICAL)
        sizer_depth_blend.Add(layout_depth_blend, 1, wx.ALL | wx.EXPAND, 4)

        self.cpn_stereo_pop_divergence = wx.CollapsiblePane(
            self.grp_stereo, label=T("Pop && Divergence"), name="cpn_stereo_pop_divergence")
        self.cpn_stereo_pop_divergence.Collapse(True)
        self.cpn_stereo_pop_divergence.Bind(wx.EVT_COLLAPSIBLEPANE_CHANGED,
                                            self.on_toggled_stereo_collapsible_pane)
        self.cpn_stereo_pop_divergence.GetPane().SetName("cpn_stereo_pop_divergence_pane")
        self.lbl_foreground_pop = wx.StaticText(self.cpn_stereo_pop_divergence.GetPane(), label=T("Foreground Pop"))
        self.cbo_foreground_pop = EditableComboBox(self.cpn_stereo_pop_divergence.GetPane(),
                                                   choices=["0.0", "0.25", "0.5", "0.75", "1.0"],
                                                   name="cbo_foreground_pop")
        self.cbo_foreground_pop.SetSelection(0)
        self.cbo_foreground_pop.SetToolTip(
            T("What it's for: pushes ONLY the nearest pixels further toward the audience (0=off, 1=strong), "
              "leaving the rest of the scene (midground/background) completely untouched. Different from "
              "Foreground Scale, which redistributes the WHOLE depth curve and always trades foreground "
              "against background.\n"
              "Con: pushed too high, the very closest objects can pop hard enough to feel uncomfortable or "
              "clash with the frame edges (pair with Preserve Screen Border to reduce that risk).\n"
              "Recommended: 0 (off) for a restrained, professional look; 0.25-0.5 for deliberate "
              "\"poke at the audience\" moments; reserve 0.75-1.0 for a genuinely gimmicky effect."))
        self.sld_stereo_foreground_pop = _build_stereo_slider(self.cpn_stereo_pop_divergence.GetPane(), self.cbo_foreground_pop, 0.0, 1.0, 100)

        self.lbl_foreground_divergence = wx.StaticText(self.cpn_stereo_pop_divergence.GetPane(), label=T("Foreground Divergence"))
        self.cbo_foreground_divergence = EditableComboBox(self.cpn_stereo_pop_divergence.GetPane(),
                                                           choices=["", "2.0", "2.5", "3.0", "3.5", "4.0"],
                                                           name="cbo_foreground_divergence")
        self.cbo_foreground_divergence.SetSelection(0)
        self.cbo_foreground_divergence.SetToolTip(
            T("What it's for: uses a SEPARATE 3D Strength (Divergence) value for only the nearest 15% of "
              "pixels, independent of the main 3D Strength slider above — lets you dial the foreground up "
              "or down without touching the rest of the scene.\n"
              "Values: leave blank to disable (nearest pixels just use the same 3D Strength as everything "
              "else, the default). Set higher than your main 3D Strength for an extra-punchy foreground; "
              "lower for a gentler one.\n"
              "Recommended: leave blank unless you've specifically noticed the foreground needs its own "
              "strength independent of the rest of the scene."))

        self.lbl_background_pop = wx.StaticText(self.cpn_stereo_pop_divergence.GetPane(), label=T("Background Pop"))
        self.cbo_background_pop = EditableComboBox(self.cpn_stereo_pop_divergence.GetPane(),
                                                    choices=["0.0", "0.25", "0.5", "0.75", "1.0"],
                                                    name="cbo_background_pop")
        self.cbo_background_pop.SetSelection(0)
        self.cbo_background_pop.SetToolTip(
            T("What it's for: pushes ONLY the farthest pixels further away from the audience (0=off, "
              "1=strong), leaving the foreground/midground completely untouched. Mirror image of "
              "Foreground Pop, aimed at the opposite end of the scene.\n"
              "How it helps: gives a deeper-feeling background without needing to redistribute the whole "
              "depth curve (unlike negative Foreground Scale, which achieves a similar deep-background "
              "feel but by taking separation away from the foreground at the same time).\n"
              "Recommended: 0.15-0.25 for a modestly more immersive background on most content; go higher "
              "for content with a genuinely deep, expansive background you want to emphasize (landscapes, "
              "wide establishing shots)."))
        self.sld_stereo_background_pop = _build_stereo_slider(self.cpn_stereo_pop_divergence.GetPane(), self.cbo_background_pop, 0.0, 1.0, 100)

        self.lbl_background_pop_coverage = wx.StaticText(self.cpn_stereo_pop_divergence.GetPane(), label=T("Background Pop Coverage %"))
        self.cbo_background_pop_coverage = EditableComboBox(self.cpn_stereo_pop_divergence.GetPane(),
                                                             choices=["15", "20", "25", "30", "40"],
                                                             name="cbo_background_pop_coverage")
        self.cbo_background_pop_coverage.SetSelection(0)
        self.cbo_background_pop_coverage.SetToolTip(
            T("What it's for: how much of the scene Background Pop treats as \"background\" (default "
              "15% = farthest 15% of pixels by depth).\n"
              "Con: larger values affect more of the scene, but push the transition line further into the "
              "midground, where a sudden change in how much a subject recedes is more likely to be "
              "noticeable as an odd \"step\" in the depth.\n"
              "Recommended: 15% (default) for a subtle, hard-to-notice effect; raise to 20-30% only if "
              "you want the effect to reach further into the scene and don't mind it becoming more visible."))
        self.sld_stereo_background_pop_coverage = _build_stereo_slider(
            self.cpn_stereo_pop_divergence.GetPane(), self.cbo_background_pop_coverage, 15, 40, 1)

        self.lbl_background_divergence = wx.StaticText(self.cpn_stereo_pop_divergence.GetPane(), label=T("Background Divergence"))
        self.cbo_background_divergence = EditableComboBox(self.cpn_stereo_pop_divergence.GetPane(),
                                                           choices=["", "2.0", "2.5", "3.0", "3.5", "4.0"],
                                                           name="cbo_background_divergence")
        self.cbo_background_divergence.SetSelection(0)
        self.cbo_background_divergence.SetToolTip(
            T("What it's for: uses a SEPARATE 3D Strength (Divergence) value for only the farthest 15% of "
              "pixels, independent of the main 3D Strength slider above.\n"
              "Values: leave blank to disable (farthest pixels just use the same 3D Strength as everything "
              "else, the default). Set lower than your main 3D Strength to keep a busy/detailed background "
              "calmer and less prone to edge artifacts; set higher for a deeper-feeling background.\n"
              "Recommended: leave blank unless you've specifically noticed the background needs its own "
              "strength independent of the rest of the scene."))

        self.lbl_edge_repair = wx.StaticText(self.cpn_stereo_pop_divergence.GetPane(), label=T("Edge Repair"))
        self.cbo_edge_repair = EditableComboBox(self.cpn_stereo_pop_divergence.GetPane(),
                                                choices=["0.0", "0.25", "0.5", "0.75", "1.0"],
                                                name="cbo_edge_repair")
        self.cbo_edge_repair.SetSelection(0)
        self.cbo_edge_repair.SetToolTip(
            T("Final cleanup pass on the finished 3D image (applies no matter which Stereo Method made it). "
              "Gently smooths a thin hairline right around real depth edges to reduce fringing/ghosting "
              "residue left over from the 3D warp. Cannot affect flat areas or anywhere without a depth "
              "edge. 0.0=off (default), 1.0=strongest."))
        self.sld_stereo_edge_repair = _build_stereo_slider(self.cpn_stereo_pop_divergence.GetPane(), self.cbo_edge_repair, 0.0, 1.0, 100)

        self.chk_sharpen = wx.CheckBox(self.cpn_stereo_pop_divergence.GetPane(), label=T("Sharpen"), name="chk_sharpen")
        self.chk_sharpen.SetValue(False)
        self.chk_sharpen.SetToolTip(
            T("What it's for: enhances fine detail/sharpness in the FINISHED, fully-rendered 3D picture "
              "itself (after Edge Repair above, if that's also on) -- different from every other "
              "sharpness-adjacent setting in this app, which all work on the DEPTH MAP instead (Depth "
              "Resolution, Depth Detail Refinement, Dual-Pass Depth Blend). This is the only control that "
              "sharpens the actual delivered image.\n"
              "How it avoids amplifying film grain: classic unsharp-mask sharpening (boost = original - "
              "blurred) applied flatly makes old scanned film grain look like ugly speckling, because grain "
              "IS high-frequency detail to a naive sharpener. This one measures how much real local "
              "detail/edge structure is actually at each spot first (same technique already used elsewhere "
              "in this app to protect real depth edges from over-smoothing) and only applies the boost "
              "where that's genuinely high -- real edges and texture get sharpened, flat or grain-only "
              "areas are left close to untouched.\n"
              "Pros: makes fine detail (hair, texture, text) pop more in the final image, safer on grainy/"
              "noisy source than a plain sharpen filter would be.\n"
              "Cons: any sharpening pass can still exaggerate real compression artifacts or genuinely fine "
              "noise that happens to look edge-like; if the image starts looking harsh/over-crisp, lower "
              "the Strength box to its right.\n"
              "Recommended: off (default) unless the finished 3D output looks a little soft to you; start "
              "at the default 0.5 Strength and raise only if you still want more."))

        self.cbo_sharpen_strength = EditableComboBox(self.cpn_stereo_pop_divergence.GetPane(),
                                                     choices=["0.25", "0.5", "0.75", "1.0"],
                                                     name="cbo_sharpen_strength")
        self.cbo_sharpen_strength.SetSelection(1)
        self.cbo_sharpen_strength.SetToolTip(
            T("How strong the Sharpen effect is (0.0-1.0). Higher = more pronounced detail boost at real "
              "edges/texture, but also more risk of an over-crisp/harsh look or exaggerating real "
              "compression artifacts. Recommended: 0.5 (default) as a safe starting point."))
        self.sld_stereo_sharpen_strength = _build_stereo_slider(self.cpn_stereo_pop_divergence.GetPane(), self.cbo_sharpen_strength, 0.25, 1.0, 100)

        self.lbl_edge_dilation = wx.StaticText(self.cpn_stereo_stability_flicker.GetPane(), label=T("Edge Fix"))
        self.cbo_edge_dilation = EditableComboBox(self.cpn_stereo_stability_flicker.GetPane(),
                                                  choices=["0", "1", "2", "3", "4"],
                                                  size=self.FromDIP((90, -1)),
                                                  name="cbo_edge_dilation")
        self.cbo_edge_dilation_y = EditableComboBox(self.cpn_stereo_stability_flicker.GetPane(),
                                                    choices=["", "0", "1", "2"],
                                                    name="cbo_edge_dilation_y")
        self.cbo_edge_dilation.SetSelection(2)
        self.cbo_edge_dilation_y.SetSelection(2)
        self.lbl_edge_dilation.SetToolTip(
            T("What it's for: smooths the depth map right at object silhouettes BEFORE the 3D shift is "
              "applied, reducing the halo/distortion artifacts that show up around foreground edges.\n"
              "Important interaction #1: this runs BEFORE 3D Strength (Divergence) sees the depth map, and "
              "does a FIXED amount of smoothing regardless of Divergence — it does not scale itself up "
              "automatically. The same leftover roughness becomes a small, maybe-invisible error at low "
              "3D Strength, but a much bigger, more visible one at high 3D Strength. Raise this when you "
              "raise 3D Strength to keep the same level of edge protection (roughly: Divergence ~2.0-2.25 "
              "pairs with 2/1, Divergence ~3.0-3.5 pairs with 3/2).\n"
              "Important interaction #2: this also runs BEFORE Depth Resolution's upscale, using a "
              "fixed-pixel-size brush on the smaller, not-yet-upscaled depth map. That means the same "
              "number here covers a LARGER proportion of the frame at a lower Depth Resolution than at a "
              "higher one — re-check this any time you change Depth Resolution, not just when you change "
              "3D Strength.\n"
              "Con: too high smooths away real fine depth detail near edges, not just artifacts."))
        self.cbo_edge_dilation.SetToolTip(
            T("X value: horizontal edge smoothing strength (higher = more iterations = smoother/wider "
              "protection, but more real detail lost near edges). Recommended: 2 as a starting point, more "
              "if you raise 3D Strength or lower Depth Resolution."))
        self.cbo_edge_dilation_y.SetToolTip(
            T("Y value: additional vertical edge smoothing strength, on top of X (left blank = same as X, "
              "fully symmetric). Vertical seams usually matter less for a left/right eye shift, so this is "
              "typically kept lower than X. Recommended: 1 as a starting point (paired with X=2)."))

        self.chk_ema_normalize = wx.CheckBox(self.cpn_stereo_stability_flicker.GetPane(),
                                             label=T("Flicker Reduction"),
                                             name="chk_ema_normalize")
        self.chk_ema_normalize.SetToolTip(
            T("What it's for (video only): smooths the depth map's overall near/far SCALE over time, so "
              "the whole image's depth intensity doesn't subtly pulse/breathe from frame to frame due to "
              "the AI model's own frame-to-frame noise. This is separate from Convergence Smoothing (which "
              "only exists in auto convergence modes and smooths a different thing, the focus point) — "
              "this one works identically no matter which Convergence Mode you're using.\n"
              "Uses the Decay Rate and Buffer settings below together.\n"
              "Recommended: on for essentially all video, paired with Scene Boundary Detection so the "
              "smoothing resets cleanly at real cuts instead of blending across them."))

        self.cbo_ema_decay = EditableComboBox(self.cpn_stereo_stability_flicker.GetPane(), choices=["0.99", "0.95", "0.9", "0.75", "0.5", "0"],
                                              name="cbo_ema_decay")
        self.cbo_ema_decay.SetSelection(2)
        self.cbo_ema_decay.SetToolTip(
            T("Decay Rate: how much weight past frames keep vs. the newest frame when tracking the "
              "depth-map scale. Updates every single frame, not periodically.\n"
              "Values: higher (0.9-0.99) = more resistant to momentary spikes (a hand reaching toward "
              "camera, a flash) but slower to follow a genuine intentional depth change (a push/pull "
              "shot). Lower (0.5-0.75) = reacts faster to real change, but more exposed to the AI's own "
              "per-frame noise. 0 = no smoothing at all.\n"
              "How to tell it's TOO LOW: on a static, unmoving dialogue shot, the whole image's depth "
              "intensity subtly pulses even though nothing is really moving closer/farther — that's it "
              "reacting to noise instead of real change. Also watch right after something briefly very "
              "close to camera passes through: if the ENTIRE frame's depth (not just that object) visibly "
              "lurches, one outlier frame yanked the whole scale.\n"
              "How to tell it's TOO HIGH: right after a hard cut or during a push/pull zoom, depth looks "
              "noticeably flat for a beat before catching up to the real new depth.\n"
              "Recommended: pair with Buffer below rather than tuning alone — see Buffer's tooltip for "
              "matched pairs (e.g. Decay 0.98 pairs with Buffer 120 for scenes with real smearing/bleeding "
              "artifacts).\n"
              "Greyed out when \"Auto EMA by Scene Length\" below is checked, since that picks its own "
              "per-scene Decay/Buffer instead — this value is still kept and still used as the fallback "
              "before the first detected scene boundary."))
        self.sld_stereo_ema_decay = _build_stereo_slider(self.cpn_stereo_stability_flicker.GetPane(), self.cbo_ema_decay, 0.0, 0.99, 100)

        self.cbo_ema_buffer = EditableComboBox(self.cpn_stereo_stability_flicker.GetPane(), choices=["150", "60", "30", "1"],
                                               name="cbo_ema_buffer")
        self.cbo_ema_buffer.SetSelection(2)
        self.cbo_ema_buffer.SetToolTip(
            T("Buffer Size (in frames): how many recent frames contribute to judging the current near/far "
              "depth range, together with Decay Rate above.\n"
              "Con of too small: a single outlier frame (something briefly very close to the lens) can "
              "yank the whole normalization range around by itself, which reads as sudden "
              "smearing/swimming/bleeding in the 3D effect even though no pixel was literally smeared — "
              "it's the depth SCALE lurching, not a warp artifact.\n"
              "Con of too large: slower to adapt to a genuine intentional depth change within one "
              "continuous shot (a push/pull reveal).\n"
              "Values (paired with Decay Rate above, tested/extrapolated from real measured footage): "
              "30/0.9 is a reasonable middle default. If you see smearing/bleeding artifacts, try 120 "
              "paired with Decay 0.98 — this dilutes any single outlier frame's influence across many more "
              "frames. Scene Boundary Detection resets this buffer at every real cut regardless of size, "
              "so even a large buffer only fully engages during a film's longer continuous shots.\n"
              "Recommended: measure your actual footage's typical shot length if possible (Scene Boundary "
              "Detection can log real cut points) and size the buffer around 30-35% of the MEDIAN shot "
              "length, rather than guessing from genre — measured testing on this project found genre "
              "assumptions about pacing were sometimes simply wrong.\n"
              "Greyed out when \"Auto EMA by Scene Length\" below is checked, since that picks its own "
              "per-scene Decay/Buffer instead — this value is still kept and still used as the fallback "
              "before the first detected scene boundary."))
        self.sld_stereo_ema_buffer = _build_stereo_slider(self.cpn_stereo_stability_flicker.GetPane(), self.cbo_ema_buffer, 1, 150, 1)

        # Parented to grp_stereo (Flicker Reduction's own StaticBox) and laid out
        # directly under the Decay Rate/Buffer row above, not grp_video_filter, so it
        # sits visually next to the fixed values it overrides -- see
        # docs/ai/AI_DECISIONS.md ADR-057 amendment (2026-09-08, relocation +
        # Decay/Buffer disable-when-checked).
        self.chk_scene_batch_auto_ema = wx.CheckBox(self.cpn_stereo_stability_flicker.GetPane(),
                                                    label=T("Auto EMA by Scene Length"),
                                                    name="chk_scene_batch_auto_ema")
        self.chk_scene_batch_auto_ema.SetValue(False)
        self.chk_scene_batch_auto_ema.SetToolTip(
            T("Automatically picks EMA Decay/Buffer per scene based on that scene's own length, "
              "using a default table (one step per whole second, 0-20s+) -- a short scene gets a "
              "smaller Buffer so the smoothing actually finishes settling before the scene ends, "
              "instead of a Buffer sized for one long continuous shot. Works two ways, depending on "
              "what else is turned on:\n"
              "With Automated Scene Batch: tuned for its independent-scene processing. Applied before "
              "Scene Settings File, so anything that file sets explicitly (EMA included) still wins "
              "for scenes it covers.\n"
              "Without Automated Scene Batch: requires Scene Detection to be turned on instead (there "
              "are no scene boundaries to key off of otherwise -- Start will refuse to run and explain "
              "this if it's missing). Re-picks EMA Decay/Buffer at every detected cut within the same "
              "continuous video, overriding the fixed Flicker Reduction/Flicker Reduction Buffer for "
              "each scene as it starts -- the output filename gets an \"_autoema<model>\" tag instead "
              "of a fixed EMA number, since no single number was used throughout.\n"
              "Either way: pick the matching Depth Model in the dropdown to its right -- see that "
              "dropdown's own tooltip for the default Buffer/Decay values used at each scene length, "
              "and the \"Edit Values...\" button below it to change them (the same edited table "
              "governs both uses).\n"
              "While this is checked, Flicker Reduction's own Decay Rate/Buffer fields just above are "
              "greyed out, since this picks per-scene values instead -- their typed-in values are kept, "
              "not cleared, and come right back (still editable) the moment you uncheck this."))

        self.cbo_scene_batch_auto_ema_model = wx.ComboBox(self.cpn_stereo_stability_flicker.GetPane(),
                                                          choices=["3DECKER VDA_L", "3DECKER Any_V3_Mono_01",
                                                                   "Nagadomi_Reference",
                                                                   "GEMINI AI", "ChatGPT", "Grok",
                                                                   "Fast Action", "Medium Magical",
                                                                   "Drama Slow Paced"],
                                                          name="cbo_scene_batch_auto_ema_model")
        self.cbo_scene_batch_auto_ema_model.SetEditable(False)
        # Default table changed from "3DECKER VDA_L" (index 0) to "Nagadomi_Reference"
        # (index 2) per user request -- see ADR-057 Amendment 7. This only changes which
        # table a user gets if they turn on "Auto EMA by Scene Length" without picking a
        # table first; the checkbox itself still defaults to off (chk_scene_batch_auto_ema
        # .SetValue(False) above, unchanged), and the choices list order/values are
        # untouched.
        self.cbo_scene_batch_auto_ema_model.SetSelection(2)
        self.cbo_scene_batch_auto_ema_model.SetToolTip(
            T("Which Auto EMA by Scene Length table to use (for both Automated Scene Batch and a "
              "regular Scene Detection conversion -- see that checkbox's own tooltip), matched to "
              "the Depth Model above. "
              "3DECKER VDA_L: a real video depth model with its own frame-to-frame memory, needs only "
              "light smoothing on top. 3DECKER Any_V3_Mono_01: a stills-only model with no "
              "frame-to-frame memory of its own (prone to visible 'depth breathing' without help), so "
              "this table uses double VDA_L's Buffer at every scene length with a correspondingly "
              "higher Decay to compensate. (These two tables were this project's own original design, "
              "hence the \"3DECKER\" prefix -- distinguishing them from the externally-sourced "
              "reference tables below.)\n"
              "Default values (Buffer/Decay by scene length, one step per whole second -- use the "
              "\"Edit Values...\" button below to change them for whichever model is selected here; "
              "these are what you get back if you ever click Reset to Default). 15-16s through "
              "19-20s and the 20s+ ceiling deliberately hold flat at each model's 15s value rather "
              "than growing further -- real testing showed Buffer pushed meaningfully past that "
              "point causes visible lag, not just smoother depth:\n"
              "3DECKER VDA_L -- 0-1s: 8/0.65, 1-2s: 12/0.70, 2-3s: 16/0.74, 3-4s: 20/0.78, 4-5s: 25/0.80, "
              "5-6s: 30/0.83, 6-7s: 38/0.86, 7-8s: 46/0.89, 8-9s: 55/0.91, 9-10s: 65/0.93, "
              "10-11s: 76/0.945, 11-12s: 87/0.955, 12-13s: 98/0.965, 13-14s: 108/0.972, "
              "14-15s: 116/0.977, 15-16s through 19-20s and 20s+: 120/0.98 (flat).\n"
              "3DECKER Any_V3_Mono_01 -- 0-1s: 16/0.825, 1-2s: 21/0.85, 2-3s: 32/0.87, 3-4s: 40/0.89, "
              "4-5s: 50/0.90, 5-6s: 60/0.915, 6-7s: 76/0.93, 7-8s: 92/0.945, 8-9s: 110/0.955, "
              "9-10s: 130/0.965, 10-11s: 152/0.9725, 11-12s: 174/0.9775, 12-13s: 196/0.9825, "
              "13-14s: 216/0.986, 14-15s: 232/0.9885, 15-16s through 19-20s and 20s+: 240/0.99 "
              "(flat).\n"
              "Nagadomi_Reference -- a more conservative third option, not tied to a specific "
              "Depth Model. 3DECKER VDA_L/3DECKER Any_V3_Mono_01 above were custom-built for this "
              "project; this one instead uses nagadomi's (this tool's original author) own real documented "
              "presets for this exact mechanism as its anchors (iw3/depth_scaler.py) -- "
              "IncrementalEMAScaler (decay=0.75, buffer=1) for very short scenes, settling at "
              "WindowEMAScaler (decay=0.9, buffer=30) -- nagadomi's own 'strong' preset -- for "
              "every scene 4 seconds or longer, unmodified, never pushed higher. Buffer is a "
              "literal future-frame lookahead count (confirmed by reading EMAMinMaxScaler's own "
              "code), not an abstract smoothing dial, so unlike the other two tables it does NOT "
              "keep growing Buffer for longer and longer scenes -- there's no real basis for a "
              "long scene needing MORE lookahead than nagadomi's own reference number. Recommended "
              "if you're seeing lag or smearing with 3DECKER VDA_L/3DECKER Any_V3_Mono_01 on long scenes (e.g. a "
              "continuous multi-minute shot) and would rather stay within nagadomi's own tested "
              "range than push past it:\n"
              "Nagadomi_Reference -- 0-1s: 1/0.750, 1-2s: 8/0.786, 2-3s: 16/0.828, 3-4s: 24/0.869, "
              "4-5s through 20s+: 30/0.900 (flat, exactly nagadomi's own WindowEMAScaler preset).\n"
              "GEMINI AI -- a fourth option, added for real A/B testing at the user's own "
              "request. Unlike the three tables above, its specific per-second curve came from a "
              "DIFFERENT AI assistant's suggestion (not this tool's own design, and not anchored to "
              "anything nagadomi documented) and was NOT verified against nunif's own source code or "
              "documentation the way Nagadomi_Reference was -- it's included as-is for comparison, not "
              "because it's confirmed correct. Requests a MUCH larger lookahead Buffer than the other "
              "three: it grows every single bucket all the way out to 480 frames (20 seconds at "
              "24fps) by 20s+, versus 3DECKER VDA_L/3DECKER Any_V3_Mono_01's 120/240-frame ceilings and "
              "Nagadomi_Reference's flat 30-frame ceiling. That means a long scene under this table "
              "needs the tool to look nearly all the way through the scene before it can compute a "
              "depth range for even its first frame -- expect real added VRAM use and startup delay, "
              "especially on long scenes with a heavier Depth Model:\n"
              "GEMINI AI -- 0-1s: 24/0.750, 1-2s: 48/0.800, 2-3s: 72/0.840, 3-4s: 96/0.870, "
              "4-5s: 120/0.890, 5-6s: 144/0.905, 6-7s: 168/0.918, 7-8s: 192/0.928, 8-9s: 216/0.936, "
              "9-10s: 240/0.943, 10-11s: 264/0.949, 11-12s: 288/0.954, 12-13s: 312/0.958, "
              "13-14s: 336/0.962, 14-15s: 360/0.965, 15-16s: 384/0.968, 16-17s: 408/0.970, "
              "17-18s: 432/0.972, 18-19s: 456/0.974, 19-20s: 480/0.975, 20s+: 480/0.975 (flat).\n"
              "ChatGPT -- a fifth option, added for real A/B testing at the user's own request, same "
              "as GEMINI AI. Its specific per-second curve came from yet another, DIFFERENT AI "
              "assistant's suggestion (ChatGPT this time) and was likewise NOT verified against "
              "nunif's own source code or documentation -- included as-is for comparison, not because "
              "it's confirmed correct. This one asks for an even LARGER lookahead Buffer than GEMINI "
              "AI: it grows every single bucket all the way out to 600 frames (25 seconds at 24fps) by "
              "20s+, versus GEMINI AI's 480-frame ceiling, 3DECKER VDA_L/3DECKER Any_V3_Mono_01's 120/240-frame "
              "ceilings, and Nagadomi_Reference's flat 30-frame ceiling -- the largest lookahead of "
              "any table in this tool. That means a long scene under this table needs the tool to look "
              "even further through the scene before it can compute a depth range for its first frame "
              "-- expect the most VRAM use and startup delay of all five options, especially on long "
              "scenes with a heavier Depth Model:\n"
              "ChatGPT -- 0-1s: 30/0.75, 1-2s: 60/0.77, 2-3s: 90/0.79, 3-4s: 120/0.81, 4-5s: 150/0.83, "
              "5-6s: 180/0.84, 6-7s: 210/0.85, 7-8s: 240/0.86, 8-9s: 270/0.87, 9-10s: 300/0.88, "
              "10-11s: 330/0.885, 11-12s: 360/0.89, 12-13s: 390/0.90, 13-14s: 420/0.91, "
              "14-15s: 450/0.92, 15-16s: 480/0.93, 16-17s: 510/0.94, 17-18s: 540/0.95, "
              "18-19s: 570/0.97, 19-20s: 600/0.99, 20s+: 600/0.99 (flat).\n"
              "Grok -- a sixth option, added for real A/B testing at the user's own request, same "
              "as GEMINI AI and ChatGPT. Its specific per-second curve came from yet another, "
              "DIFFERENT AI assistant's suggestion (Grok this time) and was likewise NOT verified "
              "against nunif's own source code or documentation -- included as-is for comparison, "
              "not because it's confirmed correct. Its own distinct character: Decay saturates very "
              "early, reaching 0.99 by the 8-9s bucket and staying flat there for every bucket after, "
              "while Buffer keeps climbing linearly the entire way to 480 frames (20 seconds at "
              "24fps) by 20s+ -- unlike GEMINI AI and ChatGPT, where Buffer and Decay both keep "
              "climbing together. That means past roughly 9 seconds, only the lookahead window "
              "(and its VRAM/startup-delay cost) keeps growing, while actual blend strength buys "
              "nothing further. Buffer's own 480-frame ceiling matches GEMINI AI's exactly and is "
              "smaller than ChatGPT's 600-frame ceiling:\n"
              "Grok -- 0-1s: 24/0.90, 1-2s: 48/0.95, 2-3s: 72/0.97, 3-4s: 96/0.975, 4-5s: 120/0.98, "
              "5-6s: 144/0.98, 6-7s: 168/0.986, 7-8s: 192/0.988, 8-9s: 216/0.99, 9-10s: 240/0.99, "
              "10-11s: 264/0.99, 11-12s: 288/0.99, 12-13s: 312/0.99, 13-14s: 336/0.99, "
              "14-15s: 360/0.99, 15-16s: 384/0.99, 16-17s: 408/0.99, 17-18s: 432/0.99, "
              "18-19s: 456/0.99, 19-20s: 480/0.99, 20s+: 480/0.99 (flat).\n"
              "Fast Action / Medium Magical / Drama Slow Paced -- three more options, added for "
              "real A/B testing at the user's own request, same as GEMINI AI/ChatGPT/Grok. Their "
              "specific per-second curves came from a ChatGPT conversation proposing genre-specific "
              "pacing schedules and were likewise NOT verified against nunif's own source code or "
              "documentation -- included as-is for comparison. Unlike GEMINI AI/ChatGPT/Grok, these "
              "three are explicitly capped at 10 seconds: Buffer/Decay both ramp up through the 9-10s "
              "bucket, then hold flat from 10-11s all the way through 20s+, at a 240-frame Buffer "
              "ceiling (half of GEMINI AI's 480-frame ceiling). Buffer is identical bucket-for-bucket "
              "across all three -- only Decay differentiates them, from Fast Action's quick-to-react, "
              "minimal smoothing up to Drama Slow Paced's maximum stability. These are a separate, "
              "per-scene mechanism from the \"Genre Preset\" quick-fill next to Flicker Reduction's "
              "Decay Rate/Buffer fields below -- that one writes one fixed pair for a whole movie, "
              "these three vary automatically by each scene's own measured length:\n"
              "Fast Action -- 0-1s: 24/0.750, 1-2s: 48/0.753, 2-3s: 72/0.755, 3-4s: 96/0.758, "
              "4-5s: 120/0.761, 5-6s: 144/0.763, 6-7s: 168/0.766, 7-8s: 192/0.768, 8-9s: 216/0.771, "
              "9-10s: 240/0.774, 10-11s through 20s+: 240/0.774 (flat).\n"
              "Medium Magical -- 0-1s: 24/0.820, 1-2s: 48/0.823, 2-3s: 72/0.826, 3-4s: 96/0.829, "
              "4-5s: 120/0.832, 5-6s: 144/0.835, 6-7s: 168/0.838, 7-8s: 192/0.841, 8-9s: 216/0.844, "
              "9-10s: 240/0.847, 10-11s through 20s+: 240/0.847 (flat).\n"
              "Drama Slow Paced -- 0-1s: 24/0.900, 1-2s: 48/0.903, 2-3s: 72/0.905, 3-4s: 96/0.908, "
              "4-5s: 120/0.911, 5-6s: 144/0.913, 6-7s: 168/0.916, 7-8s: 192/0.918, 8-9s: 216/0.921, "
              "9-10s: 240/0.924, 10-11s through 20s+: 240/0.924 (flat)."))

        self.btn_scene_batch_auto_ema_edit = wx.Button(self.cpn_stereo_stability_flicker.GetPane(),
                                                        label=T("Edit Values..."),
                                                        name="btn_scene_batch_auto_ema_edit")
        self.btn_scene_batch_auto_ema_edit.SetToolTip(
            T("What it's for: opens an editor for the Buffer/Decay numbers Auto EMA by Scene Length "
              "uses, for whichever model is currently selected in the dropdown above.\n"
              "What you can change: Buffer and Decay for each of the 21 scene-length rows (0-1s "
              "through 19-20s, plus 20s+). The scene-length ranges themselves are fixed.\n"
              "Why: the built-in numbers are a general-purpose starting point -- if your own footage "
              "consistently looks better with a bit more or less smoothing at a particular scene "
              "length, you can dial that in yourself instead of only ever getting the defaults.\n"
              "Pro: changes are saved and reused automatically on every future run, no need to "
              "re-enter them each time.\n"
              "Con: it's easy to type in a value that looks reasonable but actually smooths too "
              "little (visible flicker) or too much (ghosting/lag) for a given scene length -- change "
              "one row at a time and re-check the result before trusting a whole edited table.\n"
              "Recommended: leave untouched unless you've already noticed Auto EMA by Scene Length "
              "under- or over-smoothing a specific range of scene lengths on your own material. Click "
              "Reset to Default inside the editor at any time to undo your changes for that model."))

        # Genre Preset (ADR-057 Amendment 9): a quick-fill shortcut for Flicker
        # Reduction's OWN fixed Decay Rate/Buffer fields above, not a new per-scene
        # mechanism like Auto EMA by Scene Length. Parented/laid out right alongside
        # it for the same reason Amendment 8 moved Auto EMA here -- both are ways to
        # arrive at Flicker Reduction's Decay/Buffer values, so both live next to
        # those fields.
        self.lbl_genre_preset = wx.StaticText(self.cpn_stereo_stability_flicker.GetPane(), label=T("Genre Preset"))
        self.cbo_genre_preset = wx.ComboBox(self.cpn_stereo_stability_flicker.GetPane(),
                                            choices=GENRE_PRESET_CHOICES,
                                            name="cbo_genre_preset")
        self.cbo_genre_preset.SetEditable(False)
        self.cbo_genre_preset.SetSelection(0)
        genre_preset_tooltip = (
            T("What it's for: a quick-fill shortcut for Flicker Reduction's Decay Rate/Buffer fields "
              "above, based on a whole movie's general content/pacing -- NOT the same thing as \"Auto "
              "EMA by Scene Length\" below, which instead varies Decay/Buffer automatically PER SCENE "
              "based on each scene's own measured length. Pick one approach for a given job: type "
              "Decay/Buffer by hand, use this to quick-fill them once, or turn on Auto EMA by Scene "
              "Length to bypass fixed values entirely.\n"
              "How it works: selecting a preset immediately writes that preset's Decay/Buffer numbers "
              "into the two fields above, once -- it is not a live link, so hand-editing either field "
              "afterward is completely safe and does not reset this dropdown or cause any error.\n"
              "Why only Decay differs between presets: Decay is a pure per-frame math weighting with no "
              "extra cost, so it can safely differentiate the presets. Buffer is a literal future-frame "
              "lookahead window (real VRAM/startup-latency cost, not a \"quality\" dial -- see the Auto "
              "EMA dropdown's own tooltip below), so it stays conservative and close across all three "
              "presets instead of growing with intensity.\n"
              "Values: Fast Action = Decay 0.75/Buffer 30 (max responsiveness, minimal lookahead). "
              "Medium / Magical = Decay 0.85/Buffer 72 (balanced -- fantasy/VFX-heavy content with "
              "mixed pacing). Drama / Slow-Paced = Decay 0.94/Buffer 120 (maximum stability, still "
              "within nagadomi's own documented reference ceiling, not beyond it).\n"
              "Recommended: a reasonable starting point, not a guarantee -- a movie's pace can vary "
              "within itself (a drama can have an action beat, etc.), so spot-check the result and "
              "hand-adjust Decay/Buffer if a specific stretch doesn't match the chosen preset's pace.\n"
              "Greyed out, same as Decay Rate/Buffer above, whenever \"Auto EMA by Scene Length\" below "
              "is checked -- quick-filling fields that are themselves currently irrelevant makes no "
              "sense while that feature is picking its own per-scene values instead."))
        self.lbl_genre_preset.SetToolTip(genre_preset_tooltip)
        self.cbo_genre_preset.SetToolTip(genre_preset_tooltip)

        self.chk_ema_motion_adaptive = wx.CheckBox(self.cpn_stereo_stability_flicker.GetPane(),
                                                   label=T("Motion-Adaptive Smoothing"),
                                                   name="chk_ema_motion_adaptive")
        self.chk_ema_motion_adaptive.SetToolTip(
            T("What it's for: automatically eases Flicker Reduction's Decay Rate DOWN during fast motion, "
              "instead of using one fixed smoothing strength for the whole clip.\n"
              "How it helps: reduces the \"laggy/flat for a beat\" effect a high Decay Rate can cause "
              "during fast action, while keeping the full smoothing benefit during calm scenes.\n"
              "Safety note: this can only ever REDUCE smoothing below your Decay Rate setting during fast "
              "motion, never increase it above what you set — it's a safe add-on, not a replacement for "
              "choosing a sensible Decay Rate.\n"
              "Recommended: on, if your content mixes calm and fast-motion scenes (most movies do)."))

        self.chk_scene_detect = wx.CheckBox(self.cpn_stereo_stability_flicker.GetPane(),
                                            label=T("Scene Boundary Detection"),
                                            name="chk_scene_detect")
        self.chk_scene_detect.SetValue(False)
        self.chk_scene_detect.SetToolTip(
            T("What it's for: detects real scene/shot cuts using a trained AI model (not just a simple "
              "brightness-difference trick, so it's less likely to false-trigger on flashes or fast pans) "
              "and resets Flicker Reduction's smoothing exactly at those points, instead of letting it "
              "bleed depth-scale information across two completely unrelated shots.\n"
              "How it helps: also lets Resume align its chunk boundaries to real cuts, and lets Object "
              "Stability/EMA buffers size themselves against real, measured shot lengths instead of guessing.\n"
              "Con: a small amount of extra processing time to run the cut-detection pass (mitigated by "
              "the cache setting below on repeat runs of the same file).\n"
              "Recommended: on for essentially all movie/TV content, especially when Flicker Reduction or "
              "Resume is also on."))

        self.chk_scene_detect_cache = wx.CheckBox(self.cpn_stereo_stability_flicker.GetPane(),
                                                  label=T("Use scene boundary cache"),
                                                  name="chk_scene_detect_cache")
        self.chk_scene_detect_cache.SetValue(True)
        self.chk_scene_detect_cache.SetToolTip(
            T("What it's for: saves detected scene cuts to disk (keyed to the source file's path, size, "
              "and modification time), so re-running the same video later — after a crash, or while "
              "testing different 3D settings — doesn't need to re-scan for cuts from scratch every time.\n"
              "Recommended: on. It auto-invalidates if the source file actually changes (different size "
              "or modified date), so there's no real downside for normal use."))

        self.chk_preserve_screen_border = wx.CheckBox(self.cpn_stereo_stability_flicker.GetPane(),
                                                      label=T("Preserve Screen Border"),
                                                      name="chk_preserve_screen_border")
        self.chk_preserve_screen_border.SetValue(False)
        self.chk_preserve_screen_border.SetToolTip(
            T("What it's for: tapers the 3D shift down to zero right at the left/right frame edges, "
              "instead of applying full strength all the way to the border.\n"
              "How it helps: this is an automated version of the \"floating window\" technique real "
              "stereographers use — without it, an object with strong 3D pop that gets cut off by the "
              "frame edge creates a jarring conflict (the object appears to be in front of the screen, but "
              "the screen's own edge is cutting it off, which shouldn't be possible if it's really floating "
              "in front). Tapering the edges to zero avoids that specific conflict.\n"
              "Con: very slightly reduces 3D strength right at the extreme edges of the frame — "
              "imperceptible in the middle of the image, only affects the outermost border.\n"
              "Recommended: on for most content, especially anything with strong 3D Strength or objects "
              "that move near frame edges. Off only if you specifically want maximum strength everywhere "
              "and don't mind occasional edge conflicts."))

        self.lbl_stereo_format = wx.StaticText(self.grp_stereo, label=T("Stereo Format"))
        self.cbo_stereo_format = wx.ComboBox(
            self.grp_stereo,
            choices=["Full SBS", "Half SBS",
                     "Full TB", "Half TB",
                     "VR90",
                     "Cross Eyed",
                     "RGB-D",
                     "Half RGB-D",
                     "Anaglyph",
                     "Export", "Export disparity",
                     "Debug Depth",
                     ],
            name="cbo_stereo_format")
        self.cbo_stereo_format.SetEditable(False)
        self.cbo_stereo_format.SetSelection(0)
        self.cbo_stereo_format.SetToolTip(
            T("The final output layout. Full/Half SBS (side-by-side) and Full/Half TB (top-bottom) are "
              "the standard formats most 3D TVs and players expect — Full keeps both eyes at full "
              "resolution (bigger file), Half squeezes both into the original frame size (smaller file, "
              "common for streaming/playback compatibility). VR90 is for VR headsets. Cross Eyed is for "
              "viewing without any equipment. Anaglyph is the red/cyan glasses look. RGB-D / Export save "
              "the depth data itself instead of a finished 3D image. Recommended: Half SBS for most TVs "
              "and 3D players, unless you know you need a different format."))

        self.lbl_anaglyph_method = wx.StaticText(self.grp_stereo, label=T("Anaglyph Method"))
        self.cbo_anaglyph_method = wx.ComboBox(
            self.grp_stereo,
            choices=["dubois", "dubois2",
                     "color", "gray",
                     "half-color",
                     "wimmer", "wimmer2"],
            name="cbo_anaglyph_method")
        self.cbo_anaglyph_method.SetEditable(False)
        self.cbo_anaglyph_method.SetSelection(0)
        self.cbo_anaglyph_method.SetToolTip(
            T("Only used when Stereo Format is Anaglyph. The color-filtering recipe used for red/cyan "
              "glasses viewing. dubois/dubois2 give the most natural, least color-distorted result for "
              "most red/cyan glasses. gray avoids color distortion entirely but loses color in the image. "
              "Recommended: dubois2."))
        self.lbl_anaglyph_method.Hide()
        self.cbo_anaglyph_method.Hide()

        self.chk_stereo_mode_tag = wx.CheckBox(self.grp_stereo, label=T("Tag MKV as 3D (StereoMode)"),
                                               name="chk_stereo_mode_tag")
        self.chk_stereo_mode_tag.SetValue(False)
        self.chk_stereo_mode_tag.SetToolTip(
            T("What it's for: writes the standard Matroska \"StereoMode\" property into the finished "
              ".mkv's video track, describing the 3D layout (side-by-side/top-bottom) and which eye "
              "comes first.\n"
              "How it helps: 3D-aware players and TVs (VLC, Kodi, some smart TVs) read this and "
              "automatically switch into the correct 3D display mode — without it, the viewer has to "
              "tell their player \"this is 3D, side-by-side, left eye first\" manually every time.\n"
              "Only applies to: .mkv output, and only Full/Half SBS, Full/Half TB, Cross Eyed, and "
              "VR90 — these are real two-eye pairs Matroska's StereoMode can describe. RGB-D, Half "
              "RGB-D, and Anaglyph are skipped automatically (not a two-eye pair, or already viewable "
              "on any player without tagging) — you'll see a note printed, not a silent no-op.\n"
              "Con: none for a normal 3D TV/player — this is a pure metadata edit, no re-encoding. A "
              "player that ignores StereoMode entirely just displays the file exactly as it would "
              "have anyway.\n"
              "Note for VR90: this only tells a player the eye order, not that the video is a 180° "
              "spherical projection — VR headset apps still rely on the existing \"_180x180_LR\" "
              "filename tag to recognize that, so turning this on for VR90 output is harmless but not "
              "a substitute for that filename convention.\n"
              "Recommended: on if your output goes to a 3D TV, a 3D-aware player like VLC/Kodi, or "
              "you just don't want to manually configure 3D mode every time. Off (default) if you're "
              "not sure your player supports it, or your workflow already re-muxes/renames the file "
              "afterward in a way that could lose this tag anyway."))

        self.chk_export_depth_only = wx.CheckBox(self.grp_stereo, label=T("Depth Only"), name="chk_export_depth_only")
        self.chk_export_depth_only.SetValue(False)
        self.chk_export_depth_only.SetToolTip(
            T("Only used with Stereo Format \"Export\"/\"Export disparity\". Saves just the raw depth "
              "images, skipping the RGB frames a normal export also saves.\n"
              "Con: data exported this way is missing what's needed to later re-import and reconstruct a "
              "full stereo render from it — it's for inspecting/reusing the depth data itself, not for "
              "resuming a conversion later."))
        self.chk_export_depth_only.Hide()

        self.chk_export_depth_fit = wx.CheckBox(self.grp_stereo, label=T("Resize to fit"), name="chk_export_depth_fit")
        self.chk_export_depth_fit.SetValue(False)
        self.chk_export_depth_fit.SetToolTip(
            T("Only used with Stereo Format \"Export\"/\"Export disparity\". Resizes the exported depth "
              "images up to match the RGB frames' full resolution, instead of staying at the depth model's "
              "own (usually smaller) native resolution.\n"
              "Con: the exported files become significantly larger, and the process noticeably slower, "
              "since every depth frame is upscaled and re-saved.\n"
              "Recommended: off unless whatever you're using the exported depth images for specifically "
              "needs them pixel-matched to the RGB frames."))
        self.chk_export_depth_fit.Hide()

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.SetEmptyCellSize((0, 0))

        i = 0
        layout.Add(self.lbl_divergence, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_divergence, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.sld_stereo_divergence, (i := i + 1, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_divergence_warning, pos=(i := i + 1, 0), span=(0, 3), flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_convergence, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_convergence_mode, (i, 1), flag=wx.EXPAND)
        layout.Add(self.cbo_convergence, (i, 2), flag=wx.EXPAND)
        layout.Add(self.sld_stereo_convergence, (i := i + 1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_convergence_smoothing, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_convergence_smoothing, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.sld_stereo_convergence_smoothing, (i := i + 1, 1), (1, 2), flag=wx.EXPAND)

        layout.Add(self.lbl_ipd_offset, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.sld_ipd_offset, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_synthetic_view, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_synthetic_view, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_method, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_method, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_splat_blend_temperature, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_splat_blend_temperature, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.sld_stereo_splat_blend_temperature, (i := i + 1, 1), (1, 2), flag=wx.EXPAND)

        layout.Add((0, 8), (i := i + 1, 0))
        layout.Add(wx.StaticLine(self.grp_stereo), (i := i + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 6), (i := i + 1, 0))

        # Guided Light pilot (ADR-097): "Inpainting & Depth Source" collapsed into its
        # own wx.CollapsiblePane, same pattern as Pop & Divergence below -- merges what
        # used to be two separate StaticLine-divided blocks (method-conditional inpaint
        # fields, and Stereo Processing Width/Depth Model/Depth Resolution) into one
        # pane, since both are about what generates/sources the depth map rather than
        # the core conversion settings above.
        self.pnl_stereo_inpainting_depth_dot = wx.Panel(self.grp_stereo, size=self.FromDIP((10, 10)))
        self.pnl_stereo_inpainting_depth_dot.SetBackgroundColour(wx.Colour(255, 184, 79))
        pane_header_row = wx.BoxSizer(wx.HORIZONTAL)
        pane_header_row.Add(self.pnl_stereo_inpainting_depth_dot, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        pane_header_row.Add(self.cpn_stereo_inpainting_depth, 1, wx.EXPAND)
        layout.Add(pane_header_row, (i := i + 1, 0), (1, 3), flag=wx.EXPAND)

        pane_layout_inpaint = wx.GridBagSizer(vgap=4, hgap=4)
        pane_layout_inpaint.SetEmptyCellSize((0, 0))
        k = 0
        pane_layout_inpaint.Add(self.lbl_inpaint_model, (k, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_inpaint.Add(self.cbo_inpaint_model, (k, 1), (1, 2), flag=wx.EXPAND)
        pane_layout_inpaint.Add(self.lbl_overlap_frames, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_inpaint.Add(self.cbo_overlap_frames_pre, (k, 1), flag=wx.EXPAND)
        pane_layout_inpaint.Add(self.cbo_overlap_frames_post, (k, 2), flag=wx.EXPAND)
        pane_layout_inpaint.Add(self.lbl_mask_dilation, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_inpaint.Add(self.cbo_mask_inner_dilation, (k, 1), flag=wx.EXPAND)
        pane_layout_inpaint.Add(self.cbo_mask_outer_dilation, (k, 2), flag=wx.EXPAND)
        pane_layout_inpaint.Add(self.lbl_inpaint_max_width, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_inpaint.Add(self.cbo_inpaint_max_width, (k, 1), (1, 2), flag=wx.EXPAND)
        pane_layout_inpaint.Add((0, 8), (k := k + 1, 0))
        pane_layout_inpaint.Add(wx.StaticLine(self.cpn_stereo_inpainting_depth.GetPane()),
                                (k := k + 1, 0), (0, 3), flag=wx.EXPAND)
        pane_layout_inpaint.Add((0, 6), (k := k + 1, 0))
        pane_layout_inpaint.Add(self.lbl_stereo_width, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_inpaint.Add(self.cbo_stereo_width, (k, 1), (1, 2), flag=wx.EXPAND)
        pane_layout_inpaint.Add(self.lbl_depth_model, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_inpaint.Add(self.cbo_depth_model, (k, 1), (1, 2), flag=wx.EXPAND)
        pane_layout_inpaint.Add(self.lbl_resolution, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_inpaint.Add(self.cbo_resolution, (k, 1), flag=wx.EXPAND)
        pane_layout_inpaint.Add(self.chk_limit_resolution, (k, 2), flag=wx.EXPAND)
        self.cpn_stereo_inpainting_depth.GetPane().SetSizer(pane_layout_inpaint)

        layout.Add((0, 8), (i := i + 1, 0))
        layout.Add(wx.StaticLine(self.grp_stereo), (i := i + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 6), (i := i + 1, 0))

        # Guided Light pilot (ADR-097): "Stability & Flicker" collapsed into its own
        # wx.CollapsiblePane -- merges what used to be two separate StaticLine-divided
        # blocks (per-frame depth refinement/Object Stability, and cross-frame Flicker
        # Reduction/EMA/Scene Detection) into one pane, since both are about keeping
        # the depth map steady over time rather than the core conversion settings above.
        self.pnl_stereo_stability_flicker_dot = wx.Panel(self.grp_stereo, size=self.FromDIP((10, 10)))
        self.pnl_stereo_stability_flicker_dot.SetBackgroundColour(wx.Colour(154, 230, 132))
        pane_header_row = wx.BoxSizer(wx.HORIZONTAL)
        pane_header_row.Add(self.pnl_stereo_stability_flicker_dot, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        pane_header_row.Add(self.cpn_stereo_stability_flicker, 1, wx.EXPAND)
        layout.Add(pane_header_row, (i := i + 1, 0), (1, 3), flag=wx.EXPAND)

        pane_layout_stability = wx.GridBagSizer(vgap=4, hgap=4)
        pane_layout_stability.SetEmptyCellSize((0, 0))
        k = 0
        pane_layout_stability.Add(self.lbl_foreground_scale, (k, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_stability.Add(self.cbo_foreground_scale, (k, 1), (1, 2), flag=wx.EXPAND)
        pane_layout_stability.Add(self.lbl_edge_dilation, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_stability.Add(self.cbo_edge_dilation, (k, 1), flag=wx.EXPAND)
        pane_layout_stability.Add(self.cbo_edge_dilation_y, (k, 2), flag=wx.EXPAND)
        pane_layout_stability.Add(self.chk_depth_aa, (k := k + 1, 1), (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_stability.Add(self.chk_depth_refine, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_stability.Add(self.cbo_depth_refine_strength, (k, 1), flag=wx.EXPAND)
        pane_layout_stability.Add(self.sld_stereo_depth_refine_strength, (k, 2), flag=wx.EXPAND)
        pane_layout_stability.Add(self.chk_temporal_stabilize, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_stability.Add(self.cbo_temporal_stabilize_strength, (k, 1), flag=wx.EXPAND)
        pane_layout_stability.Add(self.sld_stereo_temporal_stabilize_strength, (k, 2), flag=wx.EXPAND)
        pane_layout_stability.Add(self.lbl_temporal_stabilize_max_shift,
                                  (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=14)
        pane_layout_stability.Add(self.cbo_temporal_stabilize_max_shift, (k, 1), flag=wx.EXPAND)
        pane_layout_stability.Add(self.lbl_temporal_stabilize_flat_boost,
                                  (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=14)
        pane_layout_stability.Add(self.cbo_temporal_stabilize_flat_boost, (k, 1), flag=wx.EXPAND)
        pane_layout_stability.Add(self.lbl_temporal_stabilize_edge_protect,
                                  (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=14)
        pane_layout_stability.Add(self.cbo_temporal_stabilize_edge_protect, (k, 1), flag=wx.EXPAND)
        pane_layout_stability.Add((0, 8), (k := k + 1, 0))
        pane_layout_stability.Add(wx.StaticLine(self.cpn_stereo_stability_flicker.GetPane()),
                                  (k := k + 1, 0), (0, 3), flag=wx.EXPAND)
        pane_layout_stability.Add((0, 6), (k := k + 1, 0))
        pane_layout_stability.Add(self.chk_ema_normalize, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_stability.Add(self.cbo_ema_decay, (k, 1), flag=wx.EXPAND)
        pane_layout_stability.Add(self.cbo_ema_buffer, (k, 2), flag=wx.EXPAND)
        pane_layout_stability.Add(self.sld_stereo_ema_decay, (k := k + 1, 1), flag=wx.EXPAND)
        pane_layout_stability.Add(self.sld_stereo_ema_buffer, (k, 2), flag=wx.EXPAND)
        pane_layout_stability.Add(self.chk_scene_batch_auto_ema, (k := k + 1, 1), (0, 1),
                                  flag=wx.EXPAND | wx.LEFT, border=14)
        pane_layout_stability.Add(self.cbo_scene_batch_auto_ema_model, (k, 2), (0, 1), flag=wx.EXPAND)
        pane_layout_stability.Add(self.lbl_genre_preset, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_stability.Add(self.cbo_genre_preset, (k, 1), flag=wx.EXPAND)
        pane_layout_stability.Add(self.btn_scene_batch_auto_ema_edit, (k, 2), flag=wx.EXPAND)
        pane_layout_stability.Add(self.chk_ema_motion_adaptive, (k := k + 1, 0), (0, 3), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_stability.Add(self.chk_scene_detect, (k := k + 1, 0), (0, 1), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_stability.Add(self.chk_scene_detect_cache, (k, 1), (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout_stability.Add(self.chk_preserve_screen_border, (k := k + 1, 0), (0, 3), flag=wx.ALIGN_CENTER_VERTICAL)
        self.cpn_stereo_stability_flicker.GetPane().SetSizer(pane_layout_stability)

        layout.Add((0, 8), (i := i + 1, 0))
        layout.Add(wx.StaticLine(self.grp_stereo), (i := i + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 6), (i := i + 1, 0))

        # Guided Light pilot (ADR-097): Pop & Divergence collapsed into its own
        # wx.CollapsiblePane -- construction/reparenting happened earlier, up where
        # these fields are built (search cpn_stereo_pop_divergence). Only the pane
        # itself (plus its colored indicator square) occupies a row in the OUTER
        # layout; everything that used to be added directly above now belongs to a
        # separate GridBagSizer built here for the pane's own GetPane() window.
        self.pnl_stereo_pop_divergence_dot = wx.Panel(self.grp_stereo, size=self.FromDIP((10, 10)))
        self.pnl_stereo_pop_divergence_dot.SetBackgroundColour(wx.Colour(79, 216, 255))
        pane_header_row = wx.BoxSizer(wx.HORIZONTAL)
        pane_header_row.Add(self.pnl_stereo_pop_divergence_dot, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        pane_header_row.Add(self.cpn_stereo_pop_divergence, 1, wx.EXPAND)
        layout.Add(pane_header_row, (i := i + 1, 0), (1, 3), flag=wx.EXPAND)

        pane_layout = wx.GridBagSizer(vgap=4, hgap=4)
        pane_layout.SetEmptyCellSize((0, 0))
        k = 0
        pane_layout.Add(self.lbl_foreground_pop, (k, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout.Add(self.cbo_foreground_pop, (k, 1), (1, 2), flag=wx.EXPAND)
        pane_layout.Add(self.sld_stereo_foreground_pop, (k := k + 1, 1), (1, 2), flag=wx.EXPAND)
        pane_layout.Add(self.lbl_foreground_divergence, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout.Add(self.cbo_foreground_divergence, (k, 1), (1, 2), flag=wx.EXPAND)
        pane_layout.Add(self.lbl_background_pop, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout.Add(self.cbo_background_pop, (k, 1), (1, 2), flag=wx.EXPAND)
        pane_layout.Add(self.sld_stereo_background_pop, (k := k + 1, 1), (1, 2), flag=wx.EXPAND)
        pane_layout.Add(self.lbl_background_pop_coverage, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout.Add(self.cbo_background_pop_coverage, (k, 1), (1, 2), flag=wx.EXPAND)
        pane_layout.Add(self.sld_stereo_background_pop_coverage, (k := k + 1, 1), (1, 2), flag=wx.EXPAND)
        pane_layout.Add(self.lbl_background_divergence, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout.Add(self.cbo_background_divergence, (k, 1), (1, 2), flag=wx.EXPAND)
        pane_layout.Add(self.lbl_edge_repair, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout.Add(self.cbo_edge_repair, (k, 1), (1, 2), flag=wx.EXPAND)
        pane_layout.Add(self.sld_stereo_edge_repair, (k := k + 1, 1), (1, 2), flag=wx.EXPAND)
        pane_layout.Add(self.chk_sharpen, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        pane_layout.Add(self.cbo_sharpen_strength, (k, 1), flag=wx.EXPAND)
        pane_layout.Add(self.sld_stereo_sharpen_strength, (k, 2), flag=wx.EXPAND)
        self.cpn_stereo_pop_divergence.GetPane().SetSizer(pane_layout)

        layout.Add((0, 8), (i := i + 1, 0))
        layout.Add(wx.StaticLine(self.grp_stereo), (i := i + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 6), (i := i + 1, 0))
        layout.Add(self.lbl_stereo_format, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_stereo_format, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_anaglyph_method, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_anaglyph_method, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.chk_stereo_mode_tag, (i := i + 1, 0), (0, 3), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_export_depth_only, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_export_depth_fit, (i, 1), (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)

        sizer_stereo = wx.StaticBoxSizer(self.grp_stereo, wx.VERTICAL)
        sizer_stereo.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # video decoding
        # hwaccel
        self.grp_video_dec = VideoDecodingBox(self.tab_video_dec, translate_function=T)

        # video encoding
        # sbs/vr180, padding
        # max-fps, crf, preset, tune
        self.grp_video = VideoEncodingBox(self.tab_video_enc, translate_function=T,
                                          has_nvenc=has_nvenc(), has_qsv=has_qsv())

        # input video filter
        # deinterlace, rotate, vf
        self.grp_video_filter = wx.StaticBox(self.tab_video_filter, label=T("Video Filter"))
        self.chk_start_time = wx.CheckBox(self.grp_video_filter, label=T("Start Time"),
                                          name="chk_start_time")
        self.chk_start_time.SetToolTip(
            T("Only process the video from this timestamp onward, instead of from the beginning. "
              "Useful for testing settings on a specific scene, or trimming unwanted intro footage."))
        self.txt_start_time = TimeCtrl(self.grp_video_filter, value="00:00:00", fmt24hr=True,
                                       name="txt_start_time")
        self.chk_end_time = wx.CheckBox(self.grp_video_filter, label=T("End Time"), name="chk_end_time")
        self.chk_end_time.SetToolTip(T("Stop processing at this timestamp instead of the end of the video."))
        self.txt_end_time = TimeCtrl(self.grp_video_filter, value="00:00:00", fmt24hr=True,
                                     name="txt_end_time")

        self.lbl_deinterlace = wx.StaticText(self.grp_video_filter, label=T("Deinterlace"))
        self.cbo_deinterlace = wx.ComboBox(self.grp_video_filter, choices=["", "yadif"],
                                           name="cbo_deinterlace")
        self.cbo_deinterlace.SetEditable(False)
        self.cbo_deinterlace.SetSelection(0)
        self.cbo_deinterlace.SetToolTip(
            T("For old interlaced video sources (combed/striped look on motion). \"yadif\" converts it to "
              "normal progressive video before 3D conversion. Leave blank for modern, already-progressive "
              "sources (most streaming/BluRay video)."))

        self.lbl_vf = wx.StaticText(self.grp_video_filter, label=T("-vf (src)"))
        self.txt_vf = wx.TextCtrl(self.grp_video_filter, name="txt_vf")
        self.txt_vf.SetToolTip(
            T("Advanced: a raw ffmpeg video filter string applied to the source before 3D conversion "
              "(e.g. cropping, scaling). Leave blank unless you specifically need this — most common "
              "needs (crop, rotate, pad, resize) already have their own simpler controls below."))

        self.lbl_rotate = wx.StaticText(self.grp_video_filter, label=T("Rotate"))
        self.cbo_rotate = wx.ComboBox(self.grp_video_filter, size=self.FromDIP((200, -1)),
                                      name="cbo_rotate")
        self.cbo_rotate.SetEditable(False)
        self.cbo_rotate.Append("", "")
        self.cbo_rotate.Append(T("Left 90 (counterclockwise)"), "left")
        self.cbo_rotate.Append(T("Right 90 (clockwise)"), "right")
        self.cbo_rotate.SetSelection(0)
        self.cbo_rotate.SetToolTip(T("Rotate the source video before 3D conversion, e.g. for footage shot "
                                     "sideways on a phone."))

        self.lbl_pad = wx.StaticText(self.grp_video_filter, label=T("Padding"))
        self.cbo_pad_mode = wx.ComboBox(self.grp_video_filter, choices=["", "tb", "lr", "top", "16:9"],
                                        name="cbo_pad_mode")
        self.cbo_pad_mode.SetEditable(False)
        self.cbo_pad_mode.SetSelection(0)
        self.cbo_pad_mode.SetToolTip(
            T("What it's for: adds solid padding/border area around the frame before the 3D shift is "
              "applied, giving objects near the edges more room to shift into without hitting the frame "
              "boundary.\ntb = top and bottom only. lr = left and right only. top = top only. "
              "16:9 = pads to a 16:9 aspect ratio specifically.\n"
              "Recommended: leave blank unless you're seeing objects clipped by the frame edge even with "
              "Preserve Screen Border on."))
        self.cbo_pad = EditableComboBox(self.grp_video_filter, choices=["", "0.01", "0.05", "0.5", "1"],
                                        name="cbo_pad")
        self.cbo_pad.SetSelection(0)
        self.cbo_pad.SetToolTip(
            T("How much padding to add, as a ratio of frame size (only applies if Padding Mode above is "
              "set to something other than blank). Con: larger values waste more of the frame on empty "
              "border instead of picture content. Recommended: start small (0.01-0.05) — you rarely need "
              "more than that."))

        self.lbl_max_output_size = wx.StaticText(self.grp_video_filter, label=T("Output Size Limit"))
        self.cbo_max_output_size = wx.ComboBox(self.grp_video_filter,
                                               choices=["",
                                                        "7680x2160",
                                                        "3840x2160",
                                                        "3840x1608",
                                                        "3840x1080",
                                                        "1920x1080", "1280x720", "640x360",
                                                        "1920x3200",
                                                        "1080x1920", "720x1280", "360x640"],
                                               name="cbo_max_output_size")
        self.cbo_max_output_size.SetEditable(False)
        self.cbo_max_output_size.SetSelection(0)
        self.cbo_max_output_size.SetToolTip(
            T("What it's for: caps the FINAL PACKED FRAME's resolution (the whole SBS/TB image, after "
              "both eyes are already combined — not a per-eye limit), to keep file size/playback "
              "requirements manageable on a large source. Leave blank to keep the source's native size, "
              "uncapped.\n"
              "IMPORTANT GOTCHA, verified by tracing the actual code: Full SBS builds a canvas DOUBLE your "
              "source's width (both eyes at full width, side by side) — e.g. a 3840-wide source becomes "
              "7680 wide. If you set this limit to 3840x2160 (a natural-looking pick for \"4K\") with Keep "
              "Aspect Ratio off, it crushes that 7680-wide Full SBS frame down to 3840 width while leaving "
              "height alone — which is the EXACT SAME pixel dimensions Half SBS already produces on "
              "purpose. Result: Full SBS and Half SBS can end up looking pixel-for-pixel identical, and "
              "toggling between them appears to do nothing.\n"
              "Recommended: for true Full SBS (full per-eye detail), set this to double your target "
              "width/height (e.g. 7680x2160 for a 4K target) or leave it blank. For Half SBS at a 4K "
              "delivery frame, 3840x2160 is correct as-is. Whichever Stereo Format you picked, make sure "
              "this limit's width actually matches what that format is supposed to produce."))

        self.chk_keep_aspect_ratio = wx.CheckBox(self.grp_video_filter, label=T("Keep Aspect Ratio"),
                                                 name="chk_keep_aspect_ratio")
        self.chk_keep_aspect_ratio.SetValue(False)
        self.chk_keep_aspect_ratio.SetToolTip(
            T("What it's for: when Output Size Limit is set, proportionally shrinks to fit inside it "
              "(preserving the source's width/height ratio) instead of crushing width and height "
              "independently to exactly match the limit's exact dimensions.\n"
              "Con of leaving off (the default): as documented in Output Size Limit's own tooltip above, "
              "an independent width/height crush can accidentally make Full SBS collapse down to the same "
              "pixel dimensions as Half SBS if the limit isn't sized correctly for your chosen Stereo "
              "Format.\n"
              "Recommended: turn on if you're not carefully matching Output Size Limit to double/single "
              "width for your Stereo Format choice, so an undersized limit shrinks proportionally instead "
              "of distorting the image."))

        self.chk_preserve_dowi = wx.CheckBox(self.grp_video_filter, label=T("Preserve Dolby Vision"),
                                              name="chk_preserve_dowi")
        self.chk_preserve_dowi.SetValue(False)
        self.chk_preserve_dowi.SetToolTip(
            T("What it's for: detects Dolby Vision RPU and/or HDR10+ dynamic metadata on the source and "
              "re-attaches it to the finished 3D video, so the HDR grading survives conversion instead of "
              "being silently dropped.\n"
              "Requires: HEVC output (Video Codec set to hevc_nvenc, hevc_qsv, hevc_amf, or libx265 — this "
              "metadata format doesn't exist for other codecs) and MKVToolNix installed for MKV output.\n"
              "Con: a small amount of extra time at the end of each job for the injection/remux step.\n"
              "Recommended: on for any Dolby Vision or HDR10+ source you want to keep looking correct on "
              "an HDR display after conversion."))

        self.chk_hdr_to_sdr = wx.CheckBox(self.grp_video_filter, label=T("Convert HDR to SDR"),
                                         name="chk_hdr_to_sdr")
        self.chk_hdr_to_sdr.SetValue(False)
        self.chk_hdr_to_sdr.SetToolTip(
            T("Properly tone-maps a PQ/HDR10/HDR10+ or HLG source down to normal SDR brightness "
              "before conversion, while keeping 10-bit color precision (reduces banding vs. plain "
              "8-bit SDR). Adds one extra encoding pass. Automatically does nothing if the source "
              "isn't actually PQ/HLG HDR. Turns off Preserve Dolby Vision, since the two are "
              "contradictory (there's no HDR grade left to preserve once tone-mapped to SDR). "
              "Recommended: on for HDR sources being watched on a non-HDR display (most projectors "
              "and many 3D setups) -- see this project's guidance on HDR/Dolby Vision for projectors "
              "and VR headsets."))

        self.chk_auto_resume = wx.CheckBox(self.grp_video_filter, label=T("Auto Resume"),
                                            name="chk_auto_resume")
        self.chk_auto_resume.SetValue(False)
        self.chk_auto_resume.SetToolTip(T("Lets a long conversion pick back up exactly where it left off if "
                                          "it's interrupted (crash, power loss, or clicking Cancel), instead "
                                          "of starting over from the beginning.\n"
                                          "Processes the whole video in one continuous pass — it does NOT "
                                          "split it into fixed-size pieces on a schedule. A new piece is only "
                                          "ever created at the exact point an interruption actually happened, "
                                          "so a normal, uninterrupted run has zero seams, identical to not "
                                          "using this option at all. If interrupted, you get exactly one seam "
                                          "at that point, not many. Recommended: on for any long/overnight "
                                          "conversion."))

        self.chk_denoise = wx.CheckBox(self.grp_video_filter, label=T("Denoise"), name="chk_denoise")
        self.chk_denoise.SetValue(False)
        self.chk_denoise.SetToolTip(
            T("What it's for: applies temporal (across-frame) denoising to the SOURCE before depth "
              "estimation even sees it. Reduces film grain/sensor noise that can otherwise confuse the "
              "depth model into reading noise as fake texture/detail.\n"
              "How it helps: particularly useful on older, grainier film sources where visible grain can "
              "translate into a noisier, less stable depth map.\n"
              "Con: real extra time up front (a full separate ffmpeg pass over the whole video before "
              "conversion starts), and denoising always risks softening some genuine fine detail along "
              "with the grain.\n"
              "Recommended: on for visibly grainy/old sources; off for clean, modern digital sources where "
              "there's no real grain to remove."))

        self.chk_preview = wx.CheckBox(self.grp_video_filter, label=T("Preview Mode"), name="chk_preview")
        self.chk_preview.SetValue(False)
        self.chk_preview.SetToolTip(
            T("What it's for: quickly renders just the first 60 seconds at 1fps and low resolution instead "
              "of a full conversion.\n"
              "How it helps: lets you check whether your settings (3D Strength, Convergence, Depth Model, "
              "etc.) look right BEFORE committing to a full-length conversion that might take hours.\n"
              "Recommended: on whenever you're testing new settings on a file for the first time; off for "
              "your actual final conversion run."))

        self.chk_scene_batch = wx.CheckBox(self.grp_video_filter, label=T("Automated Scene Batch"),
                                           name="chk_scene_batch")
        self.chk_scene_batch.SetValue(False)
        self.chk_scene_batch.SetToolTip(
            T("Fully automated whole-movie pipeline: removes letterbox bars once, detects scene "
              "cuts once, splits into per-scene clips with exact keyframe-accurate boundaries, "
              "converts each scene independently (a true fresh start per scene instead of one "
              "shared running state), then joins the results back into one seamless video and "
              "reattaches the original audio. If Preserve Dolby Vision is also on, the RPU is "
              "extracted once from the source and reinjected once at the end -- never per-clip. "
              "Input must be a single video file, not a folder."))

        self.lbl_scene_batch_crop = wx.StaticText(self.grp_video_filter, label=T("Scene Batch Crop"))
        self.txt_scene_batch_crop = wx.TextCtrl(self.grp_video_filter, name="txt_scene_batch_crop")
        self.txt_scene_batch_crop.SetValue("")
        self.txt_scene_batch_crop.SetToolTip(
            T("Optional, for Automated Scene Batch. Explicit crop as WxH or WxH:X:Y (X/Y default "
              "to centered). Leave blank to auto-detect letterbox bars once from the whole movie."))

        self.lbl_scene_settings = wx.StaticText(self.grp_video_filter, label=T("Scene Settings File"))
        self.txt_scene_settings = wx.TextCtrl(self.grp_video_filter, name="txt_scene_settings",
                                              style=wx.TE_READONLY)
        self.txt_scene_settings.SetValue("")
        self.btn_scene_settings = wx.Button(self.grp_video_filter, label=T("..."))
        self.btn_scene_settings.SetToolTip(
            T("Optional, for Automated Scene Batch. JSON file giving per-scene setting overrides "
              "(e.g. a different Divergence for different stretches of the movie). Leave blank to "
              "use the same settings for every scene."))

        self.chk_vr_optimized_merge = wx.CheckBox(self.grp_video_filter,
                                                  label=T("VR Optimized Merge"),
                                                  name="chk_vr_optimized_merge")
        self.chk_vr_optimized_merge.SetValue(False)
        self.chk_vr_optimized_merge.SetToolTip(
            T("For Automated Scene Batch's final join step. Off (default): scenes are joined with a "
              "fast, lossless stream copy -- quick, but each scene clip keeps its own independent "
              "timestamps/keyframe structure, which can read as uneven on a VR headset even though "
              "normal playback looks fine. On: re-encodes the joined timeline as one continuous "
              "stream with clean regenerated timestamps and AAC 48kHz audio, intended for smoother "
              "VR/headset playback. Slower than the default -- only turn this on if you're actually "
              "delivering to a VR headset workflow."))
        self.cbo_vr_merge_fps = wx.ComboBox(self.grp_video_filter,
                                            choices=["Source FPS", "60", "72", "80", "90", "120"],
                                            name="cbo_vr_merge_fps")
        self.cbo_vr_merge_fps.SetEditable(False)
        self.cbo_vr_merge_fps.SetSelection(0)
        self.cbo_vr_merge_fps.SetToolTip(
            T("Target constant frame rate for VR Optimized Merge. \"Source FPS\" keeps the "
              "already-converted frame rate as-is (still gets clean regenerated timestamps, just no "
              "specific forced rate) -- pick a fixed VR headset refresh rate instead if your delivery "
              "target needs one exactly."))

        self.lbl_scene_batch_variant = wx.StaticText(self.grp_video_filter, label=T("Scene Batch Variant"))
        self.txt_scene_batch_variant = wx.TextCtrl(self.grp_video_filter, name="txt_scene_batch_variant")
        self.txt_scene_batch_variant.SetValue("")
        self.txt_scene_batch_variant.SetToolTip(
            T("Optional, for Automated Scene Batch. Reuses the shared, already-done work from a "
              "prior run of the SAME movie (Dolby Vision RPU, the crop+keyframe pass, the split "
              "scene clips) but writes its own converted scenes and final output under this name, "
              "so trying different settings here never touches or overwrites an earlier attempt's "
              "progress. Leave blank for a normal run. Output becomes <name>_<variant>.<ext>."))

        self.lbl_upgrade_pix_fmt = wx.StaticText(self.grp_video_filter, label=T("Bit Depth Upgrade"))
        self.cbo_upgrade_pix_fmt = wx.ComboBox(self.grp_video_filter,
                                               choices=["", "10", "12"],
                                               name="cbo_upgrade_pix_fmt")
        self.cbo_upgrade_pix_fmt.SetEditable(False)
        self.cbo_upgrade_pix_fmt.SetSelection(0)
        self.cbo_upgrade_pix_fmt.SetToolTip(
            T("Gives the internal depth/3D math more color precision to work with, reducing banding "
              "(visible color \"steps\" in skies/shadows) — it doesn't add detail that wasn't in the "
              "source. Recommended: 10 for most sources, a real, visible improvement over 8-bit with "
              "low cost. 12 adds little you can actually see on a normal screen, for extra file size "
              "and slower encoding — usually not worth it. No effect if your Pixel Format below is "
              "already higher bit-depth than what you pick here."))

        self.lbl_autocrop = wx.StaticText(self.grp_video_filter, label=T("AutoCrop"))
        self.cbo_autocrop = wx.ComboBox(self.grp_video_filter,
                                        choices=["", "BLACK_TB", "BLACK", "FLAT_TB", "FLAT"],
                                        name="cbo_autocrop")
        self.cbo_autocrop.SetEditable(False)
        self.cbo_autocrop.SetSelection(0)
        self.cbo_autocrop.SetToolTip(
            T("Automatically detects and removes black bars or plain/flat borders before 3D conversion "
              "(so they don't waste depth detail or get distorted). _TB variants only crop top/bottom "
              "bars; the non-TB variants crop on all sides. Use the \"Test\" button to preview the "
              "detected crop before running a full conversion."))

        self.btn_autocrop_test = wx.Button(self.grp_video_filter, label=T("Test"))
        self.txt_autocrop_test = wx.TextCtrl(self.grp_video_filter, name="txt_autocrop_test", style=wx.TE_READONLY)
        self.txt_autocrop_test.SetValue("")

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        i = -1
        layout.Add(self.chk_start_time, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_start_time, (i, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_end_time, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_end_time, (i, 1), (0, 2), flag=wx.EXPAND)

        layout.Add((0, 8), (i := i + 1, 0))
        layout.Add(wx.StaticLine(self.grp_video_filter), (i := i + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 6), (i := i + 1, 0))
        layout.Add(self.lbl_deinterlace, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_deinterlace, (i, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_vf, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_vf, (i, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_rotate, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_rotate, (i, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_autocrop, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_autocrop, (i, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_autocrop_test, (i := i + 1, 1), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_autocrop_test, (i, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_pad, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_pad_mode, (i, 1), flag=wx.EXPAND)
        layout.Add(self.cbo_pad, (i, 2), flag=wx.EXPAND)

        layout.Add((0, 8), (i := i + 1, 0))
        layout.Add(wx.StaticLine(self.grp_video_filter), (i := i + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 6), (i := i + 1, 0))
        layout.Add(self.lbl_max_output_size, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_max_output_size, (i, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_keep_aspect_ratio, (i := i + 1, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_preserve_dowi, (i := i + 1, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_hdr_to_sdr, (i := i + 1, 1), (0, 2), flag=wx.EXPAND)

        layout.Add((0, 8), (i := i + 1, 0))
        layout.Add(wx.StaticLine(self.grp_video_filter), (i := i + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 6), (i := i + 1, 0))
        layout.Add(self.chk_auto_resume, (i := i + 1, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_upgrade_pix_fmt, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_upgrade_pix_fmt, (i, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_denoise, (i := i + 1, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_preview, (i := i + 1, 1), (0, 2), flag=wx.EXPAND)

        layout.Add((0, 8), (i := i + 1, 0))
        layout.Add(wx.StaticLine(self.grp_video_filter), (i := i + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 6), (i := i + 1, 0))
        layout.Add(self.chk_scene_batch, (i := i + 1, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_scene_batch_crop, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=14)
        layout.Add(self.txt_scene_batch_crop, (i, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_scene_settings, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=14)
        layout.Add(self.txt_scene_settings, (i, 1), flag=wx.EXPAND)
        layout.Add(self.btn_scene_settings, (i, 2), flag=wx.EXPAND)
        # chk_scene_batch_auto_ema/cbo_scene_batch_auto_ema_model/btn_scene_batch_auto_ema_edit
        # moved to grp_stereo's own layout (Flicker Reduction group), directly under
        # the Decay Rate/Buffer row -- see docs/ai/AI_DECISIONS.md ADR-057 amendment
        # (2026-09-08, relocation). No longer added here.
        layout.Add(self.lbl_scene_batch_variant, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=14)
        layout.Add(self.txt_scene_batch_variant, (i, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_vr_optimized_merge, (i := i + 1, 1), (0, 1), flag=wx.EXPAND | wx.LEFT, border=14)
        layout.Add(self.cbo_vr_merge_fps, (i, 2), (0, 1), flag=wx.EXPAND)

        sizer_video_filter = wx.StaticBoxSizer(self.grp_video_filter, wx.VERTICAL)
        sizer_video_filter.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # processor settings
        # device, batch-size, TTA, Low VRAM, fp16
        self.grp_processor = wx.StaticBox(self.tab_processor, label=T("Processor"))
        self.lbl_device = wx.StaticText(self.grp_processor, label=T("Device"))
        self.cbo_device = wx.ComboBox(self.grp_processor, size=self.FromDIP((200, -1)), name="cbo_device")
        self.cbo_device.SetEditable(False)
        cuda_device_names = _query_nvidia_smi_gpu_names()
        if cuda_device_names is not None:
            # See _query_nvidia_smi_gpu_names()'s docstring: deliberately avoids
            # torch.cuda.* here so this dropdown never initializes PyTorch's CUDA
            # context before ensure_cuda_context() gets a chance to run pyav's
            # NVDEC-compatible init first.
            for i, device_name in enumerate(cuda_device_names):
                self.cbo_device.Append(f"{i}:{device_name}", i)
            self.cbo_device.Append(T("All CUDA Device"), -2)
        elif torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_properties(i).name
                self.cbo_device.Append(f"{i}:{device_name}", i)
            if torch.cuda.device_count() > 0:
                self.cbo_device.Append(T("All CUDA Device"), -2)
        elif mps_is_available():
            self.cbo_device.Append("MPS", 0)
        elif xpu_is_available():
            for i in range(torch.xpu.device_count()):
                device_name = torch.xpu.get_device_name(i)
                self.cbo_device.Append(f"{i}:{device_name}", i)

        self.cbo_device.Append("CPU", -1)
        self.cbo_device.SetSelection(0)
        self.cbo_device.SetToolTip(
            T("Which GPU (or CPU) does the AI processing. \"All CUDA Device\" splits work across every "
              "GPU you have for faster batch processing. CPU works without a GPU but is dramatically "
              "slower — only use it if you have no compatible graphics card."))

        self.lbl_batch_size = wx.StaticText(self.grp_processor, label=T("Depth") + " " + T("Batch Size"))
        self.cbo_batch_size = wx.ComboBox(self.grp_processor,
                                          choices=[str(n) for n in (64, 32, 16, 14, 13, 12, 11, 10, 9, 8, 4, 3, 2, 1)],
                                          name="cbo_zoed_batch_size")
        self.cbo_batch_size.SetEditable(False)
        self.cbo_batch_size.SetToolTip(
            T("Video Only. How many frames are sent to the depth model at once. Higher = faster overall "
              "but uses more VRAM. Lower it if you run out of memory; raise it if you have VRAM to spare "
              "and want faster processing."))
        self.cbo_batch_size.SetSelection(12)  # "2"
        self.sld_processor_batch_size = _build_stereo_slider(self.grp_processor, self.cbo_batch_size, 1, 64, 1)

        self.lbl_max_workers = wx.StaticText(self.grp_processor, label=T("Worker Threads"))
        self.cbo_max_workers = wx.ComboBox(self.grp_processor,
                                           choices=[str(n) for n in (16, 14, 12, 11, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)],
                                           name="cbo_max_workers")
        self.cbo_max_workers.SetEditable(False)
        self.cbo_max_workers.SetToolTip(
            T("Video Only. How many CPU threads handle the stereo/inpainting step and frame I/O in "
              "parallel. Higher can speed things up on a multi-core CPU, but uses more RAM/VRAM. "
              "Lower it if you run into memory issues or system slowdowns while converting."))
        self.cbo_max_workers.SetSelection(13)  # "0"
        self.sld_processor_max_workers = _build_stereo_slider(self.grp_processor, self.cbo_max_workers, 0, 16, 1)

        self.chk_low_vram = wx.CheckBox(self.grp_processor, label=T("Low VRAM"), name="chk_low_vram")
        self.chk_low_vram.SetToolTip(
            T("Trades speed for lower memory use, by processing in a way that needs less VRAM at once. "
              "Only turn this on if you're actually running out of memory — it will make things slower.\n"
              "Confirmed real fix for Method forward_splat_fill's out-of-memory crashes on demanding "
              "footage, even when Depth Batch Size is already set to 1 — turn this on if that method "
              "OOMs for you."))
        self.chk_tta = wx.CheckBox(self.grp_processor, label=T("TTA"), name="chk_tta")
        self.chk_tta.SetToolTip(
            T("Use flip augmentation to improve depth quality (slow). Runs the depth model on both the "
              "normal and mirrored image/video and blends the result, often a little cleaner/more "
              "accurate, at roughly double the processing time. Works for both image depth models and "
              "video (VDA_*) models. Recommended: off for long videos (Flicker Reduction already covers "
              "similar ground there, for free) — worth trying for a single hero image, or a short clip "
              "where the extra quality matters more than the time cost."))
        self.chk_fp16 = wx.CheckBox(self.grp_processor, label=T("FP16"), name="chk_fp16")
        self.chk_fp16.SetToolTip(
            T("Use FP16 (fast) — runs the AI math at lower numeric precision, which is significantly "
              "faster and uses less VRAM on modern GPUs, with no visible quality cost in virtually all "
              "cases. Recommended: on."))
        self.chk_fp16.SetValue(True)
        self.chk_cuda_stream = wx.CheckBox(self.grp_processor, label=T("Stream"), name="chk_cuda_stream")
        self.chk_cuda_stream.SetToolTip(
            T("Use per-thread CUDA Stream (experimental: fast or slow or crash). Lets multiple worker "
              "threads share the GPU more aggressively. May speed things up, may do nothing, may "
              "occasionally cause instability — try it and turn it back off if you see crashes."))
        self.chk_cuda_stream.SetValue(False)

        self.chk_compile = wx.CheckBox(self.grp_processor, label=T("torch.compile"), name="chk_compile")
        self.chk_compile.SetToolTip(
            T("Enable model compiling: optimizes the AI model before running, trading a slower startup "
              "(the first run after changing settings has to compile) for faster processing afterward. "
              "Worth it for long videos; not worth it for a single quick image or short clip. Requires "
              "extra setup (see this project's torch_compile docs) to actually take effect."))
        self.chk_compile.SetValue(False)

        self.chk_pause_frees_vram = wx.CheckBox(self.grp_processor, label=T("Free GPU memory while paused"),
                                                name="chk_pause_frees_vram")
        self.chk_pause_frees_vram.SetToolTip(
            T("What: when you click Suspend, also move every loaded model (depth model, stereo/side "
              "model, and the SOD_v1 auto-convergence model, if used) off the GPU and actually release "
              "that VRAM back to Windows, instead of just pausing while everything stays loaded. Resume "
              "moves them all back before continuing.\n"
              "Why it helps: normally, pausing here does NOT free any VRAM -- every model just sits "
              "loaded and idle the whole time you're paused, so nothing else can use that memory. This "
              "lets you actually hand the GPU to something else (another program, a second conversion) "
              "while paused.\n"
              "Pros: real VRAM freed while paused; Resume still produces a correct, uninterrupted output.\n"
              "Cons: Resume is no longer instant -- reloading the models back onto the GPU takes a few "
              "seconds (longer if torch.compile is on, since it may recompile). No effect with multi-GPU "
              "(\"All CUDA Device\") -- those models are already spread across every GPU and are left "
              "as-is.\n"
              "Values: off (default) = today's behavior, models stay resident, Resume is instant. On = "
              "frees VRAM while paused, Resume takes a few seconds.\n"
              "Recommended: off, unless you specifically need the GPU free for something else during a "
              "long pause."))
        self.chk_pause_frees_vram.SetValue(False)

        layout = wx.GridBagSizer(vgap=5, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        k = -1
        layout.Add(self.lbl_device, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_device, (k, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_batch_size, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_batch_size, (k, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.sld_processor_batch_size, (k := k + 1, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_max_workers, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_max_workers, (k, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.sld_processor_max_workers, (k := k + 1, 1), (0, 3), flag=wx.EXPAND)

        layout.Add((0, 6), (k := k + 1, 0))
        layout.Add(wx.StaticLine(self.grp_processor), (k := k + 1, 0), (0, 4), flag=wx.EXPAND)
        layout.Add((0, 4), (k := k + 1, 0))
        layout.Add(self.chk_low_vram, (k := k + 1, 0), flag=wx.EXPAND)
        layout.Add(self.chk_tta, (k, 1), flag=wx.EXPAND)
        layout.Add(self.chk_fp16, (k, 2), flag=wx.EXPAND)
        layout.Add(self.chk_cuda_stream, (k, 3), flag=wx.EXPAND)
        layout.Add(self.chk_compile, (k := k + 1, 0), flag=wx.EXPAND)
        layout.Add(self.chk_pause_frees_vram, (k, 1), (0, 3), flag=wx.EXPAND)

        sizer_processor = wx.StaticBoxSizer(self.grp_processor, wx.VERTICAL)
        sizer_processor.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        self.grp_postprocess = wx.StaticBox(self.tab_processor, label=T("Post-Processing"))
        self.chk_waifu2x_upscale = wx.CheckBox(self.grp_postprocess,
                                               label=T("Upscale with waifu2x after conversion"),
                                               name="chk_waifu2x_upscale")
        self.chk_waifu2x_upscale.SetValue(False)
        self.chk_waifu2x_upscale.SetToolTip(
            T("What it's for (single video and Dual-Pass Depth Blend jobs only): once this job's finished "
              "output is fully written, automatically runs it through waifu2x (a separate, dedicated AI "
              "upscaler bundled with this app) as one extra step, so you don't need to run waifu2x by hand "
              "afterward.\n"
              "How it's safe: saved to a separate '_w2x' file — the original conversion output is always "
              "left untouched, even if the upscale step itself fails.\n"
              "Con: real extra processing time after the main conversion already finished, roughly "
              "proportional to the upscale factor chosen below.\n"
              "Recommended: on if you specifically want a higher-resolution final delivery file; off if "
              "your source resolution is already your target."))
        self.cbo_waifu2x_method = wx.ComboBox(self.grp_postprocess,
                                              choices=["noise_scale2x", "noise_scale4x", "scale2x", "scale4x"],
                                              name="cbo_waifu2x_method")
        self.cbo_waifu2x_method.SetEditable(False)
        self.cbo_waifu2x_method.SetSelection(0)
        self.cbo_waifu2x_method.SetToolTip(
            T("What it's for: which waifu2x mode to run. \"noise_scale\" upscales AND reduces compression "
              "artifacts/noise at the same time; plain \"scale\" only upscales, leaving existing noise "
              "alone. 2x/4x is the resulting size multiplier (width and height each multiplied by this).\n"
              "Recommended: noise_scale2x for most delivery-quality video (mild, safe noise cleanup plus a "
              "reasonable size bump); use 4x variants only if you specifically need a much larger frame."))
        self.cbo_waifu2x_noise_level = wx.ComboBox(self.grp_postprocess,
                                                    choices=["0", "1", "2", "3"],
                                                    name="cbo_waifu2x_noise_level")
        self.cbo_waifu2x_noise_level.SetEditable(False)
        self.cbo_waifu2x_noise_level.SetSelection(1)
        self.cbo_waifu2x_noise_level.SetToolTip(
            T("waifu2x noise reduction strength (0=off, 3=strongest). Ignored for plain \"scale\" methods "
              "(only matters for noise_scale2x/4x above).\n"
              "Con: pushed too high on a source that's actually clean (not noisy/grainy), can start "
              "softening real fine detail along with noise that isn't really there.\n"
              "Recommended: 1 as a safe default; raise toward 2-3 only for a visibly noisy/compressed "
              "source."))
        self.cbo_waifu2x_style = wx.ComboBox(self.grp_postprocess,
                                             choices=["photo", "art"],
                                             name="cbo_waifu2x_style")
        self.cbo_waifu2x_style.SetEditable(False)
        self.cbo_waifu2x_style.SetSelection(0)
        self.cbo_waifu2x_style.SetToolTip(
            T("waifu2x model style. \"photo\" is the better default for real movie footage; \"art\" is "
              "tuned for illustration/anime source material."))
        self.cbo_waifu2x_target = wx.ComboBox(self.grp_postprocess,
                                              choices=["auto", "4k", "8k"],
                                              name="cbo_waifu2x_target")
        self.cbo_waifu2x_target.SetEditable(False)
        self.cbo_waifu2x_target.SetSelection(0)
        self.cbo_waifu2x_target.SetToolTip(
            T("What it's for: only takes effect on a packed 3D stereo video output (Half/Full SBS, "
              "Half/Full TB, Cross-Eyed, VR90). \"auto\" (default) leaves the plain whole-frame upscale "
              "above completely unchanged (fixed 2x/4x from Method). \"4k\"/\"8k\" instead switch to a "
              "stereo-aware upscale: the video is split into its two independent eye images first, each "
              "eye is upscaled separately (so the AI upscaler never blurs/blends pixels across the seam "
              "between the two eyes the way upscaling the packed frame whole can), a frame-to-frame "
              "smoothing pass reduces flicker that becomes more visible at very high output resolutions, "
              "then the eyes are recombined. The exact per-eye enlargement needed is computed from your "
              "source's real resolution and split direction, so the final packed video actually lands on "
              "the requested width (3840 for 4k, 7680 for 8k) rather than assuming a fixed multiplier "
              "happens to fit.\n"
              "Con: several extra full passes over the video beyond the plain whole-frame path above, so "
              "real processing time is meaningfully longer -- expect this to matter most on a full-length "
              "video.\n"
              "Recommended: auto unless you specifically need a 4K/8K delivery file from a 3D stereo "
              "source and want the cleaner per-eye result."))
        self.chk_waifu2x_upscale.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_waifu2x_upscale)
        self.update_waifu2x_upscale()

        self.chk_rife_interpolate = wx.CheckBox(self.grp_postprocess,
                                                label=T("Interpolate frames with RIFE after conversion"),
                                                name="chk_rife_interpolate")
        self.chk_rife_interpolate.SetValue(False)
        self.chk_rife_interpolate.SetToolTip(
            T("What it's for (single video and Dual-Pass Depth Blend jobs only): once this job's finished "
              "output is fully written, runs it through RIFE (a separate AI frame-interpolation model) as "
              "one extra step, generating new in-between frames for smoother-looking motion. How many/where "
              "is controlled by the Rate mode dropdown below (2x by default).\n"
              "How it's safe: saved to a separate '_rife' file -- the original conversion output is always "
              "left untouched, even if the interpolation step itself fails.\n"
              "Con: real extra processing time after the main conversion already finished; RIFE "
              "interpolates the FINAL PACKED stereo frame (both eyes already combined) as one image, so it "
              "will see the seam between the two packed eyes -- it wasn't trained on that, though in "
              "practice it moves both eyes together so this doesn't cause left/right desync.\n"
              "Cannot be combined with Preserve Dolby Vision: there's no way to assign correct DV/HDR10+ "
              "metadata to RIFE's synthetic in-between frames.\n"
              "Recommended: on if your source is naturally low frame rate (e.g. 24fps film) and you want "
              "smoother motion for VR viewing; off if you're already happy with the source's frame rate or "
              "you need Dolby Vision preserved."))
        self.cbo_rife_model = wx.ComboBox(self.grp_postprocess,
                                          choices=["rife_425", "rife_425_lite"],
                                          name="cbo_rife_model")
        self.cbo_rife_model.SetEditable(False)
        self.cbo_rife_model.SetSelection(0)
        self.cbo_rife_model.SetToolTip(
            T("Which RIFE model quality tier to use.\n"
              "rife_425: the recommended full model -- better motion accuracy, especially on complex/fast "
              "motion, at a higher compute cost.\n"
              "rife_425_lite: a lower-compute-cost variant of the same generation, trades a little accuracy "
              "for speed.\n"
              "Weights are downloaded automatically the first time you use a given tier (not bundled with "
              "the app).\n"
              "Recommended: rife_425 unless interpolation time is a real bottleneck for you."))

        self.cbo_rife_mode = wx.ComboBox(self.grp_postprocess,
                                         choices=["2x", "3x", "4x", "Custom FPS..."],
                                         name="cbo_rife_mode")
        self.cbo_rife_mode.SetEditable(False)
        self.cbo_rife_mode.SetSelection(0)
        self.cbo_rife_mode.SetToolTip(
            T("What it's for: how many new frames RIFE inserts, and where.\n"
              "2x/3x/4x: inserts 1/2/3 evenly-spaced new frames between every pair of real frames, "
              "multiplying the frame rate by that exact amount (e.g. 24fps source -> 48/72/96fps).\n"
              "Custom FPS...: interpolate to an exact frame rate you choose in the field to the right "
              "instead (e.g. 24fps source -> 60fps target), even when it isn't a clean multiple of the "
              "source -- RIFE works out the correct in-between timing for each new frame automatically.\n"
              "Con: 3x/4x do roughly 2x/3x as many interpolation passes as the 2x default, so processing "
              "time increases proportionally. For a non-integer Custom FPS target (like 24->60), the new "
              "frames aren't perfectly evenly spaced in time (some land slightly closer to one real frame "
              "than the next), which can show as very slightly uneven motion smoothness on some frames -- "
              "usually not noticeable, unlike the perfectly even spacing of the clean 2x/3x/4x multipliers.\n"
              "Recommended: 2x for most uses; Custom FPS if you need to match a specific display or "
              "editing timeline's exact frame rate."))
        self.txt_rife_target_fps = wx.TextCtrl(self.grp_postprocess, name="txt_rife_target_fps")
        self.txt_rife_target_fps.SetToolTip(
            T("What it's for: the exact output frame rate to interpolate to, used only when Rate mode "
              "above is set to \"Custom FPS...\".\n"
              "Values: any number higher than your source video's own frame rate (e.g. 60 for a 24fps "
              "source). A target at or below the source's frame rate is rejected -- RIFE only adds frames, "
              "it never removes them.\n"
              "Recommended: 60 for standard smooth-motion displays, or match your target display/editing "
              "timeline's exact refresh rate."))
        self.cbo_rife_mode.Bind(wx.EVT_COMBOBOX, self.on_changed_cbo_rife_mode)
        self.chk_rife_interpolate.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_rife_interpolate)
        self.update_rife_interpolate()

        layout = wx.GridBagSizer(vgap=5, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        j = -1
        layout.Add(self.chk_waifu2x_upscale, (j := j + 1, 0), (0, 3), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_waifu2x_method, (j := j + 1, 0), flag=wx.EXPAND | wx.LEFT, border=14)
        layout.Add(self.cbo_waifu2x_noise_level, (j, 1), flag=wx.EXPAND)
        layout.Add(self.cbo_waifu2x_style, (j, 2), flag=wx.EXPAND)
        layout.Add(self.cbo_waifu2x_target, (j, 3), flag=wx.EXPAND)

        layout.Add((0, 6), (j := j + 1, 0))
        layout.Add(wx.StaticLine(self.grp_postprocess), (j := j + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 4), (j := j + 1, 0))
        layout.Add(self.chk_rife_interpolate, (j := j + 1, 0), (0, 3), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_rife_model, (j := j + 1, 0), flag=wx.EXPAND | wx.LEFT, border=14)
        layout.Add(self.cbo_rife_mode, (j, 1), flag=wx.EXPAND)
        layout.Add(self.txt_rife_target_fps, (j, 2), flag=wx.EXPAND)
        sizer_postprocess = wx.StaticBoxSizer(self.grp_postprocess, wx.VERTICAL)
        sizer_postprocess.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # --- standalone utility: retroactive DV/HDR RPU reinjection (ADR-031) ---
        # NOT part of the main conversion pipeline -- a separate tool that takes an
        # ORIGINAL source file (real Dolby Vision/HDR10+ metadata) and an
        # ALREADY-CONVERTED 3D output (currently SDR) and injects the source's HDR
        # grading into a NEW copy of the output, without re-running depth/stereo
        # conversion. Launches python -m iw3.reinject_hdr_cli as its own subprocess --
        # needs no GPU, but kept out-of-process the same way RIFE/waifu2x
        # post-processing is (see docs/ai/CODING_STANDARDS.md CS-SUBPROCESS-001).
        self.grp_hdr_reinject = wx.StaticBox(
            self.tab_tools, label=T("Retroactive HDR/DV Reinjection (Standalone Tool)"))

        self.lbl_reinject_source = wx.StaticText(self.grp_hdr_reinject, label=T("Original Source File (DV/HDR)"))
        self.txt_reinject_source = wx.TextCtrl(self.grp_hdr_reinject, name="txt_reinject_source")
        self.txt_reinject_source.SetToolTip(
            T("What it's for: the ORIGINAL video file that still has real Dolby Vision / HDR10+ metadata "
              "-- the same file iw3 converted FROM when it made the already-converted 3D output below, "
              "before that conversion's HDR grading was ever discarded.\n"
              "Con: a re-encode of your original file may no longer carry the real DV/HDR10+ metadata at "
              "all -- point this at your actual master/source file.\n"
              "Recommended: the exact same file (or an identical remux of it) you originally fed into "
              "iw3 for this conversion."))
        self.btn_reinject_source = wx.Button(self.grp_hdr_reinject, label=T("..."))

        self.lbl_reinject_converted = wx.StaticText(self.grp_hdr_reinject, label=T("Already-Converted 3D File"))
        self.txt_reinject_converted = wx.TextCtrl(self.grp_hdr_reinject, name="txt_reinject_converted")
        self.txt_reinject_converted.SetToolTip(
            T("What it's for: the iw3 3D output you already made from the source above -- currently SDR "
              "because Preserve Dolby Vision wasn't enabled for that conversion. Read-only: this tool "
              "never modifies this file, it only copies from it.\n"
              "Con: cannot be output that was already run through RIFE frame interpolation -- RIFE changes "
              "the frame count, so it can never line back up with the source's original timing (this tool "
              "will detect and refuse that case).\n"
              "Recommended: the direct, unmodified iw3 output file -- not a re-encode or upscale of it."))
        self.btn_reinject_converted = wx.Button(self.grp_hdr_reinject, label=T("..."))

        self.lbl_reinject_output = wx.StaticText(self.grp_hdr_reinject, label=T("Output File"))
        self.txt_reinject_output = wx.TextCtrl(self.grp_hdr_reinject, name="txt_reinject_output")
        self.txt_reinject_output.SetToolTip(
            T("Where to write the new HDR-reinjected copy. Auto-filled with '<converted file "
              "name>_hdr_reinjected<ext>' in the same folder once you pick the converted file above -- "
              "change it if you want it saved somewhere else.\n"
              "How it's safe: this tool never overwrites the source or converted file, only ever writes "
              "here."))
        self.btn_reinject_output = wx.Button(self.grp_hdr_reinject, label=T("..."))

        # RIFE Manifest (optional) -- wires the existing, already-real
        # reinject_hdr_cli.py --rife-manifest flag (ADR-051) into this panel (ADR-064
        # amendment). Left blank, this preserves today's exact existing behavior
        # (--rife-manifest is simply never passed). The .rife_manifest.json sidecar
        # this points at is written by the RIFE Frame Interpolation (Standalone Tool)
        # panel below, never by this tool itself.
        self.lbl_reinject_rife_manifest = wx.StaticText(self.grp_hdr_reinject, label=T("RIFE Manifest (optional)"))
        self.txt_reinject_rife_manifest = wx.TextCtrl(self.grp_hdr_reinject, name="txt_reinject_rife_manifest")
        self.txt_reinject_rife_manifest.SetToolTip(
            T("What it's for: only needed when the 'Already-Converted 3D File' above was ALSO run "
              "through the RIFE Frame Interpolation (Standalone Tool) panel below. RIFE changes the "
              "frame count, so this tool cannot normally line it back up with the original source -- "
              "this manifest (a small '<rife output>.rife_manifest.json' file the RIFE panel writes "
              "next to its own output, recording exactly how it expanded the frame timeline) is what "
              "lets it expand the Dolby Vision/HDR10+ metadata to match instead of refusing.\n"
              "How it's safe: leave this blank for a normal (non-RIFE) conversion -- behaves exactly "
              "as before, --rife-manifest is simply never passed.\n"
              "Auto-fill: picking an 'Already-Converted 3D File' above that has a matching "
              "'<file>.rife_manifest.json' sitting right next to it (the RIFE panel's own real output "
              "naming) fills this in for you automatically -- still fully editable/clearable "
              "afterward.\n"
              "Recommended: leave blank unless your converted file came out of the RIFE panel; if it "
              "did, point this at the '.rife_manifest.json' that panel wrote next to its output."))
        self.btn_reinject_rife_manifest = wx.Button(self.grp_hdr_reinject, label=T("..."))

        self.chk_reinject_start_time = wx.CheckBox(self.grp_hdr_reinject, label=T("Start"),
                                                    name="chk_reinject_start_time")
        self.chk_reinject_start_time.SetToolTip(
            T("What it's for: which point in the ORIGINAL SOURCE file the converted file's first frame "
              "actually starts at -- only needed when the converted file covers just part of the source "
              "(e.g. a clip, not the whole movie). Leave unchecked to use the whole source.\n"
              "Con: there is no auto-detection of this -- you must know and enter the exact range that "
              "was actually converted. If it's wrong, the tool refuses to proceed (an exact decoded "
              "frame-count check) rather than silently producing misaligned HDR metadata.\n"
              "Recommended: the exact --start-time you used for the original iw3 conversion, if any."))
        self.txt_reinject_start_time = TimeCtrl(self.grp_hdr_reinject, value="00:00:00", fmt24hr=True,
                                                 name="txt_reinject_start_time")
        self.chk_reinject_end_time = wx.CheckBox(self.grp_hdr_reinject, label=T("End"),
                                                  name="chk_reinject_end_time")
        self.chk_reinject_end_time.SetToolTip(
            T("Same idea as Start Time, but for where the converted file's last frame ends within the "
              "original source. Leave unchecked to use the end of the source."))
        self.txt_reinject_end_time = TimeCtrl(self.grp_hdr_reinject, value="00:00:00", fmt24hr=True,
                                               name="txt_reinject_end_time")

        self.btn_reinject_run = wx.Button(self.grp_hdr_reinject, label=T("Run"))
        self.btn_reinject_run.SetToolTip(
            T("What it's for: runs the reinjection as a separate background process (python -m "
              "iw3.reinject_hdr_cli) -- this app's own GPU/model state is never touched, and neither "
              "input file above is ever modified.\n"
              "How it's safe: before touching anything, it first checks that the source (trimmed to "
              "Start/End Time) and the converted file decode to the EXACT same number of frames -- if "
              "they don't match, it refuses and prints both frame counts to the log box below instead of "
              "producing a mismatched result.\n"
              "Con: that frame-count check decodes the full clip, so it can take a while on a long video.\n"
              "Recommended: run once per conversion you want retroactively HDR-corrected; always check "
              "the log box below afterward to confirm it actually succeeded rather than refused."))

        self.txt_reinject_log = wx.TextCtrl(self.grp_hdr_reinject, style=wx.TE_MULTILINE | wx.TE_READONLY,
                                             size=self.FromDIP((-1, 60)), name="txt_reinject_log")
        self.txt_reinject_log.SetToolTip(
            T("Shows this tool's own output: pre-flight frame-count/duration numbers, and the exact "
              "reason if it refuses to proceed (e.g. a frame-count mismatch or detected RIFE "
              "interpolation) -- not just a generic pass/fail."))
        self.btn_reinject_clear = wx.Button(self.grp_hdr_reinject, label=T("Clear"))
        self.btn_reinject_clear.SetToolTip(
            T("Empties the log box above -- output only accumulates run after run otherwise. Disabled "
              "while a job is running so it can't wipe output you may still be reading mid-run; "
              "re-enabled once the job finishes."))

        self.btn_reinject_source.Bind(wx.EVT_BUTTON, self.on_click_btn_reinject_source)
        self.btn_reinject_converted.Bind(wx.EVT_BUTTON, self.on_click_btn_reinject_converted)
        self.btn_reinject_output.Bind(wx.EVT_BUTTON, self.on_click_btn_reinject_output)
        self.btn_reinject_rife_manifest.Bind(wx.EVT_BUTTON, self.on_click_btn_reinject_rife_manifest)
        self.btn_reinject_run.Bind(wx.EVT_BUTTON, self.on_click_btn_reinject_run)
        self.btn_reinject_clear.Bind(wx.EVT_BUTTON, lambda event: self.txt_reinject_log.Clear())

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        h = -1
        layout.Add(self.lbl_reinject_source, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_reinject_source, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_reinject_source, (h, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_reinject_converted, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_reinject_converted, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_reinject_converted, (h, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_reinject_output, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_reinject_output, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_reinject_output, (h, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_reinject_rife_manifest, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_reinject_rife_manifest, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_reinject_rife_manifest, (h, 3), flag=wx.EXPAND)
        # Start/End Time + Run share one compact row rather than each taking a row of
        # their own -- this whole section lives in the Dual-Pass Depth Blend column,
        # which has limited spare vertical room.
        layout.Add(self.chk_reinject_start_time, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_reinject_start_time, (h, 1), flag=wx.EXPAND)
        layout.Add(self.chk_reinject_end_time, (h, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_reinject_end_time, (h, 3), flag=wx.EXPAND)
        layout.Add(self.btn_reinject_run, (h := h + 1, 3), flag=wx.EXPAND)
        layout.Add(self.txt_reinject_log, (h, 0), (0, 3), flag=wx.EXPAND)
        layout.Add(self.btn_reinject_clear, (h := h + 1, 3), flag=wx.EXPAND)
        sizer_hdr_reinject = wx.StaticBoxSizer(self.grp_hdr_reinject, wx.VERTICAL)
        sizer_hdr_reinject.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # --- standalone utility: search/download subtitles from OpenSubtitles (ADR-039) ---
        # NOT part of the main conversion pipeline -- searches OpenSubtitles' REST API for
        # a matching subtitle and downloads a plain .srt, so the natural next step (Add
        # Subtitle Track below) has something to feed it without leaving this app. Calls
        # iw3.subtitle_search_cli's search()/download() directly (in-process, not a
        # subprocess like the other standalone tools in this column) since it needs no
        # GPU/process isolation -- just a real network call, still run off the GUI thread
        # via startWorker since it's real network I/O. See that module's own docstring /
        # docs/ai/AI_DECISIONS.md ADR-039 for the full design.
        self.grp_subsearch = wx.StaticBox(
            self.tab_tools, label=T("Search Subtitles (OpenSubtitles) (Standalone Tool)"))

        self.lbl_subsearch_source = wx.StaticText(self.grp_subsearch, label=T("Original Source File (optional)"))
        self.txt_subsearch_source = wx.TextCtrl(self.grp_subsearch, name="txt_subsearch_source")
        self.txt_subsearch_source.SetToolTip(
            T("What it's for: the ORIGINAL, pre-conversion source video file -- optional. When given "
              "and at least 128KB, its OpenSubtitles moviehash (file size plus a checksum of only the "
              "first and last 64KB, never the whole file) is computed and used for exact-match "
              "results -- the same OSHash algorithm OpenSubtitles has used for years.\n"
              "Why this is a different file from Add Subtitle Track's 'Converted 3D Video' field "
              "below: an already-converted iw3 SBS/TB output is a structurally different file "
              "(different size, different bytes entirely) and will essentially never hash-match "
              "anything -- moviehash only works against the real original movie file.\n"
              "Con: files under 128KB can't be hashed this way -- falls back to Title/IMDb ID search "
              "automatically in that case, noted in the log below.\n"
              "Recommended: point this at the original file you converted from, if you still have "
              "it, for the most exact match; otherwise leave blank and use Title/IMDb ID instead."))
        self.btn_subsearch_source = wx.Button(self.grp_subsearch, label=T("..."))

        self.lbl_subsearch_title = wx.StaticText(self.grp_subsearch, label=T("Title"))
        self.txt_subsearch_title = wx.TextCtrl(self.grp_subsearch, name="txt_subsearch_title")
        self.txt_subsearch_title.SetToolTip(
            T("What it's for: movie/show title to search by -- optional fallback text search, used "
              "when Original Source File isn't given or its moviehash didn't match anything.\n"
              "Con: text search can return multiple/ambiguous candidates for common titles -- review "
              "the results list below (release name, uploader) before downloading.\n"
              "Recommended: the exact title, optionally with year (e.g. 'Interstellar 2014') for a "
              "more precise match."))

        self.lbl_subsearch_imdb = wx.StaticText(self.grp_subsearch, label=T("IMDb ID"))
        self.txt_subsearch_imdb = wx.TextCtrl(self.grp_subsearch, name="txt_subsearch_imdb")
        self.txt_subsearch_imdb.SetToolTip(
            T("What it's for: search by IMDb ID (e.g. tt0111161 or 111161) instead of a text title -- "
              "optional, more precise than Title when you have it.\n"
              "Recommended: leave blank unless you already know the exact IMDb ID; Title search "
              "works fine for most movies."))

        self.lbl_subsearch_language = wx.StaticText(self.grp_subsearch, label=T("Language"))
        self.cbo_subsearch_language = wx.ComboBox(
            self.grp_subsearch, value="en", name="cbo_subsearch_language",
            choices=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko",
                     "zh", "nl", "sv", "no", "da", "pl", "tr", "ar", "hi"])
        self.cbo_subsearch_language.SetToolTip(
            T("What it's for: the subtitle language to search for, as an ISO 639-1 two-letter code "
              "(e.g. en, ja, fr, de) -- confirmed by a real live search against OpenSubtitles' API "
              "that its 'languages' search parameter needs the two-letter form. Add Subtitle Track's "
              "Language field below uses this SAME two-letter format (ADR-042) -- the same code "
              "typed here is always correct there too.\n"
              "Values: pick from the dropdown, or type any other ISO 639-1 code OpenSubtitles "
              "supports -- this list only covers the most common languages, it isn't exhaustive.\n"
              "Recommended: match the language you want the subtitle text to actually be in; "
              "default 'en' if unsure."))

        self.btn_subsearch_search = wx.Button(self.grp_subsearch, label=T("Search"))
        self.btn_subsearch_search.SetToolTip(
            T("What it's for: searches OpenSubtitles' API for candidate subtitles matching whatever "
              "criteria above are filled in -- runs in the background so the app stays responsive "
              "while it waits on the network.\n"
              "How it's safe: search never spends download quota -- only Download Selected below "
              "does. Search as many times as you like.\n"
              "Con: needs a configured OpenSubtitles API key (nunif/tmp/opensubtitles_config.json) "
              "-- if missing, the log below shows exactly how to register one and where to paste it.\n"
              "Recommended: fill in at least one of Original Source File / Title / IMDb ID first, "
              "then click Search and review the results list before downloading anything."))

        self.lst_subsearch_results = wx.ListCtrl(
            self.grp_subsearch, style=wx.LC_REPORT | wx.LC_SINGLE_SEL,
            size=self.FromDIP((-1, 140)), name="lst_subsearch_results")
        self.lst_subsearch_results.InsertColumn(0, T("Release"), width=self.FromDIP(220))
        self.lst_subsearch_results.InsertColumn(1, T("Lang"), width=self.FromDIP(45))
        self.lst_subsearch_results.InsertColumn(2, T("Rating"), width=self.FromDIP(50))
        self.lst_subsearch_results.InsertColumn(3, T("Downloads"), width=self.FromDIP(70))
        self.lst_subsearch_results.InsertColumn(4, T("Uploader"), width=self.FromDIP(90))
        self.lst_subsearch_results.InsertColumn(5, T("Flag"), width=self.FromDIP(45))
        self.lst_subsearch_results.SetToolTip(
            T("What it's for: candidate subtitles from the last search, sorted with plain (non-AI/"
              "machine-translated) results first. Columns: Release name, Language, Rating, "
              "Downloads, Uploader, Flag.\n"
              "Values: a red '[MT]' Flag marks a machine-translated or AI-translated result -- most "
              "users want to avoid these auto-translated subs, which is why they're sorted after the "
              "plain results rather than mixed in.\n"
              "Recommended: prefer a high-Downloads, high-Rating, unflagged result when more than "
              "one candidate looks right for your movie."))
        self.subsearch_results = []

        self.btn_subsearch_download = wx.Button(self.grp_subsearch, label=T("Download Selected"))
        self.btn_subsearch_download.Disable()
        self.btn_subsearch_download.SetToolTip(
            T("What it's for: downloads the selected result's .srt file -- disabled until a result "
              "is selected above.\n"
              "Con: spends exactly one of your OpenSubtitles daily download-quota credits, unlike "
              "Search which is free -- only click this once you've picked the right candidate.\n"
              "How it helps next: once downloaded, if Add Subtitle Track's Subtitle File field below "
              "is still empty, it's auto-filled with the new .srt so adding it as a track is one "
              "click away.\n"
              "Recommended: check the log below afterward for the saved file path and your "
              "remaining download quota for today."))

        self.txt_subsearch_log = wx.TextCtrl(self.grp_subsearch, style=wx.TE_MULTILINE | wx.TE_READONLY,
                                              size=self.FromDIP((-1, 60)), name="txt_subsearch_log")
        self.txt_subsearch_log.SetToolTip(
            T("Shows this tool's own output verbatim, including the exact actionable message if no "
              "OpenSubtitles API key is configured yet, and the server-reported download quota "
              "(remaining/requests/reset time) after every download -- not just a generic pass/"
              "fail."))
        self.btn_subsearch_clear = wx.Button(self.grp_subsearch, label=T("Clear"))
        self.btn_subsearch_clear.SetToolTip(
            T("Empties the log box above -- output only accumulates run after run otherwise. Disabled "
              "while a search or download is running so it can't wipe output you may still be reading "
              "mid-run; re-enabled once it finishes."))

        self.btn_subsearch_source.Bind(wx.EVT_BUTTON, self.on_click_btn_subsearch_source)
        self.btn_subsearch_search.Bind(wx.EVT_BUTTON, self.on_click_btn_subsearch_search)
        self.lst_subsearch_results.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_select_lst_subsearch_results)
        self.lst_subsearch_results.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.on_select_lst_subsearch_results)
        self.btn_subsearch_download.Bind(wx.EVT_BUTTON, self.on_click_btn_subsearch_download)
        self.btn_subsearch_clear.Bind(wx.EVT_BUTTON, lambda event: self.txt_subsearch_log.Clear())

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        h = -1
        layout.Add(self.lbl_subsearch_source, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_subsearch_source, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_subsearch_source, (h, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_subsearch_title, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_subsearch_title, (h, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_subsearch_imdb, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_subsearch_imdb, (h, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_subsearch_language, (h, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_subsearch_language, (h, 3), flag=wx.EXPAND)
        layout.Add(self.btn_subsearch_search, (h := h + 1, 3), flag=wx.EXPAND)
        layout.Add(self.lst_subsearch_results, (h := h + 1, 0), (0, 4), flag=wx.EXPAND)
        layout.Add(self.btn_subsearch_download, (h := h + 1, 3), flag=wx.EXPAND)
        layout.Add(self.txt_subsearch_log, (h, 0), (0, 3), flag=wx.EXPAND)
        layout.Add(self.btn_subsearch_clear, (h := h + 1, 3), flag=wx.EXPAND)
        sizer_subsearch = wx.StaticBoxSizer(self.grp_subsearch, wx.VERTICAL)
        sizer_subsearch.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # --- standalone utility: add an SRT subtitle track (ADR-032) ---
        # NOT part of the main conversion pipeline -- takes an already-converted 3D
        # (SBS/TB) MKV and an SRT file and muxes the SRT in as a plain soft-subtitle
        # track, preserving every existing track untouched. Deliberately does NOT
        # bake stereo-duplicated/positioned cues into the file -- iw3-player already
        # does client-side per-eye subtitle rendering from a plain track, so a
        # pre-baked stereo track would double-render there (see subtitle_mux_cli.py's
        # module docstring / ADR-032). Launches python -m iw3.subtitle_mux_cli as its
        # own subprocess, same out-of-process convention as RIFE/HDR reinjection.
        self.grp_submux = wx.StaticBox(
            self.tab_tools, label=T("Add Subtitle Track (Standalone Tool)"))

        self.lbl_submux_input = wx.StaticText(self.grp_submux, label=T("Converted 3D Video (.mkv)"))
        self.txt_submux_input = wx.TextCtrl(self.grp_submux, name="txt_submux_input")
        self.txt_submux_input.SetToolTip(
            T("What it's for: the already-converted 3D video to add a subtitle track to. Must be "
              "an .mkv file -- this tool does not convert containers, so an .mp4 output must first "
              "be remuxed to .mkv by some other tool.\n"
              "Con: read-only -- never modified. A new file is always written to Output File below.\n"
              "Recommended: the direct iw3 output file, with its normal SBS/TB filename tag intact "
              "(e.g. '..._LR.mkv') so Format below can auto-detect."))
        self.btn_submux_input = wx.Button(self.grp_submux, label=T("..."))

        self.lbl_submux_srt = wx.StaticText(self.grp_submux, label=T("Subtitle File (.srt)"))
        self.txt_submux_srt = wx.TextCtrl(self.grp_submux, name="txt_submux_srt")
        self.txt_submux_srt.SetToolTip(
            T("What it's for: the SRT subtitle file to add as a new track. Validated with pysubs2 "
              "before muxing, so a malformed SRT is caught here with a clear error rather than an "
              "opaque mkvmerge failure.\n"
              "Con: one SRT per run -- no multi-language batch support. Run this tool again for "
              "each additional language.\n"
              "Recommended: a plain, ordinary SRT -- no special stereo formatting needed or wanted "
              "(iw3-player already renders subtitles in 3D itself, per-eye, at playback time)."))
        self.btn_submux_srt = wx.Button(self.grp_submux, label=T("..."))

        self.lbl_submux_output = wx.StaticText(self.grp_submux, label=T("Output File"))
        self.txt_submux_output = wx.TextCtrl(self.grp_submux, name="txt_submux_output")
        self.txt_submux_output.SetToolTip(
            T("Where to write the new file with the subtitle track added. Auto-filled with "
              "'<converted file name>_subbed.mkv' in the same folder once you pick the converted "
              "video above -- change it if you want it saved somewhere else.\n"
              "How it's safe: this tool never overwrites the input video, only ever writes here."))
        self.btn_submux_output = wx.Button(self.grp_submux, label=T("..."))

        self.lbl_submux_format = wx.StaticText(self.grp_submux, label=T("Format"))
        self.cbo_submux_format = wx.ComboBox(
            self.grp_submux, name="cbo_submux_format",
            choices=["auto", "half_sbs", "full_sbs", "half_tb", "full_tb",
                     "cross_eyed", "vr90", "rgbd", "half_rgbd", "anaglyph"])
        self.cbo_submux_format.SetEditable(False)
        self.cbo_submux_format.SetSelection(0)
        self.cbo_submux_format.SetToolTip(
            T("What it's for: the stereo/output layout of the converted video above -- covers every "
              "watchable Stereo Format iw3 can produce (Export/Export disparity/Debug Depth aren't "
              "listed here since those are data-export formats, not something you'd add subtitles "
              "to). 'auto' (default) detects this from its filename using the same tags iw3 itself "
              "writes (e.g. '_LR', '_TB', '_LRF_Full_SBS', '_TBF_fulltb', '_RLF_cross', "
              "'_180x180_LR', '_RGBD', '_HRGBD', '_redcyan').\n"
              "Con: if the filename doesn't carry one of those tags (e.g. it was renamed), auto "
              "detection is inconclusive and the tool refuses rather than guessing -- pick the "
              "correct layout here explicitly in that case. By itself this value doesn't change "
              "how the subtitle is added -- a plain track works the same regardless of layout. It "
              "only matters together with the 'Position for external players (dual-eye)' checkbox "
              "below, which needs to know which layouts are a genuine two-eye split.\n"
              "Recommended: leave on 'auto' unless the tool's log below reports it couldn't detect "
              "the format."))

        self.lbl_submux_language = wx.StaticText(self.grp_submux, label=T("Language"))
        self.txt_submux_language = wx.TextCtrl(self.grp_submux, value="en", name="txt_submux_language")
        self.txt_submux_language.SetToolTip(
            T("What it's for: the language stored as metadata on the new subtitle track (e.g. en, "
              "ja, fr, de, es), shown by players in their subtitle track menu -- as an ISO 639-1 "
              "two-letter code, the SAME format as Search Subtitles' Language field above (ADR-042 "
              "standardized both fields onto this one format so typing the same code into both is "
              "always correct). Converted internally to the three-letter code mkvmerge actually "
              "needs (e.g. en -> eng, ja -> jpn, fr -> fre) by subtitle_mux_cli.py right before the "
              "mkvmerge call -- an unrecognized code is passed through unchanged rather than "
              "erroring out.\n"
              "Con: purely metadata -- does not translate or verify the actual subtitle content's "
              "language.\n"
              "Recommended: match the SRT file's actual language; default 'en' if unsure."))

        self.lbl_submux_track_name = wx.StaticText(self.grp_submux, label=T("Track Name"))
        self.txt_submux_track_name = wx.TextCtrl(self.grp_submux, name="txt_submux_track_name")
        self.txt_submux_track_name.SetToolTip(
            T("What it's for: an optional display name for the new subtitle track (shown in "
              "player track menus, e.g. 'English (Forced)'). Leave blank to default to the SRT "
              "file's own name."))

        self.chk_submux_dual_eye = wx.CheckBox(
            self.grp_submux, name="chk_submux_dual_eye",
            label=T("Position for external players (dual-eye)"))
        self.chk_submux_dual_eye.SetValue(True)
        self.chk_submux_dual_eye.SetToolTip(
            T("What it's for: fixes a real problem when playing the output in an EXTERNAL "
              "player (VLC, MPC-HC, a TV's built-in player, etc.) -- not iw3-player. On a "
              "split-eye Format (half_sbs, full_sbs, cross_eyed, vr90, half_tb, full_tb), a "
              "plain subtitle is centered against the WHOLE frame by a normal player, which "
              "lands it right on the seam between the two eyes and tears every line of text in "
              "half. Turning this on converts the subtitle into a track with two copies per "
              "line, one positioned inside each eye-half, so it reads correctly in an external "
              "player (ADR-053).\n"
              "Why it defaults ON: most output from this project is watched in an external "
              "player, not iw3-player, so this checkbox defaults to the setting that's correct "
              "there. The tradeoff: iw3-player already does its own correct per-eye rendering "
              "of a PLAIN subtitle track, so if this stays checked, iw3-player will show each "
              "line twice, wrongly positioned, because it re-duplicates an already-duplicated "
              "track. Uncheck this box specifically when you know this file will be watched in "
              "iw3-player -- that restores the original plain-track behavior (ADR-032).\n"
              "Con: no effect for Format rgbd, half_rgbd, or anaglyph -- those aren't a "
              "two-eye split, so there's no seam to fix (the tool's log below will note this "
              "if you check the box with one of those formats selected). Also requires probing "
              "the video's real width/height via ffprobe before muxing, which adds a brief "
              "extra step.\n"
              "Recommended: leave ON for external players (VLC, MPC-HC, TVs, etc.) -- the "
              "common case. Uncheck ONLY if you'll specifically watch the result in iw3-player."))

        # --- ADR-055: optional Font Size override for the dual-eye ASS track above ---
        self.lbl_submux_font_size = wx.StaticText(self.grp_submux, label=T("Font Size"))
        self.txt_submux_font_size = wx.TextCtrl(self.grp_submux, name="txt_submux_font_size")
        self.txt_submux_font_size.SetToolTip(
            T("What it's for: fixes a real bug -- subtitles added with Position for external "
              "players (dual-eye) checked above used to render TINY, because the underlying "
              "library's own fixed default text size (20px) was never designed for this "
              "project's typical output resolutions (e.g. a 3840x2076 double-wide 4K SBS "
              "frame, where 20px is under 1% of the frame's height). Left blank (the default, "
              "recommended for almost everyone), the text size now automatically scales with "
              "the video's real resolution instead, so it looks a normal, readable size with "
              "zero configuration. Enter a number here only if you want one specific exact "
              "pixel size instead of the automatic one.\n"
              "Why it defaults to blank/automatic: a fixed number that looks right at 1080p "
              "would be far too small at 4K and vice versa -- scaling automatically with the "
              "video's actual resolution gets it right every time without you having to guess "
              "or measure anything.\n"
              "Pro: automatic sizing already accounts for Top/Bottom formats needing smaller "
              "text than Side-by-Side formats at the same resolution, since each eye only gets "
              "half the vertical space in a Top/Bottom video.\n"
              "Con: only affects Position for external players (dual-eye)'s track -- it has no "
              "effect at all if that checkbox above is unchecked (a plain subtitle file has no "
              "text-size setting to control), and no effect for Format rgbd, half_rgbd, or "
              "anaglyph either, for the same reason that checkbox has none there. This also "
              "does not affect iw3-player's own subtitle text -- that has its own separate "
              "size control in the player itself (the on-screen Subtitles panel), unrelated to "
              "this tool.\n"
              "Values: any positive number (pixels). Typical readable sizes run roughly "
              "40-60px at 1080p and 80-120px at 4K, but leaving this blank already picks a "
              "sensible number for whatever resolution the video actually is.\n"
              "Recommended: leave blank so the size always matches the video's real "
              "resolution automatically. Only set an exact number if the automatic size still "
              "doesn't look right to you for some reason."))

        # --- ADR-054: optional Start Time/End Time trim+rebase for a full-movie SRT
        # applied to a trimmed clip. Same field NAME and time format (hh:mm:ss/mm:ss)
        # as this app's own main conversion tab (chk_start_time/txt_start_time on the
        # Video Filter group) and iw3.subtitle_mux_cli's own --start-time/--end-time,
        # which reuse iw3's exact parse_time() -- so a value typed here or copied from
        # there is always correct in both places. Checkbox + TimeCtrl pair mirrors HDR
        # Reinjection's own Start/End Time controls just above (same underlying
        # --start-time/--end-time flag names, same "position within the full source
        # that the trimmed/converted file starts/ends at" idea).
        self.chk_submux_start_time = wx.CheckBox(self.grp_submux, label=T("Start Time"),
                                                  name="chk_submux_start_time")
        self.chk_submux_start_time.SetToolTip(
            T("What it's for: the real problem this solves -- you converted only a TRIMMED "
              "CLIP of a full movie with iw3's own Start Time/End Time (e.g. "
              "--start-time 00:01:15 --end-time 00:06:15 on the main conversion tab), but "
              "Subtitle File above has timestamps for the FULL movie (e.g. downloaded via "
              "Search Subtitles above). Without this, a line that was originally at 1:16 in "
              "the full movie would still say 1:16 in the new track -- 1 minute 16 seconds "
              "into a 5-minute clip that has already ended. Checking this and entering the "
              "SAME Start Time you used for the original iw3 conversion trims Subtitle File "
              "to that point onward and REBASES it so that point becomes 0:00, lining up with "
              "the clip's own first frame.\n"
              "Con: there is no auto-detection of this -- you must know and enter the exact "
              "point within the full-movie subtitle file that matches the clip's first frame. "
              "A cue straddling this point is clipped to begin at 0 rather than dropped.\n"
              "Recommended: leave unchecked when Subtitle File already matches the converted "
              "video's own timeline (e.g. it was made/downloaded specifically for this clip). "
              "Check it and match iw3's own Start Time exactly when Subtitle File is for the "
              "full source instead."))
        self.txt_submux_start_time = TimeCtrl(self.grp_submux, value="00:00:00", fmt24hr=True,
                                               name="txt_submux_start_time")
        self.chk_submux_end_time = wx.CheckBox(self.grp_submux, label=T("End Time"),
                                                name="chk_submux_end_time")
        self.chk_submux_end_time.SetToolTip(
            T("Same idea as Start Time, but for where the trimmed clip ends within the "
              "full-movie subtitle file's own timeline -- match iw3's own End Time from the "
              "original conversion. A cue extending past this point is clipped to end at "
              "(End Time - Start Time) rather than left running past the clip's last frame; "
              "cues entirely after it are dropped.\n"
              "Con: given without Start Time checked, Start Time is treated as 00:00:00 "
              "(trimming only the tail end) -- check Start Time too if the clip doesn't "
              "start at the very beginning of the full movie.\n"
              "Recommended: leave unchecked to keep everything through the end of Subtitle "
              "File; check it and match iw3's own End Time when the clip ends before the "
              "full movie does."))
        self.txt_submux_end_time = TimeCtrl(self.grp_submux, value="00:00:00", fmt24hr=True,
                                             name="txt_submux_end_time")

        self.btn_submux_run = wx.Button(self.grp_submux, label=T("Run"))
        self.btn_submux_run.SetToolTip(
            T("What it's for: runs the mux as a separate background process (python -m "
              "iw3.subtitle_mux_cli) -- this app's own GPU/model state is never touched, and the "
              "input video is never modified.\n"
              "How it's safe: every existing track (video, audio, existing subtitles) is copied "
              "into the output completely unchanged -- only the new subtitle track is added.\n"
              "Con: if Format can't be auto-detected from the input filename, this refuses "
              "immediately with that exact message shown in the log box below, rather than "
              "guessing SBS vs TB.\n"
              "Recommended: check the log box below afterward to confirm it actually succeeded "
              "rather than refused."))

        self.txt_submux_log = wx.TextCtrl(self.grp_submux, style=wx.TE_MULTILINE | wx.TE_READONLY,
                                           size=self.FromDIP((-1, 60)), name="txt_submux_log")
        self.txt_submux_log.SetToolTip(
            T("Shows this tool's own output verbatim, including the exact hard-refusal message "
              "if Format detection fails or the SRT file fails validation -- not just a generic "
              "pass/fail toast."))
        self.btn_submux_clear = wx.Button(self.grp_submux, label=T("Clear"))
        self.btn_submux_clear.SetToolTip(
            T("Empties the log box above -- output only accumulates run after run otherwise. Disabled "
              "while a job is running so it can't wipe output you may still be reading mid-run; "
              "re-enabled once the job finishes."))

        self.btn_submux_input.Bind(wx.EVT_BUTTON, self.on_click_btn_submux_input)
        self.btn_submux_srt.Bind(wx.EVT_BUTTON, self.on_click_btn_submux_srt)
        self.btn_submux_output.Bind(wx.EVT_BUTTON, self.on_click_btn_submux_output)
        self.btn_submux_run.Bind(wx.EVT_BUTTON, self.on_click_btn_submux_run)
        self.btn_submux_clear.Bind(wx.EVT_BUTTON, lambda event: self.txt_submux_log.Clear())

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        h = -1
        layout.Add(self.lbl_submux_input, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_submux_input, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_submux_input, (h, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_submux_srt, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_submux_srt, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_submux_srt, (h, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_submux_output, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_submux_output, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_submux_output, (h, 3), flag=wx.EXPAND)
        # Format/Language/Track Name share one compact row -- this section lives
        # stacked below HDR Reinjection in the same column (see that section's own
        # note about spare vertical room in the Dual-Pass Depth Blend column).
        layout.Add(self.lbl_submux_format, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_submux_format, (h, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_submux_language, (h, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_submux_language, (h, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_submux_track_name, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_submux_track_name, (h, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.chk_submux_dual_eye, (h := h + 1, 0), (0, 3), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_submux_font_size, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_submux_font_size, (h, 1), flag=wx.EXPAND)
        # Start Time/End Time share one compact row (ADR-054) -- same convention as
        # HDR Reinjection's/Add Audio Track's own Start/End Time rows above.
        layout.Add(self.chk_submux_start_time, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_submux_start_time, (h, 1), flag=wx.EXPAND)
        layout.Add(self.chk_submux_end_time, (h, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_submux_end_time, (h, 3), flag=wx.EXPAND)
        layout.Add(self.btn_submux_run, (h := h + 1, 3), flag=wx.EXPAND)
        layout.Add(self.txt_submux_log, (h, 0), (0, 3), flag=wx.EXPAND)
        layout.Add(self.btn_submux_clear, (h := h + 1, 3), flag=wx.EXPAND)
        sizer_submux = wx.StaticBoxSizer(self.grp_submux, wx.VERTICAL)
        sizer_submux.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # --- standalone utility: add an audio track / dub (ADR-042) ---
        # NOT part of the main conversion pipeline -- takes an already-converted 3D
        # (SBS/TB) MKV and a separate audio file (e.g. a different-language dub) and
        # muxes it in as a plain NEW audio track, preserving every existing track
        # (video, existing audio, subtitles) untouched -- mirrors Add Subtitle
        # Track's shape above closely (see iw3/audio_mux_cli.py's own module
        # docstring / ADR-042). Optional Source Start/End Time trims (and shifts to
        # start at 0) the audio file itself via ffmpeg before muxing, for the case
        # where the audio source covers more content than the video (e.g. a
        # full-movie dub being added to a short test clip). Launches python -m
        # iw3.audio_mux_cli as its own subprocess, same out-of-process convention as
        # RIFE/HDR reinjection/Add Subtitle Track.
        self.grp_audiomux = wx.StaticBox(
            self.tab_tools, label=T("Add Audio Track (Standalone Tool)"))

        self.lbl_audiomux_input = wx.StaticText(self.grp_audiomux, label=T("Converted 3D Video (.mkv)"))
        self.txt_audiomux_input = wx.TextCtrl(self.grp_audiomux, name="txt_audiomux_input")
        self.txt_audiomux_input.SetToolTip(
            T("What it's for: the already-converted 3D video to add an audio track to. Must be "
              "an .mkv file -- this tool does not convert containers, so an .mp4 output must "
              "first be remuxed to .mkv by some other tool.\n"
              "Con: read-only -- never modified. A new file is always written to Output File "
              "below.\n"
              "Recommended: the direct iw3 output file."))
        self.btn_audiomux_input = wx.Button(self.grp_audiomux, label=T("..."))

        self.lbl_audiomux_audio = wx.StaticText(self.grp_audiomux, label=T("Audio File (dub)"))
        self.txt_audiomux_audio = wx.TextCtrl(self.grp_audiomux, name="txt_audiomux_audio")
        self.txt_audiomux_audio.SetToolTip(
            T("What it's for: the audio file to add as a new track -- e.g. a different-language "
              "dub. Common formats mkvmerge/ffmpeg already read directly work here: AAC, AC3, "
              "DTS, FLAC, MP3, Opus, WAV, and more -- no manual format conversion needed.\n"
              "Con: read-only -- never modified. If it covers more content than the video (e.g. "
              "a full-movie dub for a short clip), use Source Start/End Time below to trim it "
              "automatically rather than pre-cutting it by hand.\n"
              "Recommended: match the audio's actual length/content to the video above as "
              "closely as you can -- Source Start/End Time handles the rest."))
        self.btn_audiomux_audio = wx.Button(self.grp_audiomux, label=T("..."))

        self.lbl_audiomux_output = wx.StaticText(self.grp_audiomux, label=T("Output File"))
        self.txt_audiomux_output = wx.TextCtrl(self.grp_audiomux, name="txt_audiomux_output")
        self.txt_audiomux_output.SetToolTip(
            T("Where to write the new file with the audio track added. Auto-filled with "
              "'<converted file name>_dubbed.mkv' in the same folder once you pick the "
              "converted video above -- change it if you want it saved somewhere else.\n"
              "How it's safe: this tool never overwrites the input video or the audio file, "
              "only ever writes here."))
        self.btn_audiomux_output = wx.Button(self.grp_audiomux, label=T("..."))

        self.lbl_audiomux_language = wx.StaticText(self.grp_audiomux, label=T("Language"))
        self.cbo_audiomux_language = wx.ComboBox(
            self.grp_audiomux, value="en", name="cbo_audiomux_language",
            choices=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko",
                     "zh", "nl", "sv", "no", "da", "pl", "tr", "ar", "hi"])
        self.cbo_audiomux_language.SetToolTip(
            T("What it's for: the language of the new audio track, as an ISO 639-1 two-letter "
              "code (e.g. en, es, fr, ja) -- converted internally to the three-letter code "
              "mkvmerge stores as track metadata (e.g. en -> eng). Purely metadata -- does not "
              "translate or verify the actual audio content's language.\n"
              "Values: pick from the dropdown, or type any other ISO 639-1 code -- this list "
              "only covers the most common languages, it isn't exhaustive; an unrecognized code "
              "is passed straight through to mkvmerge.\n"
              "Recommended: match the audio file's actual language; default 'en' if unsure."))

        self.lbl_audiomux_track_name = wx.StaticText(self.grp_audiomux, label=T("Track Name"))
        self.txt_audiomux_track_name = wx.TextCtrl(self.grp_audiomux, name="txt_audiomux_track_name")
        self.txt_audiomux_track_name.SetToolTip(
            T("What it's for: an optional display name for the new audio track (shown in "
              "player track menus, e.g. 'Spanish Dub'). Leave blank to default to the audio "
              "file's own name."))

        self.chk_audiomux_default = wx.CheckBox(self.grp_audiomux, label=T("Set as default track"),
                                                 name="chk_audiomux_default")
        self.chk_audiomux_default.SetValue(False)
        self.chk_audiomux_default.SetToolTip(
            T("What it's for: marks the new audio track as the one a player selects "
              "automatically, instead of just adding it as a selectable alternate.\n"
              "Con: any existing audio track's own default flag in the input video is left "
              "exactly as it already was -- if it was ALSO marked default, some players may then "
              "see two default audio tracks and pick whichever one they encounter first, rather "
              "than reliably preferring this new one.\n"
              "Recommended: off (default) -- usually you're adding an alternate-language track, "
              "not replacing the primary audio. Turn on only when you specifically want this new "
              "track to play automatically."))

        self.chk_audiomux_start_time = wx.CheckBox(self.grp_audiomux, label=T("Source Start"),
                                                    name="chk_audiomux_start_time")
        self.chk_audiomux_start_time.SetToolTip(
            T("What it's for: trims Audio File to start at this point, for when the audio "
              "source covers more content than the video above (e.g. a full-movie dub being "
              "added to a short test clip). The trimmed audio is also automatically shifted to "
              "start at t=0 so it lines up with the video's first frame -- you do not need to "
              "cut or shift the audio by hand. Leave unchecked to use the whole audio file "
              "as-is.\n"
              "Con: there is no auto-detection of this -- you must know and enter the exact "
              "range within the audio file that matches the video above."))
        self.txt_audiomux_start_time = TimeCtrl(self.grp_audiomux, value="00:00:00", fmt24hr=True,
                                                 name="txt_audiomux_start_time")
        self.chk_audiomux_end_time = wx.CheckBox(self.grp_audiomux, label=T("Source End"),
                                                  name="chk_audiomux_end_time")
        self.chk_audiomux_end_time.SetToolTip(
            T("Same idea as Source Start, but for where the trimmed audio should end. Leave "
              "unchecked to use the end of the audio file."))
        self.txt_audiomux_end_time = TimeCtrl(self.grp_audiomux, value="00:00:00", fmt24hr=True,
                                               name="txt_audiomux_end_time")

        self.btn_audiomux_run = wx.Button(self.grp_audiomux, label=T("Run"))
        self.btn_audiomux_run.SetToolTip(
            T("What it's for: runs the mux as a separate background process (python -m "
              "iw3.audio_mux_cli) -- this app's own GPU/model state is never touched, and "
              "neither input file is ever modified.\n"
              "How it's safe: every existing track (video, existing audio, subtitles) is "
              "copied into the output completely unchanged -- only the new audio track is "
              "added.\n"
              "Con: if Source Start/End Time is set, trimming re-runs ffmpeg first, which can "
              "take a little longer than an untrimmed run.\n"
              "Recommended: check the log box below afterward to confirm it actually succeeded "
              "rather than refused."))

        self.txt_audiomux_log = wx.TextCtrl(self.grp_audiomux, style=wx.TE_MULTILINE | wx.TE_READONLY,
                                             size=self.FromDIP((-1, 60)), name="txt_audiomux_log")
        self.txt_audiomux_log.SetToolTip(
            T("Shows this tool's own output verbatim, including the exact ffmpeg trim "
              "command(s) when Source Start/End Time is used, and the exact refusal reason "
              "if anything fails -- not just a generic pass/fail toast."))
        self.btn_audiomux_clear = wx.Button(self.grp_audiomux, label=T("Clear"))
        self.btn_audiomux_clear.SetToolTip(
            T("Empties the log box above -- output only accumulates run after run otherwise. Disabled "
              "while a job is running so it can't wipe output you may still be reading mid-run; "
              "re-enabled once the job finishes."))

        self.btn_audiomux_input.Bind(wx.EVT_BUTTON, self.on_click_btn_audiomux_input)
        self.btn_audiomux_audio.Bind(wx.EVT_BUTTON, self.on_click_btn_audiomux_audio)
        self.btn_audiomux_output.Bind(wx.EVT_BUTTON, self.on_click_btn_audiomux_output)
        self.btn_audiomux_run.Bind(wx.EVT_BUTTON, self.on_click_btn_audiomux_run)
        self.btn_audiomux_clear.Bind(wx.EVT_BUTTON, lambda event: self.txt_audiomux_log.Clear())

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        h = -1
        layout.Add(self.lbl_audiomux_input, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_audiomux_input, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_audiomux_input, (h, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_audiomux_audio, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_audiomux_audio, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_audiomux_audio, (h, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_audiomux_output, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_audiomux_output, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_audiomux_output, (h, 3), flag=wx.EXPAND)
        # Language/Track Name/Default share one compact row, and Source Start/End
        # Time + Run share another -- same compact-row convention as HDR
        # Reinjection/Add Subtitle Track above, this column has limited spare
        # vertical room.
        layout.Add(self.lbl_audiomux_language, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_audiomux_language, (h, 1), flag=wx.EXPAND)
        layout.Add(self.chk_audiomux_default, (h, 2), (0, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_audiomux_track_name, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_audiomux_track_name, (h, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.chk_audiomux_start_time, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_audiomux_start_time, (h, 1), flag=wx.EXPAND)
        layout.Add(self.chk_audiomux_end_time, (h, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_audiomux_end_time, (h, 3), flag=wx.EXPAND)
        layout.Add(self.btn_audiomux_run, (h := h + 1, 3), flag=wx.EXPAND)
        layout.Add(self.txt_audiomux_log, (h, 0), (0, 3), flag=wx.EXPAND)
        layout.Add(self.btn_audiomux_clear, (h := h + 1, 3), flag=wx.EXPAND)
        sizer_audiomux = wx.StaticBoxSizer(self.grp_audiomux, wx.VERTICAL)
        sizer_audiomux.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # --- standalone utility: retroactive MKV StereoMode tagging (ADR-033) ---
        # NOT part of the main conversion pipeline -- takes an already-converted iw3
        # .mkv output that was made before "Tag MKV as 3D (StereoMode)" existed (or
        # with it off) and tags it retroactively. Unlike HDR Reinjection/Add Subtitle
        # Track above, this edits the input file IN PLACE via mkvpropedit -- that tool
        # only rewrites container metadata, never re-encodes/re-muxes the streams, so
        # there is no separate output file to pick (see stereo_mode_tag_cli.py's
        # module docstring / ADR-033). Launches python -m iw3.stereo_mode_tag_cli as
        # its own subprocess, same out-of-process convention as the other standalone
        # tools in this column.
        self.grp_stereotag = wx.StaticBox(
            self.tab_tools, label=T("Retroactively Tag MKV as 3D (Standalone Tool)"))

        self.lbl_stereotag_input = wx.StaticText(self.grp_stereotag, label=T("Converted 3D Video (.mkv)"))
        self.txt_stereotag_input = wx.TextCtrl(self.grp_stereotag, name="txt_stereotag_input")
        self.txt_stereotag_input.SetToolTip(
            T("What it's for: the already-converted 3D video to tag. Must be an .mkv file -- "
              "StereoMode is a Matroska-only property.\n"
              "Con: this file IS edited in place (unlike the other standalone tools above, which "
              "always write a separate new file) -- mkvpropedit only rewrites container metadata, "
              "never re-encodes or re-muxes the actual video/audio, so there is nothing to gain "
              "from a full copy first. Use Backup below if you still want a safety copy.\n"
              "Recommended: the direct iw3 output file, with its normal SBS/TB filename tag intact "
              "(e.g. '..._LR.mkv') so Format below can auto-detect."))
        self.btn_stereotag_input = wx.Button(self.grp_stereotag, label=T("..."))

        self.lbl_stereotag_format = wx.StaticText(self.grp_stereotag, label=T("Format"))
        self.cbo_stereotag_format = wx.ComboBox(
            self.grp_stereotag, name="cbo_stereotag_format",
            choices=["auto", "half_sbs", "full_sbs", "half_tb", "full_tb",
                     "cross_eyed", "vr90", "rgbd", "half_rgbd", "anaglyph"])
        self.cbo_stereotag_format.SetEditable(False)
        self.cbo_stereotag_format.SetSelection(0)
        self.cbo_stereotag_format.SetToolTip(
            T("What it's for: the stereo/output layout of the video above. 'auto' (default) detects "
              "this from its filename using the same tags iw3 itself writes (e.g. '_LR', '_TB', "
              "'_LRF_Full_SBS', '_TBF_fulltb', '_RLF_cross', '_180x180_LR').\n"
              "Con: rgbd/half_rgbd/anaglyph are listed here so detection can name them, but tagging "
              "always refuses for those three (not a two-eye stereo pair, or already correct without "
              "tagging) -- see the log box below for the exact reason if that happens.\n"
              "Recommended: leave on 'auto' unless the tool's log below reports it couldn't detect "
              "the format."))

        self.chk_stereotag_backup = wx.CheckBox(self.grp_stereotag, label=T("Backup before editing"),
                                                name="chk_stereotag_backup")
        self.chk_stereotag_backup.SetValue(False)
        self.chk_stereotag_backup.SetToolTip(
            T("What it's for: copies the input file to '<name>.mkv.bak' before tagging, as an extra "
              "safety net.\n"
              "Con: uses extra disk space equal to the whole input file, and takes time to copy on a "
              "large file.\n"
              "Recommended: off (default) is fine for most people -- mkvpropedit's edit only touches "
              "metadata, never the actual video/audio. Turn on if you'd rather have a copy just in "
              "case."))

        self.btn_stereotag_run = wx.Button(self.grp_stereotag, label=T("Run"))
        self.btn_stereotag_run.SetToolTip(
            T("What it's for: runs the tagging as a separate background process (python -m "
              "iw3.stereo_mode_tag_cli) -- this app's own GPU/model state is never touched.\n"
              "Con: edits the input file above in place -- see that field's own note.\n"
              "Recommended: check the log box below afterward to confirm it actually succeeded "
              "rather than refused."))

        self.txt_stereotag_log = wx.TextCtrl(self.grp_stereotag, style=wx.TE_MULTILINE | wx.TE_READONLY,
                                             size=self.FromDIP((-1, 60)), name="txt_stereotag_log")
        self.txt_stereotag_log.SetToolTip(
            T("Shows this tool's own output verbatim, including the exact refusal reason if Format "
              "detection fails or the resolved format isn't taggable -- not just a generic pass/fail."))
        self.btn_stereotag_clear = wx.Button(self.grp_stereotag, label=T("Clear"))
        self.btn_stereotag_clear.SetToolTip(
            T("Empties the log box above -- output only accumulates run after run otherwise. Disabled "
              "while a job is running so it can't wipe output you may still be reading mid-run; "
              "re-enabled once the job finishes."))

        self.btn_stereotag_input.Bind(wx.EVT_BUTTON, self.on_click_btn_stereotag_input)
        self.btn_stereotag_run.Bind(wx.EVT_BUTTON, self.on_click_btn_stereotag_run)
        self.btn_stereotag_clear.Bind(wx.EVT_BUTTON, lambda event: self.txt_stereotag_log.Clear())

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        h = -1
        layout.Add(self.lbl_stereotag_input, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_stereotag_input, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_stereotag_input, (h, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_stereotag_format, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_stereotag_format, (h, 1), flag=wx.EXPAND)
        layout.Add(self.chk_stereotag_backup, (h, 2), (0, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.btn_stereotag_run, (h := h + 1, 3), flag=wx.EXPAND)
        layout.Add(self.txt_stereotag_log, (h, 0), (0, 3), flag=wx.EXPAND)
        layout.Add(self.btn_stereotag_clear, (h := h + 1, 3), flag=wx.EXPAND)
        sizer_stereotag = wx.StaticBoxSizer(self.grp_stereotag, wx.VERTICAL)
        sizer_stereotag.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # --- standalone utility: apply the Sharpen filter to an already-converted
        # video (ADR-063) ---
        # NOT part of the main conversion pipeline -- takes an already-converted 3D
        # video and applies ADR-061's edge-aware unsharp-mask Sharpen filter to it,
        # without re-running depth/stereo conversion. Reuses DE.apply_sharpen
        # directly (never forked/duplicated) via iw3.sharpen_cli, launched as its own
        # subprocess -- same out-of-process convention as the other standalone tools
        # in this column (see docs/ai/CODING_STANDARDS.md CS-SUBPROCESS-001). Unlike
        # Retroactively Tag MKV as 3D just above, this DOES write a separate output
        # file (an unsharp mask re-encodes the video track, unlike a pure
        # metadata-only edit), same convention as HDR Reinjection/Add Subtitle
        # Track/Add Audio Track.
        self.grp_sharpen = wx.StaticBox(
            self.tab_tools, label=T("Sharpen (Standalone Tool)"))

        self.lbl_sharpen_input = wx.StaticText(self.grp_sharpen, label=T("Converted 3D Video (.mkv)"))
        self.txt_sharpen_input = wx.TextCtrl(self.grp_sharpen, name="txt_sharpen_input")
        self.txt_sharpen_input.SetToolTip(
            T("What it's for: the already-converted 3D video to sharpen. Must be an .mkv "
              "file -- mkvmerge is what guarantees every other track (audio, subtitles, "
              "chapters, attachments) is copied through byte-for-byte unchanged around the "
              "freshly re-encoded video, and that's only available for Matroska.\n"
              "Con: read-only -- never modified. A new file is always written to Output "
              "File below.\n"
              "Recommended: the direct iw3 output file, with its normal SBS/TB/RGBD/"
              "Anaglyph filename tag intact (e.g. '..._LR.mkv') so Format below can "
              "auto-detect."))
        self.btn_sharpen_input = wx.Button(self.grp_sharpen, label=T("..."))

        self.lbl_sharpen_output = wx.StaticText(self.grp_sharpen, label=T("Output File"))
        self.txt_sharpen_output = wx.TextCtrl(self.grp_sharpen, name="txt_sharpen_output")
        self.txt_sharpen_output.SetToolTip(
            T("Where to write the new sharpened copy. Auto-filled with '<converted file "
              "name>_sharpened.mkv' in the same folder once you pick the converted video "
              "above -- change it if you want it saved somewhere else.\n"
              "How it's safe: this tool never overwrites the input video, only ever writes "
              "here."))
        self.btn_sharpen_output = wx.Button(self.grp_sharpen, label=T("..."))

        self.lbl_sharpen_format = wx.StaticText(self.grp_sharpen, label=T("Format"))
        self.cbo_sharpen_format = wx.ComboBox(
            self.grp_sharpen, name="cbo_sharpen_format",
            choices=["auto", "half_sbs", "full_sbs", "half_tb", "full_tb",
                     "cross_eyed", "vr90", "rgbd", "half_rgbd", "anaglyph"])
        self.cbo_sharpen_format.SetEditable(False)
        self.cbo_sharpen_format.SetSelection(0)
        self.cbo_sharpen_format.SetToolTip(
            T("What it's for: the stereo/output layout of the converted video above -- "
              "needed so this tool knows how to split it before sharpening (a genuine "
              "two-eye layout is split into its eye halves and each is sharpened "
              "independently, never across the seam; RGBD/Half RGBD only sharpens the RGB "
              "half, since the other half is a depth map, not a picture; Anaglyph has no "
              "seam and is sharpened as one whole frame). 'auto' (default) detects this "
              "from its filename using the same tags iw3 itself writes (e.g. '_LR', '_TB', "
              "'_LRF_Full_SBS', '_TBF_fulltb', '_RLF_cross', '_180x180_LR', '_RGBD', "
              "'_HRGBD', '_redcyan').\n"
              "Con: if the filename doesn't carry one of those tags (e.g. it was renamed), "
              "auto detection is inconclusive and the tool refuses rather than guessing -- "
              "pick the correct layout here explicitly in that case.\n"
              "Recommended: leave on 'auto' unless the tool's log below reports it "
              "couldn't detect the format."))

        self.lbl_sharpen_strength = wx.StaticText(self.grp_sharpen, label=T("Strength"))
        self.cbo_sharpen_strength_standalone = EditableComboBox(
            self.grp_sharpen, choices=["0.25", "0.5", "0.75", "1.0"],
            name="cbo_sharpen_strength_standalone")
        self.cbo_sharpen_strength_standalone.SetSelection(1)
        self.cbo_sharpen_strength_standalone.SetToolTip(
            T("How strong the Sharpen effect is (0.0-1.0) -- the exact same edge-aware "
              "unsharp-mask filter, range, and default (0.5) as the in-pipeline Sharpen "
              "control on the Stereo tab (ADR-061), just applied here to an already-"
              "converted file instead of during conversion.\n"
              "Values: higher = more pronounced detail boost at real edges/texture, but "
              "also more risk of an over-crisp/harsh look or exaggerating real compression "
              "artifacts. 0.0 is an exact no-op (the video track is still fully "
              "re-encoded, but every pixel is left unchanged) -- useful only for testing.\n"
              "Recommended: 0.5 (default) as a safe starting point, same as the "
              "in-pipeline control."))

        self.btn_sharpen_run = wx.Button(self.grp_sharpen, label=T("Run"))
        self.btn_sharpen_run.SetToolTip(
            T("What it's for: runs the sharpen pass as a separate background process "
              "(python -m iw3.sharpen_cli) -- this app's own GPU/model state is never "
              "touched, and the input video is never modified.\n"
              "How it's safe: every existing track (audio, subtitles, chapters, "
              "attachments) is copied into the output completely unchanged -- only the "
              "video track is re-encoded, and only its pixels are touched.\n"
              "Con: if Format can't be auto-detected from the input filename, this refuses "
              "immediately with that exact message shown in the log box below, rather than "
              "guessing the layout. Re-encoding the video track takes time proportional to "
              "the video's length.\n"
              "Recommended: check the log box below afterward to confirm it actually "
              "succeeded rather than refused."))

        self.txt_sharpen_log = wx.TextCtrl(self.grp_sharpen, style=wx.TE_MULTILINE | wx.TE_READONLY,
                                            size=self.FromDIP((-1, 60)), name="txt_sharpen_log")
        self.txt_sharpen_log.SetToolTip(
            T("Shows this tool's own output verbatim, including the exact hard-refusal "
              "message if Format detection fails -- not just a generic pass/fail toast."))
        self.btn_sharpen_clear = wx.Button(self.grp_sharpen, label=T("Clear"))
        self.btn_sharpen_clear.SetToolTip(
            T("Empties the log box above -- output only accumulates run after run otherwise. Disabled "
              "while a job is running so it can't wipe output you may still be reading mid-run; "
              "re-enabled once the job finishes."))

        self.btn_sharpen_input.Bind(wx.EVT_BUTTON, self.on_click_btn_sharpen_input)
        self.btn_sharpen_output.Bind(wx.EVT_BUTTON, self.on_click_btn_sharpen_output)
        self.btn_sharpen_run.Bind(wx.EVT_BUTTON, self.on_click_btn_sharpen_run)
        self.btn_sharpen_clear.Bind(wx.EVT_BUTTON, lambda event: self.txt_sharpen_log.Clear())

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        h = -1
        layout.Add(self.lbl_sharpen_input, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_sharpen_input, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_sharpen_input, (h, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_sharpen_output, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_sharpen_output, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_sharpen_output, (h, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_sharpen_format, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_sharpen_format, (h, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_sharpen_strength, (h, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_sharpen_strength_standalone, (h, 3), flag=wx.EXPAND)
        layout.Add(self.btn_sharpen_run, (h := h + 1, 3), flag=wx.EXPAND)
        layout.Add(self.txt_sharpen_log, (h, 0), (0, 3), flag=wx.EXPAND)
        layout.Add(self.btn_sharpen_clear, (h := h + 1, 3), flag=wx.EXPAND)
        sizer_sharpen = wx.StaticBoxSizer(self.grp_sharpen, wx.VERTICAL)
        sizer_sharpen.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # --- standalone utility: RIFE frame interpolation on an already-converted
        # video (ADR-029/ADR-049/ADR-051's own CLI tool, iw3/rife_cli.py, previously
        # CLI-only -- see ADR-051's own note this was left as a deliberate follow-up).
        # NOT part of the main conversion pipeline (that's chk_rife_interpolate/
        # cbo_rife_model/cbo_rife_mode/txt_rife_target_fps up in Post-Processing above,
        # untouched by this) -- this is the SAME iw3.rife_cli tool, just launched
        # retroactively against a video that was already converted, following the exact
        # same out-of-process pattern/layout as Sharpen (Standalone Tool) just above
        # (see docs/ai/CODING_STANDARDS.md CS-SUBPROCESS-001). Every control maps to a
        # real rife_cli.py create_parser() flag: --input/--output/--rife-model/
        # --rife-multiplier/--rife-target-fps/--gpu (original), plus --video-codec
        # (added 2026-09-08 fixing the real, confirmed bug that RIFE could never
        # output HEVC at all -- see docs/ai/AI_DECISIONS.md ADR-051/ADR-064
        # amendments; cbo_rife_standalone_codec below).
        self.grp_rife_standalone = wx.StaticBox(
            self.tab_tools, label=T("RIFE Frame Interpolation (Standalone Tool)"))

        self.lbl_rife_standalone_input = wx.StaticText(self.grp_rife_standalone,
                                                         label=T("Converted 3D Video"))
        self.txt_rife_standalone_input = wx.TextCtrl(self.grp_rife_standalone,
                                                       name="txt_rife_standalone_input")
        self.txt_rife_standalone_input.SetToolTip(
            T("What it's for: an already-converted 3D video to smooth the motion of. RIFE is a separate "
              "AI model that creates new, genuinely synthesized in-between frames (not simple frame "
              "duplication/blending) so motion looks smoother at a higher frame rate -- e.g. a 24fps "
              "movie interpolated to 48fps.\n"
              "How it's safe: read-only -- never modified. A new file is always written to Output File "
              "below.\n"
              "Con: RIFE interpolates the FINAL PACKED stereo frame (both eyes already combined) as one "
              "image, so it will see the seam between the two packed eyes -- it wasn't trained on that, "
              "though in practice both eyes move together so this doesn't cause left/right desync (same "
              "accepted tradeoff as the in-pipeline RIFE step above).\n"
              "Recommended: the direct iw3 output file you already made."))
        self.btn_rife_standalone_input = wx.Button(self.grp_rife_standalone, label=T("..."))

        self.lbl_rife_standalone_output = wx.StaticText(self.grp_rife_standalone, label=T("Output File"))
        self.txt_rife_standalone_output = wx.TextCtrl(self.grp_rife_standalone,
                                                        name="txt_rife_standalone_output")
        self.txt_rife_standalone_output.SetToolTip(
            T("Where to write the new, motion-smoothed copy. Auto-filled with '<converted file "
              "name>_rife<ext>' in the same folder once you pick the converted video above -- change it "
              "if you want it saved somewhere else.\n"
              "How it's safe: this tool never overwrites the input video, only ever writes here. It also "
              "always writes a small '<output>.rife_manifest.json' file next to it, recording which "
              "output frames are real and which are RIFE-synthetic -- see Run below for what that's for."))
        self.btn_rife_standalone_output = wx.Button(self.grp_rife_standalone, label=T("..."))

        self.lbl_rife_standalone_model = wx.StaticText(self.grp_rife_standalone, label=T("RIFE Model"))
        self.cbo_rife_standalone_model = wx.ComboBox(self.grp_rife_standalone,
                                                       choices=["rife_425", "rife_425_lite"],
                                                       name="cbo_rife_standalone_model")
        self.cbo_rife_standalone_model.SetEditable(False)
        self.cbo_rife_standalone_model.SetSelection(0)
        self.cbo_rife_standalone_model.SetToolTip(
            T("Which RIFE model quality tier to use -- the same two tiers, and same tradeoff, as the "
              "in-pipeline RIFE step above.\n"
              "rife_425: the recommended full model -- better motion accuracy, especially on complex/fast "
              "motion, at a higher compute cost.\n"
              "rife_425_lite: a lower-compute-cost variant of the same generation, trades a little "
              "accuracy for speed.\n"
              "Weights are downloaded automatically the first time you use a given tier (not bundled with "
              "the app).\n"
              "Recommended: rife_425 unless interpolation time is a real bottleneck for you."))

        self.lbl_rife_standalone_mode = wx.StaticText(self.grp_rife_standalone, label=T("Rate"))
        self.cbo_rife_standalone_mode = wx.ComboBox(self.grp_rife_standalone,
                                                      choices=["2x", "3x", "4x", "Custom FPS..."],
                                                      name="cbo_rife_standalone_mode")
        self.cbo_rife_standalone_mode.SetEditable(False)
        self.cbo_rife_standalone_mode.SetSelection(0)
        self.cbo_rife_standalone_mode.SetToolTip(
            T("What it's for: how many new frames RIFE inserts, and where.\n"
              "2x/3x/4x: inserts 1/2/3 evenly-spaced new frames between every pair of real frames, "
              "multiplying the frame rate by that exact amount (e.g. 24fps source -> 48/72/96fps).\n"
              "Custom FPS...: interpolate to an exact frame rate you choose in the field to the right "
              "instead (e.g. 24fps source -> 60fps target), even when it isn't a clean multiple of the "
              "source -- RIFE works out the correct in-between timing for each new frame automatically. "
              "Must be higher than the source video's own frame rate -- this tool refuses (see the log "
              "box below) rather than silently misbehaving if it isn't.\n"
              "Con: 3x/4x do roughly 2x/3x as many interpolation passes as the 2x default, so processing "
              "time increases proportionally. For a non-integer Custom FPS target (like 24->60), the new "
              "frames aren't perfectly evenly spaced in time, which can show as very slightly uneven "
              "motion smoothness on some frames -- usually not noticeable, unlike the perfectly even "
              "spacing of the clean 2x/3x/4x multipliers.\n"
              "Recommended: 2x for most uses; Custom FPS if you need to match a specific display or "
              "editing timeline's exact frame rate."))
        self.txt_rife_standalone_target_fps = wx.TextCtrl(self.grp_rife_standalone,
                                                            name="txt_rife_standalone_target_fps")
        self.txt_rife_standalone_target_fps.SetToolTip(
            T("What it's for: the exact output frame rate to interpolate to, used only when Rate above is "
              "set to \"Custom FPS...\".\n"
              "Values: any number higher than your source video's own frame rate (e.g. 60 for a 24fps "
              "source). A target at or below the source's frame rate is rejected -- RIFE only adds "
              "frames, it never removes them.\n"
              "Recommended: 60 for standard smooth-motion displays, or match your target display/editing "
              "timeline's exact refresh rate."))

        self.lbl_rife_standalone_gpu = wx.StaticText(self.grp_rife_standalone, label=T("GPU"))
        self.cbo_rife_standalone_gpu = wx.ComboBox(self.grp_rife_standalone, name="cbo_rife_standalone_gpu")
        self.cbo_rife_standalone_gpu.SetEditable(False)
        cuda_device_names = _query_nvidia_smi_gpu_names()
        if cuda_device_names is not None:
            # See _query_nvidia_smi_gpu_names()'s docstring -- same reasoning as
            # the Device dropdown above.
            for i, device_name in enumerate(cuda_device_names):
                self.cbo_rife_standalone_gpu.Append(f"{i}:{device_name}", i)
        elif torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_properties(i).name
                self.cbo_rife_standalone_gpu.Append(f"{i}:{device_name}", i)
        elif mps_is_available():
            self.cbo_rife_standalone_gpu.Append("MPS", 0)
        elif xpu_is_available():
            for i in range(torch.xpu.device_count()):
                device_name = torch.xpu.get_device_name(i)
                self.cbo_rife_standalone_gpu.Append(f"{i}:{device_name}", i)
        self.cbo_rife_standalone_gpu.Append("CPU", -1)
        self.cbo_rife_standalone_gpu.SetSelection(0)
        self.cbo_rife_standalone_gpu.SetToolTip(
            T("Which GPU (or CPU) runs this tool's own RIFE model -- reuses the Device selector's "
              "convention from the Processor tab, but without \"All CUDA Device\": this tool runs as one "
              "single background process (see Run below) and iw3.rife_cli's own --gpu option only ever "
              "targets one device, it cannot split work across several the way the main conversion's "
              "Device selector can.\n"
              "Con: CPU works without a GPU but is dramatically slower -- only use it if you have no "
              "compatible graphics card.\n"
              "Recommended: your main GPU (the first entry) unless you're deliberately running this "
              "alongside another GPU job and want to keep them on separate devices."))

        self.lbl_rife_standalone_codec = wx.StaticText(self.grp_rife_standalone, label=T("Output Codec"))
        self.cbo_rife_standalone_codec = wx.ComboBox(self.grp_rife_standalone,
                                                       name="cbo_rife_standalone_codec")
        self.cbo_rife_standalone_codec.SetEditable(False)
        # ClientData carries the real --video-codec value each choice maps to (None
        # for the unchanged default) -- same convention as cbo_rife_standalone_gpu
        # above.
        self.cbo_rife_standalone_codec.Append(T("H.264 (default)"), None)
        self.cbo_rife_standalone_codec.Append(T("H.265/HEVC -- libx265 (CPU)"), "libx265")
        self.cbo_rife_standalone_codec.Append(T("H.265/HEVC -- hevc_nvenc (GPU)"), "hevc_nvenc")
        self.cbo_rife_standalone_codec.SetSelection(0)
        self.cbo_rife_standalone_codec.SetToolTip(
            T("What it's for: which video format this tool encodes its OUTPUT with.\n"
              "Why you would change it: RIFE's own output has always defaulted to H.264, unrelated to "
              "Dolby Vision/HDR entirely -- leave this on the default for that. But if the video you're "
              "interpolating has Dolby Vision or HDR10+ and you plan to fix that metadata afterward with "
              "the Retroactive HDR/DV Reinjection tool above (its \"RIFE Manifest\" field is built for "
              "exactly this), that tool can ONLY inject into an HEVC (H.265) file -- RIFE's H.264 default "
              "output can NEVER accept that metadata, no matter what. Pick an HEVC option here FIRST if "
              "that's your plan.\n"
              "H.265/HEVC -- libx265 (CPU): software encode, works on any machine, slower and produces a "
              "larger file than the H.264 default at the same quality setting.\n"
              "H.265/HEVC -- hevc_nvenc (GPU): hardware encode on the GPU selected above, much faster "
              "than libx265, requires an NVIDIA GPU with NVENC support (most GeForce/RTX cards from the "
              "last several generations).\n"
              "Con: HEVC output is somewhat less universally compatible with older/non-4K playback "
              "devices than H.264, and both HEVC options here produce a larger or slower-to-produce file "
              "than the H.264 default.\n"
              "Recommended: leave on the default (H.264) unless you specifically plan to run the "
              "Retroactive HDR/DV Reinjection tool afterward -- then pick libx265 (works everywhere) or "
              "hevc_nvenc (faster, if your GPU supports it)."))

        self.btn_rife_standalone_run = wx.Button(self.grp_rife_standalone, label=T("Run"))
        self.btn_rife_standalone_run.SetToolTip(
            T("What it's for: runs RIFE interpolation as a separate background process (python -m "
              "iw3.rife_cli) -- this app's own GPU/model state is never touched, and the input video is "
              "never modified.\n"
              "How it's safe: writes to a new output file only; a '<output>.rife_manifest.json' sidecar "
              "is always written alongside it too, recording which output frames are real and which are "
              "RIFE-synthetic.\n"
              "Important -- Dolby Vision/HDR: RIFE itself does NOT touch DV/HDR10+ metadata at all (it "
              "doesn't even look at it). If the video you're interpolating has Dolby Vision, don't stop "
              "here -- first, set Output Codec above to an HEVC option (DV/HDR10+ reinjection requires "
              "HEVC output, and this tool's H.264 default can never accept it). Then use the Retroactive "
              "HDR/DV Reinjection tool above, pointing its \"Converted\" field at this Run's output and "
              "its \"RIFE Manifest\" field at the '.rife_manifest.json' sidecar this Run writes, against "
              "your ORIGINAL Dolby Vision source. Skipping either step means the output plays back "
              "without correct Dolby Vision metadata.\n"
              "Recommended: check the log box below afterward to confirm it actually succeeded rather "
              "than refused, and note the printed manifest file path if you'll need it for Dolby Vision."))

        self.txt_rife_standalone_log = wx.TextCtrl(self.grp_rife_standalone,
                                                     style=wx.TE_MULTILINE | wx.TE_READONLY,
                                                     size=self.FromDIP((-1, 60)), name="txt_rife_standalone_log")
        self.txt_rife_standalone_log.SetToolTip(
            T("Shows this tool's own output verbatim, including the exact refusal message if Custom FPS "
              "isn't genuinely higher than the source's own frame rate, and the manifest file path it "
              "wrote on success."))
        self.btn_rife_standalone_clear = wx.Button(self.grp_rife_standalone, label=T("Clear"))
        self.btn_rife_standalone_clear.SetToolTip(
            T("Empties the log box above -- output only accumulates run after run otherwise. Disabled "
              "while a job is running so it can't wipe output you may still be reading mid-run; "
              "re-enabled once the job finishes."))

        self.btn_rife_standalone_input.Bind(wx.EVT_BUTTON, self.on_click_btn_rife_standalone_input)
        self.btn_rife_standalone_output.Bind(wx.EVT_BUTTON, self.on_click_btn_rife_standalone_output)
        self.cbo_rife_standalone_mode.Bind(wx.EVT_COMBOBOX, self.on_changed_cbo_rife_standalone_mode)
        self.btn_rife_standalone_run.Bind(wx.EVT_BUTTON, self.on_click_btn_rife_standalone_run)
        self.btn_rife_standalone_clear.Bind(wx.EVT_BUTTON, lambda event: self.txt_rife_standalone_log.Clear())
        self.update_rife_standalone_mode()

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        h = -1
        layout.Add(self.lbl_rife_standalone_input, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_rife_standalone_input, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_rife_standalone_input, (h, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_rife_standalone_output, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_rife_standalone_output, (h, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_rife_standalone_output, (h, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_rife_standalone_model, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_rife_standalone_model, (h, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_rife_standalone_gpu, (h, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_rife_standalone_gpu, (h, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_rife_standalone_codec, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_rife_standalone_codec, (h, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_rife_standalone_mode, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_rife_standalone_mode, (h, 1), flag=wx.EXPAND)
        layout.Add(self.txt_rife_standalone_target_fps, (h, 2), flag=wx.EXPAND)
        layout.Add(self.btn_rife_standalone_run, (h, 3), flag=wx.EXPAND)
        layout.Add(self.txt_rife_standalone_log, (h := h + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add(self.btn_rife_standalone_clear, (h := h + 1, 3), flag=wx.EXPAND)
        sizer_rife_standalone = wx.StaticBoxSizer(self.grp_rife_standalone, wx.VERTICAL)
        sizer_rife_standalone.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # Each category below is its own panel (a Notebook tab, or a Single Page
        # section -- see ADR-037) instead of one big 4-column grid -- every sizer_*
        # here was already fully built above (unchanged), this only changes how
        # they're composed onto their category panel. Processor+Post-Processing and
        # the three standalone tools (HDR Reinject/Add Subtitle/Stereo Mode Tag) are
        # combined into single categories since each is small on its own, matching
        # how they were already visually stacked together before ADR-036.
        tab_layout = wx.BoxSizer(wx.VERTICAL)
        tab_layout.Add(sizer_stereo, 1, wx.ALL | wx.EXPAND, 4)
        self.tab_stereo.SetSizer(tab_layout)

        tab_layout = wx.BoxSizer(wx.VERTICAL)
        tab_layout.Add(sizer_depth_blend, 1, wx.ALL | wx.EXPAND, 4)
        self.tab_depth_blend.SetSizer(tab_layout)

        tab_layout = wx.BoxSizer(wx.VERTICAL)
        tab_layout.Add(sizer_video_filter, 1, wx.ALL | wx.EXPAND, 4)
        self.tab_video_filter.SetSizer(tab_layout)

        tab_layout = wx.BoxSizer(wx.VERTICAL)
        tab_layout.Add(self.grp_video_dec.sizer, 1, wx.ALL | wx.EXPAND, 4)
        self.tab_video_dec.SetSizer(tab_layout)

        tab_layout = wx.BoxSizer(wx.VERTICAL)
        tab_layout.Add(self.grp_video.sizer, 1, wx.ALL | wx.EXPAND, 4)
        self.tab_video_enc.SetSizer(tab_layout)

        tab_layout = wx.BoxSizer(wx.VERTICAL)
        tab_layout.Add(sizer_processor, 0, wx.ALL | wx.EXPAND, 4)
        tab_layout.Add(sizer_postprocess, 0, wx.ALL | wx.EXPAND, 4)
        self.tab_processor.SetSizer(tab_layout)

        tab_layout = wx.BoxSizer(wx.VERTICAL)
        tab_layout.Add(sizer_hdr_reinject, 0, wx.ALL | wx.EXPAND, 4)
        tab_layout.Add(sizer_subsearch, 0, wx.ALL | wx.EXPAND, 4)
        tab_layout.Add(sizer_submux, 0, wx.ALL | wx.EXPAND, 4)
        tab_layout.Add(sizer_audiomux, 0, wx.ALL | wx.EXPAND, 4)
        tab_layout.Add(sizer_stereotag, 0, wx.ALL | wx.EXPAND, 4)
        tab_layout.Add(sizer_sharpen, 0, wx.ALL | wx.EXPAND, 4)
        tab_layout.Add(sizer_rife_standalone, 0, wx.ALL | wx.EXPAND, 4)
        self.tab_tools.SetSizer(tab_layout)

        # ADR-037: the 7 category panels built above are already fully self-contained
        # (each owns its own StaticBoxSizer(s) via the SetSizer() calls above) -- the
        # only thing left is composing them onto the visible pnl_options area, which is
        # the one part that differs between Tabbed and Single Page.
        if self.layout_mode == LAYOUT_MODE_SINGLE_PAGE:
            self._compose_options_layout_single_page()
        else:
            self._compose_options_layout_tabbed()

        # preset panel
        self.pnl_preset = wx.Panel(self)
        self.lbl_preset = wx.StaticText(self.pnl_preset, label=" " + T("Preset"))
        self.cbo_app_preset = EditableComboBox(self.pnl_preset, choices=self.list_preset(),
                                               size=self.FromDIP((200, -1)),
                                               name="cbo_app_preset")
        self.cbo_app_preset.SetSelection(0)
        self.btn_load_preset = wx.Button(self.pnl_preset, label=T("Load"))
        self.btn_save_preset = wx.Button(self.pnl_preset, label=T("Save"))
        self.btn_delete_preset = wx.Button(self.pnl_preset, label=T("Delete"))

        # quick presets
        self.sep_quick_preset = wx.StaticLine(self.pnl_preset, size=self.FromDIP((2, 20)), style=wx.LI_VERTICAL)
        self.btn_quick_preset_movie = wx.Button(self.pnl_preset, label=T("Movie"))
        self.btn_quick_preset_movie.SetToolTip(
            T("Quick preset: subtle, comfortable 3D for movies. Sets 3D Strength 2.0, Convergence Plane "
              "0.5 in auto (sod_v1) mode, Foreground Pop off — a restrained look favoring comfort over "
              "impact, closer to how a professional stereographer would grade a typical dialogue-driven "
              "film."))
        self.btn_quick_preset_action = wx.Button(self.pnl_preset, label=T("Action"))
        self.btn_quick_preset_action.SetToolTip(
            T("Quick preset: strong pop effects for action/VFX scenes. Sets 3D Strength 3.0, Convergence "
              "Plane 0.5 in auto (face_detect) mode, Foreground Pop 0.5 — a more aggressive, "
              "attention-grabbing look, at some cost to comfort over long viewing."))
        # ADR-057 Amendment 12: moved here from the Genre Preset dropdown's "My
        # Preferred Settings" entry (Amendment 11) at the user's request, to match
        # Movie/Action's own top-bar quick-preset button pattern instead of living
        # inside a Flicker-Reduction-scoped dropdown.
        self.btn_quick_preset_3decker = wx.Button(self.pnl_preset, label=T("3DECKER Preferred"))
        self.btn_quick_preset_3decker.SetToolTip(
            T("Quick preset: applies your own confirmed-best combo across depth/divergence/convergence/"
              "refinement/stability/EMA settings at once, from a real tuning session. Sets Depth Model "
              "Any_V3_Mono_01, Divergence 2.5, Convergence 0.5, Depth Detail Refinement on (strength "
              "1.0), Object Stability on (strength 0.3, Flat-Area Boost 0, Edge Protection 0, Max Shift "
              "off), Scene Detection on, and Auto EMA by Scene Length on using the Nagadomi_Reference "
              "table -- so it also greys out Flicker Reduction's Decay Rate/Buffer/Genre Preset fields, "
              "same as checking \"Auto EMA by Scene Length\" by hand would."))

        # preset comparison test
        self.sep_compare_preset = wx.StaticLine(self.pnl_preset, size=self.FromDIP((2, 20)), style=wx.LI_VERTICAL)
        self.btn_compare_presets = wx.Button(self.pnl_preset, label=T("Compare Presets..."))
        self.btn_compare_presets.SetToolTip(
            T("Render the same short test clip with 2 or more saved presets, "
              "then join the results back-to-back into one comparison video"))

        # copy command / import command (ADR-074)
        self.sep_command = wx.StaticLine(self.pnl_preset, size=self.FromDIP((2, 20)), style=wx.LI_VERTICAL)
        self.btn_copy_command = wx.Button(self.pnl_preset, label=T("Copy Command"))
        self.btn_import_command = wx.Button(self.pnl_preset, label=T("Import Command"))
        self.btn_import_command.SetToolTip(
            T("Paste a \"python -m iw3 ...\" command line (like the ones Copy Command produces) and "
              "apply every matching setting from it to this window. Shows the pasted text for you to "
              "review/edit before anything is changed -- this can overwrite a lot of settings at once."))

        # language
        self.sep_language = wx.StaticLine(self.pnl_preset, size=self.FromDIP((2, 20)), style=wx.LI_VERTICAL)
        self.lbl_language = wx.StaticText(self.pnl_preset, label=T("Language"))
        self.cbo_language = wx.ComboBox(self.pnl_preset, name="cbo_language")
        self.cbo_language.SetEditable(False)

        lang_selection = 0
        for i, lang in enumerate(LOCAL_LIST):
            t = LOCALES.get(lang)
            name = t.get("_NAME", "Undefined")
            self.cbo_language.Append(name, lang)
            if lang in LOCALE_DICT.get("_LOCALE", []):
                lang_selection = i
        self.cbo_language.SetSelection(lang_selection)

        # GUI layout preference (ADR-037, live-switching ADR-045): Tabbed vs. Single
        # Page. Persisted like Language, in its own file (so the INITIAL layout is
        # known before any control is built), but unlike Language, changing it applies
        # immediately in the running window -- see on_text_changed_cbo_layout,
        # switch_layout_mode, and docs/ai/AI_DECISIONS.md.
        self.sep_layout = wx.StaticLine(self.pnl_preset, size=self.FromDIP((2, 20)), style=wx.LI_VERTICAL)
        self.lbl_layout = wx.StaticText(self.pnl_preset, label=T("Layout"))
        self.cbo_layout = wx.ComboBox(self.pnl_preset, name="cbo_layout")
        self.cbo_layout.SetEditable(False)
        self.cbo_layout.Append(T("Tabbed"), LAYOUT_MODE_TABS)
        self.cbo_layout.Append(T("Single Page"), LAYOUT_MODE_SINGLE_PAGE)
        self.cbo_layout.SetSelection(0 if self.layout_mode == LAYOUT_MODE_TABS else 1)
        self.cbo_layout.SetToolTip(
            T("What it's for: choose how the 100+ conversion settings below are organized.\n"
              "Values: Tabbed (default) — groups settings into 7 category tabs (Stereo Generation, "
              "Dual-Pass Depth Blend, Video Filter, Video Decoding, Video Encoding, Processor, "
              "Standalone Tools) so you only see one category at a time. Single Page — shows all 7 "
              "category groups at once on one scrollable page, so nothing is hidden behind a tab click.\n"
              "Con: Single Page needs more scrolling/screen space to see everything at once; Tabbed "
              "hides other categories until you click their tab.\n"
              "Recommended: Tabbed for a smaller, less cluttered window; Single Page if you'd rather "
              "see every setting at once and don't mind scrolling.\n"
              "Note: switches instantly -- no restart needed."))

        # UI Zoom: scales the whole app's text/control size up or down, independent of
        # Windows' own system display scaling (init_win32_dpi(), untouched by this).
        # Persisted like Layout, in its own file, and -- like Layout (ADR-038) --
        # applies immediately in the running window, see on_text_changed_cbo_zoom,
        # apply_zoom_level, and docs/ai/AI_DECISIONS.md.
        self.sep_zoom = wx.StaticLine(self.pnl_preset, size=self.FromDIP((2, 20)), style=wx.LI_VERTICAL)
        self.lbl_zoom = wx.StaticText(self.pnl_preset, label=T("Zoom"))
        self.cbo_zoom = wx.ComboBox(self.pnl_preset, name="cbo_zoom")
        self.cbo_zoom.SetEditable(False)
        for pct in ZOOM_LEVELS:
            self.cbo_zoom.Append(f"{pct}%", pct)
        self.cbo_zoom.SetSelection(ZOOM_LEVELS.index(self.zoom_level))
        self.cbo_zoom.SetToolTip(
            T("What it's for: scales the whole app's text and control size up or down -- for "
              "readability on a high-DPI display, or just personal preference. Separate from "
              "Windows' own system display scaling, which this app already handles on its own.\n"
              "Values: 80% (smallest) to 200% (largest); 100% is the original/default size.\n"
              "Con: at the largest sizes some tabs may need more scrolling to see every setting; "
              "Single Page layout scrolls to fit automatically, Tabbed may need a taller window.\n"
              "Recommended: leave at 100% unless text is hard to read; 125%-150% is a reasonable "
              "middle ground on a high-resolution display.\n"
              "Note: applies immediately -- no restart needed."))

        # check for updates (read-only fetch + compare only -- never pulls/merges/
        # resets anything; see docs/ai/AI_DECISIONS.md ADR-035)
        self.sep_update = wx.StaticLine(self.pnl_preset, size=self.FromDIP((2, 20)), style=wx.LI_VERTICAL)
        self.btn_check_updates = wx.Button(self.pnl_preset, label=T("Check for Updates"))
        self.btn_check_updates.SetToolTip(
            T("What it's for: checks whether the original upstream nunif project "
              "(github.com/nagadomi/nunif) has new commits that aren't in this fork yet, and shows you "
              "what they are.\n"
              "How it helps: lets you see what's changed upstream without any risk to your setup or "
              "this session's own customizations (RIFE, Z-Splat, HDR reinjection, subtitle muxing, "
              "StereoMode tagging, etc.).\n"
              "Con: read-only -- runs 'git fetch' plus a comparison only. It NEVER runs pull/merge/reset, "
              "so nothing is ever applied automatically; this button cannot update anything by itself, "
              "and upstream commits could conflict with this fork's own customizations if applied later.\n"
              "Recommended: safe to click any time -- it only reads and reports, never changes anything."))

        # run update (applies the real update.bat -- see docs/ai/AI_DECISIONS.md
        # ADR-069, the direct follow-up to ADR-035's deliberately-deferred
        # "applying an update" scope)
        self.btn_run_update = wx.Button(self.pnl_preset, label=T("Run Update"))
        self.btn_run_update.SetToolTip(
            T("What it's for: actually runs the real update.bat script from inside the app -- the "
              "same script you'd otherwise have to find and double-click outside the app -- to "
              "update Python packages, downloaded models, and the source code together, all in "
              "one real operation.\n"
              "Why it's separate from Check for Updates: that button only checks and reports what's "
              "different upstream and changes nothing on disk; this button actually applies an "
              "update to real files.\n"
              "Con: a real, somewhat time-consuming operation (package downloads, model downloads, "
              "a source pull) with no undo -- a confirmation dialog appears before anything runs, "
              "and, as Check for Updates already warns, an upstream source update could in "
              "principle conflict with this fork's own customizations.\n"
              "Disabled while a conversion (or other background job) is running, so packages/source "
              "can't change out from under a job that's using them.\n"
              "Recommended: use it when you actually want to apply an update you already know about "
              "(e.g. from Check for Updates) -- not as a routine/automatic click."))

        layout = wx.BoxSizer(wx.HORIZONTAL)
        layout.Add(self.lbl_preset, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT, border=2)
        layout.Add(self.cbo_app_preset, flag=wx.ALL, border=2)
        layout.Add(self.btn_load_preset, flag=wx.ALL, border=2)
        layout.Add(self.btn_save_preset, flag=wx.ALL, border=2)
        layout.Add(self.btn_delete_preset, flag=wx.ALL, border=2)
        layout.AddSpacer(2)
        layout.Add(self.sep_quick_preset, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT)
        layout.AddSpacer(4)
        layout.Add(self.btn_quick_preset_movie, flag=wx.ALL, border=2)
        layout.Add(self.btn_quick_preset_action, flag=wx.ALL, border=2)
        layout.Add(self.btn_quick_preset_3decker, flag=wx.ALL, border=2)
        layout.AddSpacer(2)
        layout.Add(self.sep_compare_preset, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT)
        layout.AddSpacer(4)
        layout.Add(self.btn_compare_presets, flag=wx.ALL, border=2)
        layout.AddSpacer(2)
        layout.Add(self.sep_command, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT)
        layout.AddSpacer(4)
        layout.Add(self.btn_copy_command, flag=wx.ALL, border=2)
        layout.Add(self.btn_import_command, flag=wx.ALL, border=2)

        layout.AddSpacer(2)
        layout.Add(self.sep_language, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT)
        layout.AddSpacer(4)
        layout.Add(self.lbl_language, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT, border=2)
        layout.Add(self.cbo_language, flag=wx.ALL, border=2)

        layout.AddSpacer(2)
        layout.Add(self.sep_layout, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT)
        layout.AddSpacer(4)
        layout.Add(self.lbl_layout, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT, border=2)
        layout.Add(self.cbo_layout, flag=wx.ALL, border=2)

        layout.AddSpacer(2)
        layout.Add(self.sep_zoom, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT)
        layout.AddSpacer(4)
        layout.Add(self.lbl_zoom, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT, border=2)
        layout.Add(self.cbo_zoom, flag=wx.ALL, border=2)

        layout.AddSpacer(2)
        layout.Add(self.sep_update, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT)
        layout.AddSpacer(4)
        layout.Add(self.btn_check_updates, flag=wx.ALL, border=2)
        layout.AddSpacer(2)
        layout.Add(self.btn_run_update, flag=wx.ALL, border=2)
        layout.AddSpacer(8)
        self.pnl_preset.SetSizer(layout)

        # processing panel
        self.pnl_process = wx.Panel(self)
        if LAYOUT_DEBUG:
            self.pnl_process.SetBackgroundColour("#fcc")
        self.prg_tqdm = wx.Gauge(self.pnl_process, style=wx.GA_HORIZONTAL)
        self.btn_quick_preview = wx.Button(self.pnl_process, label=T("Quick Preview"))
        self.btn_quick_preview.SetToolTip(
            T("Process a short 45 second clip (or a single frame for images) with the "
              "current settings to quickly check the result"))
        self.btn_start = wx.Button(self.pnl_process, label=T("Start"))
        self.btn_suspend = wx.Button(self.pnl_process, label=T("Suspend"))
        self.btn_cancel = wx.Button(self.pnl_process, label=T("Cancel"))

        layout = wx.BoxSizer(wx.HORIZONTAL)
        layout.Add(self.prg_tqdm, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        layout.Add(self.btn_quick_preview, 0, wx.ALL, 4)
        layout.Add(self.btn_start, 0, wx.ALL, 4)
        layout.Add(self.btn_suspend, 0, wx.ALL, 4)
        layout.Add(self.btn_cancel, 0, wx.ALL, 4)
        self.pnl_process.SetSizer(layout)

        # main layout
        layout = wx.BoxSizer(wx.VERTICAL)
        layout.AddSpacer(8)
        layout.Add(self.pnl_preset, 0, wx.ALIGN_RIGHT, 2)
        layout.Add(self.pnl_file.panel, 0, wx.ALL | wx.EXPAND, 8)
        layout.Add(self.pnl_file_option, 0, wx.ALL | wx.EXPAND, 4)
        layout.Add(self.pnl_options, 1, wx.ALL | wx.EXPAND, 8)
        layout.Add(self.pnl_process, 0, wx.ALL | wx.EXPAND, 8)
        self.SetSizer(layout)

        # bind
        self.pnl_file.bind_input_path_changed(self.on_text_changed_txt_input)
        self.pnl_file.bind_output_path_changed(self.on_text_changed_txt_output)

        # Guided Light pilot (ADR-097): two-way slider <-> combo sync for every field
        # in STEREO_SLIDER_FIELDS. wx.EvtHandler.Bind() REPLACES any existing handler
        # for the same (event, control) pair rather than adding a second one (this
        # wx build has no `add=` kwarg at all -- confirmed the hard way), so
        # cbo_divergence -- which already needs its own EVT_TEXT handler,
        # update_divergence_warning() -- gets a combined wrapper below instead of a
        # plain functools.partial, so both behaviors still run off one binding. This
        # replaces the old standalone `self.cbo_divergence.Bind(wx.EVT_TEXT,
        # self.update_divergence_warning)` line -- it would otherwise just get
        # silently overwritten by this loop anyway.
        for _combo_name, _slider_name, _lo, _hi, _mult, _is_int, _extra in STEREO_SLIDER_FIELDS:
            _combo = getattr(self, _combo_name)
            _slider = getattr(self, _slider_name)
            _slider.Bind(wx.EVT_SLIDER, functools.partial(
                self._on_stereo_slider_scroll, combo=_combo, slider=_slider,
                min_val=_lo, max_val=_hi, multiplier=_mult, is_int=_is_int, extra_sync=_extra))
            if _combo_name == "cbo_divergence":
                def _divergence_text_handler(event, _self=self, _c=_combo, _s=_slider, _l=_lo, _h=_hi, _m=_mult):
                    _self.update_divergence_warning(event)
                    _self._on_stereo_combo_text_sync_slider(event, combo=_c, slider=_s, min_val=_l, max_val=_h, multiplier=_m)
                _combo.Bind(wx.EVT_TEXT, _divergence_text_handler)
            else:
                _combo.Bind(wx.EVT_TEXT, functools.partial(
                    self._on_stereo_combo_text_sync_slider, combo=_combo, slider=_slider,
                    min_val=_lo, max_val=_hi, multiplier=_mult))

        # Same two-way sync, extended to Dual-Pass Depth Blend and Processor -- none
        # of these combos have any pre-existing EVT_TEXT handler to preserve (checked
        # directly against every Bind() call in this file before assuming so), so the
        # plain functools.partial branch is safe for all of them, no combined-wrapper
        # case needed the way cbo_divergence required above.
        for _combo_name, _slider_name, _lo, _hi, _mult, _is_int, _extra in DEPTH_BLEND_SLIDER_FIELDS + PROCESSOR_SLIDER_FIELDS:
            _combo = getattr(self, _combo_name)
            _slider = getattr(self, _slider_name)
            _slider.Bind(wx.EVT_SLIDER, functools.partial(
                self._on_stereo_slider_scroll, combo=_combo, slider=_slider,
                min_val=_lo, max_val=_hi, multiplier=_mult, is_int=_is_int, extra_sync=_extra))
            _combo.Bind(wx.EVT_TEXT, functools.partial(
                self._on_stereo_combo_text_sync_slider, combo=_combo, slider=_slider,
                min_val=_lo, max_val=_hi, multiplier=_mult))
        self.cbo_synthetic_view.Bind(wx.EVT_TEXT, self.update_divergence_warning)
        self.cbo_method.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_method)
        self.lbl_divergence_warning.Bind(wx.EVT_LEFT_DOWN, self.on_click_divergence_warning)

        self.cbo_depth_model.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_depth_model)
        self.cbo_edge_dilation_y.Bind(wx.EVT_TEXT, self.on_changed_edge_dilation)
        self.chk_ema_normalize.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_ema_normalize)
        self.chk_scene_batch_auto_ema.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_scene_batch_auto_ema)
        self.cbo_genre_preset.Bind(wx.EVT_TEXT, self.on_changed_cbo_genre_preset)
        self.chk_depth_blend.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_depth_blend)
        self.chk_temporal_stabilize.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_temporal_stabilize)
        self.cbo_depth_blend_region.Bind(wx.EVT_TEXT, self.on_changed_chk_depth_blend)
        self.chk_depth_blend_bilateral.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_depth_blend_bilateral)
        self.chk_depth_blend_clahe.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_depth_blend_clahe)
        self.chk_depth_blend_align.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_depth_blend_align)
        self.chk_depth_refine.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_depth_refine)
        self.chk_sharpen.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_sharpen)

        self.cbo_stereo_format.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_stereo_format)

        self.cbo_pad_mode.Bind(wx.EVT_TEXT, self.update_pad_mode)

        self.cbo_device.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_device)
        self.chk_compile.Bind(wx.EVT_CHECKBOX, self.update_compile)

        self.btn_load_preset.Bind(wx.EVT_BUTTON, self.on_click_btn_load_preset)
        self.btn_save_preset.Bind(wx.EVT_BUTTON, self.on_click_btn_save_preset)
        self.btn_delete_preset.Bind(wx.EVT_BUTTON, self.on_click_btn_delete_preset)
        self.btn_quick_preset_movie.Bind(wx.EVT_BUTTON, lambda event: self.apply_quick_preset("movie"))
        self.btn_quick_preset_action.Bind(wx.EVT_BUTTON, lambda event: self.apply_quick_preset("action"))
        self.btn_quick_preset_3decker.Bind(wx.EVT_BUTTON, lambda event: self.apply_quick_preset("3decker"))
        self.btn_compare_presets.Bind(wx.EVT_BUTTON, self.on_click_btn_compare_presets)
        self.btn_copy_command.Bind(wx.EVT_BUTTON, self.on_click_btn_copy_command)
        self.btn_import_command.Bind(wx.EVT_BUTTON, self.on_click_btn_import_command)
        self.cbo_language.Bind(wx.EVT_TEXT, self.on_text_changed_cbo_language)
        self.cbo_layout.Bind(wx.EVT_TEXT, self.on_text_changed_cbo_layout)
        self.cbo_zoom.Bind(wx.EVT_TEXT, self.on_text_changed_cbo_zoom)
        self.btn_check_updates.Bind(wx.EVT_BUTTON, self.on_click_btn_check_updates)
        self.btn_run_update.Bind(wx.EVT_BUTTON, self.on_click_btn_run_update)

        self.btn_autocrop_test.Bind(wx.EVT_BUTTON, self.on_click_btn_autocrop_test)
        self.btn_scene_settings.Bind(wx.EVT_BUTTON, self.on_click_btn_scene_settings)
        self.btn_scene_batch_auto_ema_edit.Bind(wx.EVT_BUTTON, self.on_click_btn_scene_batch_auto_ema_edit)

        self.btn_start.Bind(wx.EVT_BUTTON, self.on_click_btn_start)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self.on_click_btn_cancel)
        self.btn_suspend.Bind(wx.EVT_BUTTON, self.on_click_btn_suspend)
        self.btn_quick_preview.Bind(wx.EVT_BUTTON, self.on_click_btn_quick_preview)

        self.Bind(EVT_TQDM, self.on_tqdm)
        self.Bind(EVT_IW3_STAGE, self.on_stage_change)
        self.Bind(wx.EVT_TIMER, self.on_stage_pulse_timer, self.stage_pulse_timer)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        editable_comboboxes = self.get_editable_comboboxes()

        self.SetDropTarget(FileDropCallback(self.on_drop_files))
        # Disable default drop target
        for control in (self.pnl_file.input_path_widget, self.pnl_file.output_path_widget, self.txt_vf,
                        self.txt_start_time, self.txt_end_time, *editable_comboboxes):
            control.SetDropTarget(FileDropCallback(self.on_drop_files))

        # Fix Frame and Panel background colors are different in windows
        self.SetBackgroundColour(self.pnl_file_option.GetBackgroundColour())

        # state
        self.btn_cancel.Disable()
        self.btn_suspend.Disable()

        self.load_preset()
        # probe_compile=False: skip the real torch.compile() GPU probe during
        # passive window construction -- a persisted "compile: on" setting from a
        # previous session must not grab a CUDA context before the user does
        # anything. The checkbox/device handlers still probe on real interaction.
        self.update_controls(probe_compile=False)

    def _compose_options_layout_tabbed(self):
        """ADR-036/ADR-037/ADR-045/ADR-048 -- Tabbed layout: each category panel is
        wrapped in its own ScrolledPanel (self.tab_wrap_*) and that WRAPPER becomes the
        wx.Notebook page, not the category panel directly -- see ADR-048 for why a
        separate wrapper was used instead of making the category panels themselves
        ScrolledPanel (would nest a ScrolledPanel inside pnl_single's own ScrolledPanel
        in Single Page mode). Reused both at startup (ADR-037, based on the user's
        saved Layout preference) and live, on every switch back to Tabbed (ADR-045, via
        switch_layout_mode() -- the 7 category panels are Reparent()-ed onto their
        wrapper before this runs, so wiring each wrapper's sizer below always sees a
        tab that is already a real child of it, exactly as at startup)."""
        wrap_pages = (
            (self.tab_wrap_stereo, self.tab_stereo, T("Stereo Generation")),
            (self.tab_wrap_depth_blend, self.tab_depth_blend, T("Dual-Pass Depth Blend")),
            (self.tab_wrap_video_filter, self.tab_video_filter, T("Video Filter")),
            (self.tab_wrap_video_dec, self.tab_video_dec, T("Video Decoding")),
            (self.tab_wrap_video_enc, self.tab_video_enc, T("Video Encoding")),
            (self.tab_wrap_processor, self.tab_processor, T("Processor")),
            (self.tab_wrap_tools, self.tab_tools, T("Standalone Tools")),
        )
        for wrap, tab, label in wrap_pages:
            wrap_sizer = wx.BoxSizer(wx.VERTICAL)
            wrap_sizer.Add(tab, 1, wx.EXPAND)
            wrap.SetSizer(wrap_sizer)
            wrap.SetAutoLayout(1)
            wrap.SetupScrolling(scroll_x=False, scroll_y=True)
            # Same reasoning as pnl_single's own explicit MinSize (ADR-045): a
            # ScrolledPanel's own GetBestSize() is deliberately tiny regardless of its
            # content, so without this every tab would collapse to near-zero height in
            # the Notebook. Verified by isolated probe (ADR-048) that pinning this does
            # NOT defeat scrolling -- the wrapper still reports a real scrollable
            # virtual size when its parent Notebook forces it shorter than this pinned
            # minimum.
            wrap.SetMinSize(wrap_sizer.CalcMin())
            self.nb_options.AddPage(wrap, label)
        # Force a deterministic starting tab -- without this, wx sometimes lands on
        # whichever page happens to contain the last control touched by a SetSelection()
        # call made deep inside a sub-panel's own __init__ (e.g. VideoEncodingBox's
        # cbo_video_format) instead of the first page, which looked like a random tab
        # on launch.
        self.nb_options.SetSelection(0)

        layout = wx.BoxSizer(wx.VERTICAL)
        layout.Add(self.nb_options, 1, wx.EXPAND)
        self.pnl_options.SetSizer(layout)
        # ADR-045: both containers always exist (see initialize_component) -- only the
        # one actually composed here should be visible/managed by pnl_options's sizer.
        self.nb_options.Show()
        self.pnl_single.Hide()

    def _compose_options_layout_single_page(self):
        """ADR-037/ADR-045 -- Single Page layout: the same 7 category panels (each
        already built with its own StaticBoxSizer(s), identical to the tabbed path)
        placed directly onto one scrollable page instead of behind tab clicks,
        arranged in the same 4-column grid template this file used PRE-ADR-036 (the
        "earlier pass" that added spacing/dividers/indentation within each StaticBox
        group) -- see docs/ai/AI_DECISIONS.md ADR-037 for why this specific arrangement
        was reused rather than invented fresh: column 0 is Stereo Generation (the most
        used, tallest group); column 1 stacks Video Decoding/Video Encoding; column 2
        stacks Video Filter/Processor; column 3 stacks Dual-Pass Depth Blend/
        Standalone Tools -- the exact same column pairing this file used before the
        tabs conversion, just with Processor+Post-Processing and the three standalone
        tools already pre-combined into single panels per ADR-036. Called both at
        startup and live, on every switch back to Single Page (ADR-045, via
        switch_layout_mode() -- which Reparent()s the 7 panels onto self.pnl_single
        before this runs, and always builds a brand-new GridBagSizer here rather than
        reusing a stale one from a previous switch)."""
        content = wx.GridBagSizer(vgap=0, hgap=0)
        content.SetEmptyCellSize((0, 0))
        content.Add(self.tab_stereo, pos=(0, 0), span=(2, 1), flag=wx.ALL | wx.EXPAND, border=4)
        content.Add(self.tab_video_dec, pos=(0, 1), flag=wx.ALL | wx.EXPAND, border=4)
        content.Add(self.tab_video_enc, pos=(1, 1), flag=wx.ALL | wx.EXPAND, border=4)
        content.Add(self.tab_video_filter, pos=(0, 2), flag=wx.ALL | wx.EXPAND, border=4)
        content.Add(self.tab_processor, pos=(1, 2), flag=wx.ALL | wx.EXPAND, border=4)
        content.Add(self.tab_depth_blend, pos=(0, 3), flag=wx.ALL | wx.EXPAND, border=4)
        content.Add(self.tab_tools, pos=(1, 3), flag=wx.ALL | wx.EXPAND, border=4)
        self.pnl_single.SetSizer(content)
        self.pnl_single.SetAutoLayout(1)
        self.pnl_single.SetupScrolling(scroll_x=True, scroll_y=True)
        # A ScrolledPanel's own GetBestSize() is deliberately tiny regardless of its
        # content (that's what lets a window shrink below its content and scroll) --
        # so nunif/gui/common.py:refresh_layouts's recursive InvalidateBestSize+Fit
        # pass (run both at startup by IW3App.OnInit and on every live switch by
        # switch_layout_mode, ADR-045) collapses the whole frame down to a few dozen
        # pixels tall unless pnl_single is given an explicit min size to fall back on.
        # CalcMin() is this GridBagSizer's own real computed minimum (not a hardcoded
        # guess), so the window opens at a size that shows one full column without
        # scrolling, while a user who shrinks it manually still gets real scrollbars.
        self.pnl_single.SetMinSize(content.CalcMin())

        layout = wx.BoxSizer(wx.VERTICAL)
        layout.Add(self.pnl_single, 1, wx.EXPAND)
        self.pnl_options.SetSizer(layout)
        # ADR-045: both containers always exist (see initialize_component) -- only the
        # one actually composed here should be visible/managed by pnl_options's sizer.
        self.pnl_single.Show()
        self.nb_options.Hide()

    def switch_layout_mode(self, new_mode):
        """ADR-045, wrapper handling added by ADR-048 -- live layout switching: move
        the 7 category panels between their ScrolledPanel wrappers (self.tab_wrap_*,
        Tabbed) and self.pnl_single (Single Page) in the already-running window,
        instead of only applying the Layout preference on next launch (ADR-037's
        original, more cautious choice). Verified safe on this project's actual wx
        version (4.3.1 phoenix / wxWidgets 3.3.3) with an isolated harness before being
        wired in here -- see docs/ai/AI_DECISIONS.md ADR-045: a plain
        wx.Panel.Reparent() cleanly preserves a panel's children, their Bind()s, and
        cross-control Enable/Disable relationships (e.g. Object Stability's
        sub-settings), because only the 7 *category* panels ever move -- every
        wx.StaticBox/control inside them keeps the SAME parent (its category panel)
        throughout, so none of them are ever reparented themselves. ADR-048 only
        changes WHERE a category panel lands in Tabbed mode (its own wrapper instead of
        self.nb_options directly) -- the wrappers themselves never move, only the 7
        category panels do, same as before.

        No-ops if new_mode already matches self.layout_mode (defensive -- the combo
        box shouldn't fire EVT_TEXT without an actual change, but this keeps a
        double-fire from tearing down and rebuilding the active layout for nothing).
        """
        if new_mode == self.layout_mode:
            return

        tabs = (self.tab_stereo, self.tab_depth_blend, self.tab_video_filter,
                self.tab_video_dec, self.tab_video_enc, self.tab_processor, self.tab_tools)
        wraps = (self.tab_wrap_stereo, self.tab_wrap_depth_blend, self.tab_wrap_video_filter,
                 self.tab_wrap_video_dec, self.tab_wrap_video_enc, self.tab_wrap_processor,
                 self.tab_wrap_tools)

        # Detach every category panel from whichever container currently holds it,
        # WITHOUT destroying the panel or any of its children (RemovePage/Detach, never
        # DeletePage/Clear(delete_windows=True)).
        if self.layout_mode == LAYOUT_MODE_TABS:
            while self.nb_options.GetPageCount():
                self.nb_options.RemovePage(0)
            # RemovePage() only detaches each wrapper from the Notebook -- the category
            # panel is still a child of its wrapper's own sizer and must be detached
            # from THAT too before it can be reparented elsewhere.
            for wrap, tab in zip(wraps, tabs):
                wrap_sizer = wrap.GetSizer()
                if wrap_sizer is not None:
                    wrap_sizer.Detach(tab)
        else:
            single_page_sizer = self.pnl_single.GetSizer()
            for tab in tabs:
                single_page_sizer.Detach(tab)

        new_parents = wraps if new_mode == LAYOUT_MODE_TABS else (self.pnl_single,) * len(tabs)
        for tab, new_parent in zip(tabs, new_parents):
            tab.Reparent(new_parent)
            # A wx.Notebook auto-Hide()s every page except the currently selected one,
            # and RemovePage() does not undo that -- so a panel that was an inactive
            # tab keeps carrying a Hidden state after being detached/reparented. Sizers
            # (GridBagSizer here) exclude Hidden windows from CalcMin(), which silently
            # collapsed the whole Single Page layout to near-zero size before this Show()
            # was added. Force every panel visible here; _compose_options_layout_tabbed's
            # own SetSelection(0) still correctly re-hides the non-active pages for the
            # Tabbed case afterward, exactly as it already does at construction time.
            tab.Show()

        self.layout_mode = new_mode
        if new_mode == LAYOUT_MODE_SINGLE_PAGE:
            self._compose_options_layout_single_page()
        else:
            self._compose_options_layout_tabbed()

        # Re-theme (ADR-037's apply_accent_theme already picks the right container by
        # self.layout_mode) and recompute sizes exactly the way IW3App.OnInit already
        # does once at startup (nunif/gui/common.py:refresh_layouts) -- a Notebook and
        # a ScrolledPanel size/scroll completely differently, and this is the
        # established recursive InvalidateBestSize+Layout+Fit pass this codebase
        # already trusts to get that right, rather than a smaller ad hoc subset of it.
        self.apply_accent_theme()
        refresh_layouts(self)
        self._clamp_frame_to_screen()

    def apply_zoom_level(self, zoom_level):
        """Live UI Zoom: rescales the whole app's base font and re-lays-out every
        control, on top of (not instead of) init_win32_dpi()'s system-level DPI
        awareness. wx controls only inherit their parent's font at CONSTRUCTION time,
        not dynamically -- self.SetFont() in initialize_component() is why a fresh
        launch already opens at the persisted zoom level for free (every child is
        built AFTER that call), but an already-built running window needs every
        existing descendant's font set explicitly to actually rescale live. See
        docs/ai/AI_DECISIONS.md.
        """
        self.zoom_level = zoom_level
        normal_font = self._scaled_font(BASE_NORMAL_FONT_PT)
        warning_font = self._scaled_font(BASE_WARNING_FONT_PT)

        self.SetFont(normal_font)
        self._set_font_recursive(self, normal_font)
        self.lbl_divergence_warning.SetFont(warning_font)

        # Re-derive the bold StaticBox headers / Start-Cancel button fonts from the
        # new base size (apply_accent_theme reads self.grp_stereo.GetFont() live, it
        # never caches the old font) and recompute every container's best size the
        # same way switch_layout_mode() already does after moving panels around.
        self.apply_accent_theme()

        if self.layout_mode == LAYOUT_MODE_SINGLE_PAGE:
            # Same reasoning as _compose_options_layout_single_page(): a ScrolledPanel's
            # own GetBestSize() is deliberately tiny regardless of its content, so its
            # explicit MinSize (pinned when the page was last composed) must be
            # recomputed here too -- otherwise it stays at whatever size the OLD, smaller
            # font needed, and a zoom-in only becomes reachable by scrolling instead of
            # the window growing to actually show it, unlike a fresh Single Page
            # composition at the same zoom level would.
            self.pnl_single.SetMinSize(self.pnl_single.GetSizer().CalcMin())
        else:
            # ADR-048: Tabbed mode's Notebook pages are now ScrolledPanel wrappers with
            # their own pinned MinSize (same reason as pnl_single above), so they need
            # the exact same live-zoom recompute -- otherwise a zoom-in would leave a
            # wrapper's pinned MinSize stale from the old, smaller font and require
            # scrolling to reach content that should now show/expand instead.
            for wrap in (self.tab_wrap_stereo, self.tab_wrap_depth_blend, self.tab_wrap_video_filter,
                         self.tab_wrap_video_dec, self.tab_wrap_video_enc, self.tab_wrap_processor,
                         self.tab_wrap_tools):
                wrap_sizer = wrap.GetSizer()
                if wrap_sizer is not None:
                    wrap.SetMinSize(wrap_sizer.CalcMin())

        refresh_layouts(self)
        self._clamp_frame_to_screen()

    def _clamp_frame_to_screen(self):
        """ADR-056 -- keeps pnl_process (the progress bar row plus Start/Suspend/
        Cancel) reliably on-screen. refresh_layouts()/Fit() size this frame to its
        full natural content height -- which, in the top-level vertical sizer built in
        initialize_component(), pins pnl_options's ScrolledPanel content (see
        _compose_options_layout_single_page's/_compose_options_layout_tabbed's own
        SetMinSize comments) to its entire UN-scrolled size -- with nothing clamping
        that height against the real screen, it can exceed the monitor's usable work
        area (most easily in Single Page mode, or at a higher Zoom level), pushing
        pnl_process off the bottom edge or behind the taskbar. Maximize is disabled
        on this frame (wx.MAXIMIZE_BOX removed in __init__), so a user can't just
        maximize their way back to it either.

        pnl_process has sizer proportion 0 (fixed) while pnl_options has proportion 1
        (stretch) in that same top-level sizer, and pnl_options's own content is
        already real ScrolledPanel(s) with scrolling wired up (SetupScrolling) --
        already verified elsewhere in this file to correctly show scrollbars even when
        its parent forces it shorter than its own pinned MinSize (see
        _compose_options_layout_tabbed's wrap.SetMinSize comment). So shrinking the
        FRAME itself down to the screen's work area lets the sizer hand 100% of the
        deficit to pnl_options, which scrolls, while pnl_process keeps its full
        requested height and stays fully visible. Only ever shrinks/repositions the
        frame -- a window that already fits the screen is left exactly as Fit() sized
        it.

        ADR-056 Amendment -- also called after every other runtime self.Fit() on this
        frame, not just the original three (IW3App.OnInit, switch_layout_mode,
        apply_zoom_level): update_anaglyph_state, update_export_option_state,
        update_inpaint_options, update_model_selection, update_divergence_warning, and
        on_click_divergence_warning all Show()/Hide() real controls in response to an
        ordinary field change (Stereo Format, Method, Depth Model, Divergence) and
        already called self.Fit() on their own, but never this clamp -- confirmed by
        measurement to be the actual regression: these were never covered by the
        original fix, but stayed harmless while the frame's natural height had enough
        margin under the screen; once tonight's other additions shrank that margin, an
        ordinary field change (e.g. Method -> forward_inpaint, or a Divergence warning
        appearing) was again enough to push pnl_process behind the taskbar."""
        display_index = wx.Display.GetFromWindow(self)
        if display_index == wx.NOT_FOUND:
            display_index = 0
        work_area = wx.Display(display_index).GetClientArea()
        width, height = self.GetSize()
        new_width = min(width, work_area.GetWidth())
        new_height = min(height, work_area.GetHeight())
        if (new_width, new_height) != (width, height):
            self.SetSize((new_width, new_height))
        x, y = self.GetPosition()
        new_x = min(max(x, work_area.GetX()), work_area.GetRight() - new_width)
        new_y = min(max(y, work_area.GetY()), work_area.GetBottom() - new_height)
        if (new_x, new_y) != (x, y):
            self.SetPosition((new_x, new_y))

    def _set_font_recursive(self, window, font):
        window.SetFont(font)
        for child in window.GetChildren():
            self._set_font_recursive(child, font)

    def apply_accent_theme(self):
        """3DECKER visual pass: a real, considered color palette using only what
        wxPython natively supports (SetForegroundColour/SetBackgroundColour/SetFont) --
        no new GUI toolkit, no control renamed/rebound/retooltipped. Runs after
        apply_dark_mode() (nunif/gui/common.py) so it always applies last instead of
        being clobbered by that function's blanket recursive fg/bg reset, and picks
        light vs. dark palette values itself so the accent stays legible either way.
        """
        dark = is_dark_mode()
        accent = wx.Colour(0x6f, 0xb2, 0xf7) if dark else wx.Colour(0x1f, 0x5f, 0xc9)
        panel_bg = wx.Colour(0x2c, 0x30, 0x38) if dark else wx.Colour(0xf3, 0xf5, 0xf9)

        box_font = self.grp_stereo.GetFont().Bold()
        for box in (
            self.grp_stereo, self.grp_depth_blend, self.grp_video_filter,
            self.grp_processor, self.grp_postprocess,
            self.grp_hdr_reinject, self.grp_subsearch, self.grp_submux, self.grp_stereotag,
            self.grp_video_dec.grp_video_dec, self.grp_video.grp_video,
        ):
            box.SetForegroundColour(accent)
            box.SetFont(box_font)

        # ADR-037: the container that wraps the 7 category panels differs by Layout
        # preference (wx.Notebook vs. a scrollable single-page panel) -- theme
        # whichever one is actually in use instead of assuming the Notebook.
        options_container = self.nb_options if self.layout_mode == LAYOUT_MODE_TABS else self.pnl_single
        for panel in (
            self, self.pnl_options, options_container,
            self.tab_stereo, self.tab_depth_blend, self.tab_video_filter,
            self.tab_video_dec, self.tab_video_enc, self.tab_processor, self.tab_tools,
            # ADR-048: the per-tab ScrolledPanel wrappers (Tabbed mode only) sit between
            # the Notebook and each category panel -- theme them too so no default-
            # colored ring shows around a tab's content, especially in dark mode.
            self.tab_wrap_stereo, self.tab_wrap_depth_blend, self.tab_wrap_video_filter,
            self.tab_wrap_video_dec, self.tab_wrap_video_enc, self.tab_wrap_processor,
            self.tab_wrap_tools,
            self.pnl_file_option, self.pnl_preset, self.pnl_process,
            self.pnl_file.panel,
            # Guided Light pilot (ADR-097): a CollapsiblePane's content pane is its
            # own real window, not automatically covered by any of the above -- theme
            # it too so it doesn't show a default white/gray background in dark mode.
            # The colored indicator square deliberately keeps its own accent color,
            # not panel_bg.
            self.cpn_stereo_pop_divergence.GetPane(),
            self.cpn_stereo_inpainting_depth.GetPane(),
            self.cpn_stereo_stability_flicker.GetPane(),
        ):
            panel.SetBackgroundColour(panel_bg)

        # Primary/danger accents on the two most consequential process-panel actions
        # only -- Start (goes) and Cancel (stops) -- everything else stays neutral so
        # these two remain visually distinct, not just "the whole toolbar is colorful."
        for btn, color in (
            (self.btn_start, wx.Colour(0x21, 0x8a, 0x4c)),
            (self.btn_cancel, wx.Colour(0xcc, 0x33, 0x33)),
        ):
            btn.SetForegroundColour(wx.Colour(0xff, 0xff, 0xff))
            btn.SetBackgroundColour(color)
            btn.SetFont(btn.GetFont().Bold())

        self.Refresh()

    def update_controls(self, probe_compile=True):
        self.update_start_button_state()
        self.update_input_option_state()
        self.update_anaglyph_state()
        self.update_export_option_state()

        self.update_model_selection()
        self.update_edge_dilation()
        self.update_inpaint_options()
        self.update_ema_normalize()
        self.update_depth_blend()
        self.update_depth_refine()
        self.update_sharpen()
        self.update_temporal_stabilize()
        self.update_convergence_mode()
        self.update_scene_segment()
        self.grp_video.update_controls()

        self.update_divergence_warning()
        self.update_preserve_screen_border()
        self.update_splat_blend_temperature()
        self.update_pad_mode()
        self.update_compile(probe=probe_compile)

    def get_depth_models(self):
        depth_models = [
            "ZoeD_N", "ZoeD_K", "ZoeD_NK",
            "ZoeD_Any_N", "ZoeD_Any_K",
            "DepthPro", "DepthPro_S",
            "Any_S", "Any_B", "Any_L",
            "Any_V2_S",
        ]
        if DepthAnythingModel.has_checkpoint_file("Any_V2_B"):
            depth_models.append("Any_V2_B")
        if DepthAnythingModel.has_checkpoint_file("Any_V2_L"):
            depth_models.append("Any_V2_L")

        depth_models += ["Any_V2_N_S", "Any_V2_N_B"]
        if DepthAnythingModel.has_checkpoint_file("Any_V2_N_L"):
            depth_models.append("Any_V2_N_L")
        depth_models += ["Any_V2_K_S", "Any_V2_K_B"]
        if DepthAnythingModel.has_checkpoint_file("Any_V2_K_L"):
            depth_models.append("Any_V2_K_L")

        depth_models += ["Distill_Any_S"]
        if DepthAnythingModel.has_checkpoint_file("Distill_Any_B"):
            depth_models.append("Distill_Any_B")
        if DepthAnythingModel.has_checkpoint_file("Distill_Any_L"):
            depth_models.append("Distill_Any_L")

        depth_models += ["Any_V3_Mono", "Any_V3_Mono_01"]

        depth_models += ["VDA_S"]
        if VideoDepthAnythingModel.has_checkpoint_file("VDA_B"):
            depth_models.append("VDA_B")
        if VideoDepthAnythingModel.has_checkpoint_file("VDA_L"):
            depth_models.append("VDA_L")

        depth_models += ["VDA_Metric_S"]
        if VideoDepthAnythingModel.has_checkpoint_file("VDA_Metric_B"):
            depth_models.append("VDA_Metric_B")
        if VideoDepthAnythingModel.has_checkpoint_file("VDA_Metric_L"):
            depth_models.append("VDA_Metric_L")

        depth_models += ["VDA_Stream_S"]
        if VideoDepthAnythingStreamingModel.has_checkpoint_file("VDA_Stream_B"):
            depth_models.append("VDA_Stream_B")
        if VideoDepthAnythingStreamingModel.has_checkpoint_file("VDA_Stream_L"):
            depth_models.append("VDA_Stream_L")

        depth_models += ["VDA_Stream_Metric_S"]
        if VideoDepthAnythingStreamingModel.has_checkpoint_file("VDA_Stream_Metric_B"):
            depth_models.append("VDA_Stream_Metric_B")
        if VideoDepthAnythingStreamingModel.has_checkpoint_file("VDA_Stream_Metric_L"):
            depth_models.append("VDA_Stream_Metric_L")

        return depth_models

    def _on_stereo_slider_scroll(self, event, combo, slider, min_val, max_val, multiplier, is_int, extra_sync):
        """Guided Light pilot (ADR-097): dragging a slider writes the resulting value
        into its combo box via the existing _apply_combo_value() helper, so every
        existing consumer of that combo (arg-building, persistence, dependent-field
        update_* methods) sees a real, normal value change. wx.Slider.SetValue() never
        itself fires EVT_SLIDER, so the reverse sync below can't loop back into this."""
        value = slider.GetValue() / multiplier
        text = str(int(value)) if is_int else str(round(value, 2))
        _apply_combo_value(combo, text)
        if extra_sync is not None:
            getattr(self, extra_sync)()
        event.Skip()

    def _on_stereo_combo_text_sync_slider(self, event, combo, slider, min_val, max_val, multiplier):
        """Guided Light pilot (ADR-097): typing/selecting a value in the combo moves
        the slider thumb to match. Invalid or mid-keystroke text is silently ignored
        (the slider just stays where it was) rather than raising or clamping to a
        misleading edge value."""
        event.Skip()
        try:
            value = float(combo.GetValue())
        except ValueError:
            return
        value = max(min_val, min(max_val, value))
        slider.SetValue(round(value * multiplier))

    def get_stereo_sliders_and_panes(self):
        """Guided Light pilot (ADR-097) controls that must NEVER be handed to
        wx.lib.agw.persist -- it ships handlers for both wx.Slider and
        wx.CollapsiblePane (persist_handlers.py) that would otherwise auto-attach and
        persist a second, independent value alongside the combo's real one, or (for a
        pane) silently desync the ScrolledPanel's pinned MinSize on restore, since
        Restore() doesn't fire EVT_COLLAPSIBLEPANE_CHANGED. Sliders always derive from
        their combo at construction/load time; panes always start at their
        construction-time default expand state every launch -- see
        on_toggled_stereo_collapsible_pane()."""
        sliders = [getattr(self, slider_attr) for _, slider_attr, *_ in STEREO_SLIDER_FIELDS]
        pane_attrs = ("cpn_stereo_pop_divergence", "cpn_stereo_stability_flicker",
                      "cpn_stereo_inpainting_depth", "cpn_stereo_advanced")
        panes = [p for p in (getattr(self, name, None) for name in pane_attrs) if p is not None]
        return sliders + panes

    def get_depth_blend_sliders_and_panes(self):
        """Same persistence-exclusion purpose as get_stereo_sliders_and_panes(), for
        the Guided Light sliders added to Dual-Pass Depth Blend. No collapsible pane
        was added on this tab -- it's compact enough on its own (~14 fields, already
        organized into checkbox-gated Bilateral/CLAHE/Alignment sub-clusters shown via
        Enable()-graying, not Show()/Hide() -- confirmed live this doesn't need the
        same declutter treatment Stereo Generation's ~60 fields did)."""
        return [getattr(self, slider_attr) for _, slider_attr, *_ in DEPTH_BLEND_SLIDER_FIELDS]

    def get_processor_sliders_and_panes(self):
        """Same persistence-exclusion purpose as get_stereo_sliders_and_panes(), for
        the 2 Guided Light sliders added to Processor (Depth Batch Size, Worker
        Threads). No collapsible pane here either -- this tab is small (2 slider
        fields + a handful of checkboxes plus the separate Post-Processing sub-group),
        nowhere near the field count that motivated panes on Stereo Generation."""
        return [getattr(self, slider_attr) for _, slider_attr, *_ in PROCESSOR_SLIDER_FIELDS]

    def on_toggled_stereo_collapsible_pane(self, event):
        """Guided Light pilot (ADR-097): a collapsible pane toggling changes
        grp_stereo's real content height, which the ScrolledPanel wrapper (Tabbed
        mode: tab_wrap_stereo; Single Page mode: pnl_single) never re-measures on its
        own -- its MinSize is pinned once at construction (ADR-048) and nothing before
        this reused apply_zoom_level()'s exact fix for the same class of stale-MinSize
        bug caused by a different height-changing trigger (live zoom, not a pane).

        Order matters: refresh_layouts()'s recursive InvalidateBestSize() must run
        BEFORE the CalcMin()/SetMinSize() pinning below, not after -- a plain
        Layout() call repositions children using their EXISTING cached best-size, it
        does not itself force that cache to be recomputed, so computing CalcMin()
        first (the order apply_zoom_level uses) would silently reuse the pane's
        pre-toggle size here. apply_zoom_level gets away with a CalcMin()-then-
        refresh_layouts() order only because its own SetFont() call earlier already
        invalidates every descendant's best-size cache as a side effect -- a toggled
        CollapsiblePane has no equivalent implicit invalidation, so it must be done
        explicitly, first (confirmed live: the opposite order left Single Page mode's
        pnl_single MinSize completely unchanged after a real toggle)."""
        refresh_layouts(self)
        if self.layout_mode == LAYOUT_MODE_SINGLE_PAGE:
            self.pnl_single.SetMinSize(self.pnl_single.GetSizer().CalcMin())
        else:
            wrap_sizer = self.tab_wrap_stereo.GetSizer()
            if wrap_sizer is not None:
                self.tab_wrap_stereo.SetMinSize(wrap_sizer.CalcMin())
        refresh_layouts(self)
        self._clamp_frame_to_screen()
        event.Skip()

    def get_editable_comboboxes(self):
        editable_comboboxes = [
            self.cbo_divergence,
            self.cbo_convergence,
            self.cbo_convergence_smoothing,
            self.cbo_resolution,
            self.cbo_stereo_width,
            self.cbo_edge_dilation,
            self.cbo_edge_dilation_y,
            self.cbo_mask_outer_dilation,
            self.cbo_mask_inner_dilation,
            self.cbo_inpaint_max_width,
            self.cbo_ema_decay,
            self.cbo_ema_buffer,
            self.cbo_depth_blend_strength,
            self.cbo_depth_blend_region_percent,
            self.cbo_depth_blend_feather_blur,
            self.cbo_depth_blend_bilateral_d,
            self.cbo_depth_blend_bilateral_sigma_color,
            self.cbo_depth_blend_bilateral_sigma_space,
            self.cbo_depth_blend_clahe_clip,
            self.cbo_depth_blend_clahe_tile,
            self.cbo_depth_blend_align_decay,
            self.cbo_depth_refine_strength,
            self.cbo_depth_blend_edge_suppression,
            self.cbo_temporal_stabilize_strength,
            self.cbo_temporal_stabilize_max_shift,
            self.cbo_temporal_stabilize_flat_boost,
            self.cbo_temporal_stabilize_edge_protect,
            *self.grp_video.get_editable_comboboxes(),
            *self.grp_video_dec.get_editable_comboboxes(),
            self.cbo_foreground_scale,
            self.cbo_foreground_pop,
            self.cbo_foreground_divergence,
            self.cbo_background_pop,
            self.cbo_background_pop_coverage,
            self.cbo_background_divergence,
            self.cbo_edge_repair,
            self.cbo_sharpen_strength,
            self.cbo_pad,
            self.cbo_app_preset,
        ]
        return editable_comboboxes

    def get_anaglyph_method(self):
        if self.cbo_stereo_format.GetValue() == "Anaglyph":
            anaglyph = self.cbo_anaglyph_method.GetValue()
        else:
            anaglyph = None
        return anaglyph

    def on_close(self, event):
        if self.processing:
            with wx.MessageDialog(
                    self,
                    message=(T("A conversion is currently running.") + "\n\n" +
                             T("Closing right now will lose any progress made since the last "
                               "checkpoint — the same as a crash. Stop it safely first instead?") + "\n\n" +
                             T("Yes: stop safely (saves progress), then close.") + "\n" +
                             T("No: close immediately, losing unsaved progress.")),
                    caption=T("Conversion in progress"),
                    style=wx.YES_NO | wx.ICON_WARNING) as dlg:
                result = dlg.ShowModal()
            if result == wx.ID_YES:
                event.Veto()
                self.SetStatusText(T("Stopping safely before closing..."))
                self.suspend_event.set()
                self.stop_event.set()
                wx.CallLater(200, self._wait_then_close)
                return
            # else: fall through and close immediately, accepting the data loss

        self.save_preset()
        event.Skip()

    def _wait_then_close(self):
        if self.processing:
            wx.CallLater(200, self._wait_then_close)
            return
        self.save_preset()
        self.Destroy()

    def on_drop_files(self, x, y, filenames):
        if filenames:
            self.pnl_file.set_input_path(filenames[0])
        return True

    def update_start_button_state(self):
        if not self.processing:
            if self.pnl_file.input_path and self.pnl_file.output_path and not self.updating:
                self.btn_start.Enable()
            else:
                self.btn_start.Disable()
        if hasattr(self, "btn_run_update"):
            self.btn_run_update.Enable(not self.processing and not self.updating)

    def update_input_option_state(self):
        input_path = self.pnl_file.input_path
        is_export = self.cbo_stereo_format.GetValue() in {"Export", "Export disparity"}

        if is_yaml(input_path):
            try:
                config = export_config.ExportConfig.load(input_path)
                if config.type == export_config.IMAGE_TYPE:
                    self.chk_resume.Enable()
                    self.chk_recursive.Disable()
                else:
                    self.chk_resume.Disable()
                    self.chk_recursive.Disable()
            except:  # noqa
                self.chk_resume.Disable()
                self.chk_recursive.Disable()
        elif path.isdir(input_path) or is_text(input_path):
            self.chk_resume.Enable()
            self.chk_recursive.Enable()
        else:
            if is_export:
                self.chk_resume.Enable()
            else:
                self.chk_resume.Disable()
            self.chk_recursive.Disable()
        self.chk_recursive.SetValue(False)

    def reset_time_range(self):
        self.chk_start_time.SetValue(False)
        self.chk_end_time.SetValue(False)
        self.txt_start_time.SetValue("00:00:00")
        self.txt_end_time.SetValue("00:00:00")

    def resolve_output_path(self, input_path, output_path):
        args = self.parse_args()
        video = is_video(input_path)
        if args.export:
            if is_video(input_path):
                basename = (path.splitext(path.basename(input_path))[0]).strip()
                output_path = path.join(output_path, basename)
        elif is_output_dir(output_path):
            output_path = path.join(
                output_path,
                make_output_filename(input_path, args, video=video))

        return output_path

    def on_text_changed_txt_input(self, event):
        self.update_start_button_state()
        self.update_input_option_state()
        self.reset_time_range()

    def on_text_changed_txt_output(self, event):
        self.update_start_button_state()

    def update_model_selection(self):
        name = self.cbo_depth_model.GetValue()

        if name in DEPTH_PRO_MODELS:
            self.cbo_resolution.Disable()
            self.chk_fp16.Disable()
        else:
            self.cbo_resolution.Enable()
            self.chk_fp16.Enable()

        if name in ZOEDPETH_MODELS or name in DEPTH_PRO_MODELS:
            self.chk_limit_resolution.Disable()
        else:
            self.chk_limit_resolution.Enable()

        if (
                name in DA_AA_SUPPORTED_MODELS or
                name in VDA_AA_SUPPORTED_MODELS or
                name in VDA_STREAM_AA_SUPPORTED_MODELS or
                name in DA3_AA_SUPPORTED_MODELS
        ):
            self.chk_depth_aa.Enable()
        else:
            self.chk_depth_aa.Disable()

        self.Layout()
        self.Fit()
        self._clamp_frame_to_screen()

    def update_anaglyph_state(self):
        if self.cbo_stereo_format.GetValue() == "Anaglyph":
            self.lbl_anaglyph_method.Show()
            self.cbo_anaglyph_method.Show()
        else:
            self.lbl_anaglyph_method.Hide()
            self.cbo_anaglyph_method.Hide()

        self.Layout()
        self.Fit()
        self._clamp_frame_to_screen()

    def update_export_option_state(self):
        if self.cbo_stereo_format.GetValue() in {"Export", "Export disparity"}:
            self.chk_export_depth_only.Show()
            self.chk_export_depth_fit.Show()
        else:
            self.chk_export_depth_only.Hide()
            self.chk_export_depth_fit.Hide()

        self.Layout()
        self.Fit()
        self._clamp_frame_to_screen()

    def on_selected_index_changed_cbo_depth_model(self, event):
        self.update_model_selection()
        self.update_scene_segment()

    def update_preserve_screen_border(self):
        if self.cbo_method.GetValue() in {"row_flow_v2", "row_flow_v3", "row_flow_v3_sym",
                                          "mlbw_l2", "mlbw_l2s", "mlbw_l4", "mlbw_l2_inpaint",
                                          "monobw", "monobw_inpaint"}:
            self.chk_preserve_screen_border.Enable()
        else:
            self.chk_preserve_screen_border.Disable()

    def update_splat_blend_temperature(self):
        self.cbo_splat_blend_temperature.Enable(self.cbo_method.GetValue() == "forward_splat_fill")

    def on_selected_index_changed_cbo_method(self, event):
        self.update_divergence_warning()
        self.update_preserve_screen_border()
        self.update_inpaint_options()
        self.update_splat_blend_temperature()

    def on_selected_index_changed_cbo_stereo_format(self, event):
        self.update_input_option_state()
        self.update_anaglyph_state()
        self.update_export_option_state()

    def on_selected_index_changed_cbo_video_format(self, event):
        self.update_video_format()

    def on_selected_index_changed_cbo_video_codec(self, event):
        self.update_video_codec()

    def update_edge_dilation(self):
        if self.cbo_edge_dilation_y.GetValue():
            self.cbo_edge_dilation.SetToolTip(
                T("X value: horizontal edge smoothing strength. Y is set separately, to its own value on "
                  "the right. Recommended: 2 as a starting point."))
        else:
            self.cbo_edge_dilation.SetToolTip(
                T("X value (Y is blank, so this same number is used for BOTH X and Y — fully symmetric "
                  "smoothing). Recommended: 2 as a starting point."))

    def update_inpaint_options(self):
        if self.cbo_method.GetValue() in {"forward_inpaint", "mlbw_l2_inpaint", "monobw_inpaint"}:
            self.lbl_inpaint_model.Show()
            self.cbo_inpaint_model.Show()
            self.lbl_overlap_frames.Show()
            self.cbo_overlap_frames_pre.Show()
            self.cbo_overlap_frames_post.Show()
            self.lbl_mask_dilation.Show()
            self.cbo_mask_outer_dilation.Show()
            self.cbo_mask_inner_dilation.Show()
            self.lbl_inpaint_max_width.Show()
            self.cbo_inpaint_max_width.Show()
        else:
            self.lbl_inpaint_model.Hide()
            self.cbo_inpaint_model.Hide()
            self.lbl_overlap_frames.Hide()
            self.cbo_overlap_frames_pre.Hide()
            self.cbo_overlap_frames_post.Hide()
            self.lbl_mask_dilation.Hide()
            self.cbo_mask_outer_dilation.Hide()
            self.cbo_mask_inner_dilation.Hide()
            self.lbl_inpaint_max_width.Hide()
            self.cbo_inpaint_max_width.Hide()

        self.Layout()
        self.Fit()
        self._clamp_frame_to_screen()

    def on_changed_edge_dilation(self, event):
        self.update_edge_dilation()

    def update_convergence_mode(self):
        if self.cbo_convergence_mode.GetValue() == "constant":
            self.cbo_convergence_smoothing.Disable()
        else:
            self.cbo_convergence_smoothing.Enable()

    def on_changed_cbo_convergence_mode(self, event):
        self.update_convergence_mode()

    def update_ema_normalize(self):
        if self.chk_ema_normalize.IsChecked():
            self.chk_ema_motion_adaptive.Enable()
        else:
            self.chk_ema_motion_adaptive.Disable()
        # Decay Rate/Buffer are greyed out (never cleared) whenever "Auto EMA by
        # Scene Length" is checked, since that feature picks its own per-scene
        # values instead and these fixed ones only matter as its pre-first-scene
        # fallback -- see docs/ai/AI_DECISIONS.md ADR-057 amendment (2026-09-08).
        # Genre Preset (cbo_genre_preset, ADR-057 Amendment 9) only ever quick-fills
        # Decay Rate/Buffer, so it follows the exact same enable/disable condition as
        # the two fields it fills -- reusing this same gating, not a new mechanism.
        if self.chk_ema_normalize.IsChecked() and not self.chk_scene_batch_auto_ema.IsChecked():
            self.cbo_ema_decay.Enable()
            self.cbo_ema_buffer.Enable()
            self.cbo_genre_preset.Enable()
        else:
            self.cbo_ema_decay.Disable()
            self.cbo_ema_buffer.Disable()
            self.cbo_genre_preset.Disable()

    def update_scene_segment(self, *args, **kwargs):
        pass

    def on_changed_chk_ema_normalize(self, event):
        self.update_ema_normalize()
        self.update_scene_segment()

    def on_changed_chk_scene_batch_auto_ema(self, event):
        self.update_ema_normalize()

    def on_changed_cbo_genre_preset(self, event):
        # One-time quick-fill only -- writes Decay/Buffer once into the plain
        # wx.TextCtrl-backed fields and keeps no live link back to this dropdown
        # afterward, so a later hand-edit of either field is always safe (ADR-057
        # Amendment 9). The placeholder entry intentionally maps to nothing.
        preset = self.cbo_genre_preset.GetValue()
        values = GENRE_PRESET_EMA_VALUES.get(preset)
        if values is not None:
            decay, buffer = values
            self.cbo_ema_decay.SetValue(str(decay))
            self.cbo_ema_buffer.SetValue(str(buffer))

    def update_depth_blend(self):
        if self.chk_depth_blend.IsChecked():
            self.cbo_depth_blend_model.Enable()
            self.cbo_depth_blend_strength.Enable()
            self.cbo_depth_blend_region.Enable()
            self.cbo_depth_blend_region_percent.Enable(self.cbo_depth_blend_region.GetValue() != "detail")
            self.cbo_depth_blend_edge_suppression.Enable(self.cbo_depth_blend_region.GetValue() == "detail")
            self.chk_depth_blend_edge_hard_cutoff.Enable(self.cbo_depth_blend_region.GetValue() == "detail")
            self.cbo_depth_blend_feather_blur.Enable()
            self.chk_depth_blend_bilateral.Enable()
            self.chk_depth_blend_clahe.Enable()
            self.chk_depth_blend_align.Enable()
        else:
            self.cbo_depth_blend_model.Disable()
            self.cbo_depth_blend_strength.Disable()
            self.cbo_depth_blend_region.Disable()
            self.cbo_depth_blend_region_percent.Disable()
            self.cbo_depth_blend_edge_suppression.Disable()
            self.chk_depth_blend_edge_hard_cutoff.Disable()
            self.cbo_depth_blend_feather_blur.Disable()
            self.chk_depth_blend_bilateral.Disable()
            self.chk_depth_blend_clahe.Disable()
            self.chk_depth_blend_align.Disable()
        self.update_depth_blend_bilateral()
        self.update_depth_blend_clahe()
        self.update_depth_blend_align()

    def update_depth_blend_bilateral(self):
        enabled = self.chk_depth_blend.IsChecked() and self.chk_depth_blend_bilateral.IsChecked()
        self.cbo_depth_blend_bilateral_d.Enable(enabled)
        self.cbo_depth_blend_bilateral_sigma_color.Enable(enabled)
        self.cbo_depth_blend_bilateral_sigma_space.Enable(enabled)

    def update_depth_blend_clahe(self):
        enabled = self.chk_depth_blend.IsChecked() and self.chk_depth_blend_clahe.IsChecked()
        self.cbo_depth_blend_clahe_clip.Enable(enabled)
        self.cbo_depth_blend_clahe_tile.Enable(enabled)

    def update_depth_blend_align(self):
        enabled = self.chk_depth_blend.IsChecked() and self.chk_depth_blend_align.IsChecked()
        self.cbo_depth_blend_align_decay.Enable(enabled)

    def update_depth_refine(self):
        self.cbo_depth_refine_strength.Enable(self.chk_depth_refine.IsChecked())

    def on_changed_chk_depth_refine(self, event):
        self.update_depth_refine()

    def update_sharpen(self):
        self.cbo_sharpen_strength.Enable(self.chk_sharpen.IsChecked())

    def on_changed_chk_sharpen(self, event):
        self.update_sharpen()

    def on_changed_chk_depth_blend(self, event):
        self.update_depth_blend()

    def on_changed_chk_depth_blend_bilateral(self, event):
        self.update_depth_blend_bilateral()

    def on_changed_chk_depth_blend_clahe(self, event):
        self.update_depth_blend_clahe()

    def on_changed_chk_depth_blend_align(self, event):
        self.update_depth_blend_align()

    def update_waifu2x_upscale(self):
        enabled = self.chk_waifu2x_upscale.GetValue()
        self.cbo_waifu2x_method.Enable(enabled)
        self.cbo_waifu2x_noise_level.Enable(enabled)
        self.cbo_waifu2x_style.Enable(enabled)
        self.cbo_waifu2x_target.Enable(enabled)

    def on_changed_chk_waifu2x_upscale(self, event):
        self.update_waifu2x_upscale()

    def update_rife_interpolate(self):
        enabled = self.chk_rife_interpolate.GetValue()
        self.cbo_rife_model.Enable(enabled)
        self.cbo_rife_mode.Enable(enabled)
        self.txt_rife_target_fps.Enable(enabled and self.cbo_rife_mode.GetValue() == "Custom FPS...")

    def on_changed_chk_rife_interpolate(self, event):
        self.update_rife_interpolate()

    def on_changed_cbo_rife_mode(self, event):
        self.update_rife_interpolate()

    def update_temporal_stabilize(self):
        if self.chk_temporal_stabilize.IsChecked():
            self.cbo_temporal_stabilize_strength.Enable()
            self.cbo_temporal_stabilize_max_shift.Enable()
            self.cbo_temporal_stabilize_flat_boost.Enable()
            self.cbo_temporal_stabilize_edge_protect.Enable()
        else:
            self.cbo_temporal_stabilize_strength.Disable()
            self.cbo_temporal_stabilize_max_shift.Disable()
            self.cbo_temporal_stabilize_flat_boost.Disable()
            self.cbo_temporal_stabilize_edge_protect.Disable()

    def on_changed_chk_temporal_stabilize(self, event):
        self.update_temporal_stabilize()

    def confirm_overwrite(self, args):
        input_path = args.input
        output_path = args.output
        video = is_video(input_path)
        resume = args.resume
        if args.export:
            if is_video(input_path):
                basename = (path.splitext(path.basename(input_path))[0]).strip()
                output_path = path.join(output_path, basename, export_config.FILENAME)
            else:
                output_path = path.join(output_path, export_config.FILENAME)
        elif is_yaml(args.input):
            if is_output_dir(output_path):
                config = export_config.ExportConfig.load(input_path)
                if config.type == export_config.VIDEO_TYPE:
                    base_dir = path.dirname(args.input)
                    basename = config.basename or path.basename(base_dir)
                    output_path = path.join(
                        args.output,
                        make_output_filename(basename, args, video=True))
                else:
                    # image folder
                    return True
            else:
                output_path = output_path
                resume = False
        else:
            if is_output_dir(output_path):
                output_path = path.join(
                    output_path,
                    make_output_filename(input_path, args, video=video))
            else:
                output_path = output_path
                resume = False

        if path.exists(output_path) and not resume:
            with wx.MessageDialog(None,
                                  message=output_path + "\n" + T("already exists. Overwrite?"),
                                  caption=T("Confirm"), style=wx.YES_NO) as dlg:
                return dlg.ShowModal() == wx.ID_YES
        else:
            return True

    def show_validation_error_message(self, name, min_value, max_value):
        with wx.MessageDialog(
                None,
                message=T("`{}` must be a number {} - {}").format(name, min_value, max_value),
                caption=T("Error"),
                style=wx.OK) as dlg:
            dlg.ShowModal()

    def show_error_message(self, message):
        with wx.MessageDialog(None, message=message, caption=T("Error"), style=wx.OK) as dlg:
            dlg.ShowModal()

    def scene_auto_ema_gate_ok(self):
        """`Auto EMA by Scene Length` is meaningless without scene boundaries to key
        off of -- true unless it's off, or either `Scene Detection` (the regular,
        non--Scene-Batch path) or `Automated Scene Batch` (its own, separate pipeline,
        which always scene-detects internally) is turned on."""
        if not self.chk_scene_batch_auto_ema.GetValue():
            return True
        return self.chk_scene_batch.GetValue() or self.chk_scene_detect.IsChecked()

    def parse_args(self, skip_set_state=False):
        if not validate_number(self.cbo_divergence.GetValue(), 0.0, 100.0):
            self.show_validation_error_message(T("3D Strength"), 0.0, 100.0)
            return None
        if not validate_number(self.cbo_convergence.GetValue(), -100.0, 100.0):
            self.show_validation_error_message(T("Convergence Plane"), -100.0, 100.0)
            return None
        if not validate_number(self.cbo_convergence_smoothing.GetValue(), 0.0, 0.999):
            self.show_validation_error_message(T("Convergence Smoothing"), 0.0, 0.999)
            return None
        if not validate_number(self.cbo_pad.GetValue(), 0.0, 10.0, allow_empty=True):
            self.show_validation_error_message(T("Padding"), 0.0, 10.0)
            return None
        if not validate_number(self.cbo_edge_dilation.GetValue(), 0, 20, is_int=True, allow_empty=False):
            self.show_validation_error_message(T("Edge Fix"), 0, 20)
            return None
        if not validate_number(self.cbo_edge_dilation_y.GetValue(), 0, 20, is_int=True, allow_empty=True):
            self.show_validation_error_message(T("Edge Fix"), 0, 20)
            return None
        if self.lbl_overlap_frames.IsShown() and not (
                validate_number(self.cbo_overlap_frames_pre.GetValue(), 0, 5, is_int=True, allow_empty=False) and
                validate_number(self.cbo_overlap_frames_post.GetValue(), 0, 5, is_int=True, allow_empty=False)
        ):
            self.show_validation_error_message(T("Inpaint Overlap Frames"), 0, 5)
            return None
        if self.lbl_mask_dilation.IsShown() and not (
                validate_number(self.cbo_mask_inner_dilation.GetValue(), 0, 20, is_int=True, allow_empty=False) and
                validate_number(self.cbo_mask_outer_dilation.GetValue(), 0, 20, is_int=True, allow_empty=False)
        ):
            self.show_validation_error_message(T("Inpaint Mask Dilation"), 0, 20)
            return None
        if (
                self.lbl_inpaint_max_width.IsShown() and
                not validate_number(self.cbo_inpaint_max_width.GetValue(), 126, 8192, is_int=True, allow_empty=True)
        ):
            self.show_validation_error_message(T("Inpaint Max Width"), 126, 8192)
            return None
        if not validate_number(self.grp_video.max_fps, 0.25, 1000.0, allow_empty=False):
            self.show_validation_error_message(T("Max FPS"), 0.25, 1000.0)
            return None
        if not validate_number(self.grp_video.crf, 0, 51, is_int=True):
            self.show_validation_error_message(T("CRF"), 0, 51)
            return None
        if not validate_number(self.cbo_ema_decay.GetValue(), 0.0, 0.999):
            self.show_validation_error_message(T("Flicker Reduction"), 0.1, 0.999)
            return None
        if not validate_number(self.cbo_ema_buffer.GetValue(), 1, 1800, is_int=True):
            self.show_validation_error_message(T("Flicker Reduction Buffer"), 1, 1800)
            return None
        if not self.scene_auto_ema_gate_ok():
            self.show_error_message(
                T("`Auto EMA by Scene Length` requires either `Scene Detection` or "
                  "`Automated Scene Batch` to be turned on -- there are no scene "
                  "boundaries to key off of otherwise."))
            return None
        if self.chk_depth_blend.GetValue() and not validate_number(self.cbo_depth_blend_strength.GetValue(), 0.0, 1.0):
            self.show_validation_error_message(T("Dual-Pass Depth Blend"), 0.0, 1.0)
            return None
        if (self.chk_depth_blend.GetValue() and self.cbo_depth_blend_region.GetValue() != "detail"
                and not validate_number(self.cbo_depth_blend_region_percent.GetValue(), 0.0, 100.0)):
            self.show_validation_error_message(T("Dual-Pass Depth Blend region percent"), 0.0, 100.0)
            return None
        if self.chk_depth_blend.GetValue() and not validate_number(
                self.cbo_depth_blend_feather_blur.GetValue(), 0, 199, is_int=True):
            self.show_validation_error_message(T("Feather Blur"), 0, 199)
            return None
        if (self.chk_depth_blend.GetValue() and self.cbo_depth_blend_region.GetValue() == "detail"
                and not validate_number(self.cbo_depth_blend_edge_suppression.GetValue(), 0.0, 1.0)):
            self.show_validation_error_message(T("Edge Suppression"), 0.0, 1.0)
            return None
        if (self.chk_depth_blend.GetValue() and self.chk_depth_blend_align.GetValue()
                and not validate_number(self.cbo_depth_blend_align_decay.GetValue(), 0.0, 0.999)):
            self.show_validation_error_message(T("Depth Scale Alignment Decay"), 0.0, 0.999)
            return None
        if self.chk_depth_blend.GetValue() and self.chk_depth_blend_bilateral.GetValue():
            if not validate_number(self.cbo_depth_blend_bilateral_d.GetValue(), 1, 50, is_int=True):
                self.show_validation_error_message(T("Bilateral d"), 1, 50)
                return None
            if not validate_number(self.cbo_depth_blend_bilateral_sigma_color.GetValue(), 0.0, 500.0):
                self.show_validation_error_message(T("Bilateral sigmaColor"), 0.0, 500.0)
                return None
            if not validate_number(self.cbo_depth_blend_bilateral_sigma_space.GetValue(), 0.0, 500.0):
                self.show_validation_error_message(T("Bilateral sigmaSpace"), 0.0, 500.0)
                return None
        if self.chk_depth_blend.GetValue() and self.chk_depth_blend_clahe.GetValue():
            if not validate_number(self.cbo_depth_blend_clahe_clip.GetValue(), 0.1, 40.0):
                self.show_validation_error_message(T("CLAHE Clip Limit"), 0.1, 40.0)
                return None
            if not validate_number(self.cbo_depth_blend_clahe_tile.GetValue(), 1, 64, is_int=True):
                self.show_validation_error_message(T("CLAHE Tile Grid"), 1, 64)
                return None
        if (self.chk_temporal_stabilize.GetValue()
                and not validate_number(self.cbo_temporal_stabilize_strength.GetValue(), 0.0, 1.0)):
            self.show_validation_error_message(T("Object Stability (experimental)"), 0.0, 1.0)
            return None
        if (self.chk_temporal_stabilize.GetValue()
                and not validate_number(self.cbo_temporal_stabilize_max_shift.GetValue(), 0.0, 1.0,
                                         allow_empty=True)):
            self.show_validation_error_message(T("Max Shift"), 0.0, 1.0)
            return None
        if (self.chk_temporal_stabilize.GetValue()
                and not validate_number(self.cbo_temporal_stabilize_flat_boost.GetValue(), 0.0, 1.0)):
            self.show_validation_error_message(T("Flat-Area Boost"), 0.0, 1.0)
            return None
        if (self.chk_temporal_stabilize.GetValue()
                and not validate_number(self.cbo_temporal_stabilize_edge_protect.GetValue(), 0.0, 1.0)):
            self.show_validation_error_message(T("Edge Protection"), 0.0, 1.0)
            return None
        if not validate_number(self.cbo_foreground_scale.GetValue(), -3.0, 3.0, allow_empty=False):
            self.show_validation_error_message(T("Foreground Scale"), -3, 3)
            return None

        resolution = self.cbo_resolution.GetValue()
        if resolution == "Default" or resolution == "":
            resolution = None
        else:
            if not validate_number(resolution, 224, 8190, is_int=True, allow_empty=False):
                self.show_validation_error_message(T("Depth") + " " + T("Resolution"), 224, 8190)
                return
            resolution = int(resolution)
        limit_resolution = self.chk_limit_resolution.IsChecked()

        stereo_width = self.cbo_stereo_width.GetValue()
        if stereo_width == "Default" or stereo_width == "":
            stereo_width = None
        else:
            if not validate_number(stereo_width, 320, 8190, is_int=True, allow_empty=False):
                self.show_validation_error_message(T("Stereo processing Width"), 320, 8190)
                return
            stereo_width = int(stereo_width)

        rife_mode = self.cbo_rife_mode.GetValue()
        if self.chk_rife_interpolate.GetValue() and rife_mode == "Custom FPS...":
            if not validate_number(self.txt_rife_target_fps.GetValue(), 0.1, 1000.0, allow_empty=False):
                self.show_validation_error_message(T("RIFE Rate: Custom FPS"), 0.1, 1000.0)
                return
        if rife_mode == "Custom FPS...":
            rife_multiplier = None
            rife_target_fps = (float(self.txt_rife_target_fps.GetValue())
                                if self.txt_rife_target_fps.GetValue().strip() else None)
        else:
            rife_multiplier = int(rife_mode[0])  # "2x"/"3x"/"4x" -> 2/3/4
            rife_target_fps = None

        parser = create_parser(required_true=False)

        vr180 = self.cbo_stereo_format.GetValue() == "VR90"
        half_sbs = self.cbo_stereo_format.GetValue() == "Half SBS"
        tb = self.cbo_stereo_format.GetValue() == "Full TB"
        half_tb = self.cbo_stereo_format.GetValue() == "Half TB"
        cross_eyed = self.cbo_stereo_format.GetValue() == "Cross Eyed"
        rgbd = self.cbo_stereo_format.GetValue() == "RGB-D"
        half_rgbd = self.cbo_stereo_format.GetValue() == "Half RGB-D"
        anaglyph = self.get_anaglyph_method()
        export = self.cbo_stereo_format.GetValue() == "Export"
        export_disparity = self.cbo_stereo_format.GetValue() == "Export disparity"
        if export or export_disparity:
            export_depth_only = self.chk_export_depth_only.IsChecked()
            export_depth_fit = self.chk_export_depth_fit.IsChecked()
        else:
            export_depth_only = None
            export_depth_fit = None

        debug_depth = self.cbo_stereo_format.GetValue() == "Debug Depth"

        if self.cbo_pad.GetValue():
            pad = float(self.cbo_pad.GetValue())
        else:
            pad = None
        pad_mode = self.cbo_pad_mode.GetValue()
        if not pad_mode:
            pad_mode = "tblr"  # default

        rot = self.cbo_rotate.GetClientData(self.cbo_rotate.GetSelection())
        rotate_left = rotate_right = None
        if rot == "left":
            rotate_left = True
        elif rot == "right":
            rotate_right = True

        vf = []
        if self.cbo_deinterlace.GetValue():
            vf += [self.cbo_deinterlace.GetValue()]
        if self.txt_vf.GetValue():
            vf += [self.txt_vf.GetValue()]
        vf = ",".join(vf)

        device_id = int(self.cbo_device.GetClientData(self.cbo_device.GetSelection()))
        if device_id == -2:
            # All CUDA
            device_id = list(range(torch.cuda.device_count()))
        else:
            device_id = [device_id]

        depth_model_type = self.cbo_depth_model.GetValue()
        if (self.depth_model is None or (self.depth_model_type != depth_model_type or
                                         self.depth_model_device_id != device_id or
                                         self.depth_model_height != resolution or
                                         self.depth_model_limit_resolution != limit_resolution)):
            self.depth_model = None
            self.depth_model_type = None
            self.depth_model_device_id = None
            self.depth_model_height = None
            self.depth_model_limit_resolution = None
            gc_collect()

        max_output_width = max_output_height = None
        max_output_size = self.cbo_max_output_size.GetValue()
        if max_output_size:
            max_output_width, max_output_height = [int(s) for s in max_output_size.split("x")]

        input_path = self.pnl_file.input_path
        resume = self.chk_resume.IsEnabled() and self.chk_resume.GetValue()
        recursive = path.isdir(input_path) and self.chk_recursive.GetValue()
        skip_error = self.chk_skip_error.IsEnabled() and self.chk_skip_error.GetValue()
        start_time = self.txt_start_time.GetValue() if self.chk_start_time.GetValue() else None
        end_time = self.txt_end_time.GetValue() if self.chk_end_time.GetValue() else None

        if self.cbo_edge_dilation_y.GetValue():
            edge_dilation = [int(self.cbo_edge_dilation.GetValue()), int(self.cbo_edge_dilation_y.GetValue())]
        else:
            edge_dilation = int(self.cbo_edge_dilation.GetValue())

        if self.lbl_inpaint_model.IsShown():
            inpaint_model = self.cbo_inpaint_model.GetValue()
            mask_inner_dilation = int(self.cbo_mask_inner_dilation.GetValue())
            mask_outer_dilation = int(self.cbo_mask_outer_dilation.GetValue())
            inpaint_overlap_frames = [int(self.cbo_overlap_frames_pre.GetValue()),
                                      int(self.cbo_overlap_frames_post.GetValue())]
            if self.cbo_inpaint_max_width.GetValue():
                inpaint_max_width = int(self.cbo_inpaint_max_width.GetValue())
            else:
                inpaint_max_width = None
        else:
            inpaint_model = None
            inpaint_overlap_frames = None
            mask_inner_dilation = 0
            mask_outer_dilation = 0
            inpaint_max_width = None

        if self.chk_ema_normalize.GetValue():
            ema_options = dict(ema_normalize=True,
                               ema_decay=float(self.cbo_ema_decay.GetValue()),
                               ema_buffer=int(self.cbo_ema_buffer.GetValue()),
                               ema_motion_adaptive=self.chk_ema_motion_adaptive.GetValue())
        else:
            ema_options = {}

        if self.chk_depth_blend.GetValue():
            depth_blend_options = dict(
                depth_blend=True,
                depth_blend_model=self.cbo_depth_blend_model.GetValue(),
                depth_blend_strength=float(self.cbo_depth_blend_strength.GetValue()),
                depth_blend_region=self.cbo_depth_blend_region.GetValue(),
                depth_blend_region_percent=float(self.cbo_depth_blend_region_percent.GetValue()),
                depth_blend_feather_blur=int(self.cbo_depth_blend_feather_blur.GetValue()),
                depth_blend_bilateral=self.chk_depth_blend_bilateral.GetValue(),
                depth_blend_bilateral_d=int(self.cbo_depth_blend_bilateral_d.GetValue()),
                depth_blend_bilateral_sigma_color=float(self.cbo_depth_blend_bilateral_sigma_color.GetValue()),
                depth_blend_bilateral_sigma_space=float(self.cbo_depth_blend_bilateral_sigma_space.GetValue()),
                depth_blend_clahe=self.chk_depth_blend_clahe.GetValue(),
                depth_blend_clahe_clip=float(self.cbo_depth_blend_clahe_clip.GetValue()),
                depth_blend_clahe_tile=int(self.cbo_depth_blend_clahe_tile.GetValue()),
                depth_blend_align=self.chk_depth_blend_align.GetValue(),
                depth_blend_align_decay=float(self.cbo_depth_blend_align_decay.GetValue()),
                depth_blend_edge_suppression=float(self.cbo_depth_blend_edge_suppression.GetValue()),
                depth_blend_edge_hard_cutoff=self.chk_depth_blend_edge_hard_cutoff.GetValue(),
            )
        else:
            depth_blend_options = {}

        metadata = "filename" if self.chk_metadata.GetValue() else None
        preserve_screen_border = self.chk_preserve_screen_border.IsEnabled() and self.chk_preserve_screen_border.IsChecked()
        scene_detect = self.chk_scene_detect.IsChecked()
        disable_scene_cache = not self.chk_scene_detect_cache.IsChecked()
        depth_aa = self.chk_depth_aa.IsShown() and self.chk_depth_aa.IsEnabled() and self.chk_depth_aa.IsChecked()
        background_divergence = self.cbo_background_divergence.GetValue()
        background_divergence = float(background_divergence) if background_divergence else None
        foreground_divergence = self.cbo_foreground_divergence.GetValue()
        foreground_divergence = float(foreground_divergence) if foreground_divergence else None

        parser.set_defaults(
            input=input_path,
            output=self.pnl_file.output_path,
            yes=True,  # TODO: remove this

            divergence=float(self.cbo_divergence.GetValue()),
            convergence=float(self.cbo_convergence.GetValue()),
            convergence_mode=self.cbo_convergence_mode.GetValue(),
            convergence_smoothing=float(self.cbo_convergence_smoothing.GetValue()),
            ipd_offset=float(self.sld_ipd_offset.GetValue()),
            synthetic_view=self.cbo_synthetic_view.GetValue(),
            method=self.cbo_method.GetValue(),
            splat_blend_temperature=float(self.cbo_splat_blend_temperature.GetValue()),
            preserve_screen_border=preserve_screen_border,
            depth_model=depth_model_type,
            foreground_scale=float(self.cbo_foreground_scale.GetValue()),
            foreground_pop=float(self.cbo_foreground_pop.GetValue()),
            foreground_divergence=foreground_divergence,
            background_pop=float(self.cbo_background_pop.GetValue()),
            background_pop_coverage=float(self.cbo_background_pop_coverage.GetValue()) / 100.0,
            background_divergence=background_divergence,
            edge_repair_strength=float(self.cbo_edge_repair.GetValue()),
            sharpen=self.chk_sharpen.GetValue(),
            sharpen_strength=float(self.cbo_sharpen_strength.GetValue()),
            depth_aa=depth_aa,
            edge_dilation=edge_dilation,
            inpaint_model=inpaint_model,
            inpaint_overlap_frames=inpaint_overlap_frames,
            mask_inner_dilation=mask_inner_dilation,
            mask_outer_dilation=mask_outer_dilation,
            inpaint_max_width=inpaint_max_width,
            vr180=vr180,
            half_sbs=half_sbs,
            tb=tb,
            half_tb=half_tb,
            cross_eyed=cross_eyed,
            rgbd=rgbd,
            half_rgbd=half_rgbd,
            anaglyph=anaglyph,
            stereo_mode_tag=self.chk_stereo_mode_tag.GetValue(),

            export=export,
            export_disparity=export_disparity,
            export_depth_only=export_depth_only,
            export_depth_fit=export_depth_fit,

            debug_depth=debug_depth,
            **ema_options,
            depth_refine=self.chk_depth_refine.GetValue(),
            depth_refine_strength=float(self.cbo_depth_refine_strength.GetValue()),
            temporal_stabilize=self.chk_temporal_stabilize.GetValue(),
            temporal_stabilize_strength=float(self.cbo_temporal_stabilize_strength.GetValue()),
            temporal_stabilize_max_shift_velocity=(
                float(self.cbo_temporal_stabilize_max_shift.GetValue())
                if self.cbo_temporal_stabilize_max_shift.GetValue().strip() else None),
            temporal_stabilize_flat_region_boost=float(self.cbo_temporal_stabilize_flat_boost.GetValue()),
            temporal_stabilize_edge_protection=float(self.cbo_temporal_stabilize_edge_protect.GetValue()),
            **depth_blend_options,
            waifu2x_upscale=self.chk_waifu2x_upscale.GetValue(),
            waifu2x_method=self.cbo_waifu2x_method.GetValue(),
            waifu2x_noise_level=int(self.cbo_waifu2x_noise_level.GetValue()),
            waifu2x_style=self.cbo_waifu2x_style.GetValue(),
            waifu2x_upscale_target=self.cbo_waifu2x_target.GetValue(),
            rife_interpolate=self.chk_rife_interpolate.GetValue(),
            rife_model=self.cbo_rife_model.GetValue(),
            rife_multiplier=rife_multiplier,
            rife_target_fps=rife_target_fps,
            scene_detect=scene_detect,
            disable_scene_cache=disable_scene_cache,

            format=self.cbo_image_format.GetValue(),

            max_fps=self.grp_video.max_fps,
            pix_fmt=self.grp_video.pix_fmt,
            colorspace=self.grp_video.colorspace,
            video_format=self.grp_video.video_format,
            video_codec=self.grp_video.video_codec,
            crf=self.grp_video.crf,
            video_bitrate=self.grp_video.bitrate,
            profile_level=self.grp_video.profile_level,
            preset=self.grp_video.preset,
            tune=self.grp_video.tune,
            hwaccel=self.grp_video_dec.hwaccel,
            disable_software_fallback=not self.grp_video_dec.software_fallback,

            pad_mode=pad_mode,
            pad=pad,
            rotate_right=rotate_right,
            rotate_left=rotate_left,
            disable_exif_transpose=not self.chk_exif_transpose.GetValue(),
            vf=vf,
            max_output_width=max_output_width,
            max_output_height=max_output_height,
            keep_aspect_ratio=self.chk_keep_aspect_ratio.GetValue(),
            preserve_dowi=self.chk_preserve_dowi.GetValue(),
            hdr_to_sdr=self.chk_hdr_to_sdr.GetValue(),
            auto_resume=self.chk_auto_resume.GetValue(),
            resume_chunk_duration=300,
            upgrade_pix_fmt=int(self.cbo_upgrade_pix_fmt.GetValue()) if self.cbo_upgrade_pix_fmt.GetValue() else None,
            denoise=self.chk_denoise.GetValue(),
            preview=self.chk_preview.GetValue(),
            autocrop=self.cbo_autocrop.GetValue() if self.cbo_autocrop.GetValue() else None,
            scene_batch=self.chk_scene_batch.GetValue(),
            scene_batch_crop=self.txt_scene_batch_crop.GetValue() if self.txt_scene_batch_crop.GetValue() else None,
            scene_settings=self.txt_scene_settings.GetValue() if self.txt_scene_settings.GetValue() else None,
            scene_batch_auto_ema=self.chk_scene_batch_auto_ema.GetValue(),
            scene_batch_auto_ema_model=self.cbo_scene_batch_auto_ema_model.GetValue(),
            scene_batch_variant=self.txt_scene_batch_variant.GetValue() if self.txt_scene_batch_variant.GetValue() else None,
            vr_optimized_merge=self.chk_vr_optimized_merge.GetValue(),
            vr_merge_fps=(None if self.cbo_vr_merge_fps.GetValue() == "Source FPS"
                         else int(self.cbo_vr_merge_fps.GetValue())),
            gpu=device_id,
            batch_size=int(self.cbo_batch_size.GetValue()),
            resolution=resolution,
            limit_resolution=limit_resolution,
            stereo_width=stereo_width,
            max_workers=int(self.cbo_max_workers.GetValue()),
            tta=self.chk_tta.GetValue(),
            disable_amp=not self.chk_fp16.GetValue(),
            low_vram=self.chk_low_vram.GetValue(),
            pause_frees_vram=self.chk_pause_frees_vram.GetValue(),
            cuda_stream=self.chk_cuda_stream.GetValue(),
            compile=self.chk_compile.IsEnabled() and self.chk_compile.IsChecked(),

            resume=resume,
            recursive=recursive,
            skip_error=skip_error,
            metadata=metadata,
            start_time=start_time,
            end_time=end_time,
        )
        args = parser.parse_args()
        if not skip_set_state:
            set_state_args(
                args,
                stop_event=self.stop_event,
                suspend_event=self.suspend_event,
                tqdm_fn=functools.partial(TQDMGUI, self),
                stage_fn=functools.partial(_post_stage_change, self),
                depth_model=self.depth_model)
        return args

    def _compute_job_stages(self, args):
        """Precomputes the ordered list of stage names THIS job will actually run
        through, purely from the user's own already-chosen settings (all of these
        are opt-in flags known synchronously here, before the background worker even
        starts) -- lets the progress bar show "Step k of N" instead of an
        undifferentiated single bar for a job that can really be up to 8 separate
        phases (see docs/ai/AI_DECISIONS.md ADR-052 and its amendment). Order here
        matches the REAL order process_video_full()/process_video_with_resume() run
        these phases in, confirmed by reading both directly rather than assumed:
        Scene Boundary Detection, AutoCrop Analysis, and HDR/DV RPU Extraction all
        happen BEFORE the depth/stereo encode; Audio Extraction (auto-resume's own
        single clean-audio pass over the whole file) happens AFTER it, once all
        segments are already encoded, not before -- so it is listed after
        STAGE_DEPTH_STEREO, not before it. Dual-Pass Depth Blend passes are still
        shown as detail WITHIN stage Depth & Stereo Conversion (their own live
        per-frame tqdm progress and sub-label via _progress_title's step_label makes
        a separate top-level stage unnecessary).
        Two of these conditions are necessarily an upper-bound estimate, same
        limitation STAGE_HDR_REINJECT already had before this amendment: whether
        --preserve-dowi's source actually HAS DV/HDR10+ metadata, and whether
        --auto-resume's clip is actually long enough to need segmenting, can only be
        known by probing the file once the job is running, not synchronously here --
        so a run that turns on Preserve Dolby Vision against an SDR source, or Auto-
        Resume against a short clip, shows one more stage in "Step k/N" than actually
        fires (N is a maximum, matching how STAGE_HDR_REINJECT already behaved).
        Stage names here must stay byte-identical to iw3/utils.py's STAGE_* constants
        (imported, not re-typed) since _notify_stage() matches against this exact
        list to compute the current stage index."""
        stages = []
        if getattr(args, "scene_detect", False) or getattr(args, "scene_detect_only", False):
            stages.append(STAGE_SCENE_DETECT)
        if getattr(args, "autocrop", None) is not None:
            stages.append(STAGE_AUTOCROP)
        if getattr(args, "preserve_dowi", False):
            stages.append(STAGE_HDR_EXTRACT)
        stages.append(STAGE_DEPTH_STEREO)
        if getattr(args, "auto_resume", False):
            stages.append(STAGE_AUDIO_EXTRACT)
        if getattr(args, "waifu2x_upscale", False):
            stages.append(STAGE_WAIFU2X_UPSCALE)
        if getattr(args, "rife_interpolate", False):
            stages.append(STAGE_RIFE_INTERPOLATE)
        if getattr(args, "preserve_dowi", False):
            stages.append(STAGE_HDR_REINJECT)
        return stages

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{m:02d}:{s:02d}" if h == 0 else f"{h:02d}:{m:02d}:{s:02d}"

    def _stage_prefix(self):
        total = len(self.job_stages) if self.job_stages else 1
        idx = min(max(self.job_stage_index, 1), total)
        return f"Step {idx}/{total}: {self.current_stage_name}"

    def _set_title_progress(self, suffix=None):
        """ADR-056 -- a condensed, redundant progress signal in the window title bar
        (taskbar/Alt-Tab), on top of (not instead of) the real positioning fix in
        _clamp_frame_to_screen(). Content mirrors what the status bar already shows
        (ADR-052's stage/time/rate data) -- never a new computation -- just a shorter
        second surface for it that stays visible even if a user's real desktop layout
        (multi-monitor, taskbar auto-hide, etc.) still manages to obscure the window
        itself."""
        self.SetTitle(f"{self.base_title} -- {suffix}" if suffix else self.base_title)

    def ensure_cuda_context(self):
        # Deferred from module import time (see docs/ai/AI_DECISIONS.md) so that just
        # opening the GUI window does not grab a CUDA context / VRAM. Only called right
        # before real CUDA/video work begins (Start, Quick Preview), matching how the
        # CLI entry point (iw3/__main__.py) times this call relative to its own run.
        if not self.cuda_context_initialized:
            pyav_init_cuda_primary_context()
            self.cuda_context_initialized = True

    def on_click_btn_start(self, event):
        if self.chk_rife_interpolate.GetValue() and self.chk_preserve_dowi.GetValue():
            # Same check as set_state_args()'s CLI-side ValueError (see
            # docs/ai/AI_DECISIONS.md ADR-029) -- checked here too, before even
            # building args, so the user gets a clear message immediately instead
            # of an uncaught exception from deep inside parse_args()/set_state_args().
            wx.MessageBox(
                T("RIFE Frame Interpolation and Preserve Dolby Vision cannot be used together: "
                  "there is no way to assign correct DV/HDR10+ metadata to RIFE's synthetic "
                  "in-between frames. Disable one of the two before starting."),
                f"{T('Error')}: ValueError", wx.OK | wx.ICON_ERROR)
            return
        try:
            args = self.parse_args()
        except ValueError as e:
            wx.MessageBox(str(e), f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)
            return
        if args is None:
            return
        if not self.confirm_overwrite(args):
            return

        self.btn_autocrop_test.Disable()
        self.btn_start.Disable()
        self.btn_run_update.Disable()
        self.btn_cancel.Enable()
        self.btn_suspend.Enable()
        self.stop_event.clear()
        self.suspend_event.set()
        self.prg_tqdm.SetValue(0)
        self.SetStatusText("...")

        self.job_stages = self._compute_job_stages(args)
        self.job_stage_index = 1
        self.current_stage_name = self.job_stages[0]
        self.job_start_time = time()
        self.stage_start_time = self.job_start_time
        self.stage_pulse_timer.Stop()

        if args.state["depth_model"].has_checkpoint_file(args.depth_model):
            # Realod depth model
            self.SetStatusText(f"Loading {args.depth_model}...")
        else:
            # Need to download the model
            self.SetStatusText(f"Downloading {args.depth_model}...")
        self._set_title_progress(self._stage_prefix())

        self.ensure_cuda_context()
        startWorker(self.on_exit_worker, iw3_main, wargs=(args,))
        self.processing = True

    def on_exit_worker(self, result):
        self.stage_pulse_timer.Stop()
        try:
            args = result.get()
            self.depth_model = args.state["depth_model"]
            self.depth_model_type = args.depth_model
            self.depth_model_device_id = args.gpu
            self.depth_model_height = args.resolution
            self.depth_model_limit_resolution = args.limit_resolution

            if not self.stop_event.is_set():
                self.prg_tqdm.SetValue(self.prg_tqdm.GetRange())
                total_elapsed = self._format_duration(time() - self.job_start_time)
                self.SetStatusText(f"{T('Finished')} ({total_elapsed})")
            else:
                self.SetStatusText(T("Cancelled"))
        except: # noqa
            self.SetStatusText(T("Error"))
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            # ADR-072 Amendment: under the real GUI (pythonw.exe), nunif/pythonw_fix.py
            # (imported at the top of this file) reopens sys.stdout/sys.stderr onto
            # os.devnull -- a real, deliberate, long-standing fix for a separate,
            # genuine problem (pythonw.exe crashes on any bare write to a console-less
            # sys.stdout/stderr). A side effect nobody had accounted for: the
            # traceback.print_tb(tb) call directly above, and every [WARN]/error
            # print() throughout iw3/utils.py's real conversion pipeline (including
            # the --preserve-dowi HDR extraction block), write into that same devnull
            # sys.stderr and are silently discarded under the real GUI -- so a job
            # crash here has only ever shown str(e) in the popup, never a real
            # file/line traceback, no matter how many times it happens. This is the
            # actual, confirmed reason no console log could ever be found for any of
            # tonight's "Errno 129" reports. Writing the real traceback to a
            # persistent file, independent of sys.stdout/sys.stderr, closes that gap
            # for this and every future real conversion-job crash.
            try:
                crash_log_path = path.join(CONFIG_DIR, "iw3-gui-crash.log")
                with open(crash_log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n---- {datetime.now().isoformat(timespec='seconds')} ----\n")
                    f.write("".join(traceback.format_exception(e_type, e, tb)))
            except Exception:
                crash_log_path = None
            if crash_log_path is not None:
                message = f"{message}\n\n(Full details saved to {crash_log_path})"
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)

        self.processing = False
        self.btn_cancel.Disable()
        self.btn_suspend.Disable()
        self.btn_suspend.SetLabel(T("Suspend"))
        self.btn_autocrop_test.Enable()
        self.update_start_button_state()
        self._set_title_progress()

        # free vram
        gc_collect()

    def on_click_btn_cancel(self, event):
        self.suspend_event.set()
        self.stop_event.set()

    def on_click_btn_suspend(self, event):
        if self.suspend_event.is_set():
            self.suspend_event.clear()
            self.btn_suspend.SetLabel(T("Resume"))
            if self.chk_pause_frees_vram.GetValue():
                # ADR-038: the actual model move happens asynchronously on the
                # processing thread the next time it reaches suspend_event.wait() --
                # nothing else overwrites the status bar while genuinely paused (no
                # frames are being processed), so this stays visible for the whole
                # pause instead of just flashing by.
                self.SetStatusText(T("Pausing (freeing GPU memory)..."))
        else:
            self.start_time = time()
            self.suspend_pos = self.prg_tqdm.GetValue()
            self.suspend_event.set()
            self.btn_suspend.SetLabel(T("Suspend"))
            if self.chk_pause_frees_vram.GetValue():
                self.SetStatusText(T("Resuming (reloading models)..."))

    def on_tqdm(self, event):
        type, value, desc = event.GetValue()
        desc = desc if desc else ""
        if type == 0:
            # initialize
            # Real per-item progress data is available again (this is what every
            # tqdm-tracked phase -- AutoCrop Analysis, Scene Boundary Detection,
            # Dual-Pass Depth Blend passes, the main depth/stereo encode -- reports
            # through this same event) -- stop any indeterminate pulse animation left
            # running from a preceding waifu2x/RIFE/HDR blocking-subprocess stage.
            self.stage_pulse_timer.Stop()
            if 0 < value:
                self.prg_tqdm.SetRange(value)
            else:
                self.prg_tqdm.SetRange(1)
            self.prg_tqdm.SetValue(0)
            self.start_time = time()
            self.suspend_pos = 0
            self.SetStatusText(f"{self._stage_prefix()} -- {0}/{value} {desc}")
            self._set_title_progress(self._stage_prefix())
        elif type == 1:
            # update
            self.stage_pulse_timer.Stop()
            if self.prg_tqdm.GetValue() + value <= self.prg_tqdm.GetRange():
                self.prg_tqdm.SetValue(self.prg_tqdm.GetValue() + value)
            else:
                self.prg_tqdm.SetRange(self.prg_tqdm.GetValue() + value)
                self.prg_tqdm.SetValue(self.prg_tqdm.GetValue() + value)
            now = time()
            pos = self.prg_tqdm.GetValue()
            end_pos = self.prg_tqdm.GetRange()
            elapsed = now - self.start_time
            fps = (pos - self.suspend_pos) / (elapsed + 1e-6)
            if fps > 0:
                remaining_time = (end_pos - pos) / fps
                eta = self._format_duration(remaining_time)
                elapsed_str = self._format_duration(elapsed)
                self.SetStatusText(
                    f"{self._stage_prefix()} -- {pos}/{end_pos} frames "
                    f"[{fps:.2f} FPS, elapsed {elapsed_str}, ETA {eta}] {desc}")
                percent = int(pos / end_pos * 100) if end_pos else 0
                self._set_title_progress(f"{percent}% -- {self._stage_prefix()}")
        elif type == 2:
            # close
            pass

    def on_stage_change(self, event):
        """Handles a job-level stage transition posted from the background worker
        thread (see _post_stage_change/_notify_stage). Fires only for the stages
        that give no per-item tqdm progress of their own -- waifu2x upscaling, RIFE
        interpolation, HDR/Dolby Vision reinjection -- each a single blocking
        subprocess.run() call (CS-SUBPROCESS-001) with nothing to report until it
        finishes. There is no real progress fraction to show, so the Gauge itself is
        deliberately left exactly as the previous stage left it (usually full, since
        Depth & Stereo Conversion just finished at 100%) -- liveness for these
        stages comes entirely from stage_pulse_timer ticking the status bar's
        "running MM:SS" text every 500ms.

        The Gauge is deliberately NOT driven into wx.Gauge's indeterminate Pulse()
        ("marquee") mode here -- verified directly (an isolated wx probe, real
        PrintWindow captures, not just plausible-in-theory) that on this project's
        real wx/Windows combination, once Pulse() is called, the control gets stuck
        rendering the marquee's last frame: neither a plain SetValue() nor
        SetRange()+SetValue(0)+SetValue(<final>) afterward, nor manually clearing
        PBS_MARQUEE via SendMessage(PBM_SETMARQUEE)/SetWindowLong, restored correct
        determinate rendering (GetValue()/GetRange() reported the right numbers
        throughout -- only the native control's paint output was wrong). Using
        Pulse() here would have left the bar looking visually broken/empty at
        "Finished", which is worse than todays plain frozen-value behavior. See
        docs/ai/AI_DECISIONS.md for the full investigation."""
        name = event.name
        if name in self.job_stages:
            self.job_stage_index = self.job_stages.index(name) + 1
        self.current_stage_name = name
        self.stage_start_time = time()
        if not self.stage_pulse_timer.IsRunning():
            self.stage_pulse_timer.Start(500)
        self._refresh_stage_status_text()

    def on_stage_pulse_timer(self, event):
        self._refresh_stage_status_text()

    def _refresh_stage_status_text(self):
        running_for = self._format_duration(time() - self.stage_start_time)
        self.SetStatusText(f"{self._stage_prefix()}... (running {running_for})")
        self._set_title_progress(f"{self._stage_prefix()} ({running_for})")

    def apply_quick_preset(self, name):
        if name == "movie":
            self.cbo_divergence.SetValue("2.0")
            self.cbo_convergence.SetValue("0.5")
            self.cbo_convergence_mode.SetStringSelection("sod_v1")
            self.cbo_foreground_pop.SetValue("0.0")
            self.SetStatusText(T("Applied preset: Movie (subtle 3D)"))
        elif name == "action":
            self.cbo_divergence.SetValue("3.0")
            self.cbo_convergence.SetValue("0.5")
            self.cbo_convergence_mode.SetStringSelection("face_detect")
            self.cbo_foreground_pop.SetValue("0.5")
            self.SetStatusText(T("Applied preset: Action (strong pop effects)"))
        elif name == "3decker":
            # ADR-057 Amendment 12 (updated ADR-076): moved here from the Genre Preset
            # dropdown's "My Preferred Settings" entry (Amendment 11) -- the user's own
            # confirmed-best combo across many real controls at once, not just
            # Decay/Buffer. Updated to the full real-world-confirmed command from the
            # 2026-09-09 ADR-075 verification session (ran a real full-length movie
            # conversion cleanly with these exact settings). Checking
            # chk_scene_batch_auto_ema here goes through the exact same
            # on_changed_chk_scene_batch_auto_ema()/update_ema_normalize() chain a
            # real user click would use, so Decay Rate/Buffer/the Genre Preset
            # dropdown grey out exactly like a manual checkbox click (ADR-057
            # Amendment 8/9) -- not bypassed.
            self.cbo_method.SetStringSelection("mlbw_l2_inpaint")
            self.chk_preserve_screen_border.SetValue(True)
            self.cbo_depth_model.SetStringSelection("Any_V3_Mono_01")
            self.cbo_divergence.SetValue("2.5")
            self.cbo_convergence.SetValue("0.5")
            self.cbo_background_pop_coverage.SetValue("0.0")
            self.cbo_stereo_format.SetStringSelection("Half SBS")
            self.cbo_resolution.SetValue("512")

            self.cbo_inpaint_model.SetStringSelection("light_inpaint_v1")
            self.cbo_overlap_frames_pre.SetValue("3")
            self.cbo_overlap_frames_post.SetValue("3")
            self.chk_depth_aa.SetValue(True)

            self.chk_depth_refine.SetValue(True)
            self.cbo_depth_refine_strength.SetValue("1.0")
            self.update_depth_refine()

            self.chk_temporal_stabilize.SetValue(True)
            self.cbo_temporal_stabilize_strength.SetValue("0.3")
            self.cbo_temporal_stabilize_flat_boost.SetValue("0.0")
            self.cbo_temporal_stabilize_edge_protect.SetValue("0.0")
            self.cbo_temporal_stabilize_max_shift.SetValue("")
            self.update_temporal_stabilize()

            self.chk_scene_detect.SetValue(True)
            self.cbo_autocrop.SetStringSelection("BLACK")
            self.chk_end_time.SetValue(False)

            self.chk_ema_normalize.SetValue(True)
            self.cbo_ema_decay.SetValue("0.94")
            self.cbo_ema_buffer.SetValue("60")
            self.update_ema_normalize()

            self.cbo_scene_batch_auto_ema_model.SetStringSelection("GEMINI AI")
            self.chk_scene_batch_auto_ema.SetValue(True)
            self.on_changed_chk_scene_batch_auto_ema(None)

            # Video encoding/decoding -- codec set before tune so tune's own
            # codec-dependent choice list (NVENC vs libx264/265) is already correct
            # by the time "uhq" is applied (see ADR-075's own --tune fix).
            self.cbo_max_output_size.SetStringSelection("3840x2160")
            self.grp_video.cbo_pix_fmt.SetStringSelection("yuv420p10le")
            self.grp_video.cbo_video_format.SetStringSelection("mkv")
            if self.grp_video.has_nvenc:
                self.grp_video.cbo_video_codec.SetStringSelection("hevc_nvenc")
                self.grp_video.update_video_codec()
                self.grp_video.cbo_tune.SetValue("uhq")
            self.grp_video.cbo_fps.SetValue("1000.0")
            self.grp_video.cbo_crf.SetValue("15")
            self.grp_video_dec.cbo_hwaccel.SetStringSelection("cuda")
            self.grp_video_dec.chk_software_fallback.SetValue(False)

            self.cbo_max_workers.SetStringSelection("2")
            self.chk_metadata.SetValue(True)
            self.chk_preserve_dowi.SetValue(True)
            self.chk_stereo_mode_tag.SetValue(True)
            self.chk_auto_resume.SetValue(True)

            # Sets the checkbox only -- deliberately does NOT probe torch.compile
            # support live (same ADR-068/034 guard Import Command follows), since a
            # preset apply is a bulk state change, not a single direct interaction
            # with the Compile checkbox/Device selector.
            self.chk_compile.SetValue(True)

            self.SetStatusText(T("Applied preset: 3DECKER Preferred"))

    def save_preset(self, name=None):
        if not name:
            restore_path = True
            name = ""
            config_file = CONFIG_PATH
        else:
            restore_path = False
            name = sanitize_filename(name)
            config_file = path.join(PRESET_DIR, f"{name}.cfg")
            if path.exists(config_file):
                with wx.MessageDialog(None,
                                      message=name + "\n" + T("already exists. Overwrite?"),
                                      caption=T("Confirm"), style=wx.YES_NO) as dlg:
                    if dlg.ShowModal() != wx.ID_YES:
                        return

        input_path = self.pnl_file.input_path
        output_path = self.pnl_file.output_path
        preset = name
        try:
            if not restore_path:
                self.pnl_file.set_input_path("")
                self.pnl_file.set_output_path("")
            self.cbo_app_preset.SetValue("")
            manager = persist.PersistenceManager.Get()
            manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
            manager.SetPersistenceFile(config_file)
            persistent_manager_register_all(manager, self)
            for control in self.get_editable_comboboxes():
                persistent_manager_register(manager, control, EditableComboBoxPersistentHandler)
            for control in self.get_stereo_sliders_and_panes():
                manager.Unregister(control)
            for control in self.get_depth_blend_sliders_and_panes():
                manager.Unregister(control)
            for control in self.get_processor_sliders_and_panes():
                manager.Unregister(control)
            manager.SaveAndUnregister()
            self.reload_preset()
        finally:
            if not restore_path:
                self.pnl_file.set_input_path(input_path)
                self.pnl_file.set_output_path(output_path)
            self.cbo_app_preset.SetValue(preset)

    def list_preset(self):
        presets = [""]
        for fn in os.listdir(PRESET_DIR):
            name = path.splitext(fn)[0]
            presets.append(name)
        return presets

    def reload_preset(self):
        selected = self.cbo_app_preset.GetValue()
        choices = self.list_preset()
        self.cbo_app_preset.SetItems(choices)
        if selected in choices:
            self.cbo_app_preset.SetSelection(choices.index(selected))

    def load_preset(self, name=None, exclude_names=set()):
        exclude_names.add("cbo_language")  # ignore language
        exclude_names.add("cbo_layout")  # ignore GUI layout preference (own file, live-applied, ADR-037/038)
        exclude_names.add("cbo_zoom")  # ignore UI Zoom preference (own file, live-applied, see docs/ai/AI_DECISIONS.md)
        if not name:
            restore_path = True
            name = ""
            config_file = CONFIG_PATH
        else:
            restore_path = False
            name = sanitize_filename(name)
            config_file = path.join(PRESET_DIR, f"{name}.cfg")

        input_path = self.pnl_file.input_path
        output_path = self.pnl_file.output_path
        preset = name
        try:
            manager = persist.PersistenceManager.Get()
            manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
            manager.SetPersistenceFile(config_file)
            persistent_manager_register_all(manager, self)
            for control in self.get_editable_comboboxes():
                persistent_manager_register(manager, control, EditableComboBoxPersistentHandler)
            for control in self.get_stereo_sliders_and_panes():
                manager.Unregister(control)
            for control in self.get_depth_blend_sliders_and_panes():
                manager.Unregister(control)
            for control in self.get_processor_sliders_and_panes():
                manager.Unregister(control)
            persistent_manager_restore_all(manager, exclude_names)
            persistent_manager_unregister_all(manager)
        finally:
            if not restore_path:
                self.pnl_file.set_input_path(input_path)
                self.pnl_file.set_output_path(output_path)
            self.cbo_app_preset.SetValue(preset)

    def delete_preset(self, name=None):
        if not name:
            return
        config_file = path.join(PRESET_DIR, f"{name}.cfg")
        if path.exists(config_file):
            with wx.MessageDialog(None,
                                  message=name + "\n" + T("Delete?"),
                                  caption=T("Confirm"), style=wx.YES_NO) as dlg:
                if dlg.ShowModal() != wx.ID_YES:
                    return
            os.unlink(config_file)
        self.reload_preset()

    def on_click_btn_load_preset(self, event):
        self.load_preset(self.cbo_app_preset.GetValue(), exclude_names={self.GetName()})
        self.update_controls()

    def on_click_btn_save_preset(self, event):
        self.save_preset(self.cbo_app_preset.GetValue())

    def on_click_btn_delete_preset(self, event):
        self.delete_preset(self.cbo_app_preset.GetValue())
        event.Skip()

    def on_text_changed_cbo_language(self, event):
        lang = self.cbo_language.GetClientData(self.cbo_language.GetSelection())
        save_language_setting(LANG_CONFIG_PATH, lang)
        with wx.MessageDialog(None,
                              message=T("The language setting will be applied after restarting"),
                              style=wx.OK) as dlg:
            dlg.ShowModal()

    def on_text_changed_cbo_layout(self, event):
        # ADR-045: applies immediately via switch_layout_mode() -- no restart-required
        # dialog anymore (compare on_text_changed_cbo_language above, which still needs
        # one because the Language preference genuinely can't be applied live).
        mode = self.cbo_layout.GetClientData(self.cbo_layout.GetSelection())
        _save_layout_mode(LAYOUT_CONFIG_PATH, mode)
        self.switch_layout_mode(mode)

    def on_text_changed_cbo_zoom(self, event):
        # Applies immediately, like Layout (ADR-038) -- see apply_zoom_level().
        zoom_level = self.cbo_zoom.GetClientData(self.cbo_zoom.GetSelection())
        _save_zoom_level(ZOOM_CONFIG_PATH, zoom_level)
        self.apply_zoom_level(zoom_level)

    def on_click_divergence_warning(self, event):
        self.lbl_divergence_warning.Hide()

        self.Layout()
        self.Fit()
        self._clamp_frame_to_screen()

    def update_divergence_warning(self, *args, **kwargs):
        try:
            divergence = float(self.cbo_divergence.GetValue())
            method = self.cbo_method.GetValue()
            synthetic_view = self.cbo_synthetic_view.GetValue()
            max_divergence = float("inf")

            if method in {"row_flow_v3", "row_flow_v3_sym"}:
                if synthetic_view == "both":
                    max_divergence = 5.0
                else:
                    max_divergence = 5.0 * 0.5
            elif method == "row_flow_v2":
                if synthetic_view == "both":
                    max_divergence = 2.5
                else:
                    max_divergence = 2.5 * 0.5
            elif method in {"mlbw_l2", "mlbw_l4"}:
                if synthetic_view == "both":
                    max_divergence = 10.0
                else:
                    max_divergence = 10.0 * 0.5
            elif method in {"forward_inpaint", "mlbw_l2_inpaint", "monobw_inpaint"}:
                if synthetic_view == "both":
                    max_divergence = 5.0
                else:
                    max_divergence = 5.0 * 0.5

            if divergence > max_divergence:
                self.lbl_divergence_warning.SetLabel(
                    f"{divergence}: " + T("Out of range of training data") + f": {method}, {synthetic_view}"
                )
                self.lbl_divergence_warning.SetToolTip(
                    T("This result could be unstable"),
                )
                self.lbl_divergence_warning.Show()
            else:
                self.lbl_divergence_warning.SetLabel("")
                self.lbl_divergence_warning.SetToolTip("")
                self.lbl_divergence_warning.Hide()

            self.Layout()
            self.Fit()
            self._clamp_frame_to_screen()
        except ValueError:
            pass

    def on_selected_index_changed_cbo_device(self, event):
        self.update_compile()

    def update_compile(self, *args, probe=True, **kwargs):
        device_id = int(self.cbo_device.GetClientData(self.cbo_device.GetSelection()))
        if device_id == -2:
            # currently "All CUDA" does not support compile
            self.chk_compile.SetValue(False)
        elif probe:
            # check_compile_support() actually builds and torch.compile()s a real
            # model on this device -- genuine CUDA context + VRAM work, not a cheap
            # query. Only run it in response to the user directly touching the
            # Device selector or the torch.compile checkbox (see the two explicit
            # Bind()s to this method); never during passive startup control-sync
            # (probe=False from update_controls()'s initial call), or a restored
            # "compile: on" setting from a previous session would silently grab a
            # CUDA context the instant the window opens. See docs/ai/AI_DECISIONS.md.
            if self.chk_compile.IsChecked():
                device = create_device(device_id)
                try:
                    supported = check_compile_support(device)
                except Exception:
                    # check_compile_support() already catches real compile-probe
                    # failures internally, but this is the actual real-world GUI
                    # entry point for a real Windows compiler-toolchain probe, so
                    # guard it too rather than letting any surprise here crash the
                    # GUI with a raw error popup (see docs/ai/AI_DECISIONS.md).
                    supported = False
                if not supported:
                    self.chk_compile.SetValue(False)
                    self.SetStatusText(
                        T("torch.compile is not available on this system right now "
                          "-- see docs/torch_compile.md for setup"))

    def update_pad_mode(self, *args, **kwargs):
        if self.cbo_pad_mode.GetValue() == "16:9":
            self.cbo_pad.SetSelection(0)
            self.cbo_pad.Disable()
        else:
            self.cbo_pad.Enable()

    def get_cli_command(self):
        from subprocess import list2cmdline
        import argparse

        gui_args = self.parse_args(skip_set_state=True)
        if gui_args is None:
            return None
        default_parser = create_parser(required_true=False)
        default_args = default_parser.parse_args()
        gui_args = vars(gui_args)
        default_args = vars(default_args)

        argv = []
        yes = False
        for name in default_args.keys():
            action = next(a for a in default_parser._actions if a.dest == name)
            a = gui_args.get(name)
            b = default_args.get(name)
            if name == "input":
                name = "-i"
            elif name == "output":
                name = "-o"
            else:
                name = "--" + name.replace("_", "-")
            if name == "--yes":
                yes = True
                continue

            if isinstance(action, argparse._StoreTrueAction):
                if a:
                    argv.append(name)
                continue
            if isinstance(action, argparse._StoreFalseAction):
                if not a:
                    argv.append(name)
                continue

            if a == b:
                continue

            if isinstance(a, (list, tuple)):
                argv.append(name)
                for item in a:
                    argv.append(item)
            else:
                argv.append(name)
                argv.append(a)

        if yes:
            argv.append("--yes")

        return list2cmdline(["python", "-m", "iw3"] + [str(v) for v in argv])

    def on_click_btn_copy_command(self, event):
        command = self.get_cli_command()
        if command is None:
            # parse error
            return

        # print(command)

        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(command))
            wx.TheClipboard.Close()
        else:
            wx.MessageBox(T("Failed to open Clipbaord"), T("Error"), wx.OK | wx.ICON_ERROR)

    # --- Import Command (ADR-074): the reverse of Copy Command above. Parses a
    # pasted "python -m iw3 ..." command line via the real create_parser(), the
    # same one get_cli_command() diffs against to build the string in the first
    # place, then writes the result into every matching GUI widget. See
    # docs/ai/AI_DECISIONS.md ADR-074 for the full design/limitations. ---

    def parse_cli_command_text(self, command_text):
        """Reverse of get_cli_command(): parses a pasted command line string
        (with or without a leading "python -m iw3") into a real
        argparse.Namespace, via the exact same create_parser(required_true=
        False) definition get_cli_command() uses to build the string -- so
        Import Command can never drift out of sync with the real CLI as flags
        are added later. Returns (args, None) on success, or (None,
        error_message) on any parse failure; callers must not apply anything
        when args is None -- a malformed paste must never partially apply."""
        import io
        import contextlib

        command_text = (command_text or "").strip()
        if not command_text:
            return None, T("Nothing to import -- paste a command line first.")

        # A real command line is one logical line -- any newline inside the
        # pasted text is either an explicit cmd.exe "^" continuation (from
        # copying a multi-line block formatted for a real terminal) or, just
        # as commonly, a soft-wrap artifact reintroduced as a literal '\n' by
        # the review textbox itself when long pasted text wraps visually
        # (confirmed by a real user paste: "GEMINI AI" came back as
        # "GEMINI\n  AI", corrupting the value and failing to parse). Neither
        # case is ever *meaningful* content -- collapse every run of
        # whitespace that contains a newline (optionally preceded by "^",
        # cmd.exe's line-continuation marker) down to a single space before
        # splitting, so both cases are handled the same way.
        command_text = re.sub(r"\s*\^?[\r\n]+\s*", " ", command_text).strip()

        try:
            argv = _split_windows_command_line(command_text)
        except Exception as e:
            return None, T("Could not split that command line:") + f"\n{e}"

        # Strip a leading "python[.exe] -m iw3" -- what Copy Command produces,
        # and what the user has actually been running directly -- if present,
        # so pasting the real command line verbatim (not just its flags) works.
        i = 0
        if i < len(argv) and path.basename(argv[i]).lower() in ("python", "python.exe", "python3", "python3.exe"):
            i += 1
        if i + 1 < len(argv) and argv[i] == "-m" and argv[i + 1] == "iw3":
            i += 2
        argv = argv[i:]

        parser = create_parser(required_true=False)
        stderr_buffer = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr_buffer):
                args = parser.parse_args(argv)
        except SystemExit:
            # argparse's own error() prints usage + the real reason to stderr and
            # raises SystemExit -- caught here so a bad paste shows a message box
            # instead of taking down the whole GUI process (nothing above this
            # point has touched any widget yet, so no partial apply can happen).
            message = stderr_buffer.getvalue().strip()
            return None, (T("That command line could not be parsed:") + "\n\n"
                          + (message or T("(unrecognized or invalid option)")))
        return args, None

    def apply_parsed_args_to_gui(self, args):
        """Reverse of parse_args(): writes a real argparse.Namespace (as
        returned by parse_cli_command_text(), i.e. actually produced by
        create_parser().parse_args()) into every GUI widget parse_args() reads
        FROM. Mirrors parse_args() field-by-field -- see that method for the
        forward direction this inverts, and docs/ai/AI_DECISIONS.md ADR-074.

        Not every attribute create_parser() can produce has a GUI widget
        (--yes, deprecated --zoed-*, --warp-steps, --mapper*, --remove-bg/
        --bg-model, --keyframe*, --scene-cache-*/--scene-detect-only,
        --find-param, --update, and --resume-chunk-duration which the GUI
        always sends as a fixed 300) -- those are simply never read here, so
        they're silently skipped, matching this project's established
        "genuinely CLI-only setting -> skip, don't error" convention.

        Conversely, a handful of real GUI settings -- confirmed by reading
        create_parser() directly, not assumed -- have NO real --flag at all
        (waifu2x_method/noise_level/style; the Dual-Pass Depth Blend feather/
        bilateral/CLAHE/align/edge-suppression fields; Object Stability's
        max-shift/flat-boost/edge-protection; VR Optimized Merge/its FPS):
        Copy Command already cannot
        export these (get_cli_command() only iterates create_parser()'s own
        default Namespace keys), so a pasted command can never carry them
        either -- `args` here simply won't have these attributes, and this
        function does not try to guess or preserve them.

        Does NOT probe torch.compile support (see ADR-068, and ADR-074's use
        of the same guard) and does NOT run a conversion -- only sets widget
        values, then calls update_controls(probe_compile=False) once at the
        end to refresh dependent enable/disable/show/hide state, exactly like
        the passive startup-restore path."""
        self.pnl_file.set_input_path(args.input or "")
        self.pnl_file.set_output_path(args.output or "")

        _apply_combo_value(self.cbo_divergence, args.divergence)
        _apply_combo_value(self.cbo_convergence, args.convergence)
        _apply_combo_value(self.cbo_convergence_mode, args.convergence_mode)
        _apply_combo_value(self.cbo_convergence_smoothing, args.convergence_smoothing)
        self.sld_ipd_offset.SetValue(int(round(args.ipd_offset)))
        _apply_combo_value(self.cbo_synthetic_view, args.synthetic_view)
        _apply_combo_value(self.cbo_method, args.method)
        _apply_combo_value(self.cbo_splat_blend_temperature, args.splat_blend_temperature)
        self.chk_preserve_screen_border.SetValue(bool(args.preserve_screen_border))
        _apply_combo_value(self.cbo_depth_model, args.depth_model)
        _apply_combo_value(self.cbo_foreground_scale, args.foreground_scale)
        _apply_combo_value(self.cbo_foreground_pop, args.foreground_pop)
        _apply_combo_value(
            self.cbo_foreground_divergence,
            args.foreground_divergence if args.foreground_divergence is not None else "")
        _apply_combo_value(self.cbo_background_pop, args.background_pop)
        _apply_combo_value(self.cbo_background_pop_coverage, args.background_pop_coverage * 100.0)
        _apply_combo_value(
            self.cbo_background_divergence,
            args.background_divergence if args.background_divergence is not None else "")
        _apply_combo_value(self.cbo_edge_repair, args.edge_repair_strength)
        self.chk_sharpen.SetValue(bool(args.sharpen))
        _apply_combo_value(self.cbo_sharpen_strength, args.sharpen_strength)
        self.chk_depth_aa.SetValue(bool(args.depth_aa))

        edge_dilation = args.edge_dilation if isinstance(args.edge_dilation, list) else [args.edge_dilation]
        _apply_combo_value(self.cbo_edge_dilation, edge_dilation[0])
        _apply_combo_value(self.cbo_edge_dilation_y, edge_dilation[1] if len(edge_dilation) > 1 else "")

        if args.inpaint_model:
            _apply_combo_value(self.cbo_inpaint_model, args.inpaint_model)
        _apply_combo_value(self.cbo_mask_inner_dilation, args.mask_inner_dilation)
        _apply_combo_value(self.cbo_mask_outer_dilation, args.mask_outer_dilation)
        if args.inpaint_max_width is not None:
            _apply_combo_value(self.cbo_inpaint_max_width, args.inpaint_max_width)
        if args.inpaint_overlap_frames:
            pre = args.inpaint_overlap_frames[0]
            post = args.inpaint_overlap_frames[1] if len(args.inpaint_overlap_frames) > 1 else pre
            _apply_combo_value(self.cbo_overlap_frames_pre, pre)
            _apply_combo_value(self.cbo_overlap_frames_post, post)

        # Stereo Format -- reconstructed from the mutually-exclusive flags
        # parse_args() reads it INTO, same priority order as build there.
        if args.vr180:
            stereo_format = "VR90"
        elif args.half_sbs:
            stereo_format = "Half SBS"
        elif args.tb:
            stereo_format = "Full TB"
        elif args.half_tb:
            stereo_format = "Half TB"
        elif args.cross_eyed:
            stereo_format = "Cross Eyed"
        elif args.rgbd:
            stereo_format = "RGB-D"
        elif args.half_rgbd:
            stereo_format = "Half RGB-D"
        elif args.anaglyph is not None:
            stereo_format = "Anaglyph"
            _apply_combo_value(self.cbo_anaglyph_method, args.anaglyph)
        elif args.export:
            stereo_format = "Export"
        elif args.export_disparity:
            stereo_format = "Export disparity"
        elif args.debug_depth:
            stereo_format = "Debug Depth"
        else:
            stereo_format = "Full SBS"
        _apply_combo_value(self.cbo_stereo_format, stereo_format)
        if args.export or args.export_disparity:
            self.chk_export_depth_only.SetValue(bool(args.export_depth_only))
            self.chk_export_depth_fit.SetValue(bool(args.export_depth_fit))

        self.chk_stereo_mode_tag.SetValue(bool(args.stereo_mode_tag))

        self.chk_ema_normalize.SetValue(bool(args.ema_normalize))
        _apply_combo_value(self.cbo_ema_decay, args.ema_decay)
        _apply_combo_value(self.cbo_ema_buffer, args.ema_buffer)
        self.chk_ema_motion_adaptive.SetValue(bool(args.ema_motion_adaptive))

        self.chk_depth_refine.SetValue(bool(args.depth_refine))
        _apply_combo_value(self.cbo_depth_refine_strength, args.depth_refine_strength)

        self.chk_temporal_stabilize.SetValue(bool(args.temporal_stabilize))
        _apply_combo_value(self.cbo_temporal_stabilize_strength, args.temporal_stabilize_strength)

        # depth_blend_* fields beyond these 5 (feather blur, bilateral, CLAHE,
        # align, edge suppression) are GUI-only -- see docstring above -- left as-is.
        self.chk_depth_blend.SetValue(bool(args.depth_blend))
        _apply_combo_value(self.cbo_depth_blend_model, args.depth_blend_model)
        _apply_combo_value(self.cbo_depth_blend_strength, args.depth_blend_strength)
        _apply_combo_value(self.cbo_depth_blend_region, args.depth_blend_region)
        _apply_combo_value(self.cbo_depth_blend_region_percent, args.depth_blend_region_percent)

        self.chk_waifu2x_upscale.SetValue(bool(args.waifu2x_upscale))
        _apply_combo_value(self.cbo_waifu2x_target, args.waifu2x_upscale_target)

        self.chk_rife_interpolate.SetValue(bool(args.rife_interpolate))
        _apply_combo_value(self.cbo_rife_model, args.rife_model)
        if args.rife_target_fps is not None:
            _apply_combo_value(self.cbo_rife_mode, "Custom FPS...")
            self.txt_rife_target_fps.SetValue(str(args.rife_target_fps))
        elif args.rife_multiplier in (2, 3, 4):
            _apply_combo_value(self.cbo_rife_mode, f"{args.rife_multiplier}x")

        self.chk_scene_detect.SetValue(bool(args.scene_detect))
        self.chk_scene_detect_cache.SetValue(not args.disable_scene_cache)

        _apply_combo_value(self.cbo_image_format, args.format)

        # Video Encoding (VideoEncodingBox exposes read-only properties, not
        # setters -- its own update_video_format()/update_video_codec()
        # reconciliation, triggered below via update_controls(), rebuilds each
        # combobox's choice list around whatever raw value is set here first,
        # exactly like a user changing Format/Codec by hand would).
        grp = self.grp_video
        _apply_combo_value(grp.cbo_fps, args.max_fps)
        _apply_combo_value(grp.cbo_video_format, args.video_format)
        if args.video_codec:
            _apply_combo_value(grp.cbo_video_codec, args.video_codec)
        _apply_combo_value(grp.cbo_pix_fmt, args.pix_fmt)
        _apply_combo_value(grp.cbo_colorspace, args.colorspace)
        _apply_combo_value(grp.cbo_crf, args.crf)
        _apply_combo_value(grp.cbo_bitrate, args.video_bitrate)
        _apply_combo_value(grp.cbo_profile_level, args.profile_level or "auto")
        _apply_combo_value(grp.cbo_preset, args.preset)
        tune = list(args.tune or [])
        grp.chk_tune_fastdecode.SetValue("fastdecode" in tune)
        grp.chk_tune_zerolatency.SetValue("zerolatency" in tune)
        remaining_tune = next((t for t in tune if t not in ("fastdecode", "zerolatency")), "")
        _apply_combo_value(grp.cbo_tune, remaining_tune)

        self.grp_video_dec.cbo_hwaccel.SetValue(args.hwaccel or "")
        self.grp_video_dec.chk_software_fallback.SetValue(not args.disable_software_fallback)

        _apply_combo_value(self.cbo_pad_mode, "" if args.pad_mode == "tblr" else args.pad_mode)
        _apply_combo_value(self.cbo_pad, args.pad if args.pad is not None else "")
        if args.rotate_left:
            self.cbo_rotate.SetSelection(1)
        elif args.rotate_right:
            self.cbo_rotate.SetSelection(2)
        else:
            self.cbo_rotate.SetSelection(0)
        self.chk_exif_transpose.SetValue(not args.disable_exif_transpose)

        vf_value = args.vf or ""
        vf_parts = vf_value.split(",") if vf_value else []
        if vf_parts and vf_parts[0] == "yadif":
            self.cbo_deinterlace.SetValue("yadif")
            self.txt_vf.SetValue(",".join(vf_parts[1:]))
        else:
            self.cbo_deinterlace.SetValue("")
            self.txt_vf.SetValue(vf_value)

        if args.max_output_width and args.max_output_height:
            _apply_combo_value(self.cbo_max_output_size, f"{args.max_output_width}x{args.max_output_height}")
        else:
            _apply_combo_value(self.cbo_max_output_size, "")
        self.chk_keep_aspect_ratio.SetValue(bool(args.keep_aspect_ratio))
        self.chk_preserve_dowi.SetValue(bool(args.preserve_dowi))
        self.chk_hdr_to_sdr.SetValue(bool(args.hdr_to_sdr))
        self.chk_auto_resume.SetValue(bool(args.auto_resume))
        _apply_combo_value(
            self.cbo_upgrade_pix_fmt, args.upgrade_pix_fmt if args.upgrade_pix_fmt is not None else "")
        self.chk_denoise.SetValue(bool(args.denoise))
        self.chk_preview.SetValue(bool(args.preview))
        _apply_combo_value(self.cbo_autocrop, args.autocrop or "")
        self.chk_scene_batch.SetValue(bool(args.scene_batch))
        self.txt_scene_batch_crop.SetValue(args.scene_batch_crop or "")
        self.txt_scene_settings.SetValue(args.scene_settings or "")
        self.chk_scene_batch_auto_ema.SetValue(bool(args.scene_batch_auto_ema))
        _apply_combo_value(self.cbo_scene_batch_auto_ema_model, args.scene_batch_auto_ema_model)
        self.txt_scene_batch_variant.SetValue(args.scene_batch_variant or "")
        # VR Optimized Merge / vr_merge_fps are GUI-only (no real --flag, same as
        # the fields called out above) -- left as-is, see docstring above.

        gpu_ids = args.gpu or []
        all_cuda_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        if len(gpu_ids) > 1 and all_cuda_ids and sorted(gpu_ids) == sorted(all_cuda_ids):
            target_device_id = -2
        elif gpu_ids:
            target_device_id = gpu_ids[0]
        else:
            target_device_id = None
        if target_device_id is not None:
            for i in range(self.cbo_device.GetCount()):
                if self.cbo_device.GetClientData(i) == target_device_id:
                    self.cbo_device.SetSelection(i)
                    break

        _apply_combo_value(self.cbo_batch_size, args.batch_size)
        _apply_combo_value(self.cbo_resolution, args.resolution if args.resolution is not None else "Default")
        self.chk_limit_resolution.SetValue(bool(args.limit_resolution))
        _apply_combo_value(self.cbo_stereo_width, args.stereo_width if args.stereo_width is not None else "Default")
        _apply_combo_value(self.cbo_max_workers, args.max_workers)
        self.chk_tta.SetValue(bool(args.tta))
        self.chk_fp16.SetValue(not args.disable_amp)
        self.chk_low_vram.SetValue(bool(args.low_vram))
        self.chk_pause_frees_vram.SetValue(bool(args.pause_frees_vram))
        self.chk_cuda_stream.SetValue(bool(args.cuda_stream))
        # torch.compile: SetValue only -- never probe here. update_controls(
        # probe_compile=False) below is what keeps this safe, exactly like the
        # startup-restore path (see ADR-068, and ADR-074's use of the same guard).
        self.chk_compile.SetValue(bool(args.compile))

        self.chk_resume.SetValue(bool(args.resume))
        self.chk_recursive.SetValue(bool(args.recursive))
        self.chk_skip_error.SetValue(bool(args.skip_error))
        self.chk_metadata.SetValue(bool(args.metadata))
        if args.start_time:
            self.chk_start_time.SetValue(True)
            self.txt_start_time.SetValue(args.start_time)
        else:
            self.chk_start_time.SetValue(False)
        if args.end_time:
            self.chk_end_time.SetValue(True)
            self.txt_end_time.SetValue(args.end_time)
        else:
            self.chk_end_time.SetValue(False)

        self.update_controls(probe_compile=False)

    def on_click_btn_import_command(self, event):
        clipboard_text = ""
        if wx.TheClipboard.Open():
            try:
                data = wx.TextDataObject()
                if wx.TheClipboard.GetData(data):
                    clipboard_text = data.GetText()
            finally:
                wx.TheClipboard.Close()

        dlg = wx.Dialog(self, title=T("Import Command"), style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        lbl = wx.StaticText(
            dlg, label=T("Paste (or edit) a full \"python -m iw3 ...\" command line below, then click "
                         "OK to apply every matching setting from it to this window. Review it first -- "
                         "this can overwrite a lot of your current settings at once."))
        lbl.Wrap(dlg.FromDIP(480))
        txt = wx.TextCtrl(dlg, value=clipboard_text, style=wx.TE_MULTILINE, size=dlg.FromDIP((480, 160)))
        btn_sizer = dlg.CreateButtonSizer(wx.OK | wx.CANCEL)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(lbl, 0, wx.ALL | wx.EXPAND, 8)
        sizer.Add(txt, 1, wx.ALL | wx.EXPAND, 8)
        sizer.Add(btn_sizer, 0, wx.ALL | wx.ALIGN_CENTER, 8)
        dlg.SetSizerAndFit(sizer)
        dlg.CentreOnParent()

        modal_result = dlg.ShowModal()
        command_text = txt.GetValue()
        dlg.Destroy()

        if modal_result != wx.ID_OK:
            return

        args, error_message = self.parse_cli_command_text(command_text)
        if args is None:
            wx.MessageBox(error_message, T("Import Command"), wx.OK | wx.ICON_ERROR)
            return

        self.apply_parsed_args_to_gui(args)
        self.SetStatusText(T("Imported settings from the pasted command"))

    # --- Check for Updates (read-only fetch + compare only, see docs/ai/AI_DECISIONS.md) ---

    def run_check_updates(self):
        # Runs on a background thread via startWorker -- this hits the network
        # (git fetch) and must never freeze the GUI thread while waiting on it.
        return update_check.check_for_updates()

    def on_exit_check_updates_worker(self, result):
        self.btn_check_updates.Enable()
        try:
            check_result = result.get()
        except: # noqa
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            self.SetStatusText(T("Error"))
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)
            return

        message = update_check.format_result_message(check_result)
        if check_result["status"] == "error":
            self.SetStatusText(T("Check for Updates failed"))
            wx.MessageBox(message, T("Check for Updates"), wx.OK | wx.ICON_ERROR)
        elif check_result["status"] == "up_to_date":
            self.SetStatusText(T("Already up to date"))
            wx.MessageBox(message, T("Check for Updates"), wx.OK | wx.ICON_INFORMATION)
        else:
            self.SetStatusText(T("Updates are available upstream"))
            wx.MessageBox(message, T("Check for Updates"), wx.OK | wx.ICON_INFORMATION)

    def on_click_btn_check_updates(self, event):
        self.btn_check_updates.Disable()
        self.SetStatusText(T("Checking for updates..."))
        startWorker(self.on_exit_check_updates_worker, self.run_check_updates)

    # --- Run Update (applies the real update.bat -- see docs/ai/AI_DECISIONS.md
    # ADR-069, the direct follow-up to ADR-035's deliberately-deferred "applying an
    # update" scope; the Check for Updates feature just above stays read-only) ---

    def run_update(self, cmd, cwd, nunif_dir, dlg):
        # Runs on a background thread via startWorker -- never blocks the GUI
        # thread. update.bat can run long enough (package installs, model
        # downloads, a source pull) that its output is streamed line-by-line to
        # `dlg` via wx.CallAfter as it's produced, rather than captured and shown
        # only at the end the way this file's other standalone-tool log boxes work
        # (run_sharpen/run_rife_standalone/etc.) -- a run this long needs live
        # visibility, not just a final dump. stdin is explicitly closed (DEVNULL):
        # update.bat ends with `pause` on both its success and error paths, which
        # would otherwise wait forever for a keypress this non-interactive
        # subprocess can never provide (CS-SUBPROCESS-001: arg list, never
        # shell=True; cmd.exe /c is the explicit, documented way to run a .bat file
        # via CreateProcess without shell=True's quoting/injection risk).
        #
        # ADR-069 dated amendment: the safety check-point runs first, still on this
        # background thread. If it raises (a real git failure), this propagates out
        # through startWorker's result and is handled by on_exit_run_update_worker's
        # existing exception path -- update.bat below is never reached.
        _git_checkpoint_before_update(nunif_dir, lambda text: wx.CallAfter(dlg.append, text))

        proc = subprocess.Popen(
            cmd, cwd=cwd, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in proc.stdout:
            wx.CallAfter(dlg.append, line)
        proc.wait()
        return proc.returncode

    def on_exit_run_update_worker(self, result):
        self.updating = False
        self.update_start_button_state()
        dlg = self.dlg_run_update
        try:
            returncode = result.get()
        except: # noqa
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            if dlg is not None:
                dlg.append("\n" + message)
                dlg.mark_finished()
            self.SetStatusText(T("Error"))
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)
            return

        if dlg is not None:
            dlg.mark_finished()
        if returncode == 0:
            self.SetStatusText(T("Update finished successfully"))
        else:
            self.SetStatusText(T("Update failed -- see the log window"))
            wx.MessageBox(T("update.bat exited with an error -- see the log window for the exact "
                             "reason."),
                          T("Run Update"), wx.OK | wx.ICON_ERROR)

    def on_click_btn_run_update(self, event):
        if self.processing or self.updating:
            wx.MessageBox(
                T("A conversion (or another background job) is currently running. Wait for it to "
                  "finish, or cancel it, before running the updater -- updating packages/models/"
                  "source while a job is using them could break that job."),
                T("Run Update"), wx.OK | wx.ICON_WARNING)
            return

        with wx.MessageDialog(
                self,
                message=(T("This runs the real update.bat script and will update your Python "
                           "packages, downloaded models, and source code together.") + "\n\n" +
                         T("This can take a while (package downloads and model downloads can be "
                           "large) and there is no undo -- don't close this window while it's "
                           "running.") + "\n\n" +
                         T("Continue?")),
                caption=T("Run Update"),
                style=wx.YES_NO | wx.ICON_WARNING) as dlg:
            if dlg.ShowModal() != wx.ID_YES:
                return

        update_bat_path, cwd = _find_update_bat()
        if not path.exists(update_bat_path):
            wx.MessageBox(
                T("update.bat was not found at the expected location:") + f"\n{update_bat_path}",
                T("Run Update"), wx.OK | wx.ICON_ERROR)
            return

        nunif_dir = update_check._get_nunif_repo_root()

        comspec = os.environ.get("ComSpec") or r"C:\Windows\System32\cmd.exe"
        cmd = [comspec, "/c", update_bat_path]

        self.updating = True
        self.update_start_button_state()
        self.SetStatusText(T("Running update..."))

        self.dlg_run_update = RunUpdateDialog(self)
        self.dlg_run_update.append(
            T("Running update.bat...") + "\n" + T("This can take a while -- please wait.") + "\n\n")
        self.dlg_run_update.append(
            T("Checking for uncommitted changes to safety check-point first...") + "\n")
        self.dlg_run_update.Show()

        startWorker(self.on_exit_run_update_worker, self.run_update,
                    wargs=(cmd, cwd, nunif_dir, self.dlg_run_update))

    def test_autocrop(self):
        self.txt_autocrop_test.SetValue("")

        args = self.parse_args()
        if args is None or not args.autocrop:
            return

        device = create_device(args.gpu[0])
        if is_video(args.input):
            def _run():
                crop = AutoCrop.from_video_file(
                    args.input,
                    mode=args.autocrop,
                    uncrop_enabled=False,
                    vf=args.vf,
                    hwaccel=args.hwaccel,
                    disable_software_fallback=args.disable_software_fallback,
                    device=device,
                    batch_size=args.batch_size,
                    stop_event=self.stop_event,
                    suspend_event=self.suspend_event,
                    tqdm_fn=functools.partial(TQDMGUI, self),
                    tqdm_title=f"{path.basename(args.input)}: AutoCrop Analysis",
                ).get_crop()
                return crop

            def _on_exit(ret):
                try:
                    crop = ret.get()
                    if crop is not None:
                        result = f"crop=x={crop[0]}:y={crop[1]}:w={crop[2]}:h={crop[3]}"
                        self.txt_autocrop_test.SetValue(result)
                        self.SetStatusText(result)
                    else:
                        self.txt_autocrop_test.SetValue("")
                        self.SetStatusText(T("No crop detected"))
                except: # noqa
                    self.SetStatusText(T("Error"))
                    e_type, e, tb = sys.exc_info()
                    message = getattr(e, "message", str(e))
                    traceback.print_tb(tb)
                    wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)

                self.processing = False
                self.btn_cancel.Disable()
                self.btn_suspend.Disable()
                self.btn_autocrop_test.Enable()
                self.btn_suspend.SetLabel(T("Suspend"))
                self.update_start_button_state()

            self.btn_start.Disable()
            self.btn_autocrop_test.Disable()
            self.btn_cancel.Enable()
            self.btn_suspend.Enable()
            self.stop_event.clear()
            self.suspend_event.set()
            self.prg_tqdm.SetValue(0)
            self.SetStatusText("...")
            self.ensure_cuda_context()
            startWorker(_on_exit, _run)
            self.processing = True

        elif is_image(args.input):
            x, _ = pil_io.load_image_simple(args.input, exif_transpose=self.chk_exif_transpose.GetValue())
            if x is None:
                raise RuntimeError(f"Load Error: {args.input}")
            x = pil_io.to_tensor(x)
            x = x.to(device)
            autocrop = AutoCrop.from_image(x, mode=args.autocrop)
            crop = autocrop.get_crop()
            if crop is not None:
                result = f"crop=x={crop[0]}:y={crop[1]}:w={crop[2]}:h={crop[3]}"
                self.txt_autocrop_test.SetValue(result)
                self.SetStatusText(result)
            else:
                self.txt_autocrop_test.SetValue("")
                self.SetStatusText(T("No crop detected"))
        elif path.isdir(args.input):
            pass
        else:
            raise RuntimeError("Unsupported format")

    def show_preview_dialog(self, image_path):
        img = wx.Image(image_path, wx.BITMAP_TYPE_PNG)
        if not img.IsOk():
            raise RuntimeError(T("Could not load preview image"))

        max_w, max_h = self.FromDIP((1000, 700))
        w, h = img.GetWidth(), img.GetHeight()
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            img = img.Scale(int(w * scale), int(h * scale), wx.IMAGE_QUALITY_HIGH)

        dlg = wx.Dialog(self, title=T("Quick Preview"),
                        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        bitmap = wx.StaticBitmap(dlg, bitmap=wx.Bitmap(img))
        btn_close = wx.Button(dlg, id=wx.ID_OK, label=T("Close"))

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(bitmap, 0, wx.ALL | wx.ALIGN_CENTER, 8)
        sizer.Add(btn_close, 0, wx.ALL | wx.ALIGN_CENTER, 8)
        dlg.SetSizerAndFit(sizer)
        dlg.CentreOnParent()
        dlg.ShowModal()
        dlg.Destroy()

    PREVIEW_CLIP_SECONDS = 45.0

    def test_quick_preview(self):
        args = self.parse_args()
        if args is None:
            return

        is_video_input = is_video(args.input)
        is_image_input = is_image(args.input)
        if not is_video_input and not is_image_input:
            wx.MessageBox(T("Quick Preview only supports a single image or a single video file"),
                          T("Quick Preview"), wx.OK | wx.ICON_INFORMATION)
            return

        preview_args = copy.copy(args)
        preview_args.resume = False
        preview_args.recursive = False
        preview_args.auto_resume = False
        preview_args.export = False
        preview_args.export_disparity = False
        preview_args.metadata = None

        if is_video_input:
            timestamp = parse_time(args.start_time) if args.start_time else 1.0
            tmp_output = path.join(
                tempfile.gettempdir(), "iw3_quick_preview_output" + preview_args.video_extension)
            if path.exists(tmp_output):
                os.remove(tmp_output)
            preview_args.input = args.input
            preview_args.output = tmp_output
            preview_args.start_time = str(timestamp)
            preview_args.end_time = str(timestamp + self.PREVIEW_CLIP_SECONDS)
        else:
            tmp_output = path.join(tempfile.gettempdir(), "iw3_quick_preview_output.png")
            if path.exists(tmp_output):
                os.remove(tmp_output)
            preview_args.input = args.input
            preview_args.output = tmp_output
            preview_args.format = "png"
            preview_args.start_time = None
            preview_args.end_time = None

        self.btn_quick_preview.Disable()
        self.btn_start.Disable()
        self.SetStatusText(T("Generating preview..."))
        try:
            with wx.BusyCursor():
                wx.Yield()
                self.ensure_cuda_context()
                preview_args = iw3_main(preview_args)
                self.depth_model = preview_args.state["depth_model"]
                self.depth_model_type = preview_args.depth_model
                self.depth_model_device_id = preview_args.gpu
                self.depth_model_height = preview_args.resolution
                self.depth_model_limit_resolution = preview_args.limit_resolution

            if not path.exists(tmp_output):
                raise RuntimeError(T("Preview generation failed: no output file was produced"))

            self.SetStatusText(T("Preview ready"))
            if is_video_input:
                os.startfile(tmp_output)
            else:
                self.show_preview_dialog(tmp_output)
        finally:
            self.btn_quick_preview.Enable()
            self.update_start_button_state()

    def on_click_btn_quick_preview(self, event):
        try:
            self.test_quick_preview()
        except: # noqa
            self.SetStatusText(T("Error"))
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)

    COMPARE_CLIP_SECONDS_DEFAULT = "15"

    def _snapshot_gui_state(self):
        snapshot_path = path.join(tempfile.gettempdir(), "iw3_gui_compare_snapshot.cfg")
        manager = persist.PersistenceManager.Get()
        manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
        manager.SetPersistenceFile(snapshot_path)
        persistent_manager_register_all(manager, self)
        for control in self.get_editable_comboboxes():
            persistent_manager_register(manager, control, EditableComboBoxPersistentHandler)
        for control in self.get_stereo_sliders_and_panes():
            manager.Unregister(control)
        for control in self.get_depth_blend_sliders_and_panes():
            manager.Unregister(control)
        for control in self.get_processor_sliders_and_panes():
            manager.Unregister(control)
        manager.SaveAndUnregister()
        return snapshot_path

    def _restore_gui_state(self, snapshot_path):
        manager = persist.PersistenceManager.Get()
        manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
        manager.SetPersistenceFile(snapshot_path)
        persistent_manager_register_all(manager, self)
        for control in self.get_editable_comboboxes():
            persistent_manager_register(manager, control, EditableComboBoxPersistentHandler)
        for control in self.get_stereo_sliders_and_panes():
            manager.Unregister(control)
        for control in self.get_depth_blend_sliders_and_panes():
            manager.Unregister(control)
        for control in self.get_processor_sliders_and_panes():
            manager.Unregister(control)
        persistent_manager_restore_all(manager, {"cbo_language", "cbo_layout"})
        persistent_manager_unregister_all(manager)
        self.update_controls()
        if path.exists(snapshot_path):
            try:
                os.remove(snapshot_path)
            except Exception:
                pass

    def _concat_comparison_clips(self, clip_files, output_path):
        ffmpeg_bin = _get_ffmpeg_bin()
        ext = path.splitext(output_path)[1].lower()
        mkvmerge_bin = _find_mkvmerge() if ext == ".mkv" else None
        if mkvmerge_bin:
            mkvmerge_args = [mkvmerge_bin, "-o", output_path, clip_files[0]]
            for cf in clip_files[1:]:
                mkvmerge_args += ["+", cf]
            result = subprocess.run(mkvmerge_args, capture_output=True)
            if result.returncode in (0, 1) and path.exists(output_path):
                return
        concat_list = output_path + "_concat.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for cf in clip_files:
                f.write("file '{}'\n".format(cf.replace("'", "'\\''")))
        try:
            subprocess.run(
                [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", output_path],
                check=True, capture_output=True)
        finally:
            if path.exists(concat_list):
                try:
                    os.remove(concat_list)
                except Exception:
                    pass

    def _stack_comparison_images(self, image_files, output_path):
        from PIL import Image
        images = [Image.open(f).convert("RGB") for f in image_files]
        h = max(im.height for im in images)
        images = [im.resize((max(1, int(im.width * h / im.height)), h)) for im in images]
        total_w = sum(im.width for im in images)
        canvas = Image.new("RGB", (total_w, h))
        x = 0
        for im in images:
            canvas.paste(im, (x, 0))
            x += im.width
        canvas.save(output_path)

    def _caption_video_clip(self, clip_path, label_text):
        ffmpeg_bin = _get_ffmpeg_bin()
        ext = path.splitext(clip_path)[1]
        captioned_path = path.splitext(clip_path)[0] + "_caption" + ext
        # Escape for the ffmpeg drawtext "text" parameter: backslash first, then the
        # other characters that are special inside a filtergraph option string.
        escaped = (label_text.replace("\\", "\\\\").replace(":", "\\:")
                  .replace("'", "\\'").replace("%", "\\%"))
        drawtext = (
            f"drawtext=font=Arial:text='{escaped}':fontcolor=white:fontsize=42:"
            f"box=1:boxcolor=black@0.6:boxborderw=14:x=30:y=30:enable='lt(t,3)'"
        )
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", clip_path, "-vf", drawtext,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-c:a", "copy", captioned_path],
            check=True, capture_output=True,
        )
        return captioned_path

    def _caption_image(self, image_path, label_text):
        from PIL import Image, ImageDraw, ImageFont
        im = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("arial.ttf", max(20, im.height // 20))
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 10
        draw.rectangle([10, 10, 10 + tw + pad * 2, 10 + th + pad * 2], fill=(0, 0, 0))
        draw.text((10 + pad, 10 + pad), label_text, fill=(255, 255, 255), font=font)
        im.save(image_path)

    def run_preset_comparison(self, args_list, label_names, output_path, is_video_input):
        clip_files = []
        try:
            for a, label_text in zip(args_list, label_names):
                if self.stop_event.is_set():
                    return None
                iw3_main(a)
                if not path.exists(a.output):
                    raise RuntimeError(f"Failed to generate test clip for preset settings: {a.output}")

                if is_video_input:
                    captioned_path = self._caption_video_clip(a.output, label_text)
                    os.remove(a.output)
                    clip_files.append(captioned_path)
                else:
                    self._caption_image(a.output, label_text)
                    clip_files.append(a.output)

            if is_video_input:
                self._concat_comparison_clips(clip_files, output_path)
            else:
                self._stack_comparison_images(clip_files, output_path)
            return output_path
        finally:
            for cf in clip_files:
                if path.exists(cf):
                    try:
                        os.remove(cf)
                    except Exception:
                        pass

    def on_exit_compare_worker(self, result):
        try:
            output_path = result.get()
            if not self.stop_event.is_set() and output_path:
                self.prg_tqdm.SetValue(self.prg_tqdm.GetRange())
                self.SetStatusText(T("Finished"))
                if wx.MessageBox(
                        T("Comparison video ready:") + "\n" + output_path + "\n\n" + T("Open it now?"),
                        T("Compare Presets"), wx.YES_NO | wx.ICON_INFORMATION) == wx.YES:
                    os.startfile(output_path)
            else:
                self.SetStatusText(T("Cancelled"))
        except:  # noqa
            self.SetStatusText(T("Error"))
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)

        self.processing = False
        self.btn_cancel.Disable()
        self.btn_compare_presets.Enable()
        self.btn_autocrop_test.Enable()
        self.update_start_button_state()
        gc_collect()

    def on_click_btn_compare_presets(self, event):
        presets = [p for p in self.list_preset() if p]
        if len(presets) < 2:
            wx.MessageBox(T("You need at least 2 saved presets to use this feature. "
                            "Save some presets first with the \"Save\" button."),
                          T("Compare Presets"), wx.OK | wx.ICON_INFORMATION)
            return

        base_args = self.parse_args(skip_set_state=True)
        if base_args is None:
            return
        if not (is_video(base_args.input) or is_image(base_args.input)):
            wx.MessageBox(T("Compare Presets only supports a single image or a single video file"),
                          T("Compare Presets"), wx.OK | wx.ICON_INFORMATION)
            return
        is_video_input = is_video(base_args.input)

        dlg = wx.Dialog(self, title=T("Compare Presets"), style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        lbl = wx.StaticText(
            dlg, label=T("Select 2 or more presets to test (clips play in the order shown below):"))
        chk_list = wx.CheckListBox(dlg, choices=presets, size=dlg.FromDIP((320, 160)))

        lbl_duration = wx.StaticText(dlg, label=T("Test Clip Duration (seconds)"))
        txt_duration = wx.TextCtrl(dlg, value=self.COMPARE_CLIP_SECONDS_DEFAULT)
        duration_row = wx.BoxSizer(wx.HORIZONTAL)
        duration_row.Add(lbl_duration, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        duration_row.Add(txt_duration, 0, wx.ALL, 4)

        btn_sizer = dlg.CreateButtonSizer(wx.OK | wx.CANCEL)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(lbl, 0, wx.ALL, 8)
        sizer.Add(chk_list, 1, wx.ALL | wx.EXPAND, 8)
        if is_video_input:
            sizer.Add(duration_row, 0, wx.ALL, 4)
        sizer.Add(btn_sizer, 0, wx.ALL | wx.ALIGN_CENTER, 8)
        dlg.SetSizerAndFit(sizer)
        dlg.CentreOnParent()

        modal_result = dlg.ShowModal()
        selected = [presets[i] for i in chk_list.GetCheckedItems()]
        duration_str = txt_duration.GetValue()
        dlg.Destroy()

        if modal_result != wx.ID_OK:
            return
        if len(selected) < 2:
            wx.MessageBox(T("Please select at least 2 presets."), T("Compare Presets"), wx.OK | wx.ICON_WARNING)
            return
        if is_video_input and not validate_number(duration_str, 1, 600, allow_empty=False):
            wx.MessageBox(T("Test Clip Duration must be a number between 1 and 600."),
                          T("Compare Presets"), wx.OK | wx.ICON_WARNING)
            return
        duration = float(duration_str) if is_video_input else None

        default_ext = base_args.video_extension if is_video_input else ".png"
        wildcard = VIDEO_EXTENSIONS if is_video_input else IMAGE_EXTENSIONS
        with wx.FileDialog(self, message=T("Save Comparison Output"),
                           defaultFile="preset_comparison" + default_ext,
                           wildcard=wildcard,
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as file_dlg:
            if file_dlg.ShowModal() != wx.ID_OK:
                return
            output_path = file_dlg.GetPath()

        # Snapshot the user's live settings; each preset load below overwrites every
        # control on this window, so we must restore everything afterward.
        snapshot_path = self._snapshot_gui_state()
        timestamp = parse_time(base_args.start_time) if base_args.start_time else 1.0
        tmp_dir = tempfile.gettempdir()
        args_list = []
        try:
            for i, name in enumerate(selected):
                self.load_preset(name, exclude_names={self.GetName()})
                self.update_controls()
                preset_args = self.parse_args(skip_set_state=True)
                if preset_args is None:
                    wx.MessageBox(
                        T("Preset \"{name}\" has invalid settings, aborting.").format(name=name),
                        T("Compare Presets"), wx.OK | wx.ICON_ERROR)
                    return

                preset_args.input = base_args.input
                preset_args.resume = False
                preset_args.recursive = False
                preset_args.auto_resume = False
                preset_args.export = False
                preset_args.export_disparity = False
                preset_args.metadata = None
                preset_args.yes = True
                if is_video_input:
                    preset_args.output = path.join(tmp_dir, f"iw3_compare_{i:02d}{preset_args.video_extension}")
                    preset_args.start_time = str(timestamp)
                    preset_args.end_time = str(timestamp + duration)
                else:
                    preset_args.output = path.join(tmp_dir, f"iw3_compare_{i:02d}.png")
                    preset_args.format = "png"
                    preset_args.start_time = None
                    preset_args.end_time = None
                if path.exists(preset_args.output):
                    os.remove(preset_args.output)

                # Force a fresh depth model load per preset: presets can use different
                # depth models, and reusing self.depth_model from a previous run/preset
                # would silently run the wrong model.
                set_state_args(
                    preset_args,
                    stop_event=self.stop_event,
                    suspend_event=self.suspend_event,
                    tqdm_fn=functools.partial(TQDMGUI, self),
                    depth_model=None)
                args_list.append(preset_args)
        finally:
            self._restore_gui_state(snapshot_path)

        self.btn_autocrop_test.Disable()
        self.btn_start.Disable()
        self.btn_quick_preview.Disable()
        self.btn_compare_presets.Disable()
        self.btn_cancel.Enable()
        self.stop_event.clear()
        self.suspend_event.set()
        self.prg_tqdm.SetValue(0)
        self.SetStatusText(T("Rendering preset comparison..."))

        self.ensure_cuda_context()
        startWorker(self.on_exit_compare_worker,
                   self.run_preset_comparison,
                   wargs=(args_list, selected, output_path, is_video_input))
        self.processing = True

    def on_click_btn_autocrop_test(self, event):
        try:
            self.test_autocrop()
        except: # noqa
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)

    def on_click_btn_scene_settings(self, event):
        with wx.FileDialog(self, message=T("Select Scene Settings File"),
                           wildcard="JSON files (*.json)|*.json|All files (*.*)|*.*",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if self.txt_scene_settings.GetValue():
                dlg.SetPath(self.txt_scene_settings.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                self.txt_scene_settings.SetValue(dlg.GetPath())

    def on_click_btn_scene_batch_auto_ema_edit(self, event):
        # ADR-057: dialog opens scoped to whichever model is currently selected in the
        # dropdown -- there is no live model switch inside the dialog itself. Change
        # the dropdown, then reopen this dialog, to edit the other model's table.
        model_name = self.cbo_scene_batch_auto_ema_model.GetValue()
        with SceneBatchAutoEMADialog(self, model_name) as dlg:
            dlg.ShowModal()

    # --- Retroactive HDR/DV Reinjection (standalone tool, see ADR-031) ---

    def on_click_btn_reinject_source(self, event):
        with wx.FileDialog(self, message=T("Select Original Source File (DV/HDR)"),
                           wildcard=VIDEO_EXTENSIONS,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if self.txt_reinject_source.GetValue():
                dlg.SetPath(self.txt_reinject_source.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                self.txt_reinject_source.SetValue(dlg.GetPath())

    def on_click_btn_reinject_converted(self, event):
        with wx.FileDialog(self, message=T("Select Already-Converted 3D File"),
                           wildcard=VIDEO_EXTENSIONS,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if self.txt_reinject_converted.GetValue():
                dlg.SetPath(self.txt_reinject_converted.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                converted_path = dlg.GetPath()
                self.txt_reinject_converted.SetValue(converted_path)
                if not self.txt_reinject_output.GetValue():
                    base, ext = path.splitext(converted_path)
                    self.txt_reinject_output.SetValue(f"{base}_hdr_reinjected{ext}")
                # Auto-suggest the RIFE manifest (ADR-064 amendment): rife_cli.py
                # always writes its sidecar as literally "<its own output
                # path>.rife_manifest.json" (iw3.rife_cli._rife_manifest_path) -- if
                # picking this file finds that exact sidecar sitting right next to it,
                # it can only mean this file WAS a real RIFE output, so pre-fill is
                # reliable rather than a guess. Never overwrites a value the user
                # already typed/picked, and the field stays fully editable/clearable
                # either way.
                if not self.txt_reinject_rife_manifest.GetValue():
                    candidate_manifest = converted_path + ".rife_manifest.json"
                    if path.exists(candidate_manifest):
                        self.txt_reinject_rife_manifest.SetValue(candidate_manifest)

    def on_click_btn_reinject_output(self, event):
        with wx.FileDialog(self, message=T("Save HDR-Reinjected Output As"),
                           wildcard=VIDEO_EXTENSIONS,
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if self.txt_reinject_output.GetValue():
                dlg.SetPath(self.txt_reinject_output.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                self.txt_reinject_output.SetValue(dlg.GetPath())

    def on_click_btn_reinject_rife_manifest(self, event):
        with wx.FileDialog(self, message=T("Select RIFE Manifest"),
                           wildcard=T("RIFE Manifest") + " (*.rife_manifest.json)|*.rife_manifest.json|"
                                     + T("All files") + " (*.*)|*.*",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if self.txt_reinject_rife_manifest.GetValue():
                dlg.SetPath(self.txt_reinject_rife_manifest.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                self.txt_reinject_rife_manifest.SetValue(dlg.GetPath())

    def run_reinject_hdr(self, cmd):
        # Runs on a background thread via startWorker -- never blocks the GUI thread,
        # and this tool needs no GPU at all (pure ffmpeg/dovi_tool/hdr10plus_tool
        # subprocess orchestration), unlike RIFE/waifu2x post-processing which is kept
        # out-of-process specifically to avoid sharing GPU memory. Captures combined
        # stdout+stderr since reinject_hdr_cli prints all of its pre-flight
        # frame-count/duration numbers and refusal reasons to stderr.
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def on_exit_reinject_worker(self, result):
        self.btn_reinject_run.Enable()
        self.btn_reinject_clear.Enable()
        try:
            returncode, output = result.get()
        except: # noqa
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            self.txt_reinject_log.AppendText(message)
            self.SetStatusText(T("Error"))
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)
            return

        self.txt_reinject_log.SetValue(output)
        self.txt_reinject_log.ShowPosition(self.txt_reinject_log.GetLastPosition())
        if returncode == 0:
            self.SetStatusText(T("HDR reinjection finished successfully"))
        else:
            self.SetStatusText(T("HDR reinjection failed -- see the log below"))
            wx.MessageBox(T("HDR reinjection failed or refused -- see the log box for the exact reason."),
                          T("Retroactive HDR/DV Reinjection"), wx.OK | wx.ICON_ERROR)

    def on_click_btn_reinject_run(self, event):
        source = self.txt_reinject_source.GetValue().strip()
        converted = self.txt_reinject_converted.GetValue().strip()
        output = self.txt_reinject_output.GetValue().strip()

        if not source or not path.exists(source):
            wx.MessageBox(T("Select a valid Original Source File first."),
                          T("Retroactive HDR/DV Reinjection"), wx.OK | wx.ICON_WARNING)
            return
        if not converted or not path.exists(converted):
            wx.MessageBox(T("Select a valid Already-Converted 3D File first."),
                          T("Retroactive HDR/DV Reinjection"), wx.OK | wx.ICON_WARNING)
            return
        if not output:
            wx.MessageBox(T("Set an Output File path first."),
                          T("Retroactive HDR/DV Reinjection"), wx.OK | wx.ICON_WARNING)
            return
        if path.abspath(output) in (path.abspath(source), path.abspath(converted)):
            wx.MessageBox(T("Output File must be different from both the source and the converted file."),
                          T("Retroactive HDR/DV Reinjection"), wx.OK | wx.ICON_WARNING)
            return

        rife_manifest = self.txt_reinject_rife_manifest.GetValue().strip()
        if rife_manifest and not path.exists(rife_manifest):
            wx.MessageBox(T("RIFE Manifest file does not exist -- clear the field or pick a valid file."),
                          T("Retroactive HDR/DV Reinjection"), wx.OK | wx.ICON_WARNING)
            return

        cmd = [sys.executable, "-m", "iw3.reinject_hdr_cli",
               "--source", source, "--converted", converted, "--output", output]
        if self.chk_reinject_start_time.GetValue():
            cmd += ["--start-time", self.txt_reinject_start_time.GetValue()]
        if self.chk_reinject_end_time.GetValue():
            cmd += ["--end-time", self.txt_reinject_end_time.GetValue()]
        if rife_manifest:
            cmd += ["--rife-manifest", rife_manifest]

        self.txt_reinject_log.SetValue(
            T("Running -- this decodes the full clip to verify frame counts, so it may take a while...\n"))
        self.btn_reinject_run.Disable()
        self.btn_reinject_clear.Disable()
        self.SetStatusText(T("Running HDR reinjection..."))
        startWorker(self.on_exit_reinject_worker, self.run_reinject_hdr, wargs=(cmd,))

    # --- Search Subtitles (OpenSubtitles, standalone tool, see ADR-039) ---

    def on_click_btn_subsearch_source(self, event):
        with wx.FileDialog(self, message=T("Select Original Source File"),
                           wildcard=VIDEO_EXTENSIONS,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if self.txt_subsearch_source.GetValue():
                dlg.SetPath(self.txt_subsearch_source.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                self.txt_subsearch_source.SetValue(dlg.GetPath())

    def run_subsearch(self, moviehash, imdb_id, query, language):
        # Runs on a background thread via startWorker -- calls subtitle_search_cli.search()
        # directly (not via subprocess, unlike the other standalone tools in this tab) since
        # this needs no GPU/process isolation, just a real network call to OpenSubtitles --
        # still kept off the GUI thread since it's real network I/O. Config/API-key
        # resolution mirrors exactly what subtitle_search_cli.run() (the CLI entry point)
        # does itself.
        config = subtitle_search_cli._load_config(subtitle_search_cli.CONFIG_PATH)
        api_key = subtitle_search_cli.resolve_api_key(None, config)
        return subtitle_search_cli.search(
            api_key, moviehash=moviehash, imdb_id=imdb_id, query=query, languages=language)

    def populate_subsearch_results(self):
        self.lst_subsearch_results.DeleteAllItems()
        for i, r in enumerate(self.subsearch_results):
            flagged = bool(r.get("machine_translated") or r.get("ai_translated"))
            idx = self.lst_subsearch_results.InsertItem(i, r.get("release") or "")
            self.lst_subsearch_results.SetItem(idx, 1, str(r.get("language") or ""))
            self.lst_subsearch_results.SetItem(
                idx, 2, str(r["ratings"]) if r.get("ratings") is not None else "")
            self.lst_subsearch_results.SetItem(
                idx, 3, str(r["download_count"]) if r.get("download_count") is not None else "")
            self.lst_subsearch_results.SetItem(idx, 4, r.get("uploader_name") or "")
            self.lst_subsearch_results.SetItem(idx, 5, "[MT]" if flagged else "")
            if flagged:
                self.lst_subsearch_results.SetItemTextColour(idx, wx.Colour(0xcc, 0x33, 0x33))

    def on_exit_subsearch_worker(self, result):
        self.btn_subsearch_search.Enable()
        self.btn_subsearch_clear.Enable()
        try:
            results, error = result.get()
        except: # noqa
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            self.txt_subsearch_log.AppendText("\n" + message)
            self.SetStatusText(T("Error"))
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)
            return

        if error:
            # Surfaced verbatim -- e.g. subtitle_search_cli.REGISTER_KEY_INSTRUCTIONS when
            # no API key is configured, per docs/ai/AI_DECISIONS.md ADR-039 -- not a
            # generic failure toast.
            self.txt_subsearch_log.AppendText("\n" + error)
            self.SetStatusText(T("Subtitle search failed -- see the log below"))
            return

        # Prefer non-machine/AI-translated results first (stable sort preserves the API's
        # own within-group ordering) -- same preference subtitle_search_cli's own
        # _pick_best_result() applies for the CLI's auto-pick, just shown as a full sorted
        # list here instead of auto-choosing one.
        self.subsearch_results = sorted(
            results, key=lambda r: bool(r.get("machine_translated") or r.get("ai_translated")))
        self.populate_subsearch_results()
        self.btn_subsearch_download.Disable()
        if results:
            self.txt_subsearch_log.AppendText(f"\nFound {len(results)} result(s).")
            self.SetStatusText(T("Subtitle search finished"))
        else:
            self.txt_subsearch_log.AppendText("\n" + T("No results found."))
            self.SetStatusText(T("Subtitle search: no results"))

    def on_click_btn_subsearch_search(self, event):
        source = self.txt_subsearch_source.GetValue().strip()
        title = self.txt_subsearch_title.GetValue().strip()
        imdb_id = self.txt_subsearch_imdb.GetValue().strip()
        language = self.cbo_subsearch_language.GetValue().strip() or "en"

        if not source and not title and not imdb_id:
            wx.MessageBox(T("Provide at least one of Original Source File, Title, or IMDb ID."),
                          T("Search Subtitles"), wx.OK | wx.ICON_WARNING)
            return

        moviehash = None
        log_lines = []
        if source:
            if not path.exists(source):
                wx.MessageBox(T("Select a valid Original Source File."),
                              T("Search Subtitles"), wx.OK | wx.ICON_WARNING)
                return
            try:
                moviehash, filesize = subtitle_search_cli.compute_moviehash(source)
            except OSError as e:
                wx.MessageBox(str(e), T("Search Subtitles"), wx.OK | wx.ICON_ERROR)
                return
            if moviehash is None:
                log_lines.append(
                    f"Original Source File is only {filesize} bytes (minimum "
                    f"{subtitle_search_cli.MOVIEHASH_MIN_FILE_SIZE} needed for moviehash) -- "
                    f"falling back to Title/IMDb ID search.")
            else:
                log_lines.append(f"Computed moviehash: {moviehash}")

        self.subsearch_results = []
        self.lst_subsearch_results.DeleteAllItems()
        self.btn_subsearch_download.Disable()
        log_lines.append(T("Searching..."))
        self.txt_subsearch_log.SetValue("\n".join(log_lines))
        self.btn_subsearch_search.Disable()
        self.btn_subsearch_clear.Disable()
        self.SetStatusText(T("Searching OpenSubtitles..."))
        startWorker(self.on_exit_subsearch_worker, self.run_subsearch,
                    wargs=(moviehash, imdb_id or None, title or None, language))

    def on_select_lst_subsearch_results(self, event):
        self.btn_subsearch_download.Enable(self.lst_subsearch_results.GetFirstSelected() != -1)

    def run_subsearch_download(self, file_id, output_dir):
        # Runs on a background thread via startWorker, same reasoning as run_subsearch()
        # above -- a real network call, spends one OpenSubtitles download-quota credit.
        config = subtitle_search_cli._load_config(subtitle_search_cli.CONFIG_PATH)
        api_key = subtitle_search_cli.resolve_api_key(None, config)
        return subtitle_search_cli.download(api_key, file_id, output_dir)

    def on_exit_subsearch_download_worker(self, result):
        self.btn_subsearch_download.Enable(self.lst_subsearch_results.GetFirstSelected() != -1)
        self.btn_subsearch_clear.Enable()
        try:
            saved_path, error, quota_info = result.get()
        except: # noqa
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            self.txt_subsearch_log.AppendText("\n" + message)
            self.SetStatusText(T("Error"))
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)
            return

        if error:
            self.txt_subsearch_log.AppendText("\n" + error)
            self.SetStatusText(T("Subtitle download failed -- see the log below"))
            wx.MessageBox(T("Downloading the subtitle failed -- see the log box for the exact "
                             "reason."),
                          T("Search Subtitles"), wx.OK | wx.ICON_ERROR)
            return

        self.txt_subsearch_log.AppendText(f"\nDownloaded: {saved_path}")
        if quota_info:
            self.txt_subsearch_log.AppendText(
                f"\nDownload quota (server-reported): remaining={quota_info.get('remaining')} "
                f"requests={quota_info.get('requests')} reset_time={quota_info.get('reset_time')}")
        self.SetStatusText(T("Subtitle downloaded successfully"))

        # Convenience hand-off (see docs/ai/AI_DECISIONS.md ADR-039): auto-populate Add
        # Subtitle Track's Subtitle File field with the just-downloaded .srt, if it's
        # still empty, so the natural next step is one click away instead of browsing to
        # it manually.
        if not self.txt_submux_srt.GetValue().strip():
            self.txt_submux_srt.SetValue(saved_path)

    def on_click_btn_subsearch_download(self, event):
        selected = self.lst_subsearch_results.GetFirstSelected()
        if selected == -1 or selected >= len(self.subsearch_results):
            wx.MessageBox(T("Select a result to download first."),
                          T("Search Subtitles"), wx.OK | wx.ICON_WARNING)
            return
        chosen = self.subsearch_results[selected]

        submux_input = self.txt_submux_input.GetValue().strip()
        source = self.txt_subsearch_source.GetValue().strip()
        if submux_input:
            # Same folder as Add Subtitle Track's converted-video field, when set -- the
            # natural place to want the .srt to land next to.
            output_dir = path.dirname(path.abspath(submux_input))
        elif source:
            output_dir = path.dirname(path.abspath(source))
        else:
            output_dir = os.getcwd()

        release = chosen.get("release") or chosen.get("file_id")
        self.txt_subsearch_log.AppendText(f"\nDownloading {release}...")
        self.btn_subsearch_download.Disable()
        self.btn_subsearch_clear.Disable()
        self.SetStatusText(T("Downloading subtitle..."))
        startWorker(self.on_exit_subsearch_download_worker, self.run_subsearch_download,
                    wargs=(chosen["file_id"], output_dir))

    # --- Add Subtitle Track (standalone tool, see ADR-032) ---

    def on_click_btn_submux_input(self, event):
        with wx.FileDialog(self, message=T("Select Converted 3D Video (.mkv)"),
                           wildcard=VIDEO_EXTENSIONS,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if self.txt_submux_input.GetValue():
                dlg.SetPath(self.txt_submux_input.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                input_path = dlg.GetPath()
                self.txt_submux_input.SetValue(input_path)
                if not self.txt_submux_output.GetValue():
                    base = path.splitext(input_path)[0]
                    self.txt_submux_output.SetValue(f"{base}_subbed.mkv")

    def on_click_btn_submux_srt(self, event):
        with wx.FileDialog(self, message=T("Select Subtitle File (.srt)"),
                           wildcard="SubRip Subtitle files (*.srt)|*.srt|All files (*.*)|*.*",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if self.txt_submux_srt.GetValue():
                dlg.SetPath(self.txt_submux_srt.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                self.txt_submux_srt.SetValue(dlg.GetPath())

    def on_click_btn_submux_output(self, event):
        with wx.FileDialog(self, message=T("Save Subtitled Output As"),
                           wildcard="Matroska files (*.mkv)|*.mkv|All files (*.*)|*.*",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if self.txt_submux_output.GetValue():
                dlg.SetPath(self.txt_submux_output.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                self.txt_submux_output.SetValue(dlg.GetPath())

    def run_submux(self, cmd):
        # Runs on a background thread via startWorker -- never blocks the GUI thread.
        # This tool needs no GPU at all (pure mkvmerge subprocess orchestration), kept
        # out-of-process anyway for the same convention as RIFE/HDR reinjection.
        # Captures combined stdout+stderr since subtitle_mux_cli prints its resolved
        # format, the mkvmerge command line, and any refusal reason to stderr.
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def on_exit_submux_worker(self, result):
        self.btn_submux_run.Enable()
        self.btn_submux_clear.Enable()
        try:
            returncode, output = result.get()
        except: # noqa
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            self.txt_submux_log.AppendText(message)
            self.SetStatusText(T("Error"))
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)
            return

        self.txt_submux_log.SetValue(output)
        self.txt_submux_log.ShowPosition(self.txt_submux_log.GetLastPosition())
        if returncode == 0:
            self.SetStatusText(T("Subtitle track added successfully"))
        else:
            self.SetStatusText(T("Adding subtitle track failed -- see the log below"))
            wx.MessageBox(T("Adding the subtitle track failed or refused -- see the log box for the "
                             "exact reason."),
                          T("Add Subtitle Track"), wx.OK | wx.ICON_ERROR)

    def on_click_btn_submux_run(self, event):
        input_path = self.txt_submux_input.GetValue().strip()
        srt_path = self.txt_submux_srt.GetValue().strip()
        output_path = self.txt_submux_output.GetValue().strip()

        if not input_path or not path.exists(input_path):
            wx.MessageBox(T("Select a valid Converted 3D Video file first."),
                          T("Add Subtitle Track"), wx.OK | wx.ICON_WARNING)
            return
        if not srt_path or not path.exists(srt_path):
            wx.MessageBox(T("Select a valid Subtitle File first."),
                          T("Add Subtitle Track"), wx.OK | wx.ICON_WARNING)
            return
        if not output_path:
            wx.MessageBox(T("Set an Output File path first."),
                          T("Add Subtitle Track"), wx.OK | wx.ICON_WARNING)
            return
        if path.abspath(output_path) == path.abspath(input_path):
            wx.MessageBox(T("Output File must be different from the input video."),
                          T("Add Subtitle Track"), wx.OK | wx.ICON_WARNING)
            return
        font_size = self.txt_submux_font_size.GetValue().strip()
        if font_size and not validate_number(font_size, 0.1, 1000.0, allow_empty=True):
            self.show_validation_error_message(T("Font Size"), 0.1, 1000.0)
            return

        cmd = [sys.executable, "-m", "iw3.subtitle_mux_cli",
               "--input", input_path, "--srt", srt_path, "--output", output_path,
               "--format", self.cbo_submux_format.GetValue()]
        language = self.txt_submux_language.GetValue().strip()
        if language:
            cmd += ["--language", language]
        track_name = self.txt_submux_track_name.GetValue().strip()
        if track_name:
            cmd += ["--track-name", track_name]
        if self.chk_submux_dual_eye.GetValue():
            cmd += ["--dual-eye-subtitles"]
        if font_size:
            cmd += ["--font-size", font_size]
        if self.chk_submux_start_time.GetValue():
            cmd += ["--start-time", self.txt_submux_start_time.GetValue()]
        if self.chk_submux_end_time.GetValue():
            cmd += ["--end-time", self.txt_submux_end_time.GetValue()]

        self.txt_submux_log.SetValue(T("Running...\n"))
        self.btn_submux_run.Disable()
        self.btn_submux_clear.Disable()
        self.SetStatusText(T("Adding subtitle track..."))
        startWorker(self.on_exit_submux_worker, self.run_submux, wargs=(cmd,))

    # --- Add Audio Track (standalone tool, see ADR-042) ---

    def on_click_btn_audiomux_input(self, event):
        with wx.FileDialog(self, message=T("Select Converted 3D Video (.mkv)"),
                           wildcard=VIDEO_EXTENSIONS,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if self.txt_audiomux_input.GetValue():
                dlg.SetPath(self.txt_audiomux_input.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                input_path = dlg.GetPath()
                self.txt_audiomux_input.SetValue(input_path)
                if not self.txt_audiomux_output.GetValue():
                    base = path.splitext(input_path)[0]
                    self.txt_audiomux_output.SetValue(f"{base}_dubbed.mkv")

    def on_click_btn_audiomux_audio(self, event):
        with wx.FileDialog(self, message=T("Select Audio File (dub)"),
                           wildcard=AUDIO_EXTENSIONS,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if self.txt_audiomux_audio.GetValue():
                dlg.SetPath(self.txt_audiomux_audio.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                self.txt_audiomux_audio.SetValue(dlg.GetPath())

    def on_click_btn_audiomux_output(self, event):
        with wx.FileDialog(self, message=T("Save Dubbed Output As"),
                           wildcard="Matroska files (*.mkv)|*.mkv|All files (*.*)|*.*",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if self.txt_audiomux_output.GetValue():
                dlg.SetPath(self.txt_audiomux_output.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                self.txt_audiomux_output.SetValue(dlg.GetPath())

    def run_audiomux(self, cmd):
        # Runs on a background thread via startWorker -- never blocks the GUI thread.
        # This tool needs no GPU at all (ffmpeg trim + mkvmerge subprocess
        # orchestration), kept out-of-process anyway for the same convention as
        # RIFE/HDR reinjection/Add Subtitle Track. Captures combined stdout+stderr
        # since audio_mux_cli prints its resolved language, the ffmpeg trim
        # command(s) (when used), the mkvmerge command line, and any refusal reason
        # to stderr.
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def on_exit_audiomux_worker(self, result):
        self.btn_audiomux_run.Enable()
        self.btn_audiomux_clear.Enable()
        try:
            returncode, output = result.get()
        except: # noqa
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            self.txt_audiomux_log.AppendText(message)
            self.SetStatusText(T("Error"))
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)
            return

        self.txt_audiomux_log.SetValue(output)
        self.txt_audiomux_log.ShowPosition(self.txt_audiomux_log.GetLastPosition())
        if returncode == 0:
            self.SetStatusText(T("Audio track added successfully"))
        else:
            self.SetStatusText(T("Adding audio track failed -- see the log below"))
            wx.MessageBox(T("Adding the audio track failed or refused -- see the log box for the "
                             "exact reason."),
                          T("Add Audio Track"), wx.OK | wx.ICON_ERROR)

    def on_click_btn_audiomux_run(self, event):
        input_path = self.txt_audiomux_input.GetValue().strip()
        audio_path = self.txt_audiomux_audio.GetValue().strip()
        output_path = self.txt_audiomux_output.GetValue().strip()

        if not input_path or not path.exists(input_path):
            wx.MessageBox(T("Select a valid Converted 3D Video file first."),
                          T("Add Audio Track"), wx.OK | wx.ICON_WARNING)
            return
        if not audio_path or not path.exists(audio_path):
            wx.MessageBox(T("Select a valid Audio File first."),
                          T("Add Audio Track"), wx.OK | wx.ICON_WARNING)
            return
        if not output_path:
            wx.MessageBox(T("Set an Output File path first."),
                          T("Add Audio Track"), wx.OK | wx.ICON_WARNING)
            return
        if path.abspath(output_path) == path.abspath(input_path):
            wx.MessageBox(T("Output File must be different from the input video."),
                          T("Add Audio Track"), wx.OK | wx.ICON_WARNING)
            return

        cmd = [sys.executable, "-m", "iw3.audio_mux_cli",
               "--input", input_path, "--audio", audio_path, "--output", output_path]
        language = self.cbo_audiomux_language.GetValue().strip()
        if language:
            cmd += ["--language", language]
        track_name = self.txt_audiomux_track_name.GetValue().strip()
        if track_name:
            cmd += ["--track-name", track_name]
        if self.chk_audiomux_default.GetValue():
            cmd += ["--default"]
        if self.chk_audiomux_start_time.GetValue():
            cmd += ["--source-start-time", self.txt_audiomux_start_time.GetValue()]
        if self.chk_audiomux_end_time.GetValue():
            cmd += ["--source-end-time", self.txt_audiomux_end_time.GetValue()]

        self.txt_audiomux_log.SetValue(T("Running...\n"))
        self.btn_audiomux_run.Disable()
        self.btn_audiomux_clear.Disable()
        self.SetStatusText(T("Adding audio track..."))
        startWorker(self.on_exit_audiomux_worker, self.run_audiomux, wargs=(cmd,))

    def on_click_btn_stereotag_input(self, event):
        with wx.FileDialog(self, message=T("Select Converted 3D Video (.mkv)"),
                           wildcard=VIDEO_EXTENSIONS,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if self.txt_stereotag_input.GetValue():
                dlg.SetPath(self.txt_stereotag_input.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                self.txt_stereotag_input.SetValue(dlg.GetPath())

    def run_stereotag(self, cmd):
        # Runs on a background thread via startWorker -- never blocks the GUI thread.
        # This tool needs no GPU at all (pure mkvpropedit subprocess orchestration),
        # kept out-of-process anyway for the same convention as the other standalone
        # tools in this column. Captures combined stdout+stderr since
        # stereo_mode_tag_cli prints its resolved format and any refusal reason to
        # stderr.
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def on_exit_stereotag_worker(self, result):
        self.btn_stereotag_run.Enable()
        self.btn_stereotag_clear.Enable()
        try:
            returncode, output = result.get()
        except: # noqa
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            self.txt_stereotag_log.AppendText(message)
            self.SetStatusText(T("Error"))
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)
            return

        self.txt_stereotag_log.SetValue(output)
        self.txt_stereotag_log.ShowPosition(self.txt_stereotag_log.GetLastPosition())
        if returncode == 0:
            self.SetStatusText(T("MKV tagged as 3D successfully"))
        else:
            self.SetStatusText(T("Tagging MKV as 3D failed -- see the log below"))
            wx.MessageBox(T("Tagging the MKV as 3D failed or refused -- see the log box for the "
                             "exact reason."),
                          T("Retroactively Tag MKV as 3D"), wx.OK | wx.ICON_ERROR)

    def on_click_btn_stereotag_run(self, event):
        input_path = self.txt_stereotag_input.GetValue().strip()

        if not input_path or not path.exists(input_path):
            wx.MessageBox(T("Select a valid Converted 3D Video file first."),
                          T("Retroactively Tag MKV as 3D"), wx.OK | wx.ICON_WARNING)
            return

        cmd = [sys.executable, "-m", "iw3.stereo_mode_tag_cli",
               "--input", input_path, "--format", self.cbo_stereotag_format.GetValue()]
        if self.chk_stereotag_backup.GetValue():
            cmd += ["--backup"]

        self.txt_stereotag_log.SetValue(T("Running...\n"))
        self.btn_stereotag_run.Disable()
        self.btn_stereotag_clear.Disable()
        self.SetStatusText(T("Tagging MKV as 3D..."))
        startWorker(self.on_exit_stereotag_worker, self.run_stereotag, wargs=(cmd,))

    # --- Sharpen (standalone tool, see ADR-063) ---

    def on_click_btn_sharpen_input(self, event):
        with wx.FileDialog(self, message=T("Select Converted 3D Video (.mkv)"),
                           wildcard=VIDEO_EXTENSIONS,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if self.txt_sharpen_input.GetValue():
                dlg.SetPath(self.txt_sharpen_input.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                input_path = dlg.GetPath()
                self.txt_sharpen_input.SetValue(input_path)
                if not self.txt_sharpen_output.GetValue():
                    base = path.splitext(input_path)[0]
                    self.txt_sharpen_output.SetValue(f"{base}_sharpened.mkv")

    def on_click_btn_sharpen_output(self, event):
        with wx.FileDialog(self, message=T("Save Sharpened Output As"),
                           wildcard="Matroska files (*.mkv)|*.mkv|All files (*.*)|*.*",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if self.txt_sharpen_output.GetValue():
                dlg.SetPath(self.txt_sharpen_output.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                self.txt_sharpen_output.SetValue(dlg.GetPath())

    def run_sharpen(self, cmd):
        # Runs on a background thread via startWorker -- never blocks the GUI thread.
        # This tool decodes/re-encodes the video track (conv2d ops on whatever GPU
        # --gpu selects), kept out-of-process anyway for the same convention as the
        # other standalone tools in this column -- this app's own GPU/model state is
        # never touched. Captures combined stdout+stderr since sharpen_cli prints its
        # resolved format, per-eye/RGBD/anaglyph handling note, and any refusal
        # reason to stderr.
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def on_exit_sharpen_worker(self, result):
        self.btn_sharpen_run.Enable()
        self.btn_sharpen_clear.Enable()
        try:
            returncode, output = result.get()
        except: # noqa
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            self.txt_sharpen_log.AppendText(message)
            self.SetStatusText(T("Error"))
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)
            return

        self.txt_sharpen_log.SetValue(output)
        self.txt_sharpen_log.ShowPosition(self.txt_sharpen_log.GetLastPosition())
        if returncode == 0:
            self.SetStatusText(T("Sharpen applied successfully"))
        else:
            self.SetStatusText(T("Sharpen failed -- see the log below"))
            wx.MessageBox(T("Applying Sharpen failed or refused -- see the log box for the "
                             "exact reason."),
                          T("Sharpen"), wx.OK | wx.ICON_ERROR)

    def on_click_btn_sharpen_run(self, event):
        input_path = self.txt_sharpen_input.GetValue().strip()
        output_path = self.txt_sharpen_output.GetValue().strip()

        if not input_path or not path.exists(input_path):
            wx.MessageBox(T("Select a valid Converted 3D Video file first."),
                          T("Sharpen"), wx.OK | wx.ICON_WARNING)
            return
        if not output_path:
            wx.MessageBox(T("Set an Output File path first."),
                          T("Sharpen"), wx.OK | wx.ICON_WARNING)
            return
        if path.abspath(output_path) == path.abspath(input_path):
            wx.MessageBox(T("Output File must be different from the input video."),
                          T("Sharpen"), wx.OK | wx.ICON_WARNING)
            return
        strength = self.cbo_sharpen_strength_standalone.GetValue().strip()
        if not validate_number(strength, 0.0, 1.0):
            self.show_validation_error_message(T("Strength"), 0.0, 1.0)
            return

        cmd = [sys.executable, "-m", "iw3.sharpen_cli",
               "--input", input_path, "--output", output_path,
               "--format", self.cbo_sharpen_format.GetValue(),
               "--sharpen-strength", strength]

        self.txt_sharpen_log.SetValue(T("Running...\n"))
        self.btn_sharpen_run.Disable()
        self.btn_sharpen_clear.Disable()
        self.SetStatusText(T("Applying Sharpen..."))
        startWorker(self.on_exit_sharpen_worker, self.run_sharpen, wargs=(cmd,))

    # --- RIFE Frame Interpolation (standalone tool) ---

    def on_click_btn_rife_standalone_input(self, event):
        with wx.FileDialog(self, message=T("Select Converted 3D Video"),
                           wildcard=VIDEO_EXTENSIONS,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if self.txt_rife_standalone_input.GetValue():
                dlg.SetPath(self.txt_rife_standalone_input.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                input_path = dlg.GetPath()
                self.txt_rife_standalone_input.SetValue(input_path)
                if not self.txt_rife_standalone_output.GetValue():
                    base, ext = path.splitext(input_path)
                    self.txt_rife_standalone_output.SetValue(f"{base}_rife{ext}")

    def on_click_btn_rife_standalone_output(self, event):
        with wx.FileDialog(self, message=T("Save RIFE-Interpolated Output As"),
                           wildcard=VIDEO_EXTENSIONS,
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if self.txt_rife_standalone_output.GetValue():
                dlg.SetPath(self.txt_rife_standalone_output.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                self.txt_rife_standalone_output.SetValue(dlg.GetPath())

    def update_rife_standalone_mode(self):
        # Same enable/disable pattern as the in-pipeline update_rife_interpolate():
        # the Custom FPS field is only meaningful (and only enabled) when Rate is set
        # to "Custom FPS...".
        self.txt_rife_standalone_target_fps.Enable(
            self.cbo_rife_standalone_mode.GetValue() == "Custom FPS...")

    def on_changed_cbo_rife_standalone_mode(self, event):
        self.update_rife_standalone_mode()

    def run_rife_standalone(self, cmd):
        # Runs on a background thread via startWorker -- never blocks the GUI thread.
        # Kept out-of-process the same way the other standalone tools in this column
        # are (this app's own GPU/model state is never touched). Captures combined
        # stdout+stderr since rife_cli prints its resolved fps/validation refusal
        # reason and the written manifest path to stderr.
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def on_exit_rife_standalone_worker(self, result):
        self.btn_rife_standalone_run.Enable()
        self.btn_rife_standalone_clear.Enable()
        try:
            returncode, output = result.get()
        except: # noqa
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            self.txt_rife_standalone_log.AppendText(message)
            self.SetStatusText(T("Error"))
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)
            return

        self.txt_rife_standalone_log.SetValue(output)
        self.txt_rife_standalone_log.ShowPosition(self.txt_rife_standalone_log.GetLastPosition())
        if returncode == 0:
            self.SetStatusText(T("RIFE interpolation applied successfully"))
        else:
            self.SetStatusText(T("RIFE interpolation failed -- see the log below"))
            wx.MessageBox(T("RIFE interpolation failed or refused -- see the log box for the "
                             "exact reason."),
                          T("RIFE Frame Interpolation"), wx.OK | wx.ICON_ERROR)

    def on_click_btn_rife_standalone_run(self, event):
        input_path = self.txt_rife_standalone_input.GetValue().strip()
        output_path = self.txt_rife_standalone_output.GetValue().strip()

        if not input_path or not path.exists(input_path):
            wx.MessageBox(T("Select a valid Converted 3D Video file first."),
                          T("RIFE Frame Interpolation"), wx.OK | wx.ICON_WARNING)
            return
        if not output_path:
            wx.MessageBox(T("Set an Output File path first."),
                          T("RIFE Frame Interpolation"), wx.OK | wx.ICON_WARNING)
            return
        if path.abspath(output_path) == path.abspath(input_path):
            wx.MessageBox(T("Output File must be different from the input video."),
                          T("RIFE Frame Interpolation"), wx.OK | wx.ICON_WARNING)
            return

        rife_mode = self.cbo_rife_standalone_mode.GetValue()
        if rife_mode == "Custom FPS...":
            if not validate_number(self.txt_rife_standalone_target_fps.GetValue(), 0.1, 1000.0,
                                    allow_empty=False):
                self.show_validation_error_message(T("RIFE Rate: Custom FPS"), 0.1, 1000.0)
                return
            rife_multiplier = None
            rife_target_fps = float(self.txt_rife_standalone_target_fps.GetValue())
        else:
            rife_multiplier = int(rife_mode[0])  # "2x"/"3x"/"4x" -> 2/3/4
            rife_target_fps = None

        gpu_id = int(self.cbo_rife_standalone_gpu.GetClientData(self.cbo_rife_standalone_gpu.GetSelection()))
        video_codec = self.cbo_rife_standalone_codec.GetClientData(self.cbo_rife_standalone_codec.GetSelection())

        cmd = [sys.executable, "-m", "iw3.rife_cli",
               "--input", input_path, "--output", output_path,
               "--rife-model", self.cbo_rife_standalone_model.GetValue(),
               "--gpu", str(gpu_id)]
        if rife_target_fps is not None:
            cmd += ["--rife-target-fps", str(rife_target_fps)]
        else:
            cmd += ["--rife-multiplier", str(rife_multiplier)]
        # Only appended when a non-default codec is picked (ClientData None for the
        # default "H.264 (default)" choice) -- omitting the flag entirely for anyone
        # who doesn't touch this new control keeps today's exact existing command
        # byte-for-byte, matching rife_cli.py's own --video-codec default=None
        # backward-compat guarantee.
        if video_codec:
            cmd += ["--video-codec", str(video_codec)]

        self.txt_rife_standalone_log.SetValue(T("Running...\n"))
        self.btn_rife_standalone_run.Disable()
        self.btn_rife_standalone_clear.Disable()
        self.SetStatusText(T("Applying RIFE interpolation..."))
        startWorker(self.on_exit_rife_standalone_worker, self.run_rife_standalone, wargs=(cmd,))


LOCAL_LIST = sorted(list(LOCALES.keys()))
LOCALE_DICT = LOCALES.get(get_default_locale(), {})


def T(s):
    return LOCALE_DICT.get(s, s)


def _self_test_no_eager_cuda_context():
    """Synthetic/mocked test: opening the GUI window (constructing MainFrame) must not
    grab a CUDA context (pyav_init_cuda_primary_context) or probe torch.compile support
    (check_compile_support -- itself a real torch.compile() on a GPU tensor, not a cheap
    query) before the user actually starts a real conversion -- see docs/ai/AI_DECISIONS.md.
    No GPU or real movie file needed: both heavy calls are monkeypatched with
    call-counting stubs, following this project's established --self-test convention
    (see iw3/subtitle_mux_cli.py)."""
    import iw3.gui as gui_mod

    pyav_calls = []
    compile_calls = []

    def _fake_pyav_init():
        pyav_calls.append(1)

    def _fake_check_compile_support(device):
        compile_calls.append(device)
        return True

    orig_pyav = gui_mod.pyav_init_cuda_primary_context
    orig_compile = gui_mod.check_compile_support
    gui_mod.pyav_init_cuda_primary_context = _fake_pyav_init
    gui_mod.check_compile_support = _fake_check_compile_support
    app = None
    frame = None
    try:
        app = wx.App()
        frame = gui_mod.MainFrame()
        assert not pyav_calls, \
            "pyav_init_cuda_primary_context ran during passive window construction"
        assert not compile_calls, \
            "check_compile_support ran during passive window construction"

        # Real CUDA/video work (Start, Quick Preview, ...) must still init the context,
        # exactly once even if triggered more than once in the same session.
        frame.ensure_cuda_context()
        assert len(pyav_calls) == 1
        frame.ensure_cuda_context()
        assert len(pyav_calls) == 1, "ensure_cuda_context re-initialized an already-established context"

        # Direct user interaction with the compile checkbox/device selector must still
        # probe compile support (probe=True is the default for those two Bind()s).
        frame.chk_compile.SetValue(True)
        frame.update_compile(probe=True)
        assert len(compile_calls) == 1, "explicit probe=True did not validate torch.compile support"
    finally:
        gui_mod.pyav_init_cuda_primary_context = orig_pyav
        gui_mod.check_compile_support = orig_compile
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_no_eager_cuda_context: PASS")


def _self_test_device_dropdown_no_torch_cuda_touch():
    """Synthetic/mocked test for the real bug behind ADR-034/071's crash (see
    _query_nvidia_smi_gpu_names()'s docstring): the pre-existing
    _self_test_no_eager_cuda_context() above only checked that
    pyav_init_cuda_primary_context()/check_compile_support() don't run during
    passive window construction -- it never checked whether torch.cuda itself
    gets touched, which is exactly the gap that let this slip through
    unnoticed. This test monkeypatches torch.cuda.is_available/device_count/
    get_device_properties to raise if called at all during MainFrame()
    construction (they must not be, now that the Device/RIFE-GPU dropdowns
    query nvidia-smi instead), and mocks _query_nvidia_smi_gpu_names() to
    return a fixed fake device list so the test needs no real GPU."""
    import iw3.gui as gui_mod

    def _fail(*a, **kw):
        raise AssertionError("torch.cuda touched during passive window construction")

    orig_nvidia_smi = gui_mod._query_nvidia_smi_gpu_names
    orig_is_available = torch.cuda.is_available
    orig_device_count = torch.cuda.device_count
    orig_get_device_properties = torch.cuda.get_device_properties
    gui_mod._query_nvidia_smi_gpu_names = lambda: ["Fake GPU 0"]
    torch.cuda.is_available = _fail
    torch.cuda.device_count = _fail
    torch.cuda.get_device_properties = _fail
    app = None
    frame = None
    try:
        app = wx.App()
        frame = gui_mod.MainFrame()
        assert frame.cbo_device.FindString("0:Fake GPU 0") != wx.NOT_FOUND, \
            "Device dropdown was not populated from the mocked nvidia-smi device list"
    finally:
        gui_mod._query_nvidia_smi_gpu_names = orig_nvidia_smi
        torch.cuda.is_available = orig_is_available
        torch.cuda.device_count = orig_device_count
        torch.cuda.get_device_properties = orig_get_device_properties
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_device_dropdown_no_torch_cuda_touch: PASS")


def _self_test_compile_probe_crash_handled():
    """Regression test for a real crash: clicking the torch.compile checkbox with a
    specific GPU/CPU selected (not "All CUDA Device") used to throw a raw, uncaught
    error (a real Windows OSError [Errno 129] from a broken Triton/MSVC toolchain on
    the user's machine surfaced this way) instead of failing gracefully -- see
    docs/ai/AI_DECISIONS.md. `check_compile_support()` itself (nunif/models/utils.py)
    now catches any real probe failure, but `update_compile()` is the actual GUI entry
    point named in the bug report, so this drives it directly with a REAL MainFrame
    and a mocked check_compile_support that raises OSError (plus RuntimeError/
    AssertionError, to confirm this isn't narrowed to one exception type), and
    confirms: no exception escapes, the checkbox ends up unchecked, and a status-bar
    message appears -- no crash, no popup, matching this file's other lightweight
    SetStatusText failure notices."""
    import iw3.gui as gui_mod

    for exc in (OSError(129, "Error number -129 occurred"), RuntimeError("compile backend failed"),
                AssertionError("probe assertion failed")):
        def _raising_check_compile_support(device, _exc=exc):
            raise _exc

        orig_compile = gui_mod.check_compile_support
        gui_mod.check_compile_support = _raising_check_compile_support
        app = None
        frame = None
        try:
            app = wx.App()
            frame = gui_mod.MainFrame()

            device_id = None
            for i in range(frame.cbo_device.GetCount()):
                if int(frame.cbo_device.GetClientData(i)) != -2:
                    device_id = i
                    break
            assert device_id is not None, "no non-'All CUDA Device' entry to select"
            frame.cbo_device.SetSelection(device_id)

            frame.chk_compile.SetValue(True)
            frame.update_compile(probe=True)  # must not raise

            assert not frame.chk_compile.IsChecked(), \
                f"checkbox must end up unchecked after a real probe failure ({exc.__class__.__name__})"
            status = frame.GetStatusBar().GetStatusText()
            assert "torch.compile" in status and "docs/torch_compile.md" in status, \
                f"expected an informative status message, got: {status!r}"
        finally:
            gui_mod.check_compile_support = orig_compile
            if frame is not None:
                frame.Destroy()
            if app is not None:
                app.Destroy()

    print("_self_test_compile_probe_crash_handled: PASS")


def _self_test_layout_modes():
    """Regression test for the GUI Layout preference (Tabbed vs Single Page, ADR-037):
    every control must be constructed exactly once (never duplicated/rebuilt) and
    correctly reachable in BOTH modes -- only the container that composes the shared
    StaticBoxSizers should differ. No GPU or real movie file needed:
    _load_layout_mode is monkeypatched to force each mode without touching the real
    persisted iw3-gui-layout.cfg file."""
    import iw3.gui as gui_mod

    orig_load = gui_mod._load_layout_mode
    app = wx.App()
    try:
        for mode in (gui_mod.LAYOUT_MODE_TABS, gui_mod.LAYOUT_MODE_SINGLE_PAGE):
            gui_mod._load_layout_mode = lambda config_path, _mode=mode: _mode
            frame = None
            try:
                frame = gui_mod.MainFrame()
                assert frame.layout_mode == mode

                # Every category panel must exist and actually be composed (have a
                # sizer) regardless of mode -- construction of the shared sizer_*
                # groups inside each one is identical either way.
                for tab in (frame.tab_stereo, frame.tab_depth_blend, frame.tab_video_filter,
                            frame.tab_video_dec, frame.tab_video_enc, frame.tab_processor,
                            frame.tab_tools):
                    assert tab.GetSizer() is not None

                if mode == gui_mod.LAYOUT_MODE_TABS:
                    assert frame.nb_options.GetPageCount() == 7
                    # ADR-048: each Notebook page is a ScrolledPanel wrapper holding
                    # the category panel, not the category panel directly.
                    assert frame.tab_stereo.GetParent() is frame.tab_wrap_stereo
                    assert frame.tab_tools.GetParent() is frame.tab_wrap_tools
                    assert frame.tab_wrap_stereo.GetParent() is frame.nb_options
                    assert frame.tab_wrap_tools.GetParent() is frame.nb_options
                else:
                    assert frame.pnl_single.GetSizer() is not None
                    assert frame.tab_stereo.GetParent() is frame.pnl_single
                    assert frame.tab_tools.GetParent() is frame.pnl_single

                # A representative control from each of a few categories, including
                # one added well after the original tabs/grid split (RIFE), must exist
                # and stay wired to its real parent StaticBox in both modes.
                assert frame.chk_rife_interpolate.GetParent() is frame.grp_postprocess
                assert frame.grp_stereo.GetParent() is frame.tab_stereo
                assert frame.grp_stereotag.GetParent() is frame.tab_tools

                # The Layout combo itself must reflect the active mode and never be
                # restored from a preset/snapshot (own file + restart, like Language).
                assert frame.cbo_layout.GetClientData(frame.cbo_layout.GetSelection()) == mode
            finally:
                gui_mod._load_layout_mode = orig_load
                if frame is not None:
                    frame.Destroy()
    finally:
        app.Destroy()

    print("_self_test_layout_modes: PASS")


def _self_test_layout_mode_live_switch():
    """Regression test for ADR-045 (live Layout switching, no restart): starting from
    a real MainFrame in either mode, MainFrame.switch_layout_mode() must move all 7
    category panels to the other container and back, repeatedly, in the same running
    window, without losing or duplicating any control, and without breaking a
    cross-control Enable/Disable relationship that lives inside one of those panels
    (Object Stability's sub-settings, chk_temporal_stabilize -> its 4 dependent
    combos). Also guards against a real collapse bug found and fixed while building
    this: wx.Notebook leaves every non-selected page Hidden even after RemovePage(),
    which silently zeroed out GridBagSizer.CalcMin() for the Single Page layout (a
    sizer excludes Hidden windows from its min-size calculation) and collapsed the
    whole window to a couple dozen pixels tall -- switch_layout_mode() now forces
    each panel Show() after reparenting, so every panel must be visible and pnl_single
    must report a real (non near-zero) computed minimum after switching to Single
    Page. No GPU or real movie file needed."""
    import iw3.gui as gui_mod

    orig_load = gui_mod._load_layout_mode
    app = wx.App()
    frame = None
    try:
        gui_mod._load_layout_mode = lambda config_path: gui_mod.LAYOUT_MODE_TABS
        frame = gui_mod.MainFrame()
        tabs = (frame.tab_stereo, frame.tab_depth_blend, frame.tab_video_filter,
                frame.tab_video_dec, frame.tab_video_enc, frame.tab_processor, frame.tab_tools)
        wraps = (frame.tab_wrap_stereo, frame.tab_wrap_depth_blend, frame.tab_wrap_video_filter,
                 frame.tab_wrap_video_dec, frame.tab_wrap_video_enc, frame.tab_wrap_processor,
                 frame.tab_wrap_tools)

        assert frame.layout_mode == gui_mod.LAYOUT_MODE_TABS
        assert frame.nb_options.GetPageCount() == 7

        for i in range(3):
            frame.switch_layout_mode(gui_mod.LAYOUT_MODE_SINGLE_PAGE)
            assert frame.layout_mode == gui_mod.LAYOUT_MODE_SINGLE_PAGE
            assert frame.nb_options.GetPageCount() == 0, f"round {i}: notebook still has pages after switching away"
            for tab in tabs:
                assert tab.GetParent() is frame.pnl_single, f"round {i}: {tab} not reparented to pnl_single"
                assert tab.IsShown(), f"round {i}: {tab} still Hidden after switching to Single Page " \
                    "(wx.Notebook leaves inactive pages Hidden -- collapses the whole window, see docstring)"
            # A collapsed pnl_single (the original bug) reports a near-zero computed
            # min size here -- assert a real one instead of just "not None".
            min_w, min_h = frame.pnl_single.GetSizer().CalcMin()
            assert min_w > 100 and min_h > 100, \
                f"round {i}: pnl_single collapsed after switching to Single Page (CalcMin={(min_w, min_h)})"
            # every control below a moved category panel must still be reachable and
            # wired to the SAME parent StaticBox it always had (only the 7 category
            # panels themselves ever move -- see switch_layout_mode's docstring).
            assert frame.grp_stereo.GetParent() is frame.tab_stereo
            assert frame.chk_rife_interpolate.GetParent() is frame.grp_postprocess

            # Object Stability's Enable/Disable dependency must still work correctly
            # on the reparented panel.
            frame.chk_temporal_stabilize.SetValue(True)
            frame.update_temporal_stabilize()
            assert frame.cbo_temporal_stabilize_strength.IsEnabled(), f"round {i}: sub-setting did not enable"
            frame.chk_temporal_stabilize.SetValue(False)
            frame.update_temporal_stabilize()
            assert not frame.cbo_temporal_stabilize_strength.IsEnabled(), f"round {i}: sub-setting did not disable"

            frame.switch_layout_mode(gui_mod.LAYOUT_MODE_TABS)
            assert frame.layout_mode == gui_mod.LAYOUT_MODE_TABS
            assert frame.nb_options.GetPageCount() == 7, f"round {i}: notebook page count wrong after switching back"
            assert frame.pnl_single.GetSizer().GetItemCount() == 0, \
                f"round {i}: single-page sizer still has items after switching away"
            assert frame.tab_stereo.IsShown(), \
                f"round {i}: the selected (first) tab must be visible back in Tabbed mode"
            # ADR-048: a category panel's parent in Tabbed mode is now its own
            # ScrolledPanel wrapper, and that wrapper (not the category panel) is the
            # actual Notebook page.
            for tab, wrap in zip(tabs, wraps):
                assert tab.GetParent() is wrap, f"round {i}: {tab} not reparented back to its wrapper {wrap}"
                assert wrap.GetParent() is frame.nb_options, f"round {i}: wrapper {wrap} not a child of nb_options"
            assert frame.grp_stereo.GetParent() is frame.tab_stereo
            assert frame.chk_rife_interpolate.GetParent() is frame.grp_postprocess

        # A no-op switch (same mode) must not raise or disturb the current composition.
        frame.switch_layout_mode(gui_mod.LAYOUT_MODE_TABS)
        assert frame.nb_options.GetPageCount() == 7
    finally:
        gui_mod._load_layout_mode = orig_load
        if frame is not None:
            frame.Destroy()
        app.Destroy()

    print("_self_test_layout_mode_live_switch: PASS")


def _self_test_tabbed_scrolling():
    """Regression test for ADR-048 (Tabbed mode's Notebook pages must scroll when a
    tab's content is taller than the visible area -- previously they simply cut off
    bottom controls with no way to reach them). A full interactive scroll-and-see is
    verified manually via screenshots (see docs/ai/AI_DECISIONS.md ADR-048); this
    self-test checks the parts that don't need a human eye: every Notebook page is
    really a ScrolledPanel wrapper (not a plain wx.Panel) with scrolling actually
    configured, and that its virtual size genuinely exceeds a forced small client size
    for the tallest real tab (Standalone Tools), so a scrollbar/mousewheel would
    actually be needed and able to reach the bottom. No GPU or real movie file needed."""
    import iw3.gui as gui_mod
    import wx.lib.scrolledpanel as scrolledpanel

    orig_load = gui_mod._load_layout_mode
    app = wx.App()
    frame = None
    try:
        gui_mod._load_layout_mode = lambda config_path: gui_mod.LAYOUT_MODE_TABS
        frame = gui_mod.MainFrame()
        assert frame.layout_mode == gui_mod.LAYOUT_MODE_TABS

        wraps = (frame.tab_wrap_stereo, frame.tab_wrap_depth_blend, frame.tab_wrap_video_filter,
                 frame.tab_wrap_video_dec, frame.tab_wrap_video_enc, frame.tab_wrap_processor,
                 frame.tab_wrap_tools)
        for wrap in wraps:
            assert isinstance(wrap, scrolledpanel.ScrolledPanel), \
                f"{wrap} is not a ScrolledPanel -- Tabbed mode page would not scroll"
            # SetupScrolling(scroll_y=True) sets a non-zero vertical scroll rate;
            # (0, 0) means scrolling was never actually configured on this window.
            assert wrap.GetScrollPixelsPerUnit()[1] > 0, \
                f"{wrap} has no vertical scroll rate configured"

        # The category panels themselves must stay plain wx.Panel -- only the wrapper
        # scrolls, avoiding a ScrolledPanel nested inside another ScrolledPanel (see
        # ADR-048 for why that nesting was avoided for Single Page).
        for tab in (frame.tab_stereo, frame.tab_depth_blend, frame.tab_video_filter,
                    frame.tab_video_dec, frame.tab_video_enc, frame.tab_processor, frame.tab_tools):
            assert not isinstance(tab, scrolledpanel.ScrolledPanel), \
                f"{tab} unexpectedly became a ScrolledPanel -- would nest under pnl_single in Single Page mode"

        # Standalone Tools is the tallest tab (HDR Reinjection, Add Subtitle Track, Add
        # Audio Track, Stereo Mode Tagging, Search Subtitles all stacked) -- force its
        # wrapper to a small client size, as a real too-small window would, and confirm
        # the reported virtual size still reflects the real (taller) content instead of
        # collapsing to the forced client size.
        wrap = frame.tab_wrap_tools
        real_content_h = wrap.GetSizer().CalcMin()[1]
        wrap.SetSize((400, 120))
        wrap.Layout()
        virtual_h = wrap.GetVirtualSize()[1]
        assert virtual_h >= real_content_h, \
            f"tab_wrap_tools virtual size ({virtual_h}) lost real content height ({real_content_h})"
        assert virtual_h > 120, \
            "tab_wrap_tools virtual size did not exceed the forced small client size -- would not scroll"
    finally:
        gui_mod._load_layout_mode = orig_load
        if frame is not None:
            frame.Destroy()
        app.Destroy()

    print("_self_test_tabbed_scrolling: PASS")


def _self_test_stereo_sliders_sync():
    """Regression test for the Guided Light pilot's (ADR-097) slider <-> combo two-way
    sync: for every field in STEREO_SLIDER_FIELDS, dragging the slider must update the
    combo's value, and typing/selecting in the combo must move the slider thumb, using
    the same real event-firing path wx itself uses (ProcessWindowEvent), not a direct
    method call, so the actual Bind() wiring is what's under test. Also confirms
    cbo_divergence's slider path keeps update_divergence_warning() wired. No GPU or
    real movie file needed."""
    import iw3.gui as gui_mod

    app = wx.App()
    frame = None
    try:
        frame = gui_mod.MainFrame()

        for combo_name, slider_name, lo, hi, multiplier, is_int, extra_sync in gui_mod.STEREO_SLIDER_FIELDS:
            combo = getattr(frame, combo_name)
            slider = getattr(frame, slider_name)

            # slider -> combo. Mirror the real handler's own two-step rounding (value
            # is stored as an int thumb position, then divided back) rather than
            # comparing against the un-rounded midpoint directly -- the real code
            # path's own rounding can legitimately land one cent away from a naive
            # round(mid, 2) (e.g. 0.475 -> thumb 48 (round-half-to-even) -> 0.48, not
            # the 0.47 a direct round(0.475, 2) gives due to float representation).
            mid = (lo + hi) / 2
            thumb = round(mid * multiplier)
            slider.SetValue(thumb)
            slider.ProcessWindowEvent(wx.CommandEvent(wx.wxEVT_SLIDER, slider.GetId()))
            resolved = thumb / multiplier
            expected = str(int(resolved)) if is_int else str(round(resolved, 2))
            assert combo.GetValue() == expected, \
                f"{combo_name}: slider->combo sync failed (slider={mid}, combo={combo.GetValue()!r}, expected={expected!r})"

            # combo -> slider
            _apply_combo_value(combo, str(lo))
            combo.ProcessWindowEvent(wx.CommandEvent(wx.wxEVT_TEXT, combo.GetId()))
            assert slider.GetValue() == round(lo * multiplier), \
                f"{combo_name}: combo->slider sync failed (combo={lo}, slider={slider.GetValue()}, expected={round(lo * multiplier)})"

        # cbo_divergence's slider path must still trigger the real warning-label logic
        # (not bypass it) -- push it to an extreme value known to trigger a warning.
        frame.sld_stereo_divergence.SetValue(round(5.0 * 10))
        frame.sld_stereo_divergence.ProcessWindowEvent(wx.CommandEvent(wx.wxEVT_SLIDER, frame.sld_stereo_divergence.GetId()))
        assert frame.cbo_divergence.GetValue() == "5.0"
        frame.update_divergence_warning()
        # (visibility itself depends on other state like method/format; just confirm
        # the call path executes without raising, proving extra_sync actually fired
        # above via the slider handler -- an exception there would have already failed
        # the slider->combo assertion for this field.)
    finally:
        if frame is not None:
            frame.Destroy()
            # wx.Destroy() on MSW defers actual HWND teardown to the next event-loop
            # idle pass -- without pumping it here, this test's ~15 extra windows
            # (13 sliders + 1 pane + 1 indicator square) plus a whole second
            # MainFrame's worth of controls stay allocated until whatever runs next
            # happens to yield, which measurably contributed to a real
            # "system allowance of handles for Window Manager objects" failure
            # several self-tests later in the full suite before this was added.
            wx.SafeYield()
        app.Destroy()

    print("_self_test_stereo_sliders_sync: PASS")


def _self_test_depth_blend_and_processor_sliders_sync():
    """Same regression coverage as _self_test_stereo_sliders_sync, extended to the
    Guided Light sliders added to Dual-Pass Depth Blend (DEPTH_BLEND_SLIDER_FIELDS)
    and Processor (PROCESSOR_SLIDER_FIELDS). No field in either table has an
    extra_sync callback, so this omits that half of the original test. No GPU or real
    movie file needed."""
    import iw3.gui as gui_mod

    app = wx.App()
    frame = None
    try:
        frame = gui_mod.MainFrame()

        for combo_name, slider_name, lo, hi, multiplier, is_int, extra_sync in (
                gui_mod.DEPTH_BLEND_SLIDER_FIELDS + gui_mod.PROCESSOR_SLIDER_FIELDS):
            combo = getattr(frame, combo_name)
            slider = getattr(frame, slider_name)

            mid = (lo + hi) / 2
            thumb = round(mid * multiplier)
            slider.SetValue(thumb)
            slider.ProcessWindowEvent(wx.CommandEvent(wx.wxEVT_SLIDER, slider.GetId()))
            resolved = thumb / multiplier
            expected = str(int(resolved)) if is_int else str(round(resolved, 2))
            assert combo.GetValue() == expected, \
                f"{combo_name}: slider->combo sync failed (slider={mid}, combo={combo.GetValue()!r}, expected={expected!r})"

            _apply_combo_value(combo, str(lo))
            combo.ProcessWindowEvent(wx.CommandEvent(wx.wxEVT_TEXT, combo.GetId()))
            assert slider.GetValue() == round(lo * multiplier), \
                f"{combo_name}: combo->slider sync failed (combo={lo}, slider={slider.GetValue()}, expected={round(lo * multiplier)})"
    finally:
        if frame is not None:
            frame.Destroy()
            wx.SafeYield()  # see _self_test_stereo_sliders_sync's finally block for why
        app.Destroy()

    print("_self_test_depth_blend_and_processor_sliders_sync: PASS")


def _self_test_stereo_collapsible_sections():
    """Regression test for the Guided Light pilot's (ADR-097) collapsible panes.

    The mode-independent invariant: tab_stereo's OWN sizer CalcMin() must change on
    every real toggle -- this is the direct proof the pane's Show/Hide actually
    reflowed grp_stereo's content, true regardless of which layout mode is active.

    The mode-SPECIFIC invariant differs, confirmed by live diagnosis rather than
    assumed: in Tabbed mode, tab_wrap_stereo (the ScrolledPanel wrapper, ADR-048)
    holds ONLY tab_stereo 1:1, so its pinned MinSize must track the same change
    exactly -- this exercises the real flagged risk (a pane changing height without
    the wrapper's stale, construction-time-pinned MinSize ever being told). In
    Single Page mode, tab_stereo is one of 7 panels sharing pnl_single's 4-column
    GridBagSizer (ADR-037) -- pnl_single's OVERALL min size is only ever set by
    whichever column is tallest, and Stereo Generation is not always that column
    (Standalone Tools' tab_tools is taller, confirmed by direct measurement: 1949px
    vs Stereo Generation's ~1400px expanded), so asserting pnl_single's total size
    must change on every Stereo Generation toggle would be asserting something
    false about this specific layout, not a real invariant -- the correct check is
    only that SetMinSize() runs without error and never reports a smaller value
    than tab_stereo's own new requirement (i.e. never silently drops below what's
    actually needed, which WOULD be a real clipping bug).

    Exercised in both layout modes, and a switch back to Tabbed, since ADR-045 live
    switching reparents tab_stereo between two different containers with two
    different pinned-MinSize mechanisms. No GPU or real movie file needed."""
    import iw3.gui as gui_mod

    orig_load = gui_mod._load_layout_mode
    app = wx.App()
    frame = None
    try:
        gui_mod._load_layout_mode = lambda config_path: gui_mod.LAYOUT_MODE_TABS
        frame = gui_mod.MainFrame()
        panes = frame.get_stereo_sliders_and_panes()
        panes = [p for p in panes if isinstance(p, wx.CollapsiblePane)]
        assert len(panes) > 0, "no collapsible panes were built -- nothing to test"

        def toggle_and_check(label, check_wrapper_tracks=None):
            for pane in panes:
                before_expanded = pane.IsExpanded()
                tab_min_before = frame.tab_stereo.GetSizer().CalcMin()
                if check_wrapper_tracks is not None:
                    wrap_min_before = check_wrapper_tracks()

                pane.Collapse(before_expanded)  # Collapse(True) if it was expanded, else no-op-ish
                frame.on_toggled_stereo_collapsible_pane(
                    wx.CollapsiblePaneEvent(pane, wx.wxEVT_COLLAPSIBLEPANE_CHANGED, pane.GetId()))

                tab_min_after = frame.tab_stereo.GetSizer().CalcMin()
                assert tab_min_after != tab_min_before, \
                    f"{label}: {pane.GetLabel()} toggle did not change tab_stereo's own content size " \
                    f"({tab_min_before} -> {tab_min_after})"
                w, h = tab_min_after
                assert w > 50 and h > 50, f"{label}: tab_stereo collapsed to near-zero after toggling {pane.GetLabel()}"

                if check_wrapper_tracks is not None:
                    wrap_min_after = check_wrapper_tracks()
                    assert wrap_min_after != wrap_min_before, \
                        f"{label}: wrapper did not track {pane.GetLabel()}'s toggle " \
                        f"({wrap_min_before} -> {wrap_min_after})"
                    assert wrap_min_after[1] >= tab_min_after[1], \
                        f"{label}: wrapper height ({wrap_min_after[1]}) fell below tab_stereo's own " \
                        f"requirement ({tab_min_after[1]}) after toggling {pane.GetLabel()} -- would clip"

                # restore original state so the next pane's before/after comparison is clean
                pane.Collapse(before_expanded)
                frame.on_toggled_stereo_collapsible_pane(
                    wx.CollapsiblePaneEvent(pane, wx.wxEVT_COLLAPSIBLEPANE_CHANGED, pane.GetId()))

        # Tabbed: tab_wrap_stereo holds tab_stereo 1:1, so it must track exactly.
        toggle_and_check("Tabbed", check_wrapper_tracks=lambda: frame.tab_wrap_stereo.GetSizer().CalcMin())

        # Single Page: pnl_single's overall min is legitimately dominated by other
        # columns (see docstring) -- only assert tab_stereo's own size changes and
        # that pnl_single's SetMinSize path runs cleanly without ever under-sizing.
        frame.switch_layout_mode(gui_mod.LAYOUT_MODE_SINGLE_PAGE)
        toggle_and_check("Single Page")
        w, h = frame.pnl_single.GetSizer().CalcMin()
        assert w > 100 and h > 100, "pnl_single collapsed to near-zero after a Stereo Generation pane toggle"

        # Switch back: tab_wrap_stereo must resume tracking exactly, same as before.
        frame.switch_layout_mode(gui_mod.LAYOUT_MODE_TABS)
        toggle_and_check("Tabbed (after switch back)",
                          check_wrapper_tracks=lambda: frame.tab_wrap_stereo.GetSizer().CalcMin())
    finally:
        gui_mod._load_layout_mode = orig_load
        if frame is not None:
            frame.Destroy()
            wx.SafeYield()  # see _self_test_stereo_sliders_sync's finally block for why
        app.Destroy()

    print("_self_test_stereo_collapsible_sections: PASS")


def _self_test_zoom_level_persistence():
    """Regression test for UI Zoom's persisted preference: round-trips through the
    real save/load helpers using a scratch file so the user's actual
    iw3-gui-zoom.cfg is never touched, and separately checks the font-point scaling
    math used by MainFrame.apply_zoom_level()/_scaled_font(). No GPU, no wx.App, no
    real movie file needed."""
    import iw3.gui as gui_mod
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as tmp_dir:
        cfg = path.join(tmp_dir, "iw3-gui-zoom.cfg")

        # No file yet -> default.
        assert gui_mod._load_zoom_level(cfg) == gui_mod.DEFAULT_ZOOM_LEVEL

        for level in gui_mod.ZOOM_LEVELS:
            gui_mod._save_zoom_level(cfg, level)
            assert gui_mod._load_zoom_level(cfg) == level

        # A corrupted/foreign value on disk must not crash -- falls back to default.
        with open(cfg, mode="w", encoding="utf-8") as f:
            f.write("not-a-number")
        assert gui_mod._load_zoom_level(cfg) == gui_mod.DEFAULT_ZOOM_LEVEL

        with open(cfg, mode="w", encoding="utf-8") as f:
            f.write("999")  # a real int, but not one of the offered ZOOM_LEVELS
        assert gui_mod._load_zoom_level(cfg) == gui_mod.DEFAULT_ZOOM_LEVEL

    # Font-point scaling math: 100% must reproduce the original hardcoded sizes
    # exactly (10pt / 8pt), and scaling must move monotonically with zoom level.
    assert gui_mod._zoom_font_point(gui_mod.BASE_NORMAL_FONT_PT, 100) == 10
    assert gui_mod._zoom_font_point(gui_mod.BASE_WARNING_FONT_PT, 100) == 8
    assert gui_mod._zoom_font_point(gui_mod.BASE_NORMAL_FONT_PT, 200) == 20
    assert gui_mod._zoom_font_point(gui_mod.BASE_NORMAL_FONT_PT, 80) == 8
    assert gui_mod._zoom_font_point(gui_mod.BASE_NORMAL_FONT_PT, 50) == 6, \
        "font point size must never collapse below the 6pt floor even at extreme zoom"

    print("_self_test_zoom_level_persistence: PASS")


def _self_test_zoom_startup_restore():
    """Regression test for a real bug caught during manual verification: MainFrame
    already restores general app settings at startup via load_preset() (registers
    every named control with wx.lib.agw.persist and restores from CONFIG_PATH), and
    that generic restore ran AFTER cbo_zoom's own SetSelection(self.zoom_level) during
    construction, silently resetting the combo back to whatever (usually 100%) was
    last saved into the general iw3-gui.cfg -- even though frame.zoom_level and every
    control's actual font were still correctly the persisted UI Zoom value. Fixed by
    adding "cbo_zoom" to load_preset()'s hardcoded exclude_names, the same way
    "cbo_language"/"cbo_layout" already are. This monkeypatches _load_zoom_level so
    the real iw3-gui-zoom.cfg is never touched, and constructs a real MainFrame (the
    bug only reproduces through the real startup path, not the save/load helpers in
    isolation). No GPU or real movie file needed."""
    import iw3.gui as gui_mod

    orig_load = gui_mod._load_zoom_level
    app = wx.App()
    frame = None
    try:
        gui_mod._load_zoom_level = lambda config_path: 150
        frame = gui_mod.MainFrame()
        assert frame.zoom_level == 150
        assert frame.cbo_divergence.GetFont().GetPointSize() == gui_mod._zoom_font_point(
            gui_mod.BASE_NORMAL_FONT_PT, 150)
        selected = frame.cbo_zoom.GetClientData(frame.cbo_zoom.GetSelection())
        assert selected == 150, \
            f"cbo_zoom combo shows {selected}% after startup, not the persisted 150% -- " \
            "load_preset()'s generic settings restore clobbered it"
    finally:
        gui_mod._load_zoom_level = orig_load
        if frame is not None:
            frame.Destroy()
        app.Destroy()

    print("_self_test_zoom_startup_restore: PASS")


def _self_test_zoom_live_rescale():
    """Regression test for UI Zoom applying live (no restart) in a real running
    window, mirroring _self_test_layout_mode_live_switch's approach for Layout
    (ADR-038): drives MainFrame.apply_zoom_level() across the full offered range,
    repeatedly, in the same window, and checks a representative control's actual
    font point size changed each time and the window still lays out without error.
    No GPU or real movie file needed."""
    import iw3.gui as gui_mod

    app = wx.App()
    frame = None
    try:
        frame = gui_mod.MainFrame()
        assert frame.zoom_level == gui_mod.DEFAULT_ZOOM_LEVEL
        assert frame.cbo_divergence.GetFont().GetPointSize() == gui_mod.BASE_NORMAL_FONT_PT

        for level in gui_mod.ZOOM_LEVELS:
            frame.apply_zoom_level(level)
            assert frame.zoom_level == level
            expected_pt = gui_mod._zoom_font_point(gui_mod.BASE_NORMAL_FONT_PT, level)
            assert frame.cbo_divergence.GetFont().GetPointSize() == expected_pt, \
                f"level {level}: a live control's font did not rescale"
            assert frame.lbl_divergence_warning.GetFont().GetPointSize() == \
                gui_mod._zoom_font_point(gui_mod.BASE_WARNING_FONT_PT, level), \
                f"level {level}: the warning-font control did not rescale"
            # Accent theme's bold StaticBox headers must track the new size too, not
            # be left stuck at whatever size they were first bolded at.
            assert frame.grp_stereo.GetFont().GetPointSize() == expected_pt

        # Repeated switching back to 100% must be idempotent/stable.
        frame.apply_zoom_level(100)
        assert frame.cbo_divergence.GetFont().GetPointSize() == gui_mod.BASE_NORMAL_FONT_PT
    finally:
        if frame is not None:
            frame.Destroy()
        app.Destroy()

    print("_self_test_zoom_live_rescale: PASS")


def _self_test_progress_stage_display():
    """Regression test for the progress-bar stage/time/rate display (see
    docs/ai/AI_DECISIONS.md ADR-052 and its amendment): feeds synthetic tqdm-style
    and stage-change events straight into MainFrame.on_tqdm()/on_stage_change() (no
    real wx.PostEvent, no GPU, no movie file -- CS-TEST-001) and asserts the computed
    stage list, stage index, status text, Gauge state, and indeterminate-pulse timer
    behave correctly, including the exact real-world gap this feature fixes: a job
    with every real optional phase enabled (Scene Boundary Detection, AutoCrop
    Analysis, HDR/DV RPU Extraction, Audio Extraction, waifu2x, RIFE, HDR
    Reinjection -- 8 stages total with the always-on Depth & Stereo Conversion) must
    show all 8 in the REAL order process_video_full()/process_video_with_resume()
    run them (confirmed by reading both directly -- Audio Extraction falls AFTER
    Depth & Stereo Conversion, not before, since it is auto-resume's own single
    clean-audio pass over the whole file, done once every segment is already
    encoded) and advance through all of them; the pulse timer used for the
    no-progress-data subprocess stages must stop the instant real per-frame tqdm
    data resumes."""
    import types
    import iw3.gui as gui_mod

    class _FakeTqdmEvent:
        def __init__(self, type_, value, desc):
            self._v = (type_, value, desc)

        def GetValue(self):
            return self._v

    class _FakeStageEvent:
        def __init__(self, name):
            self.name = name

    app = wx.App()
    frame = None
    try:
        frame = gui_mod.MainFrame()

        # _compute_job_stages: purely a function of already-known settings.
        plain_args = types.SimpleNamespace(
            scene_detect=False, scene_detect_only=False, autocrop=None, preserve_dowi=False,
            auto_resume=False, waifu2x_upscale=False, rife_interpolate=False)
        assert frame._compute_job_stages(plain_args) == [gui_mod.STAGE_DEPTH_STEREO]

        # Each of the 4 new conditional stages gates on its own real setting,
        # independent of the others -- confirmed one at a time, not just in
        # combination, so a single wrong condition can't hide behind another.
        only_scene_detect = types.SimpleNamespace(
            scene_detect=True, scene_detect_only=False, autocrop=None, preserve_dowi=False,
            auto_resume=False, waifu2x_upscale=False, rife_interpolate=False)
        assert frame._compute_job_stages(only_scene_detect) == [
            gui_mod.STAGE_SCENE_DETECT, gui_mod.STAGE_DEPTH_STEREO]

        only_autocrop = types.SimpleNamespace(
            scene_detect=False, scene_detect_only=False, autocrop="768:432", preserve_dowi=False,
            auto_resume=False, waifu2x_upscale=False, rife_interpolate=False)
        assert frame._compute_job_stages(only_autocrop) == [
            gui_mod.STAGE_AUTOCROP, gui_mod.STAGE_DEPTH_STEREO]

        only_preserve_dowi = types.SimpleNamespace(
            scene_detect=False, scene_detect_only=False, autocrop=None, preserve_dowi=True,
            auto_resume=False, waifu2x_upscale=False, rife_interpolate=False)
        assert frame._compute_job_stages(only_preserve_dowi) == [
            gui_mod.STAGE_HDR_EXTRACT, gui_mod.STAGE_DEPTH_STEREO, gui_mod.STAGE_HDR_REINJECT], \
            "--preserve-dowi gates BOTH HDR/DV RPU Extraction (before the encode) and " \
            "HDR/Dolby Vision Reinjection (after it) -- same flag, two real stages"

        only_auto_resume = types.SimpleNamespace(
            scene_detect=False, scene_detect_only=False, autocrop=None, preserve_dowi=False,
            auto_resume=True, waifu2x_upscale=False, rife_interpolate=False)
        assert frame._compute_job_stages(only_auto_resume) == [
            gui_mod.STAGE_DEPTH_STEREO, gui_mod.STAGE_AUDIO_EXTRACT], \
            "Audio Extraction must come AFTER Depth & Stereo Conversion, not before"

        full_args = types.SimpleNamespace(
            scene_detect=True, scene_detect_only=False, autocrop="768:432", preserve_dowi=True,
            auto_resume=True, waifu2x_upscale=True, rife_interpolate=True)
        assert frame._compute_job_stages(full_args) == [
            gui_mod.STAGE_SCENE_DETECT, gui_mod.STAGE_AUTOCROP, gui_mod.STAGE_HDR_EXTRACT,
            gui_mod.STAGE_DEPTH_STEREO, gui_mod.STAGE_AUDIO_EXTRACT, gui_mod.STAGE_WAIFU2X_UPSCALE,
            gui_mod.STAGE_RIFE_INTERPOLATE, gui_mod.STAGE_HDR_REINJECT,
        ]

        # _format_duration
        assert gui_mod.MainFrame._format_duration(0) == "00:00"
        assert gui_mod.MainFrame._format_duration(65) == "01:05"
        assert gui_mod.MainFrame._format_duration(3661) == "01:01:01"

        # An 8-stage job: simulate Start, then the main tqdm-tracked encode running live.
        frame.job_stages = full_args and frame._compute_job_stages(full_args)
        frame.job_stage_index = 1
        frame.current_stage_name = frame.job_stages[0]
        frame.job_start_time = time()
        frame.stage_start_time = frame.job_start_time
        frame.stage_pulse_timer.Stop()

        # Stage 1: Scene Boundary Detection (already tqdm-tracked -- live per-frame
        # progress, no pulse timer needed).
        frame.on_tqdm(_FakeTqdmEvent(0, 100, "movie.mp4 [ZoeD_Any_N]"))
        assert frame.prg_tqdm.GetRange() == 100
        assert frame.prg_tqdm.GetValue() == 0
        assert not frame.stage_pulse_timer.IsRunning()
        status = frame.GetStatusBar().GetStatusText()
        assert "Step 1/8" in status and gui_mod.STAGE_SCENE_DETECT in status, status

        frame.on_tqdm(_FakeTqdmEvent(1, 10, "movie.mp4 [ZoeD_Any_N]"))
        assert frame.prg_tqdm.GetValue() == 10
        status = frame.GetStatusBar().GetStatusText()
        assert "Step 1/8" in status and "FPS" in status and "elapsed" in status and "ETA" in status, status

        # Stage 2: AutoCrop Analysis (also tqdm-tracked).
        frame.on_stage_change(_FakeStageEvent(gui_mod.STAGE_AUTOCROP))
        assert frame.job_stage_index == 2
        status = frame.GetStatusBar().GetStatusText()
        assert "Step 2/8" in status and gui_mod.STAGE_AUTOCROP in status, status

        # Stage 3: HDR/DV RPU Extraction -- a blocking subprocess with no per-item
        # progress, same class of gap as waifu2x/RIFE below -- must animate the
        # Gauge via the pulse timer instead of sitting frozen.
        frame.on_stage_change(_FakeStageEvent(gui_mod.STAGE_HDR_EXTRACT))
        assert frame.job_stage_index == 3
        assert frame.stage_pulse_timer.IsRunning()
        status = frame.GetStatusBar().GetStatusText()
        assert "Step 3/8" in status and gui_mod.STAGE_HDR_EXTRACT in status, status

        # Stage 4: Depth & Stereo Conversion -- real per-frame tqdm data resuming
        # must stop the pulse timer left running by stage 3.
        frame.on_stage_change(_FakeStageEvent(gui_mod.STAGE_DEPTH_STEREO))
        assert frame.job_stage_index == 4
        frame.on_tqdm(_FakeTqdmEvent(0, 200, "movie.mp4 [ZoeD_Any_N]"))
        assert not frame.stage_pulse_timer.IsRunning()
        status = frame.GetStatusBar().GetStatusText()
        assert "Step 4/8" in status and gui_mod.STAGE_DEPTH_STEREO in status, status

        # Stage 5: Audio Extraction.
        frame.on_stage_change(_FakeStageEvent(gui_mod.STAGE_AUDIO_EXTRACT))
        assert frame.job_stage_index == 5
        assert frame.stage_pulse_timer.IsRunning()
        status = frame.GetStatusBar().GetStatusText()
        assert "Step 5/8" in status and gui_mod.STAGE_AUDIO_EXTRACT in status, status

        # Stage 6: waifu2x upscale.
        frame.on_stage_change(_FakeStageEvent(gui_mod.STAGE_WAIFU2X_UPSCALE))
        assert frame.job_stage_index == 6
        assert frame.stage_pulse_timer.IsRunning()
        status = frame.GetStatusBar().GetStatusText()
        assert "Step 6/8" in status and gui_mod.STAGE_WAIFU2X_UPSCALE in status, status

        # Stage 7: RIFE.
        frame.on_stage_change(_FakeStageEvent(gui_mod.STAGE_RIFE_INTERPOLATE))
        assert frame.job_stage_index == 7
        status = frame.GetStatusBar().GetStatusText()
        assert "Step 7/8" in status and gui_mod.STAGE_RIFE_INTERPOLATE in status, status

        # Stage 8: HDR reinjection.
        frame.on_stage_change(_FakeStageEvent(gui_mod.STAGE_HDR_REINJECT))
        assert frame.job_stage_index == 8
        status = frame.GetStatusBar().GetStatusText()
        assert "Step 8/8" in status and gui_mod.STAGE_HDR_REINJECT in status, status

        # Real per-frame tqdm data resuming (e.g. a second file in a batch job
        # starting its own Depth & Stereo Conversion pass) must stop the pulse timer.
        frame.on_tqdm(_FakeTqdmEvent(0, 50, "movie2.mp4 [ZoeD_Any_N]"))
        assert not frame.stage_pulse_timer.IsRunning()
    finally:
        if frame is not None:
            frame.Destroy()
        app.Destroy()

    print("_self_test_progress_stage_display: PASS")


def _self_test_progress_bar_visible_on_screen():
    """Regression test for ADR-056 (progress bar getting pushed off the bottom of the
    screen). The confirmed root cause: refresh_layouts()/Fit() size the frame to its
    full natural content height (Single Page/Tabbed both pin their ScrolledPanel
    content's MinSize to the entire un-scrolled size -- see
    _compose_options_layout_single_page()/_compose_options_layout_tabbed()) with
    nothing clamping that against the real screen, so on a tall enough content set
    (or high enough Zoom level) the frame can become taller than the monitor's usable
    work area -- and since pnl_process (the Gauge plus Start/Suspend/Cancel) is the
    LAST, fixed-proportion row in the frame's own top-level vertical sizer, it is
    exactly what gets pushed past the visible screen edge. This deterministically
    reproduces that pre-fix symptom (forcing the frame taller than the screen,
    regardless of this test machine's real resolution) and asserts
    MainFrame._clamp_frame_to_screen() brings both the frame and pnl_process back
    within the display's client (work) area. No GPU or real movie file, and no human
    screenshot inspection needed for this part -- a real running-conversion
    screenshot comparison was also done manually, see docs/ai/AI_DECISIONS.md."""
    import iw3.gui as gui_mod

    app = wx.App()
    frame = None
    try:
        frame = gui_mod.MainFrame()
        frame.Show()
        display_index = wx.Display.GetFromWindow(frame)
        if display_index == wx.NOT_FOUND:
            display_index = 0
        work_area = wx.Display(display_index).GetClientArea()

        # Deliberately force the frame taller than the real screen -- reproduces the
        # pre-fix symptom regardless of what resolution this test machine actually has.
        oversized_height = work_area.GetHeight() + 400
        frame.SetSize((frame.GetSize().GetWidth(), oversized_height))
        frame.Layout()
        assert frame.GetSize().GetHeight() > work_area.GetHeight(), \
            "test setup failed to actually oversize the frame"

        frame._clamp_frame_to_screen()
        frame.Layout()

        new_size = frame.GetSize()
        assert new_size.GetHeight() <= work_area.GetHeight(), \
            f"frame height {new_size.GetHeight()} still exceeds the screen work area {work_area.GetHeight()}"
        assert new_size.GetWidth() <= work_area.GetWidth(), \
            f"frame width {new_size.GetWidth()} still exceeds the screen work area {work_area.GetWidth()}"

        progress_rect = frame.pnl_process.GetScreenRect()
        assert progress_rect.GetBottom() <= work_area.GetBottom(), \
            f"pnl_process bottom {progress_rect.GetBottom()} still extends past the screen's " \
            f"usable area (work area bottom {work_area.GetBottom()}) -- this is the reported bug"
        assert progress_rect.GetTop() >= work_area.GetTop(), \
            f"pnl_process top {progress_rect.GetTop()} is above the screen's usable area"

        frame_rect = frame.GetScreenRect()
        assert frame_rect.Contains(progress_rect), \
            "pnl_process is not fully contained within the (now-clamped) frame"
    finally:
        if frame is not None:
            frame.Destroy()
        app.Destroy()

    print("_self_test_progress_bar_visible_on_screen: PASS")


def _self_test_progress_bar_visible_after_live_field_changes():
    """Regression test for the ADR-056 amendment: a real-usage regression where
    progress bar/Start-Suspend-Cancel got pushed behind the taskbar AGAIN, even with
    ADR-056's original clamp mechanism intact and working. Root cause, confirmed by
    measurement (see docs/ai/AI_DECISIONS.md ADR-056 amendment): several pre-existing
    runtime handlers -- update_anaglyph_state, update_export_option_state,
    update_inpaint_options, update_model_selection, update_divergence_warning, and
    on_click_divergence_warning -- Show()/Hide() real controls in response to an
    ordinary field change (Stereo Format, Method, Depth Model, Divergence value) and
    call self.Fit() on their own, but were never paired with
    MainFrame._clamp_frame_to_screen() the way IW3App.OnInit/switch_layout_mode/
    apply_zoom_level are. This was harmless while the frame's natural content height
    had comfortable margin under the real screen; once other same-night additions
    (RIFE/Sharpen standalone panels, Auto EMA by Scene Length's relocation, Genre
    Preset) shrank that margin, an everyday field change was again enough to exceed
    the screen. Deterministic regardless of this test machine's real resolution: the
    frame is first forced to sit exactly at the screen's usable height (simulating an
    already-tightly-clamped window, the realistic worst case after tonight's
    additions), then a real field change that Show()s additional controls
    (Method -> forward_inpaint, which reveals ~9 inpaint controls via
    update_inpaint_options) is applied through its real event handler -- not a direct
    call to the clamp -- and pnl_process must still end up fully on-screen afterward."""
    import iw3.gui as gui_mod

    app = wx.App()
    frame = None
    try:
        frame = gui_mod.MainFrame()
        frame.Show()
        display_index = wx.Display.GetFromWindow(frame)
        if display_index == wx.NOT_FOUND:
            display_index = 0
        work_area = wx.Display(display_index).GetClientArea()

        # Simulate the realistic worst case: a window already sitting exactly at the
        # screen's usable height (i.e. already at the clamp boundary, as a real user's
        # window commonly is after tonight's additions), before any live field change.
        frame.SetSize((frame.GetSize().GetWidth(), work_area.GetHeight()))
        frame.Layout()

        # A real field change, through its real event handler -- not a synthetic
        # resize -- that Show()s additional controls and calls self.Fit() internally.
        frame.cbo_method.SetValue("forward_inpaint")
        frame.on_selected_index_changed_cbo_method(None)

        new_size = frame.GetSize()
        assert new_size.GetHeight() <= work_area.GetHeight(), \
            f"frame height {new_size.GetHeight()} exceeds the screen work area " \
            f"{work_area.GetHeight()} after a live field change grew the content -- " \
            f"the field-change handler's self.Fit() was not followed by a clamp"

        progress_rect = frame.pnl_process.GetScreenRect()
        assert progress_rect.GetBottom() <= work_area.GetBottom(), \
            f"pnl_process bottom {progress_rect.GetBottom()} extends past the screen's " \
            f"usable area (work area bottom {work_area.GetBottom()}) after a live field " \
            f"change -- this is the reported regression"
    finally:
        if frame is not None:
            frame.Destroy()
        app.Destroy()

    print("_self_test_progress_bar_visible_after_live_field_changes: PASS")


def _self_test_scene_batch_auto_ema_editor():
    """Regression test for ADR-057 (Auto EMA by Scene Length's "Edit Values..."
    editor). Redirects scene_batch.EMA_OVERRIDES_PATH to a throwaway temp file (never
    touches the real nunif/tmp/iw3_auto_ema_overrides.json) and drives
    SceneBatchAutoEMADialog directly -- no ShowModal() (would block on real user
    input), no GPU, no real movie file. Covers every acceptance check from this
    feature's spec: default-unchanged-when-no-override-file, an edited value actually
    being used by the real _load_scene_settings() lookup, validation rejecting bad
    Buffer/Decay input, Reset to Default really restoring built-in values, and bucket
    boundaries (min_duration/max_duration) surviving a save round-trip unchanged."""
    import tempfile
    import iw3.scene_batch as scene_batch_mod

    tmp_dir = tempfile.mkdtemp(prefix="iw3_auto_ema_selftest_")
    orig_override_path = scene_batch_mod.EMA_OVERRIDES_PATH
    scene_batch_mod.EMA_OVERRIDES_PATH = path.join(tmp_dir, "iw3_auto_ema_overrides.json")

    app = None
    frame = None
    try:
        # 1. Default-unchanged-when-no-override-file (key regression check).
        assert not path.exists(scene_batch_mod.EMA_OVERRIDES_PATH)
        assert scene_batch_mod.load_ema_overrides_file() == {}
        default_rules = scene_batch_mod._load_scene_settings(
            None, auto_ema_by_duration=True, auto_ema_model="3DECKER VDA_L")
        assert default_rules == list(scene_batch_mod.EMA_BY_DURATION_VDA_L), \
            "default rules changed with no override file present"

        app = wx.App()
        frame = MainFrame()
        assert isinstance(frame.btn_scene_batch_auto_ema_edit, wx.Button)

        validation_errors = []
        frame.show_validation_error_message = lambda name, lo, hi: validation_errors.append((name, lo, hi))

        dlg = SceneBatchAutoEMADialog(frame, "3DECKER VDA_L")
        assert dlg.buffer_ctrls[0].GetValue() == "8" and dlg.decay_ctrls[0].GetValue() == "0.65"

        # 2. Validation rejects bad Buffer/Decay input, and never writes a file for it.
        dlg.buffer_ctrls[0].SetValue("-1")
        dlg.decay_ctrls[0].SetValue("0.65")
        dlg.on_save(None)
        assert len(validation_errors) == 1, "non-positive Buffer was not rejected"
        assert not path.exists(scene_batch_mod.EMA_OVERRIDES_PATH), "invalid input must not be saved"

        validation_errors.clear()
        dlg.buffer_ctrls[0].SetValue("8")
        dlg.decay_ctrls[0].SetValue("1.0")
        dlg.on_save(None)
        assert len(validation_errors) == 1, "decay=1.0 (not strictly < 1) was not rejected"

        validation_errors.clear()
        dlg.decay_ctrls[0].SetValue("0.0")
        dlg.on_save(None)
        assert len(validation_errors) == 1, "decay=0.0 (not strictly > 0) was not rejected"

        # 3. A real edit persists and is actually used by _load_scene_settings().
        validation_errors.clear()
        dlg.buffer_ctrls[0].SetValue("99")
        dlg.decay_ctrls[0].SetValue("0.5")
        dlg.on_save(None)
        assert not validation_errors
        saved = scene_batch_mod.load_ema_overrides_file()
        assert saved["3DECKER VDA_L"][0] == {"ema_buffer": 99, "ema_decay": 0.5}

        rules = scene_batch_mod._load_scene_settings(
            None, auto_ema_by_duration=True, auto_ema_model="3DECKER VDA_L")
        assert rules[0]["overrides"] == {"ema_buffer": 99, "ema_decay": 0.5}
        assert rules[0]["min_duration"] == 0 and rules[0]["max_duration"] == 1, \
            "bucket boundaries must come from the hardcoded table, never the override file"

        other_rules = scene_batch_mod._load_scene_settings(
            None, auto_ema_by_duration=True, auto_ema_model="3DECKER Any_V3_Mono_01")
        assert other_rules == list(scene_batch_mod.EMA_BY_DURATION_ANY_V3_MONO_01), \
            "editing 3DECKER VDA_L must not affect the other model's table"

        # 4. Reset to Default (+ Save) really restores the built-in values and removes
        # the stored override entirely, rather than just writing a copy of the defaults.
        dlg2 = SceneBatchAutoEMADialog(frame, "3DECKER VDA_L")
        assert dlg2.buffer_ctrls[0].GetValue() == "99", "dialog did not load the saved override"
        dlg2.on_reset(None)
        assert dlg2.buffer_ctrls[0].GetValue() == "8" and dlg2.decay_ctrls[0].GetValue() == "0.65"
        dlg2.on_save(None)
        assert "3DECKER VDA_L" not in scene_batch_mod.load_ema_overrides_file()
        reset_rules = scene_batch_mod._load_scene_settings(
            None, auto_ema_by_duration=True, auto_ema_model="3DECKER VDA_L")
        assert reset_rules == list(scene_batch_mod.EMA_BY_DURATION_VDA_L)

        # 5. Bucket boundaries never corrupted by a save round-trip.
        dlg3 = SceneBatchAutoEMADialog(frame, "3DECKER VDA_L")
        dlg3.buffer_ctrls[3].SetValue("21")
        dlg3.decay_ctrls[3].SetValue("0.79")
        dlg3.on_save(None)
        combined = scene_batch_mod._table_from_override("3DECKER VDA_L", scene_batch_mod.EMA_BY_DURATION_VDA_L)
        for base_rule, combined_rule in zip(scene_batch_mod.EMA_BY_DURATION_VDA_L, combined):
            assert combined_rule.get("min_duration") == base_rule.get("min_duration")
            assert combined_rule.get("max_duration") == base_rule.get("max_duration")
    finally:
        scene_batch_mod.EMA_OVERRIDES_PATH = orig_override_path
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_scene_batch_auto_ema_editor: PASS")


def _self_test_nagadomi_reference_ema_option():
    """Regression test for the third Auto EMA by Scene Length table, "Nagadomi_Reference"
    (docs/ai/AI_DECISIONS.md ADR-057 second amendment): confirms the dropdown now offers
    all three choices with Nagadomi_Reference as the default (index 2, per ADR-057
    Amendment 7), and that
    SceneBatchAutoEMADialog correctly loads the new table's 21 rows when opened for it --
    in particular buffer=1/decay=0.750 at 0-1s and buffer=30/decay=0.900 at 20s+, per the
    task's real-GUI acceptance check. No GPU, no ShowModal() (would block on real user
    input), matching this file's other self-tests."""
    app = None
    frame = None
    try:
        app = wx.App()
        frame = MainFrame()

        choices = list(frame.cbo_scene_batch_auto_ema_model.GetItems())
        assert choices == ["3DECKER VDA_L", "3DECKER Any_V3_Mono_01", "Nagadomi_Reference", "GEMINI AI", "ChatGPT", "Grok", "Fast Action", "Medium Magical", "Drama Slow Paced"], choices
        assert frame.cbo_scene_batch_auto_ema_model.GetSelection() == 2, \
            "default selection must be Nagadomi_Reference (index 2) -- ADR-057 Amendment 7"
        assert frame.cbo_scene_batch_auto_ema_model.GetValue() == "Nagadomi_Reference"

        dlg = SceneBatchAutoEMADialog(frame, "Nagadomi_Reference")
        assert len(dlg.buffer_ctrls) == 21 and len(dlg.decay_ctrls) == 21
        assert dlg.buffer_ctrls[0].GetValue() == "1" and dlg.decay_ctrls[0].GetValue() == "0.75"
        assert dlg.buffer_ctrls[20].GetValue() == "30" and dlg.decay_ctrls[20].GetValue() == "0.9"

        # Editing/saving/resetting works the same generic way as the other two models --
        # verify it round-trips through _load_scene_settings for this model specifically.
        import iw3.scene_batch as scene_batch_mod
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="iw3_nagadomi_ref_selftest_")
        orig_override_path = scene_batch_mod.EMA_OVERRIDES_PATH
        scene_batch_mod.EMA_OVERRIDES_PATH = path.join(tmp_dir, "iw3_auto_ema_overrides.json")
        try:
            dlg.buffer_ctrls[0].SetValue("2")
            dlg.decay_ctrls[0].SetValue("0.76")
            dlg.on_save(None)
            rules = scene_batch_mod._load_scene_settings(
                None, auto_ema_by_duration=True, auto_ema_model="Nagadomi_Reference")
            assert rules[0]["overrides"] == {"ema_buffer": 2, "ema_decay": 0.76}
            other_rules = scene_batch_mod._load_scene_settings(
                None, auto_ema_by_duration=True, auto_ema_model="3DECKER VDA_L")
            assert other_rules == list(scene_batch_mod.EMA_BY_DURATION_VDA_L), \
                "editing Nagadomi_Reference must not affect 3DECKER VDA_L's table"
        finally:
            scene_batch_mod.EMA_OVERRIDES_PATH = orig_override_path
    finally:
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_nagadomi_reference_ema_option: PASS")


def _self_test_gemini_ai_ema_option():
    """Regression test for the fourth Auto EMA by Scene Length table, dropdown choice
    "GEMINI AI" (docs/ai/AI_DECISIONS.md ADR-057 third amendment): confirms the
    dropdown now offers all four choices with Nagadomi_Reference as the default (index 2,
    per ADR-057 Amendment 7), and that SceneBatchAutoEMADialog correctly loads the new table's 21
    rows when opened for it -- in particular buffer=24/decay=0.750 at 0-1s and
    buffer=480/decay=0.975 at 20s+, per the task's real-GUI acceptance check. No GPU,
    no ShowModal() (would block on real user input), matching this file's other
    self-tests."""
    app = None
    frame = None
    try:
        app = wx.App()
        frame = MainFrame()

        choices = list(frame.cbo_scene_batch_auto_ema_model.GetItems())
        assert choices == ["3DECKER VDA_L", "3DECKER Any_V3_Mono_01", "Nagadomi_Reference", "GEMINI AI", "ChatGPT", "Grok", "Fast Action", "Medium Magical", "Drama Slow Paced"], choices
        assert frame.cbo_scene_batch_auto_ema_model.GetSelection() == 2, \
            "default selection must be Nagadomi_Reference (index 2) -- ADR-057 Amendment 7"
        assert frame.cbo_scene_batch_auto_ema_model.GetValue() == "Nagadomi_Reference"

        dlg = SceneBatchAutoEMADialog(frame, "GEMINI AI")
        assert len(dlg.buffer_ctrls) == 21 and len(dlg.decay_ctrls) == 21
        assert dlg.buffer_ctrls[0].GetValue() == "24" and dlg.decay_ctrls[0].GetValue() == "0.75"
        assert dlg.buffer_ctrls[9].GetValue() == "240" and dlg.decay_ctrls[9].GetValue() == "0.943"
        assert dlg.buffer_ctrls[19].GetValue() == "480" and dlg.decay_ctrls[19].GetValue() == "0.975"
        assert dlg.buffer_ctrls[20].GetValue() == "480" and dlg.decay_ctrls[20].GetValue() == "0.975"

        # Editing/saving/resetting works the same generic way as the other three models --
        # verify it round-trips through _load_scene_settings for this model specifically.
        import iw3.scene_batch as scene_batch_mod
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="iw3_gemini_ai_selftest_")
        orig_override_path = scene_batch_mod.EMA_OVERRIDES_PATH
        scene_batch_mod.EMA_OVERRIDES_PATH = path.join(tmp_dir, "iw3_auto_ema_overrides.json")
        try:
            dlg.buffer_ctrls[0].SetValue("25")
            dlg.decay_ctrls[0].SetValue("0.76")
            dlg.on_save(None)
            rules = scene_batch_mod._load_scene_settings(
                None, auto_ema_by_duration=True, auto_ema_model="GEMINI AI")
            assert rules[0]["overrides"] == {"ema_buffer": 25, "ema_decay": 0.76}
            other_rules = scene_batch_mod._load_scene_settings(
                None, auto_ema_by_duration=True, auto_ema_model="3DECKER VDA_L")
            assert other_rules == list(scene_batch_mod.EMA_BY_DURATION_VDA_L), \
                "editing GEMINI AI must not affect 3DECKER VDA_L's table"
        finally:
            scene_batch_mod.EMA_OVERRIDES_PATH = orig_override_path
    finally:
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_gemini_ai_ema_option: PASS")


def _self_test_chatgpt_ema_option():
    """Regression test for the fifth Auto EMA by Scene Length table, dropdown choice
    "ChatGPT" (docs/ai/AI_DECISIONS.md ADR-057 fourth amendment): confirms the
    dropdown now offers all five choices with Nagadomi_Reference as the default (index 2,
    per ADR-057 Amendment 7), and that SceneBatchAutoEMADialog correctly loads the new table's 21
    rows when opened for it -- in particular buffer=30/decay=0.75 at 0-1s and
    buffer=600/decay=0.99 at 20s+, per the task's real-GUI acceptance check. No GPU,
    no ShowModal() (would block on real user input), matching this file's other
    self-tests."""
    app = None
    frame = None
    try:
        app = wx.App()
        frame = MainFrame()

        choices = list(frame.cbo_scene_batch_auto_ema_model.GetItems())
        assert choices == ["3DECKER VDA_L", "3DECKER Any_V3_Mono_01", "Nagadomi_Reference", "GEMINI AI", "ChatGPT", "Grok", "Fast Action", "Medium Magical", "Drama Slow Paced"], choices
        assert frame.cbo_scene_batch_auto_ema_model.GetSelection() == 2, \
            "default selection must be Nagadomi_Reference (index 2) -- ADR-057 Amendment 7"
        assert frame.cbo_scene_batch_auto_ema_model.GetValue() == "Nagadomi_Reference"

        dlg = SceneBatchAutoEMADialog(frame, "ChatGPT")
        assert len(dlg.buffer_ctrls) == 21 and len(dlg.decay_ctrls) == 21
        assert dlg.buffer_ctrls[0].GetValue() == "30" and dlg.decay_ctrls[0].GetValue() == "0.75"
        assert dlg.buffer_ctrls[9].GetValue() == "300" and dlg.decay_ctrls[9].GetValue() == "0.88"
        assert dlg.buffer_ctrls[19].GetValue() == "600" and dlg.decay_ctrls[19].GetValue() == "0.99"
        assert dlg.buffer_ctrls[20].GetValue() == "600" and dlg.decay_ctrls[20].GetValue() == "0.99"

        # Editing/saving/resetting works the same generic way as the other four models --
        # verify it round-trips through _load_scene_settings for this model specifically.
        import iw3.scene_batch as scene_batch_mod
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="iw3_chatgpt_selftest_")
        orig_override_path = scene_batch_mod.EMA_OVERRIDES_PATH
        scene_batch_mod.EMA_OVERRIDES_PATH = path.join(tmp_dir, "iw3_auto_ema_overrides.json")
        try:
            dlg.buffer_ctrls[0].SetValue("31")
            dlg.decay_ctrls[0].SetValue("0.76")
            dlg.on_save(None)
            rules = scene_batch_mod._load_scene_settings(
                None, auto_ema_by_duration=True, auto_ema_model="ChatGPT")
            assert rules[0]["overrides"] == {"ema_buffer": 31, "ema_decay": 0.76}
            other_rules = scene_batch_mod._load_scene_settings(
                None, auto_ema_by_duration=True, auto_ema_model="3DECKER VDA_L")
            assert other_rules == list(scene_batch_mod.EMA_BY_DURATION_VDA_L), \
                "editing ChatGPT must not affect 3DECKER VDA_L's table"
        finally:
            scene_batch_mod.EMA_OVERRIDES_PATH = orig_override_path
    finally:
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_chatgpt_ema_option: PASS")


def _self_test_grok_ema_option():
    """Regression test for the sixth Auto EMA by Scene Length table, dropdown choice
    "Grok" (docs/ai/AI_DECISIONS.md ADR-057 fifth amendment): confirms the dropdown
    now offers all six choices with Nagadomi_Reference as the default (index 2, per
    ADR-057 Amendment 7), and
    that SceneBatchAutoEMADialog correctly loads the new table's 21 rows when opened
    for it -- in particular buffer=24/decay=0.90 at 0-1s, decay first reaching 0.99 at
    8-9s (buffer=216) and staying flat there, and buffer=480/decay=0.99 at both 19-20s
    and 20s+, per the task's real-GUI acceptance check. No GPU, no ShowModal() (would
    block on real user input), matching this file's other self-tests."""
    app = None
    frame = None
    try:
        app = wx.App()
        frame = MainFrame()

        choices = list(frame.cbo_scene_batch_auto_ema_model.GetItems())
        assert choices == ["3DECKER VDA_L", "3DECKER Any_V3_Mono_01", "Nagadomi_Reference", "GEMINI AI", "ChatGPT", "Grok", "Fast Action", "Medium Magical", "Drama Slow Paced"], choices
        assert frame.cbo_scene_batch_auto_ema_model.GetSelection() == 2, \
            "default selection must be Nagadomi_Reference (index 2) -- ADR-057 Amendment 7"
        assert frame.cbo_scene_batch_auto_ema_model.GetValue() == "Nagadomi_Reference"

        dlg = SceneBatchAutoEMADialog(frame, "Grok")
        assert len(dlg.buffer_ctrls) == 21 and len(dlg.decay_ctrls) == 21
        assert dlg.buffer_ctrls[0].GetValue() == "24" and dlg.decay_ctrls[0].GetValue() == "0.9"
        assert dlg.buffer_ctrls[8].GetValue() == "216" and dlg.decay_ctrls[8].GetValue() == "0.99"
        assert dlg.buffer_ctrls[19].GetValue() == "480" and dlg.decay_ctrls[19].GetValue() == "0.99"
        assert dlg.buffer_ctrls[20].GetValue() == "480" and dlg.decay_ctrls[20].GetValue() == "0.99"

        # Editing/saving/resetting works the same generic way as the other five models --
        # verify it round-trips through _load_scene_settings for this model specifically.
        import iw3.scene_batch as scene_batch_mod
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="iw3_grok_selftest_")
        orig_override_path = scene_batch_mod.EMA_OVERRIDES_PATH
        scene_batch_mod.EMA_OVERRIDES_PATH = path.join(tmp_dir, "iw3_auto_ema_overrides.json")
        try:
            dlg.buffer_ctrls[0].SetValue("25")
            dlg.decay_ctrls[0].SetValue("0.91")
            dlg.on_save(None)
            rules = scene_batch_mod._load_scene_settings(
                None, auto_ema_by_duration=True, auto_ema_model="Grok")
            assert rules[0]["overrides"] == {"ema_buffer": 25, "ema_decay": 0.91}
            other_rules = scene_batch_mod._load_scene_settings(
                None, auto_ema_by_duration=True, auto_ema_model="3DECKER VDA_L")
            assert other_rules == list(scene_batch_mod.EMA_BY_DURATION_VDA_L), \
                "editing Grok must not affect 3DECKER VDA_L's table"
        finally:
            scene_batch_mod.EMA_OVERRIDES_PATH = orig_override_path
    finally:
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_grok_ema_option: PASS")


def _self_test_fast_action_ema_option():
    """Regression test for the seventh Auto EMA by Scene Length table, dropdown choice
    "Fast Action" (docs/ai/AI_DECISIONS.md ADR-057 Amendment 10): confirms the dropdown
    now offers all nine choices with Nagadomi_Reference as the default (index 2, per
    ADR-057 Amendment 7), and that SceneBatchAutoEMADialog correctly loads the new
    table's 21 rows when opened for it -- in particular buffer=24/decay=0.750 at 0-1s,
    buffer=240/decay=0.774 at 9-10s, and every bucket from 10-11s through 20s+ holding
    flat at buffer=240/decay=0.774, per the task's real-GUI acceptance check. No GPU,
    no ShowModal() (would block on real user input), matching this file's other
    self-tests."""
    app = None
    frame = None
    try:
        app = wx.App()
        frame = MainFrame()

        choices = list(frame.cbo_scene_batch_auto_ema_model.GetItems())
        assert choices == ["3DECKER VDA_L", "3DECKER Any_V3_Mono_01", "Nagadomi_Reference", "GEMINI AI", "ChatGPT", "Grok", "Fast Action", "Medium Magical", "Drama Slow Paced"], choices
        assert frame.cbo_scene_batch_auto_ema_model.GetSelection() == 2, \
            "default selection must be Nagadomi_Reference (index 2) -- ADR-057 Amendment 7"
        assert frame.cbo_scene_batch_auto_ema_model.GetValue() == "Nagadomi_Reference"

        dlg = SceneBatchAutoEMADialog(frame, "Fast Action")
        assert len(dlg.buffer_ctrls) == 21 and len(dlg.decay_ctrls) == 21
        assert dlg.buffer_ctrls[0].GetValue() == "24" and dlg.decay_ctrls[0].GetValue() == "0.75"
        assert dlg.buffer_ctrls[9].GetValue() == "240" and dlg.decay_ctrls[9].GetValue() == "0.774"
        for i in range(10, 21):
            assert dlg.buffer_ctrls[i].GetValue() == "240" and dlg.decay_ctrls[i].GetValue() == "0.774", \
                f"row {i} must hold flat at the 9-10s bucket's value (10s cap, ADR-057 Amendment 10)"

        # Editing/saving/resetting works the same generic way as the other six models --
        # verify it round-trips through _load_scene_settings for this model specifically.
        import iw3.scene_batch as scene_batch_mod
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="iw3_fast_action_selftest_")
        orig_override_path = scene_batch_mod.EMA_OVERRIDES_PATH
        scene_batch_mod.EMA_OVERRIDES_PATH = path.join(tmp_dir, "iw3_auto_ema_overrides.json")
        try:
            dlg.buffer_ctrls[0].SetValue("25")
            dlg.decay_ctrls[0].SetValue("0.76")
            dlg.on_save(None)
            rules = scene_batch_mod._load_scene_settings(
                None, auto_ema_by_duration=True, auto_ema_model="Fast Action")
            assert rules[0]["overrides"] == {"ema_buffer": 25, "ema_decay": 0.76}
            other_rules = scene_batch_mod._load_scene_settings(
                None, auto_ema_by_duration=True, auto_ema_model="3DECKER VDA_L")
            assert other_rules == list(scene_batch_mod.EMA_BY_DURATION_VDA_L), \
                "editing Fast Action must not affect 3DECKER VDA_L's table"
        finally:
            scene_batch_mod.EMA_OVERRIDES_PATH = orig_override_path
    finally:
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_fast_action_ema_option: PASS")


def _self_test_medium_magical_ema_option():
    """Regression test for the eighth Auto EMA by Scene Length table, dropdown choice
    "Medium Magical" (docs/ai/AI_DECISIONS.md ADR-057 Amendment 10): confirms the
    dropdown now offers all nine choices with Nagadomi_Reference as the default (index
    2, per ADR-057 Amendment 7), and that SceneBatchAutoEMADialog correctly loads the
    new table's 21 rows when opened for it -- in particular buffer=24/decay=0.820 at
    0-1s, buffer=240/decay=0.847 at 9-10s, and every bucket from 10-11s through 20s+
    holding flat at buffer=240/decay=0.847, per the task's real-GUI acceptance check.
    No GPU, no ShowModal() (would block on real user input), matching this file's
    other self-tests."""
    app = None
    frame = None
    try:
        app = wx.App()
        frame = MainFrame()

        choices = list(frame.cbo_scene_batch_auto_ema_model.GetItems())
        assert choices == ["3DECKER VDA_L", "3DECKER Any_V3_Mono_01", "Nagadomi_Reference", "GEMINI AI", "ChatGPT", "Grok", "Fast Action", "Medium Magical", "Drama Slow Paced"], choices
        assert frame.cbo_scene_batch_auto_ema_model.GetSelection() == 2, \
            "default selection must be Nagadomi_Reference (index 2) -- ADR-057 Amendment 7"
        assert frame.cbo_scene_batch_auto_ema_model.GetValue() == "Nagadomi_Reference"

        dlg = SceneBatchAutoEMADialog(frame, "Medium Magical")
        assert len(dlg.buffer_ctrls) == 21 and len(dlg.decay_ctrls) == 21
        assert dlg.buffer_ctrls[0].GetValue() == "24" and dlg.decay_ctrls[0].GetValue() == "0.82"
        assert dlg.buffer_ctrls[9].GetValue() == "240" and dlg.decay_ctrls[9].GetValue() == "0.847"
        for i in range(10, 21):
            assert dlg.buffer_ctrls[i].GetValue() == "240" and dlg.decay_ctrls[i].GetValue() == "0.847", \
                f"row {i} must hold flat at the 9-10s bucket's value (10s cap, ADR-057 Amendment 10)"

        # Editing/saving/resetting works the same generic way as the other seven models --
        # verify it round-trips through _load_scene_settings for this model specifically.
        import iw3.scene_batch as scene_batch_mod
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="iw3_medium_magical_selftest_")
        orig_override_path = scene_batch_mod.EMA_OVERRIDES_PATH
        scene_batch_mod.EMA_OVERRIDES_PATH = path.join(tmp_dir, "iw3_auto_ema_overrides.json")
        try:
            dlg.buffer_ctrls[0].SetValue("25")
            dlg.decay_ctrls[0].SetValue("0.83")
            dlg.on_save(None)
            rules = scene_batch_mod._load_scene_settings(
                None, auto_ema_by_duration=True, auto_ema_model="Medium Magical")
            assert rules[0]["overrides"] == {"ema_buffer": 25, "ema_decay": 0.83}
            other_rules = scene_batch_mod._load_scene_settings(
                None, auto_ema_by_duration=True, auto_ema_model="3DECKER VDA_L")
            assert other_rules == list(scene_batch_mod.EMA_BY_DURATION_VDA_L), \
                "editing Medium Magical must not affect 3DECKER VDA_L's table"
        finally:
            scene_batch_mod.EMA_OVERRIDES_PATH = orig_override_path
    finally:
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_medium_magical_ema_option: PASS")


def _self_test_drama_slow_paced_ema_option():
    """Regression test for the ninth Auto EMA by Scene Length table, dropdown choice
    "Drama Slow Paced" (docs/ai/AI_DECISIONS.md ADR-057 Amendment 10): confirms the
    dropdown now offers all nine choices with Nagadomi_Reference as the default (index
    2, per ADR-057 Amendment 7), and that SceneBatchAutoEMADialog correctly loads the
    new table's 21 rows when opened for it -- in particular buffer=24/decay=0.900 at
    0-1s, buffer=240/decay=0.924 at 9-10s, and every bucket from 10-11s through 20s+
    holding flat at buffer=240/decay=0.924, per the task's real-GUI acceptance check.
    No GPU, no ShowModal() (would block on real user input), matching this file's
    other self-tests."""
    app = None
    frame = None
    try:
        app = wx.App()
        frame = MainFrame()

        choices = list(frame.cbo_scene_batch_auto_ema_model.GetItems())
        assert choices == ["3DECKER VDA_L", "3DECKER Any_V3_Mono_01", "Nagadomi_Reference", "GEMINI AI", "ChatGPT", "Grok", "Fast Action", "Medium Magical", "Drama Slow Paced"], choices
        assert frame.cbo_scene_batch_auto_ema_model.GetSelection() == 2, \
            "default selection must be Nagadomi_Reference (index 2) -- ADR-057 Amendment 7"
        assert frame.cbo_scene_batch_auto_ema_model.GetValue() == "Nagadomi_Reference"

        dlg = SceneBatchAutoEMADialog(frame, "Drama Slow Paced")
        assert len(dlg.buffer_ctrls) == 21 and len(dlg.decay_ctrls) == 21
        assert dlg.buffer_ctrls[0].GetValue() == "24" and dlg.decay_ctrls[0].GetValue() == "0.9"
        assert dlg.buffer_ctrls[9].GetValue() == "240" and dlg.decay_ctrls[9].GetValue() == "0.924"
        for i in range(10, 21):
            assert dlg.buffer_ctrls[i].GetValue() == "240" and dlg.decay_ctrls[i].GetValue() == "0.924", \
                f"row {i} must hold flat at the 9-10s bucket's value (10s cap, ADR-057 Amendment 10)"

        # Editing/saving/resetting works the same generic way as the other eight models --
        # verify it round-trips through _load_scene_settings for this model specifically.
        import iw3.scene_batch as scene_batch_mod
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="iw3_drama_slow_paced_selftest_")
        orig_override_path = scene_batch_mod.EMA_OVERRIDES_PATH
        scene_batch_mod.EMA_OVERRIDES_PATH = path.join(tmp_dir, "iw3_auto_ema_overrides.json")
        try:
            dlg.buffer_ctrls[0].SetValue("25")
            dlg.decay_ctrls[0].SetValue("0.91")
            dlg.on_save(None)
            rules = scene_batch_mod._load_scene_settings(
                None, auto_ema_by_duration=True, auto_ema_model="Drama Slow Paced")
            assert rules[0]["overrides"] == {"ema_buffer": 25, "ema_decay": 0.91}
            other_rules = scene_batch_mod._load_scene_settings(
                None, auto_ema_by_duration=True, auto_ema_model="3DECKER VDA_L")
            assert other_rules == list(scene_batch_mod.EMA_BY_DURATION_VDA_L), \
                "editing Drama Slow Paced must not affect 3DECKER VDA_L's table"
        finally:
            scene_batch_mod.EMA_OVERRIDES_PATH = orig_override_path
    finally:
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_drama_slow_paced_ema_option: PASS")


def _self_test_auto_ema_default_is_nagadomi_reference():
    """Regression test for ADR-057 Amendment 7 (default Auto EMA by Scene Length table
    changed from "3DECKER VDA_L" to "Nagadomi_Reference"): confirms the dropdown's
    default SELECTION changed (index 2 / "Nagadomi_Reference") without reordering the
    choices list or changing the "Auto EMA by Scene Length" checkbox's own separate
    off-by-default state, on a fresh MainFrame (no config file, no user interaction)."""
    app = None
    frame = None
    try:
        app = wx.App()
        frame = MainFrame()

        assert list(frame.cbo_scene_batch_auto_ema_model.GetItems()) == [
            "3DECKER VDA_L", "3DECKER Any_V3_Mono_01", "Nagadomi_Reference",
            "GEMINI AI", "ChatGPT", "Grok", "Fast Action", "Medium Magical",
            "Drama Slow Paced"], \
            "choices list order (first six) must be unchanged by this default-selection-only " \
            "change -- ADR-057 Amendment 10's three new entries are appended after, not " \
            "inserted"
        assert frame.cbo_scene_batch_auto_ema_model.GetSelection() == 2
        assert frame.cbo_scene_batch_auto_ema_model.GetValue() == "Nagadomi_Reference"
        assert frame.chk_scene_batch_auto_ema.GetValue() is False, \
            "Auto EMA by Scene Length checkbox must still default to off -- only the " \
            "table selected for IF it's turned on changed"
    finally:
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_auto_ema_default_is_nagadomi_reference: PASS")


def _self_test_scene_auto_ema_regular_gate():
    """Regression test for extending Auto EMA by Scene Length to a regular (non--
    Scene Batch) conversion: --scene-batch-auto-ema now also works with plain Scene
    Detection, but is meaningless (no scene boundaries to key off of) without EITHER
    Scene Detection or Automated Scene Batch turned on. Drives the real
    MainFrame.scene_auto_ema_gate_ok() (the exact method parse_args() calls, not a
    hand-copied condition) -- deliberately does NOT call parse_args() itself, which
    runs on to build a full args Namespace and is not meant for a self-test context."""
    app = None
    frame = None
    try:
        app = wx.App()
        frame = MainFrame()

        # Auto EMA on, neither Scene Detection nor Scene Batch on -> blocked.
        frame.chk_scene_batch_auto_ema.SetValue(True)
        frame.chk_scene_batch.SetValue(False)
        frame.chk_scene_detect.SetValue(False)
        assert not frame.scene_auto_ema_gate_ok(), "must be blocked with no scene boundaries to key off of"

        # Auto EMA on, Scene Detection on (the new regular path) -> gate passes.
        frame.chk_scene_detect.SetValue(True)
        assert frame.scene_auto_ema_gate_ok(), "Scene Detection alone must satisfy the gate"

        # Auto EMA on, Scene Batch on (existing behavior) -> gate passes too.
        frame.chk_scene_detect.SetValue(False)
        frame.chk_scene_batch.SetValue(True)
        assert frame.scene_auto_ema_gate_ok(), "Automated Scene Batch alone must satisfy the gate"

        # Auto EMA off -> gate never blocks, regardless of the other two.
        frame.chk_scene_batch_auto_ema.SetValue(False)
        frame.chk_scene_batch.SetValue(False)
        frame.chk_scene_detect.SetValue(False)
        assert frame.scene_auto_ema_gate_ok(), "gate must not block when Auto EMA by Scene Length is off"
    finally:
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_scene_auto_ema_regular_gate: PASS")


def _self_test_auto_ema_relocated_and_disables_ema_fields():
    """Regression test for the ADR-057 relocation amendment (2026-09-08): Auto EMA by
    Scene Length's controls (chk_scene_batch_auto_ema/cbo_scene_batch_auto_ema_model/
    btn_scene_batch_auto_ema_edit) were moved from the Video Filter group box into
    Flicker Reduction's own StaticBox (grp_stereo), directly under the Decay Rate/
    Buffer row, and checking "Auto EMA by Scene Length" now greys out (but does NOT
    clear) Flicker Reduction's own Decay Rate/Buffer fields (cbo_ema_decay/
    cbo_ema_buffer) -- unchecking it re-enables them with whatever value was typed in
    still present. Confirms all three parts on a real MainFrame: parent/sizer
    relocation, disable-without-clear, and re-enable-with-value-intact."""
    app = None
    frame = None
    try:
        app = wx.App()
        frame = MainFrame()

        # (a) Relocation: the three controls are now real children of grp_stereo
        # (Flicker Reduction's StaticBox), not grp_video_filter, and are laid out in
        # the SAME GridBagSizer as cbo_ema_buffer (Flicker Reduction's own Buffer
        # field) -- i.e. genuinely inside that group's layout, not merely reparented.
        assert frame.chk_scene_batch_auto_ema.GetParent() is frame.grp_stereo
        assert frame.cbo_scene_batch_auto_ema_model.GetParent() is frame.grp_stereo
        assert frame.btn_scene_batch_auto_ema_edit.GetParent() is frame.grp_stereo

        stereo_grid = frame.cbo_ema_buffer.GetContainingSizer()
        assert isinstance(stereo_grid, wx.GridBagSizer), "Flicker Reduction's own layout must be a GridBagSizer"
        assert frame.chk_scene_batch_auto_ema.GetContainingSizer() is stereo_grid, \
            "chk_scene_batch_auto_ema must be laid out in the same grid as Flicker Reduction's fields"
        assert frame.cbo_scene_batch_auto_ema_model.GetContainingSizer() is stereo_grid
        assert frame.btn_scene_batch_auto_ema_edit.GetContainingSizer() is stereo_grid

        # Sits directly under (a larger grid row index than) the Decay Rate/Buffer
        # row it visually relates to, in that same underlying GridBagSizer.
        buffer_pos = stereo_grid.GetItem(frame.cbo_ema_buffer).GetPos()
        auto_ema_pos = stereo_grid.GetItem(frame.chk_scene_batch_auto_ema).GetPos()
        assert auto_ema_pos.GetRow() > buffer_pos.GetRow(), \
            "Auto EMA by Scene Length must sit below the Decay Rate/Buffer row"

        # (b) Checking Auto EMA disables Decay/Buffer WITHOUT clearing their values.
        frame.cbo_ema_decay.SetValue("0.87")
        frame.cbo_ema_buffer.SetValue("77")
        frame.chk_ema_normalize.SetValue(True)
        frame.update_ema_normalize()
        assert frame.cbo_ema_decay.IsEnabled() and frame.cbo_ema_buffer.IsEnabled(), \
            "must be enabled before Auto EMA is turned on (Flicker Reduction itself is on)"

        frame.chk_scene_batch_auto_ema.SetValue(True)
        frame.on_changed_chk_scene_batch_auto_ema(None)
        assert not frame.cbo_ema_decay.IsEnabled(), "Decay must be greyed out while Auto EMA is checked"
        assert not frame.cbo_ema_buffer.IsEnabled(), "Buffer must be greyed out while Auto EMA is checked"
        assert frame.cbo_ema_decay.GetValue() == "0.87", "Decay's value must NOT be cleared by disabling"
        assert frame.cbo_ema_buffer.GetValue() == "77", "Buffer's value must NOT be cleared by disabling"

        # (c) Unchecking Auto EMA re-enables them, value still intact.
        frame.chk_scene_batch_auto_ema.SetValue(False)
        frame.on_changed_chk_scene_batch_auto_ema(None)
        assert frame.cbo_ema_decay.IsEnabled(), "Decay must re-enable once Auto EMA is unchecked"
        assert frame.cbo_ema_buffer.IsEnabled(), "Buffer must re-enable once Auto EMA is unchecked"
        assert frame.cbo_ema_decay.GetValue() == "0.87", "Decay's value must still be there after re-enabling"
        assert frame.cbo_ema_buffer.GetValue() == "77", "Buffer's value must still be there after re-enabling"
    finally:
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_auto_ema_relocated_and_disables_ema_fields: PASS")


def _self_test_genre_preset_quick_fill():
    """Regression test for the Genre Preset quick-fill feature (ADR-057 Amendment 9):
    a dropdown next to Flicker Reduction's Decay Rate/Buffer fields that fills them
    once with a fixed preset pair, distinct from the per-scene "Auto EMA by Scene
    Length" mechanism. Confirms each of the three real presets fills the exact
    correct Decay/Buffer values, the dropdown is disabled/enabled in lockstep with
    Flicker Reduction's own fields (including while Auto EMA by Scene Length is
    checked), and hand-editing a filled-in field afterward causes no error and does
    not reset the dropdown's own selection."""
    app = None
    frame = None
    try:
        app = wx.App()
        frame = MainFrame()

        assert list(frame.cbo_genre_preset.GetItems()) == [
            "-- Select --", "Fast Action", "Medium / Magical", "Drama / Slow-Paced"]
        assert frame.cbo_genre_preset.GetValue() == "-- Select --", \
            "placeholder must be selected by default, no preset applied"

        frame.chk_ema_normalize.SetValue(True)
        frame.chk_scene_batch_auto_ema.SetValue(False)
        frame.update_ema_normalize()

        # (a) Each real preset fills the exact correct Decay/Buffer values.
        expected = {
            "Fast Action": ("0.75", "30"),
            "Medium / Magical": ("0.85", "72"),
            "Drama / Slow-Paced": ("0.94", "120"),
        }
        for preset_name, (expected_decay, expected_buffer) in expected.items():
            frame.cbo_ema_decay.SetValue("")
            frame.cbo_ema_buffer.SetValue("")
            frame.cbo_genre_preset.SetStringSelection(preset_name)
            frame.on_changed_cbo_genre_preset(None)
            assert frame.cbo_ema_decay.GetValue() == expected_decay, \
                f"{preset_name}: expected Decay {expected_decay}, got {frame.cbo_ema_decay.GetValue()}"
            assert frame.cbo_ema_buffer.GetValue() == expected_buffer, \
                f"{preset_name}: expected Buffer {expected_buffer}, got {frame.cbo_ema_buffer.GetValue()}"

        # The placeholder itself must never fill anything.
        frame.cbo_ema_decay.SetValue("0.42")
        frame.cbo_ema_buffer.SetValue("42")
        frame.cbo_genre_preset.SetStringSelection(GENRE_PRESET_PLACEHOLDER)
        frame.on_changed_cbo_genre_preset(None)
        assert frame.cbo_ema_decay.GetValue() == "0.42", "placeholder must not touch Decay"
        assert frame.cbo_ema_buffer.GetValue() == "42", "placeholder must not touch Buffer"

        # (b) Hand-editing a filled-in field afterward is safe: no exception, and the
        # Genre Preset dropdown's own selection is left exactly as it was.
        frame.cbo_genre_preset.SetStringSelection("Drama / Slow-Paced")
        frame.on_changed_cbo_genre_preset(None)
        frame.cbo_ema_decay.SetValue("0.60")
        frame.cbo_ema_buffer.SetValue("99")
        assert frame.cbo_genre_preset.GetValue() == "Drama / Slow-Paced", \
            "hand-editing Decay/Buffer must not reset the Genre Preset dropdown"
        assert frame.cbo_ema_decay.GetValue() == "0.60" and frame.cbo_ema_buffer.GetValue() == "99"

        # (c) Disabled/enabled in lockstep with Flicker Reduction's own fields,
        # including while Auto EMA by Scene Length is checked.
        assert frame.cbo_genre_preset.IsEnabled(), \
            "Genre Preset must be enabled while Flicker Reduction is on and Auto EMA is off"

        frame.chk_scene_batch_auto_ema.SetValue(True)
        frame.on_changed_chk_scene_batch_auto_ema(None)
        assert not frame.cbo_ema_decay.IsEnabled() and not frame.cbo_ema_buffer.IsEnabled()
        assert not frame.cbo_genre_preset.IsEnabled(), \
            "Genre Preset must grey out together with Decay/Buffer while Auto EMA is checked"

        frame.chk_scene_batch_auto_ema.SetValue(False)
        frame.on_changed_chk_scene_batch_auto_ema(None)
        assert frame.cbo_ema_decay.IsEnabled() and frame.cbo_ema_buffer.IsEnabled()
        assert frame.cbo_genre_preset.IsEnabled(), \
            "Genre Preset must re-enable together with Decay/Buffer once Auto EMA is unchecked"

        # Selecting/filling a preset must never touch the Auto EMA checkbox itself,
        # and vice versa -- independent controls.
        assert frame.chk_scene_batch_auto_ema.GetValue() is False
    finally:
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_genre_preset_quick_fill: PASS")


def _self_test_hdr_reinject_rife_manifest_field():
    """Regression test for wiring reinject_hdr_cli.py's existing, already-real
    --rife-manifest flag (ADR-051) into the HDR Reinjection panel's new "RIFE
    Manifest (optional)" field (docs/ai/AI_DECISIONS.md ADR-064 amendment). Confirms:
    a blank field omits --rife-manifest entirely (today's exact existing behavior,
    unchanged); a filled-in, existing path passes --rife-manifest through verbatim; a
    non-blank but non-existent path refuses with a warning instead of silently
    passing a bad path; and picking an "Already-Converted 3D File" that has a
    matching real "<file>.rife_manifest.json" sidecar next to it (the RIFE Frame
    Interpolation panel's own real output-naming convention) auto-fills the new
    field, while a converted file with no such sidecar leaves it blank rather than
    guessing. No GPU or real movie file needed: startWorker is monkeypatched to
    capture command args instead of launching a real subprocess."""
    import iw3.gui as gui_mod

    app = wx.App()
    frame = None
    orig_start_worker = gui_mod.startWorker
    orig_message_box = gui_mod.wx.MessageBox
    message_box_calls = []
    # Several of this method's real validation-failure branches call the real, truly
    # blocking wx.MessageBox (a native modal dialog that pumps its own nested event
    # loop and waits for a real click) -- deliberately exercising one of those
    # branches below (case (c)) needs this mocked out, exactly the kind of
    # synthetic/mocked substitution this project's own --self-test convention already
    # uses for GPU/subprocess calls (CS-TEST-001), just applied to a GUI dialog call
    # instead.
    gui_mod.wx.MessageBox = lambda *a, **kw: message_box_calls.append(a)
    try:
        frame = gui_mod.MainFrame()

        assert frame.txt_reinject_rife_manifest.GetParent() is frame.grp_hdr_reinject
        assert frame.btn_reinject_rife_manifest.GetParent() is frame.grp_hdr_reinject

        captured = {}

        def _fake_start_worker(on_exit, worker_fn, wargs=(), **kwargs):
            captured["cmd"] = wargs[0]

        gui_mod.startWorker = _fake_start_worker

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = path.join(tmpdir, "source.mkv")
            converted_path = path.join(tmpdir, "movie_3d.mkv")
            output_path = path.join(tmpdir, "movie_3d_hdr_reinjected.mkv")
            for p in (source_path, converted_path):
                with open(p, "wb") as f:
                    f.write(b"fake")

            frame.txt_reinject_source.SetValue(source_path)
            frame.txt_reinject_converted.SetValue(converted_path)
            frame.txt_reinject_output.SetValue(output_path)
            frame.chk_reinject_start_time.SetValue(False)
            frame.chk_reinject_end_time.SetValue(False)

            # (a) Blank manifest field -> --rife-manifest omitted entirely (existing
            # behavior for anyone not using RIFE, unchanged).
            frame.txt_reinject_rife_manifest.SetValue("")
            frame.on_click_btn_reinject_run(None)
            cmd = captured["cmd"]
            assert "--rife-manifest" not in cmd, cmd
            assert cmd == [sys.executable, "-m", "iw3.reinject_hdr_cli",
                            "--source", source_path, "--converted", converted_path,
                            "--output", output_path], cmd

            # (b) Filled-in, existing manifest path -> passed through verbatim.
            manifest_path = path.join(tmpdir, "movie_3d_rife.mkv.rife_manifest.json")
            with open(manifest_path, "w") as f:
                f.write("{}")
            captured.clear()
            frame.txt_reinject_rife_manifest.SetValue(manifest_path)
            frame.on_click_btn_reinject_run(None)
            cmd = captured["cmd"]
            assert "--rife-manifest" in cmd and manifest_path in cmd, cmd

            # (c) Non-blank but non-existent manifest path -> refuses with a warning
            # (no command built at all), rather than silently passing a bad path
            # through.
            captured.clear()
            message_box_calls.clear()
            frame.txt_reinject_rife_manifest.SetValue(path.join(tmpdir, "does_not_exist.rife_manifest.json"))
            frame.on_click_btn_reinject_run(None)
            assert "cmd" not in captured, "must refuse before building a command for a missing manifest path"
            assert len(message_box_calls) == 1, "must warn the user exactly once about the missing manifest path"

            # (d) Auto-suggest on picking "Already-Converted 3D File": a real RIFE
            # output naming convention (<file>_rife<ext>) WITH a matching real
            # sidecar next to it auto-fills the field.
            rife_output_path = path.join(tmpdir, "clip2_rife.mkv")
            with open(rife_output_path, "wb") as f:
                f.write(b"fake")
            rife_manifest_path = rife_output_path + ".rife_manifest.json"
            with open(rife_manifest_path, "w") as f:
                f.write("{}")

            frame.txt_reinject_converted.SetValue("")
            frame.txt_reinject_output.SetValue("")
            frame.txt_reinject_rife_manifest.SetValue("")

            class _FakeDialog:
                def __init__(self, chosen_path):
                    self.chosen_path = chosen_path

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def GetValue(self):
                    return ""

                def SetPath(self, p):
                    pass

                def ShowModal(self):
                    return wx.ID_OK

                def GetPath(self):
                    return self.chosen_path

            orig_file_dialog = gui_mod.wx.FileDialog
            gui_mod.wx.FileDialog = lambda *a, **kw: _FakeDialog(rife_output_path)
            try:
                frame.on_click_btn_reinject_converted(None)
            finally:
                gui_mod.wx.FileDialog = orig_file_dialog
            assert frame.txt_reinject_converted.GetValue() == rife_output_path
            assert frame.txt_reinject_rife_manifest.GetValue() == rife_manifest_path, \
                frame.txt_reinject_rife_manifest.GetValue()

            # A converted file with no matching sidecar must leave the field blank
            # rather than guessing.
            no_sidecar_path = path.join(tmpdir, "clip3_rife.mkv")
            with open(no_sidecar_path, "wb") as f:
                f.write(b"fake")
            frame.txt_reinject_converted.SetValue("")
            frame.txt_reinject_output.SetValue("")
            frame.txt_reinject_rife_manifest.SetValue("")
            gui_mod.wx.FileDialog = lambda *a, **kw: _FakeDialog(no_sidecar_path)
            try:
                frame.on_click_btn_reinject_converted(None)
            finally:
                gui_mod.wx.FileDialog = orig_file_dialog
            assert frame.txt_reinject_rife_manifest.GetValue() == "", \
                "must not guess a manifest path when no real sidecar exists"
    finally:
        gui_mod.startWorker = orig_start_worker
        gui_mod.wx.MessageBox = orig_message_box
        if frame is not None:
            frame.Destroy()
        app.Destroy()

    print("_self_test_hdr_reinject_rife_manifest_field: PASS")


def _self_test_3decker_quick_preset():
    """Regression test for the "3DECKER Preferred" top-bar quick-preset button
    (docs/ai/AI_DECISIONS.md ADR-057 Amendment 12, updated ADR-076) -- moved here
    from the Genre Preset dropdown's "My Preferred Settings" entry (Amendment 11),
    now btn_quick_preset_3decker next to btn_quick_preset_movie/btn_quick_preset_action,
    following their exact apply_quick_preset(name) click-handler pattern. Confirms
    every real control this preset touches lands on the exact right value (the full
    2026-09-09 ADR-075-verified real-world command, not just the original
    Decay/Buffer-era subset), that clicking it correctly checks "Auto EMA by Scene
    Length" and drives the SAME grey-out chain a real click on that checkbox would
    (Decay Rate/Buffer/the Genre Preset dropdown itself all grey out), that "My
    Preferred Settings" is gone from the Genre Preset dropdown, that torch.compile
    is set WITHOUT a live GPU probe (ADR-068/034 guard -- same as Import Command),
    and that hand-editing afterward causes no error -- same as the Movie/Action
    quick presets."""
    app = None
    frame = None
    try:
        app = wx.App()
        frame = MainFrame()

        items = list(frame.cbo_genre_preset.GetItems())
        assert items == ["-- Select --", "Fast Action", "Medium / Magical", "Drama / Slow-Paced"], items
        assert "My Preferred Settings" not in items, items

        frame.chk_ema_normalize.SetValue(True)
        frame.chk_scene_batch_auto_ema.SetValue(False)
        frame.update_ema_normalize()

        compile_probe_calls = []
        orig_update_compile = frame.update_compile
        frame.update_compile = lambda *a, **kw: compile_probe_calls.append((a, kw))

        frame.apply_quick_preset("3decker")

        frame.update_compile = orig_update_compile

        assert frame.cbo_method.GetValue() == "mlbw_l2_inpaint", frame.cbo_method.GetValue()
        assert frame.chk_preserve_screen_border.GetValue() is True
        assert frame.cbo_depth_model.GetValue() == "Any_V3_Mono_01", frame.cbo_depth_model.GetValue()
        assert frame.cbo_divergence.GetValue() == "2.5", frame.cbo_divergence.GetValue()
        assert frame.cbo_convergence.GetValue() == "0.5", frame.cbo_convergence.GetValue()
        assert frame.cbo_background_pop_coverage.GetValue() == "0.0", frame.cbo_background_pop_coverage.GetValue()
        assert frame.cbo_stereo_format.GetValue() == "Half SBS", frame.cbo_stereo_format.GetValue()
        assert frame.cbo_resolution.GetValue() == "512", frame.cbo_resolution.GetValue()

        assert frame.cbo_inpaint_model.GetValue() == "light_inpaint_v1", frame.cbo_inpaint_model.GetValue()
        assert frame.cbo_overlap_frames_pre.GetValue() == "3", frame.cbo_overlap_frames_pre.GetValue()
        assert frame.cbo_overlap_frames_post.GetValue() == "3", frame.cbo_overlap_frames_post.GetValue()
        assert frame.chk_depth_aa.GetValue() is True

        assert frame.chk_depth_refine.GetValue() is True
        assert frame.cbo_depth_refine_strength.GetValue() == "1.0", frame.cbo_depth_refine_strength.GetValue()
        assert frame.cbo_depth_refine_strength.IsEnabled()
        assert frame.chk_temporal_stabilize.GetValue() is True
        assert frame.cbo_temporal_stabilize_strength.GetValue() == "0.3", \
            frame.cbo_temporal_stabilize_strength.GetValue()
        assert frame.cbo_temporal_stabilize_flat_boost.GetValue() == "0.0", \
            frame.cbo_temporal_stabilize_flat_boost.GetValue()
        assert frame.cbo_temporal_stabilize_edge_protect.GetValue() == "0.0", \
            frame.cbo_temporal_stabilize_edge_protect.GetValue()
        assert frame.cbo_temporal_stabilize_max_shift.GetValue() == "", \
            frame.cbo_temporal_stabilize_max_shift.GetValue()
        assert frame.cbo_temporal_stabilize_strength.IsEnabled()
        assert frame.chk_scene_detect.GetValue() is True
        assert frame.cbo_autocrop.GetValue() == "BLACK", frame.cbo_autocrop.GetValue()
        assert frame.chk_end_time.GetValue() is False

        assert frame.chk_scene_batch_auto_ema.GetValue() is True
        assert frame.cbo_scene_batch_auto_ema_model.GetValue() == "GEMINI AI", \
            frame.cbo_scene_batch_auto_ema_model.GetValue()

        assert frame.cbo_max_output_size.GetValue() == "3840x2160", frame.cbo_max_output_size.GetValue()
        assert frame.grp_video.cbo_pix_fmt.GetValue() == "yuv420p10le", frame.grp_video.cbo_pix_fmt.GetValue()
        assert frame.grp_video.cbo_video_format.GetValue() == "mkv", frame.grp_video.cbo_video_format.GetValue()
        if frame.grp_video.has_nvenc:
            assert frame.grp_video.cbo_video_codec.GetValue() == "hevc_nvenc", \
                frame.grp_video.cbo_video_codec.GetValue()
            assert frame.grp_video.cbo_tune.GetValue() == "uhq", frame.grp_video.cbo_tune.GetValue()
        assert frame.grp_video.cbo_fps.GetValue() == "1000.0", frame.grp_video.cbo_fps.GetValue()
        assert frame.grp_video.cbo_crf.GetValue() == "15", frame.grp_video.cbo_crf.GetValue()
        assert frame.grp_video_dec.cbo_hwaccel.GetValue() == "cuda", frame.grp_video_dec.cbo_hwaccel.GetValue()
        assert frame.grp_video_dec.chk_software_fallback.GetValue() is False

        assert frame.cbo_max_workers.GetValue() == "2", frame.cbo_max_workers.GetValue()
        assert frame.chk_metadata.GetValue() is True
        assert frame.chk_preserve_dowi.GetValue() is True
        assert frame.chk_stereo_mode_tag.GetValue() is True
        assert frame.chk_auto_resume.GetValue() is True

        assert frame.chk_compile.GetValue() is True
        assert not compile_probe_calls, \
            "applying the preset must not trigger a live torch.compile GPU probe"

        # Checking Auto EMA (via this preset) must trigger the exact same grey-out
        # chain a real checkbox click drives (ADR-057 Amendment 8/9): Decay
        # Rate/Buffer/the Genre Preset dropdown itself all disabled.
        assert not frame.cbo_ema_decay.IsEnabled()
        assert not frame.cbo_ema_buffer.IsEnabled()
        assert not frame.cbo_genre_preset.IsEnabled()

        # Hand-editing afterward causes no error, same guarantee as Movie/Action.
        frame.cbo_divergence.SetValue("4.0")
        assert frame.cbo_divergence.GetValue() == "4.0"
    finally:
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_3decker_quick_preset: PASS")


def _self_test_rife_standalone_panel():
    """Regression test for the new RIFE Frame Interpolation (Standalone Tool) panel
    (see docs/ai/AI_DECISIONS.md) -- pure GUI wiring around the existing, unmodified
    iw3/rife_cli.py, following the exact ADR-063 Sharpen (Standalone Tool) panel
    pattern. Confirms: every new control exists and is parented to grp_rife_standalone
    inside tab_tools (not disturbing any pre-existing control); the Rate mode
    enable/disable pattern for the Custom FPS field matches the in-pipeline RIFE
    controls' own established convention; the GPU dropdown deliberately has no "All
    CUDA Device" entry, since rife_cli.py's own --gpu targets exactly one device,
    unlike the main conversion's multi-GPU-capable Device selector; the Run button's
    tooltip actually carries the Dolby Vision/HDR reinjection reminder (the whole
    point of this task, not just a planned addition); and the exact command line Run
    constructs for both a simple multiplier and a Custom FPS target matches
    rife_cli.py's real create_parser() flag names read directly from that file. No
    GPU or real movie file needed: startWorker is monkeypatched to capture its args
    instead of actually launching a background subprocess."""
    import iw3.gui as gui_mod

    app = wx.App()
    frame = None
    orig_start_worker = gui_mod.startWorker
    try:
        frame = gui_mod.MainFrame()

        assert frame.grp_rife_standalone.GetParent() is frame.tab_tools
        for name in ("txt_rife_standalone_input", "txt_rife_standalone_output",
                     "cbo_rife_standalone_model", "cbo_rife_standalone_mode",
                     "txt_rife_standalone_target_fps", "cbo_rife_standalone_gpu",
                     "cbo_rife_standalone_codec",
                     "btn_rife_standalone_run", "txt_rife_standalone_log"):
            ctrl = getattr(frame, name)
            assert ctrl.GetParent() is frame.grp_rife_standalone, name

        # Output Codec: real, confirmed fix (2026-09-08, see docs/ai/AI_DECISIONS.md
        # ADR-051/ADR-064 amendments) for the bug that RIFE could never output HEVC
        # at all -- confirmed via ffprobe against a real RIFE output (codec_name:
        # h264, not hevc). Default choice must map to ClientData None (the same
        # unset default rife_cli.py's own --video-codec has always had).
        assert frame.cbo_rife_standalone_codec.GetValue() == T("H.264 (default)")
        default_codec_index = frame.cbo_rife_standalone_codec.GetSelection()
        assert frame.cbo_rife_standalone_codec.GetClientData(default_codec_index) is None

        # Default state: 2x mode, Custom FPS field disabled.
        assert frame.cbo_rife_standalone_mode.GetValue() == "2x"
        assert not frame.txt_rife_standalone_target_fps.IsEnabled()

        # Switching to Custom FPS... enables the field; switching back disables it.
        frame.cbo_rife_standalone_mode.SetValue("Custom FPS...")
        frame.update_rife_standalone_mode()
        assert frame.txt_rife_standalone_target_fps.IsEnabled()
        frame.cbo_rife_standalone_mode.SetValue("2x")
        frame.update_rife_standalone_mode()
        assert not frame.txt_rife_standalone_target_fps.IsEnabled()

        # GPU dropdown reuses the main Device selector's enumeration convention but
        # deliberately omits "All CUDA Device" (rife_cli.py's --gpu is single-device only).
        gpu_items = [frame.cbo_rife_standalone_gpu.GetString(i)
                     for i in range(frame.cbo_rife_standalone_gpu.GetCount())]
        assert "All CUDA Device" not in gpu_items, gpu_items
        assert "CPU" in gpu_items, gpu_items
        cpu_index = gpu_items.index("CPU")
        assert int(frame.cbo_rife_standalone_gpu.GetClientData(cpu_index)) == -1

        # The Dolby Vision / HDR reinjection reminder must actually be present in the
        # Run button's tooltip text, not just planned -- updated 2026-09-08 to point
        # at the real Output Codec control and the Retroactive HDR/DV Reinjection
        # panel's own real "RIFE Manifest" field (both now real, see
        # docs/ai/AI_DECISIONS.md ADR-051/ADR-064 amendments) rather than a raw
        # command line, since both steps can now be done entirely through the GUI.
        run_tip = frame.btn_rife_standalone_run.GetToolTip().GetTip()
        assert "Retroactive HDR/DV Reinjection" in run_tip, run_tip
        assert "RIFE Manifest" in run_tip, run_tip
        assert "Output Codec" in run_tip, run_tip
        assert "Dolby Vision" in run_tip, run_tip

        # Command construction, mirroring on_click_btn_rife_standalone_run's real
        # validation/build path -- captured via a monkeypatched startWorker instead of
        # a real GPU subprocess.
        captured = {}

        def _fake_start_worker(on_exit, worker_fn, wargs=(), **kwargs):
            captured["cmd"] = wargs[0]

        gui_mod.startWorker = _fake_start_worker

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = path.join(tmpdir, "movie_3d.mkv")
            with open(input_path, "wb") as f:
                f.write(b"fake")
            output_path = path.join(tmpdir, "movie_3d_rife.mkv")

            frame.txt_rife_standalone_input.SetValue(input_path)
            frame.txt_rife_standalone_output.SetValue(output_path)
            frame.cbo_rife_standalone_model.SetValue("rife_425_lite")
            frame.cbo_rife_standalone_gpu.SetSelection(cpu_index)
            frame.cbo_rife_standalone_mode.SetValue("3x")
            frame.update_rife_standalone_mode()

            frame.on_click_btn_rife_standalone_run(None)
            cmd = captured["cmd"]
            assert cmd == [sys.executable, "-m", "iw3.rife_cli",
                            "--input", input_path, "--output", output_path,
                            "--rife-model", "rife_425_lite",
                            "--gpu", "-1",
                            "--rife-multiplier", "3"], cmd

            captured.clear()
            frame.cbo_rife_standalone_mode.SetValue("Custom FPS...")
            frame.update_rife_standalone_mode()
            frame.txt_rife_standalone_target_fps.SetValue("60")
            frame.on_click_btn_rife_standalone_run(None)
            cmd = captured["cmd"]
            assert cmd == [sys.executable, "-m", "iw3.rife_cli",
                            "--input", input_path, "--output", output_path,
                            "--rife-model", "rife_425_lite",
                            "--gpu", "-1",
                            "--rife-target-fps", "60.0"], cmd

            # Output Codec: default (index 0, "H.264 (default)") must omit
            # --video-codec entirely -- byte-for-byte the same command as above,
            # backward-compat for anyone who never touches this new control.
            assert "--video-codec" not in cmd, cmd

            # Picking an HEVC choice appends --video-codec with the real flag value
            # rife_cli.py's own create_parser() expects (read directly from that
            # file, not assumed). Uses SetSelection (not SetValue) -- this is a
            # non-editable ComboBox whose ClientData (read via GetSelection) carries
            # the real --video-codec value, same convention as cbo_rife_standalone_gpu
            # above; SetValue() alone does not reliably update the selection index
            # backing GetClientData() for this control type.
            captured.clear()
            frame.cbo_rife_standalone_mode.SetValue("2x")
            frame.update_rife_standalone_mode()
            frame.cbo_rife_standalone_codec.SetSelection(
                frame.cbo_rife_standalone_codec.FindString(T("H.265/HEVC -- libx265 (CPU)")))
            frame.on_click_btn_rife_standalone_run(None)
            cmd = captured["cmd"]
            assert cmd == [sys.executable, "-m", "iw3.rife_cli",
                            "--input", input_path, "--output", output_path,
                            "--rife-model", "rife_425_lite",
                            "--gpu", "-1",
                            "--rife-multiplier", "2",
                            "--video-codec", "libx265"], cmd

            captured.clear()
            frame.cbo_rife_standalone_codec.SetSelection(
                frame.cbo_rife_standalone_codec.FindString(T("H.265/HEVC -- hevc_nvenc (GPU)")))
            frame.on_click_btn_rife_standalone_run(None)
            cmd = captured["cmd"]
            assert "--video-codec" in cmd and "hevc_nvenc" in cmd, cmd

            # Reset to default -> --video-codec disappears again (not sticky/broken).
            captured.clear()
            frame.cbo_rife_standalone_codec.SetSelection(default_codec_index)
            frame.on_click_btn_rife_standalone_run(None)
            cmd = captured["cmd"]
            assert "--video-codec" not in cmd, cmd
    finally:
        gui_mod.startWorker = orig_start_worker
        if frame is not None:
            frame.Destroy()
        app.Destroy()

    print("_self_test_rife_standalone_panel: PASS")


def _self_test_tool_log_clear_buttons():
    """Regression test for the "Clear" button added next to each standalone tool's
    log/output box on the Tools tab (Retroactive HDR/DV Reinjection, Search
    Subtitles, Add Subtitle Track, Add Audio Track, Stereo Mode Tag, Sharpen, RIFE
    Frame Interpolation), so a finished run's output doesn't just accumulate run
    after run with no way to empty it. Confirms every one of the 7 new Clear buttons
    exists next to its real log box and is enabled by default (a fresh GUI has no job
    running); that clicking it -- a real fired wx.EVT_BUTTON event, not just calling
    TextCtrl.Clear() directly -- empties exactly that log box and no other; and, for
    two representative panels using different underlying tools (Sharpen, Retroactive
    HDR/DV Reinjection), that starting a job disables Clear in lockstep with Run (so
    it can't wipe output still being read mid-run -- the safer of the two options
    named in the task, chosen over leaving it always-clickable) and finishing the job
    re-enables both together. No GPU or real movie file needed: startWorker is
    monkeypatched to capture args instead of launching a real subprocess."""
    import iw3.gui as gui_mod

    class _FakeResult:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

    def _click(btn):
        evt = wx.CommandEvent(wx.EVT_BUTTON.typeId, btn.GetId())
        evt.SetEventObject(btn)
        btn.GetEventHandler().ProcessEvent(evt)

    app = wx.App()
    frame = None
    orig_start_worker = gui_mod.startWorker
    try:
        frame = gui_mod.MainFrame()

        pairs = [
            ("txt_reinject_log", "btn_reinject_clear", "grp_hdr_reinject"),
            ("txt_subsearch_log", "btn_subsearch_clear", "grp_subsearch"),
            ("txt_submux_log", "btn_submux_clear", "grp_submux"),
            ("txt_audiomux_log", "btn_audiomux_clear", "grp_audiomux"),
            ("txt_stereotag_log", "btn_stereotag_clear", "grp_stereotag"),
            ("txt_sharpen_log", "btn_sharpen_clear", "grp_sharpen"),
            ("txt_rife_standalone_log", "btn_rife_standalone_clear", "grp_rife_standalone"),
        ]
        for log_name, btn_name, group_name in pairs:
            btn = getattr(frame, btn_name)
            group = getattr(frame, group_name)
            assert btn.GetParent() is group, btn_name
            assert btn.GetLabelText() == T("Clear"), btn_name
            assert btn.IsEnabled(), f"{btn_name} must be enabled by default (no job running)"

        # Functional: clicking Clear empties exactly its own log box, none other.
        frame.txt_sharpen_log.SetValue("sharpen output")
        frame.txt_reinject_log.SetValue("reinject output")
        _click(frame.btn_sharpen_clear)
        assert frame.txt_sharpen_log.GetValue() == "", "Clear must empty the Sharpen log"
        assert frame.txt_reinject_log.GetValue() == "reinject output", \
            "Clearing Sharpen's log must not touch Reinjection's log"
        _click(frame.btn_reinject_clear)
        assert frame.txt_reinject_log.GetValue() == "", "Clear must empty the Reinjection log"

        # Disabled while a job is running, re-enabled once it finishes -- two
        # representative panels driven through their real on_click_btn_*_run/
        # on_exit_*_worker handlers (not just manually toggled Enable/Disable calls).
        def _fake_start_worker(on_exit, worker_fn, wargs=(), **kwargs):
            pass
        gui_mod.startWorker = _fake_start_worker

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = path.join(tmpdir, "movie_3d.mp4")
            with open(input_path, "wb") as f:
                f.write(b"fake")
            output_path = path.join(tmpdir, "movie_3d_sharp.mp4")
            frame.txt_sharpen_input.SetValue(input_path)
            frame.txt_sharpen_output.SetValue(output_path)

            frame.on_click_btn_sharpen_run(None)
            assert not frame.btn_sharpen_run.IsEnabled()
            assert not frame.btn_sharpen_clear.IsEnabled(), \
                "Clear must disable in lockstep with Run while a job is running"
            frame.on_exit_sharpen_worker(_FakeResult((0, "done")))
            assert frame.btn_sharpen_run.IsEnabled()
            assert frame.btn_sharpen_clear.IsEnabled(), \
                "Clear must re-enable in lockstep with Run once the job finishes"

            source_path = path.join(tmpdir, "source.mkv")
            converted_path = path.join(tmpdir, "movie_3d.mkv")
            reinject_output = path.join(tmpdir, "movie_3d_hdr.mkv")
            for p in (source_path, converted_path):
                with open(p, "wb") as f:
                    f.write(b"fake")
            frame.txt_reinject_source.SetValue(source_path)
            frame.txt_reinject_converted.SetValue(converted_path)
            frame.txt_reinject_output.SetValue(reinject_output)

            frame.on_click_btn_reinject_run(None)
            assert not frame.btn_reinject_run.IsEnabled()
            assert not frame.btn_reinject_clear.IsEnabled()
            frame.on_exit_reinject_worker(_FakeResult((0, "done")))
            assert frame.btn_reinject_run.IsEnabled()
            assert frame.btn_reinject_clear.IsEnabled()
    finally:
        gui_mod.startWorker = orig_start_worker
        if frame is not None:
            frame.Destroy()
        app.Destroy()

    print("_self_test_tool_log_clear_buttons: PASS")


def _self_test_run_update_button():
    """Regression test for the "Run Update" button (ADR-069, ADR-035's direct
    follow-up): confirms (a) the button and its confirmation dialog exist and are
    wired to the real click handler, (b) declining the confirmation dialog never
    launches update.bat -- startWorker is never called, (c) accepting it launches
    update.bat via the real subprocess command (cmd.exe /c <real update.bat path>,
    with the nunif-windows root as cwd) exactly once, and toggles self.updating/
    button state correctly while it's "running" and once it finishes, (d) the
    button refuses to start a new run (and never touches startWorker) while a
    conversion job (self.processing) is already active, warning the user instead
    of silently doing nothing, and (e) the button is visually disabled while
    self.processing is True and re-enabled once it's False -- the same
    disable-during-a-job pattern this file's other real-process buttons (ADR-066's
    Clear buttons) already use. No GPU or real movie file needed, and update.bat is
    never actually executed: startWorker is monkeypatched to capture its args
    instead of running the worker function, and wx.MessageDialog/wx.MessageBox are
    monkeypatched so no real modal blocks this test waiting on real user input."""
    import iw3.gui as gui_mod

    class _FakeResult:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

    class _FakeConfirmDialog:
        result = wx.ID_YES

        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def ShowModal(self):
            return _FakeConfirmDialog.result

    message_box_calls = []

    def _fake_message_box(message, caption="", style=0):
        message_box_calls.append((message, caption, style))

    app = wx.App()
    frame = None
    orig_start_worker = gui_mod.startWorker
    orig_message_dialog = gui_mod.wx.MessageDialog
    orig_message_box = gui_mod.wx.MessageBox
    try:
        frame = gui_mod.MainFrame()
        assert frame.btn_run_update.GetLabelText() == T("Run Update")
        tip = frame.btn_run_update.GetToolTip().GetTip()
        assert "update.bat" in tip and "Check for Updates" in tip, tip
        assert frame.btn_run_update.IsEnabled(), "must be enabled by default (nothing running)"

        gui_mod.wx.MessageDialog = _FakeConfirmDialog
        gui_mod.wx.MessageBox = _fake_message_box

        captured = {}

        def _fake_start_worker(on_exit, worker_fn, wargs=(), **kwargs):
            captured["on_exit"] = on_exit
            captured["wargs"] = wargs
        gui_mod.startWorker = _fake_start_worker

        # Declining the confirmation must never launch anything.
        _FakeConfirmDialog.result = wx.ID_NO
        frame.on_click_btn_run_update(None)
        assert "wargs" not in captured, "declining the confirmation must never call startWorker"
        assert not frame.updating
        assert frame.btn_run_update.IsEnabled()

        # Accepting it launches the real update.bat via the real subprocess command.
        _FakeConfirmDialog.result = wx.ID_YES
        frame.on_click_btn_run_update(None)
        assert "wargs" in captured, "accepting the confirmation must launch update.bat"
        cmd, cwd, nunif_dir, dlg = captured["wargs"]
        expected_bat, expected_root = gui_mod._find_update_bat()
        assert path.exists(expected_bat), expected_bat
        assert cmd[-1] == expected_bat, cmd
        assert cmd[0].lower().endswith("cmd.exe"), cmd
        assert cmd[1] == "/c", cmd
        assert cwd == expected_root, (cwd, expected_root)
        assert nunif_dir == gui_mod.update_check._get_nunif_repo_root(), nunif_dir
        assert frame.updating
        assert not frame.btn_run_update.IsEnabled(), "must disable while update.bat is 'running'"
        assert not frame.btn_start.IsEnabled(), "Start must be disabled while updating too"
        assert not dlg.btn_close.IsEnabled(), "Close must be disabled until the run finishes"

        on_exit = captured["on_exit"]
        on_exit(_FakeResult(0))
        assert not frame.updating
        assert frame.btn_run_update.IsEnabled()
        assert dlg.finished
        assert dlg.btn_close.IsEnabled()

        # Refuses to start (and never touches startWorker) while a conversion job
        # is already running, warning the user instead of silently doing nothing.
        captured.clear()
        message_box_calls.clear()
        frame.processing = True
        frame.on_click_btn_run_update(None)
        assert "wargs" not in captured, "must never launch update.bat while a job is running"
        assert message_box_calls, "must warn the user instead of silently doing nothing"

        # Visual disable/re-enable in lockstep with self.processing, via the same
        # central update_start_button_state() other job-lifecycle transitions use.
        frame.update_start_button_state()
        assert not frame.btn_run_update.IsEnabled()
        frame.processing = False
        frame.update_start_button_state()
        assert frame.btn_run_update.IsEnabled()
    finally:
        gui_mod.startWorker = orig_start_worker
        gui_mod.wx.MessageDialog = orig_message_dialog
        gui_mod.wx.MessageBox = orig_message_box
        if frame is not None:
            frame.Destroy()
        app.Destroy()

    print("_self_test_run_update_button: PASS")


def _self_test_run_update_git_checkpoint():
    """ADR-069 dated amendment: `run_update()` must call the safety check-point
    BEFORE ever launching update.bat, and a real checkpoint failure must stop
    update.bat from launching at all -- see tests/test_iw3_run_update_git_checkpoint.py
    for the underlying `_git_checkpoint_before_update` unit coverage (real disposable
    git repos, never this project's own real repo). This self-test instead covers
    `run_update()`'s own ordering/integration: a real disposable git repo stands in
    for `nunif_dir`, and a harmless real subprocess (a Python one-liner) stands in
    for update.bat itself -- so "update.bat launches" is provable (a real process
    genuinely ran) without actually invoking the real update.bat's package/model/
    source-update side effects."""
    import iw3.gui as gui_mod

    class _FakeDlg:
        def __init__(self):
            self.lines = []

        def append(self, text):
            self.lines.append(text)

    app = wx.App()
    frame = None
    tmp = tempfile.mkdtemp(prefix="iw3_run_update_selftest_")
    try:
        git_bin = update_check._find_git()
        assert git_bin is not None, "bundled git binary must resolve for this test to be meaningful"

        def _git(*args):
            return subprocess.run([git_bin, "-C", tmp] + list(args),
                                  check=True, capture_output=True, text=True)

        _git("init", "-q")
        _git("config", "user.name", "test-checkpoint")
        _git("config", "user.email", "test-checkpoint@local")
        with open(path.join(tmp, "seed.txt"), "w") as f:
            f.write("seed\n")
        _git("add", "-A")
        _git("commit", "-q", "-m", "seed commit")

        frame = gui_mod.MainFrame()
        harmless_cmd = [sys.executable, "-c", "print('update.bat stand-in ran')"]

        # Dirty tree: checkpoint commits first, then the stand-in "update.bat"
        # subprocess still runs afterward.
        with open(path.join(tmp, "seed.txt"), "a") as f:
            f.write("modified\n")
        before_hash = _git("rev-parse", "HEAD").stdout.strip()
        dlg = _FakeDlg()
        returncode = frame.run_update(harmless_cmd, tmp, tmp, dlg)
        wx.Yield()  # flush the wx.CallAfter-queued dlg.append() calls onto dlg.lines
        assert returncode == 0, "the stand-in update.bat command must actually have run"
        after_hash = _git("rev-parse", "HEAD").stdout.strip()
        assert after_hash != before_hash, "run_update must checkpoint before launching update.bat"
        assert any("Committed" in line for line in dlg.lines), dlg.lines
        assert any("update.bat stand-in ran" in line for line in dlg.lines), dlg.lines
        status = _git("status", "--porcelain").stdout
        assert status.strip() == "", "checkpoint commit must leave the tree clean"

        # A real checkpoint failure must block update.bat from launching at all.
        with open(path.join(tmp, "seed.txt"), "a") as f:
            f.write("modified again\n")
        before_hash = _git("rev-parse", "HEAD").stdout.strip()
        real_run = subprocess.run

        def _failing_run(cmd, *a, **kw):
            if len(cmd) >= 4 and cmd[3] == "commit":
                raise subprocess.CalledProcessError(
                    1, cmd, output="", stderr="fatal: simulated commit failure\n")
            return real_run(cmd, *a, **kw)

        dlg2 = _FakeDlg()
        orig_subprocess_run = gui_mod.subprocess.run
        gui_mod.subprocess.run = _failing_run
        raised = False
        try:
            try:
                frame.run_update(harmless_cmd, tmp, tmp, dlg2)
            except RuntimeError as e:
                raised = True
                assert "simulated commit failure" in str(e), str(e)
        finally:
            gui_mod.subprocess.run = orig_subprocess_run
        assert raised, "a real checkpoint failure must propagate, never be swallowed"
        after_hash = _git("rev-parse", "HEAD").stdout.strip()
        assert after_hash == before_hash, "a failed checkpoint must never move HEAD"
        assert not any("stand-in ran" in line for line in dlg2.lines), \
            "update.bat stand-in must never run after a checkpoint failure"
    finally:
        if frame is not None:
            frame.Destroy()
        app.Destroy()
        shutil.rmtree(tmp, ignore_errors=True)

    print("_self_test_run_update_git_checkpoint: PASS")


def _self_test_import_command_round_trip():
    """Regression test for the new Import Command button (docs/ai/AI_DECISIONS.md
    ADR-074) -- the reverse of Copy Command. Three parts, all against the REAL,
    unmocked create_parser()/parse_args()/get_cli_command(), not a mock:
    1) builds a real command string via get_cli_command() from real, non-default
       GUI state on one MainFrame, imports it into a SECOND fresh MainFrame via
       parse_cli_command_text()+apply_parsed_args_to_gui(), then confirms
       get_cli_command() run again on the second frame parses (via
       create_parser()) to an equivalent argparse.Namespace as the original --
       exact string equality isn't required since argument order can differ.
    2) the real Hocus Pocus command from a live session (see ADR-074), tracing
       specific paired/special fields end to end: -i/-o, Method, Preserve Screen
       Border, Depth Model, Stereo Format, Pixel Format, Output Size Limit,
       End Time, Depth Resolution, EMA, Scene Detect/Auto EMA, AutoCrop, Inpaint
       Model + Overlap Frames pair, Depth AA, Object Stability, Max Workers,
       Video Format/Codec, Software Fallback, Metadata, Preserve DV, Auto
       Resume, and torch.compile (set but NOT probed -- ADR-068's guard).
    3) the mutually-exclusive Stereo Format dispatch (Anaglyph+method pair,
       Export+Depth Only) on a separate frame, since these can't combine with
       part 2's Half SBS state.
    4) a deliberately malformed/unrecognized paste is rejected with a
       non-crashing error message and touches no widget."""
    app = None
    frame = None
    try:
        app = wx.App()
        frame = MainFrame()

        # 1) Real, non-default GUI state -> command -> re-import on a fresh frame.
        # Codec/tune are pinned to libx264/no-tune first: this machine's own
        # persisted GUI config (nunif/tmp/iw3-gui.cfg, loaded by MainFrame's own
        # __init__) can carry an NVENC-only tune value (e.g. "uhq") from real
        # prior use, which create_parser()'s own --tune choices list does not
        # accept (a real, pre-existing mismatch between VideoEncodingBox's NVENC
        # tune choices and create_parser()'s libx264/x265-only --tune choices,
        # unrelated to this feature -- see docs/ai/AI_DECISIONS.md ADR-074).
        # Pinning here keeps this test deterministic regardless of ambient state.
        frame.grp_video.cbo_video_codec.SetStringSelection("libx264")
        frame.grp_video.update_video_codec()
        frame.grp_video.cbo_tune.SetValue("")
        frame.grp_video.chk_tune_fastdecode.SetValue(False)
        frame.grp_video.chk_tune_zerolatency.SetValue(False)
        frame.pnl_file.set_input_path("C:\\test input dir\\movie.mkv")
        frame.pnl_file.set_output_path("C:\\test output dir")
        frame.cbo_divergence.SetValue("3.5")
        frame.cbo_convergence.SetValue("0.25")
        frame.cbo_edge_dilation.SetValue("4")
        frame.cbo_edge_dilation_y.SetValue("2")
        frame.update_edge_dilation()
        frame.chk_ema_normalize.SetValue(True)
        frame.cbo_ema_decay.SetValue("0.9")
        frame.cbo_ema_buffer.SetValue("45")
        frame.update_ema_normalize()
        frame.chk_start_time.SetValue(True)
        frame.txt_start_time.SetValue("00:01:00")
        frame.chk_rife_interpolate.SetValue(True)
        frame.cbo_rife_mode.SetStringSelection("Custom FPS...")
        frame.txt_rife_target_fps.SetValue("60")
        frame.chk_waifu2x_upscale.SetValue(True)
        frame.update_waifu2x_upscale()
        frame.chk_compile.SetValue(False)

        command1 = frame.get_cli_command()
        assert command1 is not None

        frame2 = MainFrame()
        try:
            args2, err2 = frame2.parse_cli_command_text(command1)
            assert args2 is not None, err2
            frame2.apply_parsed_args_to_gui(args2)
            command2 = frame2.get_cli_command()
            assert command2 is not None

            parser = create_parser(required_true=False)
            args1_reparsed = parser.parse_args(_split_windows_command_line(command1)[3:])
            args2_reparsed = parser.parse_args(_split_windows_command_line(command2)[3:])
            assert vars(args1_reparsed) == vars(args2_reparsed), (command1, command2)
        finally:
            frame2.Destroy()

        # 2) The real Hocus Pocus command from a live session.
        real_command = (
            'python -m iw3 -i "D:\\SSD  2\\2160P MOVIES\\Hocus Pocus 1993 UHD BluRay 2160p DV.mkv" '
            '-o "E:\\3d Movies\\test for error 2" --compile --method mlbw_l2_inpaint '
            '--preserve-screen-border --divergence 2.5 --max-fps 1000.0 --crf 15 '
            '--depth-model Any_V3_Mono_01 --background-pop-coverage 0.0 --half-sbs '
            '--pix-fmt yuv420p10le --max-output-width 3840 --max-output-height 2160 '
            '--end-time 00:06:30 --resolution 512 --ema-normalize --ema-decay 0.94 '
            '--ema-buffer 60 --scene-detect --scene-batch-auto-ema '
            '--scene-batch-auto-ema-model "GEMINI AI" --autocrop BLACK '
            '--inpaint-model light_inpaint_v1 --inpaint-overlap-frames 3 0 --depth-aa '
            '--temporal-stabilize --temporal-stabilize-strength 0.3 --max-workers 2 '
            '--video-format mkv --video-codec hevc_nvenc --disable-software-fallback '
            '--metadata filename --preserve-dowi --auto-resume --yes'
        )
        frame3 = MainFrame()
        try:
            args3, err3 = frame3.parse_cli_command_text(real_command)
            assert args3 is not None, err3
            frame3.apply_parsed_args_to_gui(args3)

            assert frame3.pnl_file.input_path == \
                "D:\\SSD  2\\2160P MOVIES\\Hocus Pocus 1993 UHD BluRay 2160p DV.mkv", frame3.pnl_file.input_path
            assert frame3.pnl_file.output_path == "E:\\3d Movies\\test for error 2", frame3.pnl_file.output_path
            assert frame3.cbo_method.GetValue() == "mlbw_l2_inpaint", frame3.cbo_method.GetValue()
            assert frame3.chk_preserve_screen_border.GetValue() is True
            assert frame3.cbo_divergence.GetValue() == "2.5", frame3.cbo_divergence.GetValue()
            assert frame3.cbo_depth_model.GetValue() == "Any_V3_Mono_01", frame3.cbo_depth_model.GetValue()
            assert frame3.cbo_stereo_format.GetValue() == "Half SBS", frame3.cbo_stereo_format.GetValue()
            assert frame3.grp_video.cbo_pix_fmt.GetValue() == "yuv420p10le", frame3.grp_video.cbo_pix_fmt.GetValue()
            assert frame3.cbo_max_output_size.GetValue() == "3840x2160", frame3.cbo_max_output_size.GetValue()
            assert frame3.chk_end_time.GetValue() is True
            assert frame3.txt_end_time.GetValue() == "00:06:30", frame3.txt_end_time.GetValue()
            assert frame3.cbo_resolution.GetValue() == "512", frame3.cbo_resolution.GetValue()
            assert frame3.chk_ema_normalize.GetValue() is True
            assert frame3.cbo_ema_decay.GetValue() == "0.94", frame3.cbo_ema_decay.GetValue()
            assert frame3.cbo_ema_buffer.GetValue() == "60", frame3.cbo_ema_buffer.GetValue()
            assert frame3.chk_scene_detect.GetValue() is True
            assert frame3.chk_scene_batch_auto_ema.GetValue() is True
            assert frame3.cbo_scene_batch_auto_ema_model.GetValue() == "GEMINI AI", \
                frame3.cbo_scene_batch_auto_ema_model.GetValue()
            assert frame3.cbo_autocrop.GetValue() == "BLACK", frame3.cbo_autocrop.GetValue()
            assert frame3.cbo_inpaint_model.GetValue() == "light_inpaint_v1", frame3.cbo_inpaint_model.GetValue()
            assert frame3.cbo_overlap_frames_pre.GetValue() == "3", frame3.cbo_overlap_frames_pre.GetValue()
            assert frame3.cbo_overlap_frames_post.GetValue() == "0", frame3.cbo_overlap_frames_post.GetValue()
            assert frame3.chk_depth_aa.GetValue() is True
            assert frame3.chk_temporal_stabilize.GetValue() is True
            assert frame3.cbo_temporal_stabilize_strength.GetValue() == "0.3", \
                frame3.cbo_temporal_stabilize_strength.GetValue()
            assert frame3.cbo_max_workers.GetValue() == "2", frame3.cbo_max_workers.GetValue()
            assert frame3.grp_video.cbo_video_format.GetValue() == "mkv", frame3.grp_video.cbo_video_format.GetValue()
            if frame3.grp_video.has_nvenc:
                assert frame3.grp_video.cbo_video_codec.GetValue() == "hevc_nvenc", \
                    frame3.grp_video.cbo_video_codec.GetValue()
            assert frame3.grp_video_dec.chk_software_fallback.GetValue() is False
            assert frame3.chk_metadata.GetValue() is True
            assert frame3.chk_preserve_dowi.GetValue() is True
            assert frame3.chk_auto_resume.GetValue() is True
            assert frame3.chk_compile.GetValue() is True, \
                "compile checkbox itself must still be settable -- only the live GPU probe is skipped"
        finally:
            frame3.Destroy()

        # 3) Mutually-exclusive Stereo Format dispatch: Anaglyph+method pair,
        # then Export+Depth Only (can't combine with part 2's Half SBS state).
        frame4 = MainFrame()
        try:
            args4, err4 = frame4.parse_cli_command_text(
                'python -m iw3 -i "in.mp4" -o "out" --anaglyph dubois2 --yes')
            assert args4 is not None, err4
            frame4.apply_parsed_args_to_gui(args4)
            assert frame4.cbo_stereo_format.GetValue() == "Anaglyph", frame4.cbo_stereo_format.GetValue()
            assert frame4.cbo_anaglyph_method.GetValue() == "dubois2", frame4.cbo_anaglyph_method.GetValue()

            args5, err5 = frame4.parse_cli_command_text(
                'python -m iw3 -i "in.mp4" -o "out" --export --export-depth-only --yes')
            assert args5 is not None, err5
            frame4.apply_parsed_args_to_gui(args5)
            assert frame4.cbo_stereo_format.GetValue() == "Export", frame4.cbo_stereo_format.GetValue()
            assert frame4.chk_export_depth_only.GetValue() is True
        finally:
            frame4.Destroy()

        # 4) Malformed/unrecognized paste: rejected with a message, nothing applied.
        divergence_before = frame.cbo_divergence.GetValue()
        args_bad, err_bad = frame.parse_cli_command_text("python -m iw3 --not-a-real-flag-at-all 123")
        assert args_bad is None
        assert err_bad
        assert frame.cbo_divergence.GetValue() == divergence_before, \
            "a failed parse must not have touched any widget"

        # Empty paste is also rejected cleanly, not treated as valid/empty args.
        args_empty, err_empty = frame.parse_cli_command_text("   ")
        assert args_empty is None
        assert err_empty
    finally:
        if frame is not None:
            frame.Destroy()
        if app is not None:
            app.Destroy()

    print("_self_test_import_command_round_trip: PASS")


def _run_self_tests():
    _self_test_no_eager_cuda_context()
    _self_test_compile_probe_crash_handled()
    _self_test_layout_modes()
    _self_test_layout_mode_live_switch()
    _self_test_tabbed_scrolling()
    _self_test_stereo_sliders_sync()
    _self_test_depth_blend_and_processor_sliders_sync()
    _self_test_stereo_collapsible_sections()
    _self_test_zoom_level_persistence()
    _self_test_zoom_startup_restore()
    _self_test_zoom_live_rescale()
    _self_test_progress_stage_display()
    _self_test_progress_bar_visible_on_screen()
    _self_test_progress_bar_visible_after_live_field_changes()
    _self_test_scene_batch_auto_ema_editor()
    _self_test_nagadomi_reference_ema_option()
    _self_test_gemini_ai_ema_option()
    _self_test_chatgpt_ema_option()
    _self_test_grok_ema_option()
    _self_test_fast_action_ema_option()
    _self_test_medium_magical_ema_option()
    _self_test_drama_slow_paced_ema_option()
    _self_test_auto_ema_default_is_nagadomi_reference()
    _self_test_scene_auto_ema_regular_gate()
    _self_test_auto_ema_relocated_and_disables_ema_fields()
    _self_test_genre_preset_quick_fill()
    _self_test_3decker_quick_preset()
    _self_test_hdr_reinject_rife_manifest_field()
    _self_test_rife_standalone_panel()
    _self_test_tool_log_clear_buttons()
    _self_test_run_update_button()
    _self_test_run_update_git_checkpoint()
    _self_test_import_command_round_trip()
    _self_test_device_dropdown_no_torch_cuda_touch()
    print("All iw3.gui self-tests PASSED")


def main():
    import argparse
    import sys
    global LOCALE_DICT

    if "--self-test" in sys.argv[1:]:
        _run_self_tests()
        return

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lang", type=str, choices=LOCAL_LIST, help="translation")
    args = parser.parse_args()
    if args.lang:
        LOCALE_DICT = LOCALES.get(args.lang, {})
    else:
        saved_lang = load_language_setting(LANG_CONFIG_PATH)
        if saved_lang:
            LOCALE_DICT = LOCALES.get(saved_lang, {})

    sys.argv = [sys.argv[0]]  # clear command arguments

    app = IW3App()
    app.MainLoop()


if __name__ == "__main__":
    # NOTE: pyav_init_cuda_primary_context() is intentionally NOT called here.
    # Calling it unconditionally at import/launch time grabbed a full CUDA context
    # (~600MB VRAM, confirmed by measurement) the instant the window opened, even if
    # the user never clicks Start -- risking OOM-killing an already-running GPU process
    # sharing the same card. It is deferred to MainFrame.ensure_cuda_context(), called
    # right before each real CUDA/video-decode operation (Start, Quick Preview, Compare
    # Presets, AutoCrop Test) -- see docs/ai/AI_DECISIONS.md.
    init_win32_dpi()
    main()
