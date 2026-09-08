import nunif.pythonw_fix  # noqa
import nunif.gui.subprocess_patch  # noqa
import sys
import os
from os import path
import traceback
import functools
import tempfile
import subprocess
import copy
from time import time
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
)
from . import update_check
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
CONFIG_DIR = ensure_home_dir("iw3", path.join(path.dirname(__file__), "..", "tmp"))
CONFIG_PATH = path.join(CONFIG_DIR, "iw3-gui.cfg")
LANG_CONFIG_PATH = path.join(CONFIG_DIR, "iw3-gui-lang.cfg")
PRESET_DIR = path.join(CONFIG_DIR, "presets")
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(PRESET_DIR, exist_ok=True)

# GUI Layout preference (ADR-037): Tabbed (default, ADR-036's wx.Notebook) vs Single
# Page (every category StaticBox visible at once, restart required to switch -- see
# on_text_changed_cbo_layout). Persisted the same way as the Language setting above:
# a dedicated plain-text file, read once before any control is constructed, since the
# choice decides which parent widget the category StaticBoxes get built into.
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


LAYOUT_DEBUG = False


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
        self.processing = False
        self.start_time = 0
        self.input_type = None
        self.cuda_context_initialized = False
        self.stop_event = threading.Event()
        self.suspend_event = threading.Event()
        self.suspend_pos = 0
        self.suspend_event.set()
        self.depth_model = None
        self.depth_model_type = None
        self.depth_model_device_id = None
        self.depth_model_height = None
        self.depth_model_limit_resolution = None
        self.layout_mode = _load_layout_mode(LAYOUT_CONFIG_PATH)
        self.initialize_component()
        if is_dark_mode():
            apply_dark_mode(self)
        self.apply_accent_theme()

    def initialize_component(self):
        NORMAL_FONT = wx.Font(10, family=wx.FONTFAMILY_MODERN, style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL)
        WARNING_FONT = wx.Font(8, family=wx.FONTFAMILY_MODERN, style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL)
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
        # ADR-037: which WIDGET parents these 7 category panels depends on the user's
        # Layout preference (self.layout_mode, loaded before this method runs) -- Tabbed
        # parents them to a wx.Notebook page each (ADR-036's original design); Single
        # Page parents them directly to one scrollable panel so every category is
        # visible at once. Everything below this branch (every StaticBox/sizer built
        # inside each category, and each category panel's own SetSizer() call) is
        # identical either way; only the final composition step (see
        # _compose_options_layout_tabbed / _compose_options_layout_single_page near the
        # end of this method) differs.
        if self.layout_mode == LAYOUT_MODE_SINGLE_PAGE:
            self.pnl_single = scrolledpanel.ScrolledPanel(self.pnl_options)
            tabs_parent = self.pnl_single
        else:
            self.nb_options = wx.Notebook(self.pnl_options)
            tabs_parent = self.nb_options
        self.tab_stereo = wx.Panel(tabs_parent)
        self.tab_depth_blend = wx.Panel(tabs_parent)
        self.tab_video_filter = wx.Panel(tabs_parent)
        self.tab_video_dec = wx.Panel(tabs_parent)
        self.tab_video_enc = wx.Panel(tabs_parent)
        self.tab_processor = wx.Panel(tabs_parent)
        self.tab_tools = wx.Panel(tabs_parent)

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

        self.lbl_convergence_smoothing = wx.StaticText(self.grp_stereo, label=T("Convergence Smoothing"))
        self.cbo_convergence_smoothing = EditableComboBox(
            self.grp_stereo, choices=["0.95", "0.9", "0.75", "0.5", "0.25", "0"],
            name="cbo_convergence_smoothing")
        self.cbo_convergence_smoothing.SetSelection(1)
        self.cbo_convergence_smoothing.SetToolTip(
            T("Only affects sod_v1 / Face Detect convergence modes. Controls how quickly the automatic "
              "convergence point reacts to scene changes. Higher = smoother but slower to react. Lower = "
              "more aggressive/dynamic, reacts faster but may jitter more. 0 = no smoothing at all."))

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
              "nearer one fully overwriting the other — softer, less jagged edges than forward_fill, still "
              "no AI model/extra GPU cost involved. New and not yet extensively tested on real footage — "
              "try it if forward_fill's edges look too rough for your taste.\n"
              "monobw: a simpler, lighter backward-warp method than the mlbw family — a faster middle "
              "ground when mlbw is too slow but forward_fill's quality isn't good enough.\n"
              "Recommended: mlbw_l2_inpaint or forward_inpaint for the best quality on a real GPU; "
              "row_flow_v3_sym or mlbw_l2s if you need more speed or have limited VRAM."))

        self.lbl_inpaint_model = wx.StaticText(self.grp_stereo, label=T("Inpainting Model"))
        self.cbo_inpaint_model = wx.ComboBox(self.grp_stereo,
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

        self.lbl_overlap_frames = wx.StaticText(self.grp_stereo, label=T("Inpaint Overlap Frames"))
        self.cbo_overlap_frames_pre = EditableComboBox(self.grp_stereo,
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

        self.cbo_overlap_frames_post = EditableComboBox(self.grp_stereo,
                                                        choices=["0", "3"],
                                                        name="cbo_overlap_frames_post")
        self.cbo_overlap_frames_post.SetSelection(1)
        self.cbo_overlap_frames_post.SetToolTip(
            T("Overlap Post: same idea as Overlap Pre, but the extra context frames come from AFTER each "
              "chunk instead of before. Same values/tradeoff apply.\n"
              "Recommended: default (3)."))

        self.lbl_mask_dilation = wx.StaticText(self.grp_stereo, label=T("Inpaint Mask Dilation"))
        self.cbo_mask_inner_dilation = EditableComboBox(self.grp_stereo,
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

        self.cbo_mask_outer_dilation = EditableComboBox(self.grp_stereo,
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

        self.lbl_inpaint_max_width = wx.StaticText(self.grp_stereo, label=T("Inpaint Max Width"))
        self.cbo_inpaint_max_width = EditableComboBox(self.grp_stereo,
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

        self.lbl_stereo_width = wx.StaticText(self.grp_stereo, label=T("Stereo Processing Width"))
        self.cbo_stereo_width = EditableComboBox(self.grp_stereo,
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

        self.lbl_depth_model = wx.StaticText(self.grp_stereo, label=T("Depth Model"))
        self.cbo_depth_model = wx.ComboBox(self.grp_stereo,
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

        self.lbl_resolution = wx.StaticText(self.grp_stereo, label=T("Depth") + " " + T("Resolution"))
        self.cbo_resolution = EditableComboBox(self.grp_stereo,
                                               choices=["Default", "512"],
                                               name="cbo_zoed_resolution")
        self.cbo_resolution.SetSelection(0)
        self.cbo_resolution.SetToolTip(
            T("How much detail the depth model works with internally (its short-side resolution in "
              "pixels). \"Default\" uses ~392. Higher = finer depth detail but more VRAM/time — roughly "
              "squares the cost as you increase it. Recommended: Default for most content; try 448-512 "
              "if you have VRAM to spare and want finer depth detail."))

        self.chk_limit_resolution = wx.CheckBox(self.grp_stereo, label=T("Limit to source"),
                                                name="chk_limit_resolution")
        self.chk_limit_resolution.SetToolTip(
            T("Safety cap only: if your typed Depth Resolution is HIGHER than the source video's own "
              "resolution, this brings it back down to match the source instead of wasting time asking "
              "for detail that doesn't exist. It never raises a lower value up. Recommended: on."))

        self.lbl_foreground_scale = wx.StaticText(self.grp_stereo, label=T("Foreground Scale"))
        self.cbo_foreground_scale = EditableComboBox(self.grp_stereo,
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

        self.chk_depth_aa = wx.CheckBox(self.grp_stereo, label=T("Depth Anti-aliasing"), name="chk_depth_aa")
        self.chk_depth_aa.SetValue(False)
        self.chk_depth_aa.SetToolTip(
            T("Smooths small jagged/staircase artifacts in the depth map using a dedicated AI model, "
              "without changing the actual depth values much. Only available for certain depth models "
              "(grayed out otherwise). Recommended: on, when available — minor cost, generally cleaner result."))

        self.chk_depth_refine = wx.CheckBox(self.grp_stereo, label=T("Depth Detail Refinement"),
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
            self.grp_stereo, choices=["1.5", "1.25", "1.0", "0.75", "0.5", "0.25"],
            name="cbo_depth_refine_strength")
        self.cbo_depth_refine_strength.SetSelection(2)
        self.cbo_depth_refine_strength.SetToolTip(
            T("How strong Depth Detail Refinement's cleanup is. 1.0 = the original fixed strength this "
              "feature always used. Higher = more smoothing reach (cleaner depth boundaries, but risks "
              "softening genuinely fine depth detail if pushed too far); lower = gentler, closer to doing "
              "nothing. Recommended: 1.0 as a safe starting point; try 1.25-1.5 if you want a bit more of "
              "the \"cleaner/more solid 3D\" effect this setting gives."))

        self.chk_temporal_stabilize = wx.CheckBox(self.grp_stereo, label=T("Object Stability (experimental)"),
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

        self.cbo_temporal_stabilize_strength = EditableComboBox(self.grp_stereo,
                                                                 choices=["0.9", "0.7", "0.5", "0.3"],
                                                                 name="cbo_temporal_stabilize_strength")
        self.cbo_temporal_stabilize_strength.SetSelection(1)
        self.cbo_temporal_stabilize_strength.SetToolTip(
            T("How strongly to trust the motion-warped previous frame vs the fresh per-frame depth (0-1). "
              "Automatically tapers down during fast/unreliable motion regardless of this setting."))

        self.lbl_temporal_stabilize_max_shift = wx.StaticText(self.grp_stereo, label=T("Max Shift"))
        self.cbo_temporal_stabilize_max_shift = EditableComboBox(
            self.grp_stereo,
            choices=["", "0.01", "0.02", "0.05"],
            name="cbo_temporal_stabilize_max_shift")
        self.cbo_temporal_stabilize_max_shift.SetSelection(0)
        self.cbo_temporal_stabilize_max_shift.SetToolTip(
            T("Object Stability: hard cap on how much depth is allowed to change for the same pixel "
              "between two consecutive output frames (0-1 scale, same units as depth value). Stops a "
              "single-frame spike from ever \"popping\", no matter how strong the raw model's disagreement "
              "is. Leave blank to disable (no cap, original behavior)."))

        self.lbl_temporal_stabilize_flat_boost = wx.StaticText(self.grp_stereo, label=T("Flat-Area Boost"))
        self.cbo_temporal_stabilize_flat_boost = EditableComboBox(
            self.grp_stereo,
            choices=["0.0", "0.3", "0.5", "0.7"],
            name="cbo_temporal_stabilize_flat_boost")
        self.cbo_temporal_stabilize_flat_boost.SetSelection(0)
        self.cbo_temporal_stabilize_flat_boost.SetToolTip(
            T("Object Stability: extra smoothing specifically in areas the CURRENT frame's own depth is "
              "flat (sky, walls, floors) -- these are exactly the areas where flicker is most visible and "
              "least likely to be real motion. 0 = no extra smoothing (original behavior)."))

        self.lbl_temporal_stabilize_edge_protect = wx.StaticText(self.grp_stereo, label=T("Edge Protection"))
        self.cbo_temporal_stabilize_edge_protect = EditableComboBox(
            self.grp_stereo,
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
        self.cbo_depth_blend_bilateral_sigma_color = EditableComboBox(
            self.grp_depth_blend, choices=["50", "75", "100"], name="cbo_depth_blend_bilateral_sigma_color")
        self.cbo_depth_blend_bilateral_sigma_color.SetSelection(1)
        self.cbo_depth_blend_bilateral_sigma_color.SetToolTip(
            T("Bilateral filter's depth-value sensitivity, expressed in familiar 0-255-ish terms (matches "
              "common 8-bit filter presets; scaled internally to this pipeline's real 16-bit depth range). "
              "Higher = willing to smooth across BIGGER depth differences, which risks blurring real depth "
              "edges, not just noise; lower = only smooths very similar depth values together, safer for "
              "real edges but cleans up less noise. Recommended: 75 as a balanced starting point."))
        self.cbo_depth_blend_bilateral_sigma_space = EditableComboBox(
            self.grp_depth_blend, choices=["50", "75", "100"], name="cbo_depth_blend_bilateral_sigma_space")
        self.cbo_depth_blend_bilateral_sigma_space.SetSelection(1)
        self.cbo_depth_blend_bilateral_sigma_space.SetToolTip(
            T("Bilateral filter's spatial reach, in pixels. Higher = smooths across a physically wider "
              "area of the frame; lower = stays more localized. Recommended: 75 as a balanced starting "
              "point, paired with the diameter/color settings above."))

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
        self.cbo_depth_blend_clahe_tile = EditableComboBox(self.grp_depth_blend,
                                                            choices=["4", "8", "16"],
                                                            name="cbo_depth_blend_clahe_tile")
        self.cbo_depth_blend_clahe_tile.SetSelection(1)
        self.cbo_depth_blend_clahe_tile.SetToolTip(
            T("CLAHE tile grid size (NxN) — how finely the frame is divided up for LOCAL contrast "
              "adjustment. More tiles = more localized (small-area) contrast changes; fewer tiles = "
              "smoother, more global adjustment. Only matters if CLAHE Contrast is turned on."))

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

        layout_depth_blend.Add((0, 6), (j := j + 1, 0))
        layout_depth_blend.Add(wx.StaticLine(self.grp_depth_blend), (j := j + 1, 0), (0, 3), flag=wx.EXPAND)
        layout_depth_blend.Add((0, 4), (j := j + 1, 0))
        layout_depth_blend.Add(self.lbl_depth_blend_feather_blur, (j := j + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout_depth_blend.Add(self.cbo_depth_blend_feather_blur, (j, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.chk_depth_blend_bilateral, (j := j + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout_depth_blend.Add(self.cbo_depth_blend_bilateral_d, (j, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.cbo_depth_blend_bilateral_sigma_color, (j, 2), flag=wx.EXPAND)
        layout_depth_blend.Add(self.cbo_depth_blend_bilateral_sigma_space, (j := j + 1, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.chk_depth_blend_clahe, (j := j + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout_depth_blend.Add(self.cbo_depth_blend_clahe_clip, (j, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.cbo_depth_blend_clahe_tile, (j, 2), flag=wx.EXPAND)

        layout_depth_blend.Add((0, 6), (j := j + 1, 0))
        layout_depth_blend.Add(wx.StaticLine(self.grp_depth_blend), (j := j + 1, 0), (0, 3), flag=wx.EXPAND)
        layout_depth_blend.Add((0, 4), (j := j + 1, 0))
        layout_depth_blend.Add(self.chk_depth_blend_align, (j := j + 1, 0), (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout_depth_blend.Add(self.cbo_depth_blend_align_decay, (j, 2), flag=wx.EXPAND)

        layout_depth_blend.Add((0, 6), (j := j + 1, 0))
        layout_depth_blend.Add(wx.StaticLine(self.grp_depth_blend), (j := j + 1, 0), (0, 3), flag=wx.EXPAND)
        layout_depth_blend.Add((0, 4), (j := j + 1, 0))
        layout_depth_blend.Add(self.lbl_depth_blend_edge_suppression, (j := j + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout_depth_blend.Add(self.cbo_depth_blend_edge_suppression, (j, 1), flag=wx.EXPAND)
        layout_depth_blend.Add(self.chk_depth_blend_edge_hard_cutoff, (j, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        sizer_depth_blend = wx.StaticBoxSizer(self.grp_depth_blend, wx.VERTICAL)
        sizer_depth_blend.Add(layout_depth_blend, 1, wx.ALL | wx.EXPAND, 4)

        self.lbl_foreground_pop = wx.StaticText(self.grp_stereo, label=T("Foreground Pop"))
        self.cbo_foreground_pop = EditableComboBox(self.grp_stereo,
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

        self.lbl_foreground_divergence = wx.StaticText(self.grp_stereo, label=T("Foreground Divergence"))
        self.cbo_foreground_divergence = EditableComboBox(self.grp_stereo,
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

        self.lbl_background_pop = wx.StaticText(self.grp_stereo, label=T("Background Pop"))
        self.cbo_background_pop = EditableComboBox(self.grp_stereo,
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

        self.lbl_background_pop_coverage = wx.StaticText(self.grp_stereo, label=T("Background Pop Coverage %"))
        self.cbo_background_pop_coverage = EditableComboBox(self.grp_stereo,
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

        self.lbl_background_divergence = wx.StaticText(self.grp_stereo, label=T("Background Divergence"))
        self.cbo_background_divergence = EditableComboBox(self.grp_stereo,
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

        self.lbl_edge_repair = wx.StaticText(self.grp_stereo, label=T("Edge Repair"))
        self.cbo_edge_repair = EditableComboBox(self.grp_stereo,
                                                choices=["0.0", "0.25", "0.5", "0.75", "1.0"],
                                                name="cbo_edge_repair")
        self.cbo_edge_repair.SetSelection(0)
        self.cbo_edge_repair.SetToolTip(
            T("Final cleanup pass on the finished 3D image (applies no matter which Stereo Method made it). "
              "Gently smooths a thin hairline right around real depth edges to reduce fringing/ghosting "
              "residue left over from the 3D warp. Cannot affect flat areas or anywhere without a depth "
              "edge. 0.0=off (default), 1.0=strongest."))

        self.lbl_edge_dilation = wx.StaticText(self.grp_stereo, label=T("Edge Fix"))
        self.cbo_edge_dilation = EditableComboBox(self.grp_stereo,
                                                  choices=["0", "1", "2", "3", "4"],
                                                  size=self.FromDIP((90, -1)),
                                                  name="cbo_edge_dilation")
        self.cbo_edge_dilation_y = EditableComboBox(self.grp_stereo,
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

        self.chk_ema_normalize = wx.CheckBox(self.grp_stereo,
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

        self.cbo_ema_decay = EditableComboBox(self.grp_stereo, choices=["0.99", "0.95", "0.9", "0.75", "0.5", "0"],
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
              "artifacts)."))

        self.cbo_ema_buffer = EditableComboBox(self.grp_stereo, choices=["150", "60", "30", "1"],
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
              "assumptions about pacing were sometimes simply wrong."))

        self.chk_ema_motion_adaptive = wx.CheckBox(self.grp_stereo,
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

        self.chk_scene_detect = wx.CheckBox(self.grp_stereo,
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

        self.chk_scene_detect_cache = wx.CheckBox(self.grp_stereo,
                                                  label=T("Use scene boundary cache"),
                                                  name="chk_scene_detect_cache")
        self.chk_scene_detect_cache.SetValue(True)
        self.chk_scene_detect_cache.SetToolTip(
            T("What it's for: saves detected scene cuts to disk (keyed to the source file's path, size, "
              "and modification time), so re-running the same video later — after a crash, or while "
              "testing different 3D settings — doesn't need to re-scan for cuts from scratch every time.\n"
              "Recommended: on. It auto-invalidates if the source file actually changes (different size "
              "or modified date), so there's no real downside for normal use."))

        self.chk_preserve_screen_border = wx.CheckBox(self.grp_stereo,
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
        layout.Add(self.lbl_divergence_warning, pos=(i := i + 1, 0), span=(0, 3), flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_convergence, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_convergence_mode, (i, 1), flag=wx.EXPAND)
        layout.Add(self.cbo_convergence, (i, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_convergence_smoothing, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_convergence_smoothing, (i, 1), (1, 2), flag=wx.EXPAND)

        layout.Add(self.lbl_ipd_offset, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.sld_ipd_offset, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_synthetic_view, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_synthetic_view, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_method, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_method, (i, 1), (1, 2), flag=wx.EXPAND)

        layout.Add((0, 8), (i := i + 1, 0))
        layout.Add(wx.StaticLine(self.grp_stereo), (i := i + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 6), (i := i + 1, 0))
        layout.Add(self.lbl_inpaint_model, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_inpaint_model, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_overlap_frames, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_overlap_frames_pre, (i, 1), flag=wx.EXPAND)
        layout.Add(self.cbo_overlap_frames_post, (i, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_mask_dilation, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_mask_inner_dilation, (i, 1), flag=wx.EXPAND)
        layout.Add(self.cbo_mask_outer_dilation, (i, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_inpaint_max_width, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_inpaint_max_width, (i, 1), (1, 2), flag=wx.EXPAND)

        layout.Add((0, 8), (i := i + 1, 0))
        layout.Add(wx.StaticLine(self.grp_stereo), (i := i + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 6), (i := i + 1, 0))
        layout.Add(self.lbl_stereo_width, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_stereo_width, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_depth_model, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_depth_model, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_resolution, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_resolution, (i, 1), flag=wx.EXPAND)
        layout.Add(self.chk_limit_resolution, (i, 2), flag=wx.EXPAND)

        layout.Add((0, 8), (i := i + 1, 0))
        layout.Add(wx.StaticLine(self.grp_stereo), (i := i + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 6), (i := i + 1, 0))
        layout.Add(self.lbl_foreground_scale, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_foreground_scale, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_edge_dilation, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_edge_dilation, (i, 1), flag=wx.EXPAND)
        layout.Add(self.cbo_edge_dilation_y, (i, 2), flag=wx.EXPAND)
        layout.Add(self.chk_depth_aa, (i := i + 1, 1), (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_depth_refine, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_depth_refine_strength, (i, 1), flag=wx.EXPAND)
        layout.Add(self.chk_temporal_stabilize, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_temporal_stabilize_strength, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_temporal_stabilize_max_shift, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=14)
        layout.Add(self.cbo_temporal_stabilize_max_shift, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_temporal_stabilize_flat_boost, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=14)
        layout.Add(self.cbo_temporal_stabilize_flat_boost, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_temporal_stabilize_edge_protect, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=14)
        layout.Add(self.cbo_temporal_stabilize_edge_protect, (i, 1), flag=wx.EXPAND)

        layout.Add((0, 8), (i := i + 1, 0))
        layout.Add(wx.StaticLine(self.grp_stereo), (i := i + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 6), (i := i + 1, 0))
        layout.Add(self.lbl_foreground_pop, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_foreground_pop, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_foreground_divergence, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_foreground_divergence, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_background_pop, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_background_pop, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_background_pop_coverage, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_background_pop_coverage, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_background_divergence, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_background_divergence, (i, 1), (1, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_edge_repair, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_edge_repair, (i, 1), (1, 2), flag=wx.EXPAND)

        layout.Add((0, 8), (i := i + 1, 0))
        layout.Add(wx.StaticLine(self.grp_stereo), (i := i + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 6), (i := i + 1, 0))
        layout.Add(self.chk_ema_normalize, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_ema_decay, (i, 1), flag=wx.EXPAND)
        layout.Add(self.cbo_ema_buffer, (i, 2), flag=wx.EXPAND)
        layout.Add(self.chk_ema_motion_adaptive, (i := i + 1, 0), (0, 3), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_scene_detect, (i := i + 1, 0), (0, 1), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_scene_detect_cache, (i, 1), (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_preserve_screen_border, (i := i + 1, 0), (0, 3), flag=wx.ALIGN_CENTER_VERTICAL)

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

        self.chk_scene_batch_auto_ema = wx.CheckBox(self.grp_video_filter,
                                                    label=T("Auto EMA by Scene Length"),
                                                    name="chk_scene_batch_auto_ema")
        self.chk_scene_batch_auto_ema.SetValue(False)
        self.chk_scene_batch_auto_ema.SetToolTip(
            T("For Automated Scene Batch. Automatically picks EMA Decay/Buffer per scene based on "
              "that scene's own length, using a built-in table (one step per whole second, 0-15s+) "
              "tuned for independent-scene processing -- a short scene gets a smaller Buffer so the "
              "smoothing actually finishes settling before the scene ends, instead of a Buffer sized "
              "for one long continuous shot. Applied before Scene Settings File, so anything that "
              "file sets explicitly (EMA included) still wins for scenes it covers. Pick the "
              "matching Depth Model in the dropdown to its right."))

        self.cbo_scene_batch_auto_ema_model = wx.ComboBox(self.grp_video_filter,
                                                          choices=["VDA_L", "Any_V3_Mono_01"],
                                                          name="cbo_scene_batch_auto_ema_model")
        self.cbo_scene_batch_auto_ema_model.SetEditable(False)
        self.cbo_scene_batch_auto_ema_model.SetSelection(0)
        self.cbo_scene_batch_auto_ema_model.SetToolTip(
            T("Which Auto EMA by Scene Length table to use, matched to the Depth Model above. "
              "VDA_L: a real video depth model with its own frame-to-frame memory, needs only light "
              "smoothing on top. Any_V3_Mono_01: a stills-only model with no frame-to-frame memory "
              "of its own (prone to visible 'depth breathing' without help), so this table uses "
              "double VDA_L's Buffer at every scene length with a correspondingly higher Decay to "
              "compensate."))

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
        layout.Add(self.chk_scene_batch_auto_ema, (i := i + 1, 1), (0, 1), flag=wx.EXPAND | wx.LEFT, border=14)
        layout.Add(self.cbo_scene_batch_auto_ema_model, (i, 2), (0, 1), flag=wx.EXPAND)
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
        if torch.cuda.is_available():
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

        self.chk_low_vram = wx.CheckBox(self.grp_processor, label=T("Low VRAM"), name="chk_low_vram")
        self.chk_low_vram.SetToolTip(
            T("Trades speed for lower memory use, by processing in a way that needs less VRAM at once. "
              "Only turn this on if you're actually running out of memory — it will make things slower."))
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

        layout = wx.GridBagSizer(vgap=5, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        k = -1
        layout.Add(self.lbl_device, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_device, (k, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_batch_size, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_batch_size, (k, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_max_workers, (k := k + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_max_workers, (k, 1), (0, 3), flag=wx.EXPAND)

        layout.Add((0, 6), (k := k + 1, 0))
        layout.Add(wx.StaticLine(self.grp_processor), (k := k + 1, 0), (0, 4), flag=wx.EXPAND)
        layout.Add((0, 4), (k := k + 1, 0))
        layout.Add(self.chk_low_vram, (k := k + 1, 0), flag=wx.EXPAND)
        layout.Add(self.chk_tta, (k, 1), flag=wx.EXPAND)
        layout.Add(self.chk_fp16, (k, 2), flag=wx.EXPAND)
        layout.Add(self.chk_cuda_stream, (k, 3), flag=wx.EXPAND)
        layout.Add(self.chk_compile, (k := k + 1, 0), flag=wx.EXPAND)

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
        self.chk_waifu2x_upscale.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_waifu2x_upscale)
        self.update_waifu2x_upscale()

        self.chk_rife_interpolate = wx.CheckBox(self.grp_postprocess,
                                                label=T("Interpolate frames with RIFE after conversion"),
                                                name="chk_rife_interpolate")
        self.chk_rife_interpolate.SetValue(False)
        self.chk_rife_interpolate.SetToolTip(
            T("What it's for (single video and Dual-Pass Depth Blend jobs only): once this job's finished "
              "output is fully written, runs it through RIFE (a separate AI frame-interpolation model) as "
              "one extra step, generating a new in-between frame for every pair of real frames -- doubling "
              "the effective frame rate for smoother-looking motion.\n"
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
        self.chk_rife_interpolate.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_rife_interpolate)
        self.update_rife_interpolate()

        layout = wx.GridBagSizer(vgap=5, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        j = -1
        layout.Add(self.chk_waifu2x_upscale, (j := j + 1, 0), (0, 3), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_waifu2x_method, (j := j + 1, 0), flag=wx.EXPAND | wx.LEFT, border=14)
        layout.Add(self.cbo_waifu2x_noise_level, (j, 1), flag=wx.EXPAND)
        layout.Add(self.cbo_waifu2x_style, (j, 2), flag=wx.EXPAND)

        layout.Add((0, 6), (j := j + 1, 0))
        layout.Add(wx.StaticLine(self.grp_postprocess), (j := j + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 4), (j := j + 1, 0))
        layout.Add(self.chk_rife_interpolate, (j := j + 1, 0), (0, 3), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_rife_model, (j := j + 1, 0), flag=wx.EXPAND | wx.LEFT, border=14)
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

        self.btn_reinject_source.Bind(wx.EVT_BUTTON, self.on_click_btn_reinject_source)
        self.btn_reinject_converted.Bind(wx.EVT_BUTTON, self.on_click_btn_reinject_converted)
        self.btn_reinject_output.Bind(wx.EVT_BUTTON, self.on_click_btn_reinject_output)
        self.btn_reinject_run.Bind(wx.EVT_BUTTON, self.on_click_btn_reinject_run)

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
        # Start/End Time + Run share one compact row rather than each taking a row of
        # their own -- this whole section lives in the Dual-Pass Depth Blend column,
        # which has limited spare vertical room.
        layout.Add(self.chk_reinject_start_time, (h := h + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_reinject_start_time, (h, 1), flag=wx.EXPAND)
        layout.Add(self.chk_reinject_end_time, (h, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_reinject_end_time, (h, 3), flag=wx.EXPAND)
        layout.Add(self.btn_reinject_run, (h := h + 1, 3), flag=wx.EXPAND)
        layout.Add(self.txt_reinject_log, (h, 0), (0, 3), flag=wx.EXPAND)
        sizer_hdr_reinject = wx.StaticBoxSizer(self.grp_hdr_reinject, wx.VERTICAL)
        sizer_hdr_reinject.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

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
              "correct layout here explicitly in that case. This value doesn't change how the "
              "subtitle is added either way (a plain track works the same regardless of layout) -- "
              "it's just a safety check so the tool never guesses wrong silently.\n"
              "Recommended: leave on 'auto' unless the tool's log below reports it couldn't detect "
              "the format."))

        self.lbl_submux_language = wx.StaticText(self.grp_submux, label=T("Language"))
        self.txt_submux_language = wx.TextCtrl(self.grp_submux, value="eng", name="txt_submux_language")
        self.txt_submux_language.SetToolTip(
            T("What it's for: the ISO 639-2 language code stored as metadata on the new subtitle "
              "track (e.g. eng, jpn, fre, ger, spa) -- shown by players in their subtitle track "
              "menu.\n"
              "Con: purely metadata -- does not translate or verify the actual subtitle content's "
              "language.\n"
              "Recommended: match the SRT file's actual language; default 'eng' if unsure."))

        self.lbl_submux_track_name = wx.StaticText(self.grp_submux, label=T("Track Name"))
        self.txt_submux_track_name = wx.TextCtrl(self.grp_submux, name="txt_submux_track_name")
        self.txt_submux_track_name.SetToolTip(
            T("What it's for: an optional display name for the new subtitle track (shown in "
              "player track menus, e.g. 'English (Forced)'). Leave blank to default to the SRT "
              "file's own name."))

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

        self.btn_submux_input.Bind(wx.EVT_BUTTON, self.on_click_btn_submux_input)
        self.btn_submux_srt.Bind(wx.EVT_BUTTON, self.on_click_btn_submux_srt)
        self.btn_submux_output.Bind(wx.EVT_BUTTON, self.on_click_btn_submux_output)
        self.btn_submux_run.Bind(wx.EVT_BUTTON, self.on_click_btn_submux_run)

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
        layout.Add(self.btn_submux_run, (h := h + 1, 3), flag=wx.EXPAND)
        layout.Add(self.txt_submux_log, (h, 0), (0, 3), flag=wx.EXPAND)
        sizer_submux = wx.StaticBoxSizer(self.grp_submux, wx.VERTICAL)
        sizer_submux.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

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

        self.btn_stereotag_input.Bind(wx.EVT_BUTTON, self.on_click_btn_stereotag_input)
        self.btn_stereotag_run.Bind(wx.EVT_BUTTON, self.on_click_btn_stereotag_run)

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
        sizer_stereotag = wx.StaticBoxSizer(self.grp_stereotag, wx.VERTICAL)
        sizer_stereotag.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

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
        tab_layout.Add(sizer_submux, 0, wx.ALL | wx.EXPAND, 4)
        tab_layout.Add(sizer_stereotag, 0, wx.ALL | wx.EXPAND, 4)
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

        # preset comparison test
        self.sep_compare_preset = wx.StaticLine(self.pnl_preset, size=self.FromDIP((2, 20)), style=wx.LI_VERTICAL)
        self.btn_compare_presets = wx.Button(self.pnl_preset, label=T("Compare Presets..."))
        self.btn_compare_presets.SetToolTip(
            T("Render the same short test clip with 2 or more saved presets, "
              "then join the results back-to-back into one comparison video"))

        # copy command
        self.sep_command = wx.StaticLine(self.pnl_preset, size=self.FromDIP((2, 20)), style=wx.LI_VERTICAL)
        self.btn_copy_command = wx.Button(self.pnl_preset, label=T("Copy Command"))

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

        # GUI layout preference (ADR-037): Tabbed vs. Single Page. Persisted like
        # Language, in its own file, applied on next launch -- see
        # on_text_changed_cbo_layout and docs/ai/AI_DECISIONS.md.
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
              "Note: takes effect after restarting 3DECKER — changing it just saves the preference for "
              "next launch."))

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
        layout.AddSpacer(2)
        layout.Add(self.sep_compare_preset, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT)
        layout.AddSpacer(4)
        layout.Add(self.btn_compare_presets, flag=wx.ALL, border=2)
        layout.AddSpacer(2)
        layout.Add(self.sep_command, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT)
        layout.AddSpacer(4)
        layout.Add(self.btn_copy_command, flag=wx.ALL, border=2)

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
        layout.Add(self.sep_update, flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT)
        layout.AddSpacer(4)
        layout.Add(self.btn_check_updates, flag=wx.ALL, border=2)
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

        self.cbo_divergence.Bind(wx.EVT_TEXT, self.update_divergence_warning)
        self.cbo_synthetic_view.Bind(wx.EVT_TEXT, self.update_divergence_warning)
        self.cbo_method.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_method)
        self.lbl_divergence_warning.Bind(wx.EVT_LEFT_DOWN, self.on_click_divergence_warning)

        self.cbo_depth_model.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_depth_model)
        self.cbo_edge_dilation_y.Bind(wx.EVT_TEXT, self.on_changed_edge_dilation)
        self.chk_ema_normalize.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_ema_normalize)
        self.chk_depth_blend.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_depth_blend)
        self.chk_temporal_stabilize.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_temporal_stabilize)
        self.cbo_depth_blend_region.Bind(wx.EVT_TEXT, self.on_changed_chk_depth_blend)
        self.chk_depth_blend_bilateral.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_depth_blend_bilateral)
        self.chk_depth_blend_clahe.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_depth_blend_clahe)
        self.chk_depth_blend_align.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_depth_blend_align)
        self.chk_depth_refine.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_depth_refine)

        self.cbo_stereo_format.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_stereo_format)

        self.cbo_pad_mode.Bind(wx.EVT_TEXT, self.update_pad_mode)

        self.cbo_device.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_device)
        self.chk_compile.Bind(wx.EVT_CHECKBOX, self.update_compile)

        self.btn_load_preset.Bind(wx.EVT_BUTTON, self.on_click_btn_load_preset)
        self.btn_save_preset.Bind(wx.EVT_BUTTON, self.on_click_btn_save_preset)
        self.btn_delete_preset.Bind(wx.EVT_BUTTON, self.on_click_btn_delete_preset)
        self.btn_quick_preset_movie.Bind(wx.EVT_BUTTON, lambda event: self.apply_quick_preset("movie"))
        self.btn_quick_preset_action.Bind(wx.EVT_BUTTON, lambda event: self.apply_quick_preset("action"))
        self.btn_compare_presets.Bind(wx.EVT_BUTTON, self.on_click_btn_compare_presets)
        self.btn_copy_command.Bind(wx.EVT_BUTTON, self.on_click_btn_copy_command)
        self.cbo_language.Bind(wx.EVT_TEXT, self.on_text_changed_cbo_language)
        self.cbo_layout.Bind(wx.EVT_TEXT, self.on_text_changed_cbo_layout)
        self.btn_check_updates.Bind(wx.EVT_BUTTON, self.on_click_btn_check_updates)

        self.btn_autocrop_test.Bind(wx.EVT_BUTTON, self.on_click_btn_autocrop_test)
        self.btn_scene_settings.Bind(wx.EVT_BUTTON, self.on_click_btn_scene_settings)

        self.btn_start.Bind(wx.EVT_BUTTON, self.on_click_btn_start)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self.on_click_btn_cancel)
        self.btn_suspend.Bind(wx.EVT_BUTTON, self.on_click_btn_suspend)
        self.btn_quick_preview.Bind(wx.EVT_BUTTON, self.on_click_btn_quick_preview)

        self.Bind(EVT_TQDM, self.on_tqdm)
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
        """ADR-036/ADR-037 -- Tabbed layout: the 7 category panels each become one
        wx.Notebook page. This is exactly ADR-036's original composition step,
        unchanged, just extracted into its own method so ADR-037 can pick between it
        and _compose_options_layout_single_page() based on the user's saved Layout
        preference."""
        self.nb_options.AddPage(self.tab_stereo, T("Stereo Generation"))
        self.nb_options.AddPage(self.tab_depth_blend, T("Dual-Pass Depth Blend"))
        self.nb_options.AddPage(self.tab_video_filter, T("Video Filter"))
        self.nb_options.AddPage(self.tab_video_dec, T("Video Decoding"))
        self.nb_options.AddPage(self.tab_video_enc, T("Video Encoding"))
        self.nb_options.AddPage(self.tab_processor, T("Processor"))
        self.nb_options.AddPage(self.tab_tools, T("Standalone Tools"))
        # Force a deterministic starting tab -- without this, wx sometimes lands on
        # whichever page happens to contain the last control touched by a SetSelection()
        # call made deep inside a sub-panel's own __init__ (e.g. VideoEncodingBox's
        # cbo_video_format) instead of the first page, which looked like a random tab
        # on launch.
        self.nb_options.SetSelection(0)

        layout = wx.BoxSizer(wx.VERTICAL)
        layout.Add(self.nb_options, 1, wx.EXPAND)
        self.pnl_options.SetSizer(layout)

    def _compose_options_layout_single_page(self):
        """ADR-037 -- Single Page layout: the same 7 category panels (each already
        built with its own StaticBoxSizer(s), identical to the tabbed path) placed
        directly onto one scrollable page instead of behind tab clicks, arranged in
        the same 4-column grid template this file used PRE-ADR-036 (the "earlier
        pass" that added spacing/dividers/indentation within each StaticBox group) --
        see docs/ai/AI_DECISIONS.md ADR-037 for why this specific arrangement was
        reused rather than invented fresh: column 0 is Stereo Generation (the most
        used, tallest group); column 1 stacks Video Decoding/Video Encoding; column 2
        stacks Video Filter/Processor; column 3 stacks Dual-Pass Depth Blend/
        Standalone Tools -- the exact same column pairing this file used before the
        tabs conversion, just with Processor+Post-Processing and the three standalone
        tools already pre-combined into single panels per ADR-036."""
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

        layout = wx.BoxSizer(wx.VERTICAL)
        layout.Add(self.pnl_single, 1, wx.EXPAND)
        self.pnl_options.SetSizer(layout)

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
            self.grp_hdr_reinject, self.grp_submux, self.grp_stereotag,
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
            self.pnl_file_option, self.pnl_preset, self.pnl_process,
            self.pnl_file.panel,
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
        self.update_temporal_stabilize()
        self.update_convergence_mode()
        self.update_scene_segment()
        self.grp_video.update_controls()

        self.update_divergence_warning()
        self.update_preserve_screen_border()
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
            if self.pnl_file.input_path and self.pnl_file.output_path:
                self.btn_start.Enable()
            else:
                self.btn_start.Disable()

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

    def update_anaglyph_state(self):
        if self.cbo_stereo_format.GetValue() == "Anaglyph":
            self.lbl_anaglyph_method.Show()
            self.cbo_anaglyph_method.Show()
        else:
            self.lbl_anaglyph_method.Hide()
            self.cbo_anaglyph_method.Hide()

        self.Layout()
        self.Fit()

    def update_export_option_state(self):
        if self.cbo_stereo_format.GetValue() in {"Export", "Export disparity"}:
            self.chk_export_depth_only.Show()
            self.chk_export_depth_fit.Show()
        else:
            self.chk_export_depth_only.Hide()
            self.chk_export_depth_fit.Hide()

        self.Layout()
        self.Fit()

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

    def on_selected_index_changed_cbo_method(self, event):
        self.update_divergence_warning()
        self.update_preserve_screen_border()
        self.update_inpaint_options()

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
            self.cbo_ema_decay.Enable()
            self.cbo_ema_buffer.Enable()
            self.chk_ema_motion_adaptive.Enable()
        else:
            self.cbo_ema_decay.Disable()
            self.cbo_ema_buffer.Disable()
            self.chk_ema_motion_adaptive.Disable()

    def update_scene_segment(self, *args, **kwargs):
        pass

    def on_changed_chk_ema_normalize(self, event):
        self.update_ema_normalize()
        self.update_scene_segment()

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

    def on_changed_chk_waifu2x_upscale(self, event):
        self.update_waifu2x_upscale()

    def update_rife_interpolate(self):
        enabled = self.chk_rife_interpolate.GetValue()
        self.cbo_rife_model.Enable(enabled)

    def on_changed_chk_rife_interpolate(self, event):
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
            preserve_screen_border=preserve_screen_border,
            depth_model=depth_model_type,
            foreground_scale=float(self.cbo_foreground_scale.GetValue()),
            foreground_pop=float(self.cbo_foreground_pop.GetValue()),
            foreground_divergence=foreground_divergence,
            background_pop=float(self.cbo_background_pop.GetValue()),
            background_pop_coverage=float(self.cbo_background_pop_coverage.GetValue()) / 100.0,
            background_divergence=background_divergence,
            edge_repair_strength=float(self.cbo_edge_repair.GetValue()),
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
            rife_interpolate=self.chk_rife_interpolate.GetValue(),
            rife_model=self.cbo_rife_model.GetValue(),
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
                depth_model=self.depth_model)
        return args

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
        self.btn_cancel.Enable()
        self.btn_suspend.Enable()
        self.stop_event.clear()
        self.suspend_event.set()
        self.prg_tqdm.SetValue(0)
        self.SetStatusText("...")

        if args.state["depth_model"].has_checkpoint_file(args.depth_model):
            # Realod depth model
            self.SetStatusText(f"Loading {args.depth_model}...")
        else:
            # Need to download the model
            self.SetStatusText(f"Downloading {args.depth_model}...")

        self.ensure_cuda_context()
        startWorker(self.on_exit_worker, iw3_main, wargs=(args,))
        self.processing = True

    def on_exit_worker(self, result):
        try:
            args = result.get()
            self.depth_model = args.state["depth_model"]
            self.depth_model_type = args.depth_model
            self.depth_model_device_id = args.gpu
            self.depth_model_height = args.resolution
            self.depth_model_limit_resolution = args.limit_resolution

            if not self.stop_event.is_set():
                self.prg_tqdm.SetValue(self.prg_tqdm.GetRange())
                self.SetStatusText(T("Finished"))
            else:
                self.SetStatusText(T("Cancelled"))
        except: # noqa
            self.SetStatusText(T("Error"))
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)

        self.processing = False
        self.btn_cancel.Disable()
        self.btn_suspend.Disable()
        self.btn_suspend.SetLabel(T("Suspend"))
        self.btn_autocrop_test.Enable()
        self.update_start_button_state()

        # free vram
        gc_collect()

    def on_click_btn_cancel(self, event):
        self.suspend_event.set()
        self.stop_event.set()

    def on_click_btn_suspend(self, event):
        if self.suspend_event.is_set():
            self.suspend_event.clear()
            self.btn_suspend.SetLabel(T("Resume"))
        else:
            self.start_time = time()
            self.suspend_pos = self.prg_tqdm.GetValue()
            self.suspend_event.set()
            self.btn_suspend.SetLabel(T("Suspend"))

    def on_tqdm(self, event):
        type, value, desc = event.GetValue()
        desc = desc if desc else ""
        if type == 0:
            # initialize
            if 0 < value:
                self.prg_tqdm.SetRange(value)
            else:
                self.prg_tqdm.SetRange(1)
            self.prg_tqdm.SetValue(0)
            self.start_time = time()
            self.suspend_pos = 0
            self.SetStatusText(f"{0}/{value} {desc}")
        elif type == 1:
            # update
            if self.prg_tqdm.GetValue() + value <= self.prg_tqdm.GetRange():
                self.prg_tqdm.SetValue(self.prg_tqdm.GetValue() + value)
            else:
                self.prg_tqdm.SetRange(self.prg_tqdm.GetValue() + value)
                self.prg_tqdm.SetValue(self.prg_tqdm.GetValue() + value)
            now = time()
            pos = self.prg_tqdm.GetValue()
            end_pos = self.prg_tqdm.GetRange()
            fps = (pos - self.suspend_pos) / (now - self.start_time + 1e-6)
            if fps > 0:
                remaining_time = int((end_pos - pos) / fps)
                h = remaining_time // 3600
                m = (remaining_time - h * 3600) // 60
                s = (remaining_time - h * 3600 - m * 60)
                t = f"{m:02d}:{s:02d}" if h == 0 else f"{h:02d}:{m:02d}:{s:02d}"
                self.SetStatusText(f"{pos}/{end_pos} [ {t}, {fps:.2f}FPS ] {desc}")
        elif type == 2:
            # close
            pass

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
        exclude_names.add("cbo_layout")  # ignore GUI layout preference (own file + restart, ADR-037)
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
        mode = self.cbo_layout.GetClientData(self.cbo_layout.GetSelection())
        _save_layout_mode(LAYOUT_CONFIG_PATH, mode)
        with wx.MessageDialog(None,
                              message=T("The layout setting will be applied after restarting"),
                              style=wx.OK) as dlg:
            dlg.ShowModal()

    def on_click_divergence_warning(self, event):
        self.lbl_divergence_warning.Hide()

        self.Layout()
        self.Fit()

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
                if not check_compile_support(device):
                    self.chk_compile.SetValue(False)

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
        manager.SaveAndUnregister()
        return snapshot_path

    def _restore_gui_state(self, snapshot_path):
        manager = persist.PersistenceManager.Get()
        manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
        manager.SetPersistenceFile(snapshot_path)
        persistent_manager_register_all(manager, self)
        for control in self.get_editable_comboboxes():
            persistent_manager_register(manager, control, EditableComboBoxPersistentHandler)
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

    def on_click_btn_reinject_output(self, event):
        with wx.FileDialog(self, message=T("Save HDR-Reinjected Output As"),
                           wildcard=VIDEO_EXTENSIONS,
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if self.txt_reinject_output.GetValue():
                dlg.SetPath(self.txt_reinject_output.GetValue())
            if dlg.ShowModal() == wx.ID_OK:
                self.txt_reinject_output.SetValue(dlg.GetPath())

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

        cmd = [sys.executable, "-m", "iw3.reinject_hdr_cli",
               "--source", source, "--converted", converted, "--output", output]
        if self.chk_reinject_start_time.GetValue():
            cmd += ["--start-time", self.txt_reinject_start_time.GetValue()]
        if self.chk_reinject_end_time.GetValue():
            cmd += ["--end-time", self.txt_reinject_end_time.GetValue()]

        self.txt_reinject_log.SetValue(
            T("Running -- this decodes the full clip to verify frame counts, so it may take a while...\n"))
        self.btn_reinject_run.Disable()
        self.SetStatusText(T("Running HDR reinjection..."))
        startWorker(self.on_exit_reinject_worker, self.run_reinject_hdr, wargs=(cmd,))

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

        cmd = [sys.executable, "-m", "iw3.subtitle_mux_cli",
               "--input", input_path, "--srt", srt_path, "--output", output_path,
               "--format", self.cbo_submux_format.GetValue()]
        language = self.txt_submux_language.GetValue().strip()
        if language:
            cmd += ["--language", language]
        track_name = self.txt_submux_track_name.GetValue().strip()
        if track_name:
            cmd += ["--track-name", track_name]

        self.txt_submux_log.SetValue(T("Running...\n"))
        self.btn_submux_run.Disable()
        self.SetStatusText(T("Adding subtitle track..."))
        startWorker(self.on_exit_submux_worker, self.run_submux, wargs=(cmd,))

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
        self.SetStatusText(T("Tagging MKV as 3D..."))
        startWorker(self.on_exit_stereotag_worker, self.run_stereotag, wargs=(cmd,))


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


def _run_self_tests():
    _self_test_no_eager_cuda_context()
    _self_test_layout_modes()
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
