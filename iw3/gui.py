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
import torch
from .utils import (
    create_parser, set_state_args, iw3_main,
    is_text, is_video, is_image, is_output_dir, is_yaml, make_output_filename,
    _get_ffmpeg_bin, _find_mkvmerge,
)
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


LAYOUT_DEBUG = False


class IW3App(wx.App):
    def OnInit(self):
        main_frame = MainFrame()
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
            title=T("iw3-gui") + branch_tag + python_version_tag,
            size=(1000, 840),
            style=(wx.DEFAULT_FRAME_STYLE & ~wx.MAXIMIZE_BOX)
        )
        self.processing = False
        self.start_time = 0
        self.input_type = None
        self.stop_event = threading.Event()
        self.suspend_event = threading.Event()
        self.suspend_pos = 0
        self.suspend_event.set()
        self.depth_model = None
        self.depth_model_type = None
        self.depth_model_device_id = None
        self.depth_model_height = None
        self.depth_model_limit_resolution = None
        self.initialize_component()
        if is_dark_mode():
            apply_dark_mode(self)

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
        self.chk_resume.SetToolTip(T("Skip processing when the output file already exists"))
        self.chk_resume.SetValue(True)

        self.chk_recursive = wx.CheckBox(self.pnl_file_option, label=T("Process all subfolders"),
                                         name="chk_recursive")
        self.chk_recursive.SetValue(False)
        self.chk_recursive.SetToolTip(T("When the input is a folder, also process videos/images inside its "
                                        "subfolders, not just the top level. Recommended: on, if you organize "
                                        "your movies into subfolders."))

        self.chk_skip_error = wx.CheckBox(self.pnl_file_option, label=T("Skip Error"), name="chk_skip_erro")
        self.chk_skip_error.SetToolTip(T("Skip videos that cause errors during batch processing and those that previously encountered errors."))
        self.chk_skip_error.SetValue(False)

        self.sep_batch_options = wx.StaticLine(self.pnl_file_option, size=self.FromDIP((2, 16)), style=wx.LI_VERTICAL)

        self.chk_exif_transpose = wx.CheckBox(self.pnl_file_option, label=T("EXIF Transpose"),
                                              name="chk_exif_transpose")
        self.chk_exif_transpose.SetValue(True)
        self.chk_exif_transpose.SetToolTip(T("Transpose images according to EXIF Orientation Tag"))

        self.chk_metadata = wx.CheckBox(self.pnl_file_option, label=T("Add metadata to filename"),
                                        name="chk_metadata")
        self.chk_metadata.SetValue(False)
        self.chk_metadata.SetToolTip(T("Encode your current settings (model, 3D strength, convergence, etc.) "
                                       "into the output filename, so you can tell which settings made which "
                                       "file later, and so Auto Resume can find a matching in-progress job. "
                                       "Recommended: on."))

        self.sep_image_format = wx.StaticLine(self.pnl_file_option, size=self.FromDIP((2, 16)), style=wx.LI_VERTICAL)
        self.lbl_image_format = wx.StaticText(self.pnl_file_option, label=" " + T("Image Format"))
        self.cbo_image_format = wx.ComboBox(self.pnl_file_option, choices=["png", "jpeg", "webp"],
                                            name="cbo_image_format")
        self.cbo_image_format.SetEditable(False)
        self.cbo_image_format.SetSelection(0)
        self.cbo_image_format.SetToolTip(T("Output Image Format"))

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

        # stereo generation settings
        # divergence, convergence, method, depth_model, mapper

        self.grp_stereo = wx.StaticBox(self.pnl_options, label=T("Stereo Generation"))

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
            T("How the \"screen depth\" (which point looks like it's exactly at the screen surface) is "
              "chosen.\nconstant: you set a fixed position with the value box.\n"
              "sod_v1: AI automatically picks a focus point based on the most visually important subject.\n"
              "face_detect: automatically centers on detected faces, ignoring the value box.\n"
              "Recommended: constant for predictable results, sod_v1 for movies with a clear subject."))

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
            T("Also called IPD Offset (distance between your eyes). Fine-tunes the 3D effect for your own "
              "eye spacing. 0 is average; higher values widen the simulated eye distance. Most people can "
              "leave this at 0."))

        self.lbl_synthetic_view = wx.StaticText(self.grp_stereo, label=T("Synthetic View"))
        self.cbo_synthetic_view = wx.ComboBox(self.grp_stereo,
                                              choices=["both", "right", "left"],
                                              name="cbo_synthetic_view")
        self.cbo_synthetic_view.SetEditable(False)
        self.cbo_synthetic_view.SetSelection(0)
        self.cbo_synthetic_view.SetToolTip(
            T("Which eye view gets newly generated by AI. \"both\" generates both eyes from the original "
              "center image (recommended, most natural). \"right\"/\"left\" keep the original image as one "
              "eye unchanged and only synthesize the other — faster, but can look slightly less balanced."))

        self.lbl_method = wx.StaticText(self.grp_stereo, label=T("Method"))
        self.cbo_method = wx.ComboBox(self.grp_stereo,
                                      choices=["mlbw_l2", "mlbw_l4", "mlbw_l2s",
                                               "mlbw_l2_inpaint",
                                               "row_flow_v3", "row_flow_v3_sym", "row_flow_v2",
                                               "forward_fill", "forward_inpaint",
                                               "monobw", "monobw_inpaint",
                                               ],
                                      name="cbo_method")
        self.cbo_method.SetEditable(False)
        self.cbo_method.SetSelection(2)
        self.cbo_method.SetToolTip(
            T("How the second eye view is generated from the depth map. row_flow_v3 / row_flow_v3_sym: fast, "
              "AI-based, the default and a solid all-rounder. mlbw_l2/l4: more advanced layered AI warping, "
              "supports stronger 3D strength. *_inpaint variants additionally fill in the hidden areas behind "
              "objects instead of stretching/smearing them — slower, but cleaner edges. forward_fill / "
              "monobw / grid_sample / backward: older, simpler, non-AI methods — mainly for testing/speed. "
              "Recommended: mlbw_l2_inpaint or forward_inpaint for best quality; row_flow_v3_sym for speed."))

        self.lbl_inpaint_model = wx.StaticText(self.grp_stereo, label=T("Inpainting Model"))
        self.cbo_inpaint_model = wx.ComboBox(self.grp_stereo,
                                             choices=list(INPAINT_MODELS.keys()),
                                             name="cbo_inpaint_model")
        self.cbo_inpaint_model.SetEditable(False)
        self.cbo_inpaint_model.SetSelection(0)
        self.cbo_inpaint_model.SetToolTip(
            T("Only used by forward_inpaint / monobw_inpaint. Which AI model fills in the hidden areas "
              "behind objects. Larger/\"Large\" models give cleaner results but are slower. "
              "light_inpaint_v1 is the fast built-in default."))

        self.lbl_overlap_frames = wx.StaticText(self.grp_stereo, label=T("Inpaint Overlap Frames"))
        self.cbo_overlap_frames_pre = EditableComboBox(self.grp_stereo,
                                                       choices=["0", "3"],
                                                       name="cbo_overlap_frames_pre")
        self.cbo_overlap_frames_pre.SetSelection(1)
        self.cbo_overlap_frames_pre.SetToolTip(
            T("Overlap Pre: how many extra frames before each processed chunk are fed to the inpainting "
              "model for context, to keep filled-in areas consistent frame to frame. Higher = smoother but "
              "slower. Recommended: default (3)."))

        self.cbo_overlap_frames_post = EditableComboBox(self.grp_stereo,
                                                        choices=["0", "3"],
                                                        name="cbo_overlap_frames_post")
        self.cbo_overlap_frames_post.SetSelection(1)
        self.cbo_overlap_frames_post.SetToolTip(
            T("Overlap Post: same idea as Overlap Pre, but for extra frames after each chunk. "
              "Recommended: default (3)."))

        self.lbl_mask_dilation = wx.StaticText(self.grp_stereo, label=T("Inpaint Mask Dilation"))
        self.cbo_mask_inner_dilation = EditableComboBox(self.grp_stereo,
                                                        choices=["0", "1", "2"],
                                                        name="cbo_mask_inner_dilation")
        self.cbo_mask_inner_dilation.SetSelection(0)
        self.cbo_mask_inner_dilation.SetToolTip(
            T("Inner: grows the \"needs filling in\" area inward, slightly shrinking the foreground object's "
              "edge so the inpainting blends in more smoothly. Recommended: 0, raise only if you see a thin "
              "halo around foreground objects."))

        self.cbo_mask_outer_dilation = EditableComboBox(self.grp_stereo,
                                                        choices=["0", "1", "2"],
                                                        name="cbo_mask_outer_dilation")
        self.cbo_mask_outer_dilation.SetSelection(0)
        self.cbo_mask_outer_dilation.SetToolTip(
            T("Outer: grows the \"needs filling in\" area outward into the background, giving the "
              "inpainting model more room to work with. Recommended: 0, raise only if you still see leftover "
              "smearing right behind foreground objects."))

        self.lbl_inpaint_max_width = wx.StaticText(self.grp_stereo, label=T("Inpaint Max Width"))
        self.cbo_inpaint_max_width = EditableComboBox(self.grp_stereo,
                                                      choices=["", "1920"],
                                                      name="cbo_inpaint_max_width")
        self.cbo_inpaint_max_width.SetSelection(0)
        self.cbo_inpaint_max_width.SetToolTip(
            T("Caps the resolution the inpainting model processes at, to save VRAM/time on large videos. "
              "Leave blank for no limit (best quality). Lower it only if you run out of memory."))

        self.lbl_stereo_width = wx.StaticText(self.grp_stereo, label=T("Stereo Processing Width"))
        self.cbo_stereo_width = EditableComboBox(self.grp_stereo,
                                                 choices=["Default", "1920", "1280", "640"],
                                                 name="cbo_stereo_width")
        self.cbo_stereo_width.SetSelection(0)
        self.cbo_stereo_width.SetToolTip(
            T("Only used for row_flow_v3 and row_flow_v2. Resizes the image to this width before "
              "generating the 3D effect (separate from output resolution). Default uses the source size. "
              "Lowering it can speed things up at some quality cost."))

        self.lbl_depth_model = wx.StaticText(self.grp_stereo, label=T("Depth Model"))
        self.cbo_depth_model = wx.ComboBox(self.grp_stereo,
                                           choices=self.get_depth_models(),
                                           name="cbo_depth_model")
        self.cbo_depth_model.SetEditable(False)
        self.cbo_depth_model.SetSelection(3)
        self.cbo_depth_model.SetToolTip(
            T("Which AI model estimates depth from the image. VDA_* (Video Depth Anything) models are "
              "built for video and keep depth stable/flicker-free across frames — best for movies. "
              "Any_V2_*/Any_V3_*/Distill_Any_* are image models — sharper on single photos, but can "
              "flicker if used on video. *_Metric variants estimate real-world distances, useful for some "
              "specialized use cases. Larger (_L) = better quality but slower and more VRAM."))

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
            T("Reshapes the depth curve for the whole image (-3 to 3), affecting foreground AND background. "
              "Different from Foreground Pop, which only pushes the nearest pixels."))

        self.chk_depth_aa = wx.CheckBox(self.grp_stereo, label=T("Depth Anti-aliasing"), name="chk_depth_aa")
        self.chk_depth_aa.SetValue(False)
        self.chk_depth_aa.SetToolTip(
            T("Smooths small jagged/staircase artifacts in the depth map using a dedicated AI model, "
              "without changing the actual depth values much. Only available for certain depth models "
              "(grayed out otherwise). Recommended: on, when available — minor cost, generally cleaner result."))

        self.lbl_foreground_pop = wx.StaticText(self.grp_stereo, label=T("Foreground Pop"))
        self.cbo_foreground_pop = EditableComboBox(self.grp_stereo,
                                                   choices=["0.0", "0.25", "0.5", "0.75", "1.0"],
                                                   name="cbo_foreground_pop")
        self.cbo_foreground_pop.SetSelection(0)
        self.cbo_foreground_pop.SetToolTip(
            T("Push nearest pixels further toward the audience (0=off, 1=strong). Only affects the closest "
              "objects, leaving background untouched. Different from Foreground Scale, which reshapes the "
              "whole depth curve."))

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
        self.lbl_edge_dilation.SetToolTip(T("Reduce distortion of foreground and background edges"))
        self.cbo_edge_dilation.SetToolTip(T("X or XY"))
        self.cbo_edge_dilation_y.SetToolTip(T("Y"))

        self.chk_ema_normalize = wx.CheckBox(self.grp_stereo,
                                             label=T("Flicker Reduction"),
                                             name="chk_ema_normalize")
        self.chk_ema_normalize.SetToolTip(
            T("Video Only (experimental). Smooths the depth map over time so it doesn't flicker/wobble "
              "between frames. Uses the Decay Rate and Lookahead Buffer settings below. "
              "Recommended: on for most videos, paired with Scene Boundary Detection."))

        self.cbo_ema_decay = EditableComboBox(self.grp_stereo, choices=["0.99", "0.95", "0.9", "0.75", "0.5", "0"],
                                              name="cbo_ema_decay")
        self.cbo_ema_decay.SetSelection(2)
        self.cbo_ema_decay.SetToolTip(
            T("Decay Rate: how much the depth map is smoothed frame to frame. Higher (0.9-0.99) = smoother "
              "but slower to react, good for slow/calm footage. Lower (0.5-0.75) = reacts faster, better "
              "for fast motion/action, but may flicker more. 0 = no smoothing."))

        self.cbo_ema_buffer = EditableComboBox(self.grp_stereo, choices=["150", "60", "30", "1"],
                                               name="cbo_ema_buffer")
        self.cbo_ema_buffer.SetSelection(2)
        self.cbo_ema_buffer.SetToolTip(
            T("Lookahead Buffer Size (in frames): how many frames are looked at together to judge the "
              "near/far depth range. Bigger = more stable range but slower to adapt within a shot; "
              "smaller = reacts faster to sudden depth changes within a single continuous shot. "
              "Scene Boundary Detection resets this at every real cut regardless of this setting."))

        self.chk_scene_detect = wx.CheckBox(self.grp_stereo,
                                            label=T("Scene Boundary Detection"),
                                            name="chk_scene_detect")
        self.chk_scene_detect.SetValue(False)
        self.chk_scene_detect.SetToolTip(
            T("Detects real scene/shot cuts and resets the depth model and Flicker Reduction exactly at "
              "those points, instead of letting smoothing bleed across unrelated scenes. Also makes Auto "
              "Resume align its chunk boundaries to real cuts. Recommended: on for movies/TV, especially "
              "with Flicker Reduction or Auto Resume enabled."))

        self.chk_scene_detect_cache = wx.CheckBox(self.grp_stereo,
                                                  label=T("Use scene boundary cache"),
                                                  name="chk_scene_detect_cache")
        self.chk_scene_detect_cache.SetValue(True)
        self.chk_scene_detect_cache.SetToolTip(
            T("Saves detected scene cuts to disk so re-running the same video (e.g. after a crash, or "
              "just testing different 3D settings) doesn't need to re-scan for cuts every time. "
              "Recommended: on."))

        self.chk_preserve_screen_border = wx.CheckBox(self.grp_stereo,
                                                      label=T("Preserve Screen Border"),
                                                      name="chk_preserve_screen_border")
        self.chk_preserve_screen_border.SetValue(False)
        self.chk_preserve_screen_border.SetToolTip(
            T("Forces the very edges of the frame to have zero 3D shift (zero parallax), preventing "
              "objects from being cut off oddly at the left/right edges. Recommended: on if you notice "
              "distracting edge artifacts; off for maximum 3D strength everywhere."))

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

        self.chk_export_depth_only = wx.CheckBox(self.grp_stereo, label=T("Depth Only"), name="chk_export_depth_only")
        self.chk_export_depth_only.SetValue(False)
        self.chk_export_depth_only.SetToolTip(T("Exporting depth images only.\n"
                                                "Note that exported data with this option cannot be imported."))
        self.chk_export_depth_only.Hide()

        self.chk_export_depth_fit = wx.CheckBox(self.grp_stereo, label=T("Resize to fit"), name="chk_export_depth_fit")
        self.chk_export_depth_fit.SetValue(False)
        self.chk_export_depth_fit.SetToolTip(T("Resize depth images to the same size as rgb images.\n"
                                               "Note that the process may become very slow due to the output file becoming large."))
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

        layout.Add((0, 8), (i := i + 1, 0))
        layout.Add(wx.StaticLine(self.grp_stereo), (i := i + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 6), (i := i + 1, 0))
        layout.Add(self.lbl_foreground_pop, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_foreground_pop, (i, 1), (1, 2), flag=wx.EXPAND)

        layout.Add((0, 8), (i := i + 1, 0))
        layout.Add(wx.StaticLine(self.grp_stereo), (i := i + 1, 0), (0, 3), flag=wx.EXPAND)
        layout.Add((0, 6), (i := i + 1, 0))
        layout.Add(self.chk_ema_normalize, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_ema_decay, (i, 1), flag=wx.EXPAND)
        layout.Add(self.cbo_ema_buffer, (i, 2), flag=wx.EXPAND)
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
        layout.Add(self.chk_export_depth_only, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_export_depth_fit, (i, 1), (1, 2), flag=wx.ALIGN_CENTER_VERTICAL)

        sizer_stereo = wx.StaticBoxSizer(self.grp_stereo, wx.VERTICAL)
        sizer_stereo.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # video decoding
        # hwaccel
        self.grp_video_dec = VideoDecodingBox(self.pnl_options, translate_function=T)

        # video encoding
        # sbs/vr180, padding
        # max-fps, crf, preset, tune
        self.grp_video = VideoEncodingBox(self.pnl_options, translate_function=T,
                                          has_nvenc=has_nvenc(), has_qsv=has_qsv())

        # input video filter
        # deinterlace, rotate, vf
        self.grp_video_filter = wx.StaticBox(self.pnl_options, label=T("Video Filter"))
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
        self.cbo_pad_mode.SetToolTip(T("Padding Mode"))
        self.cbo_pad = EditableComboBox(self.grp_video_filter, choices=["", "0.01", "0.05", "0.5", "1"],
                                        name="cbo_pad")
        self.cbo_pad.SetSelection(0)
        self.cbo_pad.SetToolTip(T("Padding Ratio"))

        self.lbl_max_output_size = wx.StaticText(self.grp_video_filter, label=T("Output Size Limit"))
        self.cbo_max_output_size = wx.ComboBox(self.grp_video_filter,
                                               choices=["",
                                                        "7680x2160",
                                                        "3840x2160",
                                                        "3840x1080",
                                                        "1920x1080", "1280x720", "640x360",
                                                        "1920x3200",
                                                        "1080x1920", "720x1280", "360x640"],
                                               name="cbo_max_output_size")
        self.cbo_max_output_size.SetEditable(False)
        self.cbo_max_output_size.SetSelection(0)
        self.cbo_max_output_size.SetToolTip(
            T("Caps the final output's resolution (each eye), e.g. to keep file size/playback "
              "requirements manageable on a 4K source. Leave blank to keep the source's native size."))

        self.chk_keep_aspect_ratio = wx.CheckBox(self.grp_video_filter, label=T("Keep Aspect Ratio"),
                                                 name="chk_keep_aspect_ratio")
        self.chk_keep_aspect_ratio.SetValue(False)
        self.chk_keep_aspect_ratio.SetToolTip(
            T("When Output Size Limit is set, preserve the source's original width/height proportions "
              "instead of stretching to exactly fill the limit's dimensions."))

        self.chk_preserve_dowi = wx.CheckBox(self.grp_video_filter, label=T("Preserve Dolby Vision"),
                                              name="chk_preserve_dowi")
        self.chk_preserve_dowi.SetValue(False)
        self.chk_preserve_dowi.SetToolTip(T("Detect and preserve Dolby Vision RPU and/or HDR10+ dynamic metadata through 3D conversion. Requires HEVC output (Video Codec: hevc_nvenc or libx265) and MKVToolNix installed for MKV output."))

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
        self.chk_denoise.SetToolTip(T("Apply temporal denoising before depth estimation. Reduces film grain for better 3D quality on old films."))

        self.chk_preview = wx.CheckBox(self.grp_video_filter, label=T("Preview Mode"), name="chk_preview")
        self.chk_preview.SetValue(False)
        self.chk_preview.SetToolTip(T("Quick preview: 1fps at low resolution for the first 60 seconds. Use this to check 3D settings before full processing."))

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
        layout.Add(self.chk_start_time, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_start_time, (0, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_end_time, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_end_time, (1, 1), (0, 2), flag=wx.EXPAND)

        layout.Add(self.lbl_deinterlace, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_deinterlace, (2, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_vf, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_vf, (3, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_rotate, (4, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_rotate, (4, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_autocrop, (5, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_autocrop, (5, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.btn_autocrop_test, (6, 1), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_autocrop_test, (6, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_pad, (7, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_pad_mode, (7, 1), flag=wx.EXPAND)
        layout.Add(self.cbo_pad, (7, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_max_output_size, (8, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_max_output_size, (8, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_keep_aspect_ratio, (9, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_preserve_dowi, (10, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_hdr_to_sdr, (11, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_auto_resume, (12, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.lbl_upgrade_pix_fmt, (13, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_upgrade_pix_fmt, (13, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_denoise, (14, 1), (0, 2), flag=wx.EXPAND)
        layout.Add(self.chk_preview, (15, 1), (0, 2), flag=wx.EXPAND)

        sizer_video_filter = wx.StaticBoxSizer(self.grp_video_filter, wx.VERTICAL)
        sizer_video_filter.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # processor settings
        # device, batch-size, TTA, Low VRAM, fp16
        self.grp_processor = wx.StaticBox(self.pnl_options, label=T("Processor"))
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
                                          choices=[str(n) for n in (64, 32, 16, 14, 12, 10, 9, 8, 4, 3, 2, 1)],
                                          name="cbo_zoed_batch_size")
        self.cbo_batch_size.SetEditable(False)
        self.cbo_batch_size.SetToolTip(
            T("Video Only. How many frames are sent to the depth model at once. Higher = faster overall "
              "but uses more VRAM. Lower it if you run out of memory; raise it if you have VRAM to spare "
              "and want faster processing."))
        self.cbo_batch_size.SetSelection(10)  # "2"

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
        layout.Add(self.lbl_device, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_device, (0, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_batch_size, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_batch_size, (1, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_max_workers, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_max_workers, (2, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.chk_low_vram, (3, 0), flag=wx.EXPAND)
        layout.Add(self.chk_tta, (3, 1), flag=wx.EXPAND)
        layout.Add(self.chk_fp16, (3, 2), flag=wx.EXPAND)
        layout.Add(self.chk_cuda_stream, (3, 3), flag=wx.EXPAND)
        layout.Add(self.chk_compile, (4, 0), flag=wx.EXPAND)

        sizer_processor = wx.StaticBoxSizer(self.grp_processor, wx.VERTICAL)
        sizer_processor.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        sizer_video = wx.BoxSizer(wx.VERTICAL)
        sizer_video.Add(self.grp_video_dec.sizer, 0, wx.ALL | wx.EXPAND, border=4)
        sizer_video.Add(self.grp_video.sizer, 1, wx.ALL | wx.EXPAND, border=4)

        layout = wx.GridBagSizer(vgap=0, hgap=0)
        layout.SetEmptyCellSize((0, 0))
        layout.Add(sizer_stereo, pos=(0, 0), span=(2, 1), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_video, pos=(0, 1), span=(2, 1), flag=wx.ALL | wx.EXPAND, border=0)
        layout.Add(sizer_video_filter, pos=(0, 2), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_processor, pos=(1, 2), flag=wx.ALL | wx.EXPAND, border=4)
        self.pnl_options.SetSizer(layout)

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
        self.btn_quick_preset_movie.SetToolTip(T("Quick preset: subtle, comfortable 3D for movies"))
        self.btn_quick_preset_action = wx.Button(self.pnl_preset, label=T("Action"))
        self.btn_quick_preset_action.SetToolTip(T("Quick preset: strong pop effects for action/VFX scenes"))

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

        self.btn_autocrop_test.Bind(wx.EVT_BUTTON, self.on_click_btn_autocrop_test)

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
        self.update_controls()

    def update_controls(self):
        self.update_start_button_state()
        self.update_input_option_state()
        self.update_anaglyph_state()
        self.update_export_option_state()

        self.update_model_selection()
        self.update_edge_dilation()
        self.update_inpaint_options()
        self.update_ema_normalize()
        self.update_convergence_mode()
        self.update_scene_segment()
        self.grp_video.update_controls()

        self.update_divergence_warning()
        self.update_preserve_screen_border()
        self.update_pad_mode()
        self.update_compile()

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
            *self.grp_video.get_editable_comboboxes(),
            *self.grp_video_dec.get_editable_comboboxes(),
            self.cbo_foreground_scale,
            self.cbo_foreground_pop,
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
            self.cbo_edge_dilation.SetToolTip(T("X"))
        else:
            self.cbo_edge_dilation.SetToolTip(T("X, Y"))

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
        else:
            self.cbo_ema_decay.Disable()
            self.cbo_ema_buffer.Disable()

    def update_scene_segment(self, *args, **kwargs):
        pass

    def on_changed_chk_ema_normalize(self, event):
        self.update_ema_normalize()
        self.update_scene_segment()

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
                               ema_buffer=int(self.cbo_ema_buffer.GetValue()))
        else:
            ema_options = {}

        metadata = "filename" if self.chk_metadata.GetValue() else None
        preserve_screen_border = self.chk_preserve_screen_border.IsEnabled() and self.chk_preserve_screen_border.IsChecked()
        scene_detect = self.chk_scene_detect.IsChecked()
        disable_scene_cache = not self.chk_scene_detect_cache.IsChecked()
        depth_aa = self.chk_depth_aa.IsShown() and self.chk_depth_aa.IsEnabled() and self.chk_depth_aa.IsChecked()

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

            export=export,
            export_disparity=export_disparity,
            export_depth_only=export_depth_only,
            export_depth_fit=export_depth_fit,

            debug_depth=debug_depth,
            **ema_options,
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

    def on_click_btn_start(self, event):
        args = self.parse_args()
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

    def update_compile(self, *args, **kwargs):
        device_id = int(self.cbo_device.GetClientData(self.cbo_device.GetSelection()))
        if device_id == -2:
            # currently "All CUDA" does not support compile
            self.chk_compile.SetValue(False)
        else:
            # check compiler support
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
        persistent_manager_restore_all(manager, {"cbo_language"})
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


LOCAL_LIST = sorted(list(LOCALES.keys()))
LOCALE_DICT = LOCALES.get(get_default_locale(), {})


def T(s):
    return LOCALE_DICT.get(s, s)


def main():
    import argparse
    import sys
    global LOCALE_DICT

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
    pyav_init_cuda_primary_context()
    init_win32_dpi()
    main()
