import nunif.pythonw_fix  # noqa
import nunif.gui.subprocess_patch  # noqa
import sys
import os
from os import path
import gc
import functools
from time import time
import threading
from pprint import pprint # noqa
import torch
import wx
from wx.lib.delayedresult import startWorker
import wx.lib.agw.persist as persist
from .ui_utils import (
    create_parser, set_state_args, waifu2x_main,
    is_video, is_output_dir, is_text, make_output_filename,
    MODEL_DIR, DEFAULT_ART_MODEL_DIR,
    DEFAULT_ART_SCAN_MODEL_DIR, DEFAULT_PHOTO_MODEL_DIR)
from nunif.device import mps_is_available, xpu_is_available
from nunif.utils.image_loader import IMG_EXTENSIONS as LOADER_SUPPORTED_EXTENSIONS
from nunif.utils.video import (
    VIDEO_EXTENSIONS as KNOWN_VIDEO_EXTENSIONS,
    has_nvenc,
    has_qsv,
    pyav_init_cuda_primary_context,
)
from nunif.utils.git import get_current_branch
from nunif.utils.home_dir import ensure_home_dir
from nunif.gui import (
    TQDMGUI, FileDropCallback, EVT_TQDM, TimeCtrl,
    EditableComboBox, EditableComboBoxPersistentHandler,
    persistent_manager_register_all, persistent_manager_restore_all, persistent_manager_register,
    extension_list_to_wildcard,
    validate_number,
    set_icon_ex,
    VideoEncodingBox, VideoDecodingBox, IOPathPanel,
    get_default_locale,
    init_win32_dpi,
    set_tooltip_long_hover,
    enable_persistent_tooltips,
)
from .locales import LOCALES
from . import models # noqa
from .external_sr import available_onnx_sr_methods


IMAGE_EXTENSIONS = extension_list_to_wildcard(LOADER_SUPPORTED_EXTENSIONS)
VIDEO_EXTENSIONS = extension_list_to_wildcard(KNOWN_VIDEO_EXTENSIONS)
CONFIG_DIR = ensure_home_dir("waifu2x", path.join(path.dirname(__file__), "..", "tmp"))
CONFIG_PATH = path.join(CONFIG_DIR, "waifu2x-gui.cfg")
os.makedirs(CONFIG_DIR, exist_ok=True)


LAYOUT_DEBUG = False


class Waifu2xApp(wx.App):
    def OnInit(self):
        set_tooltip_long_hover()
        main_frame = MainFrame()
        enable_persistent_tooltips(main_frame)
        self.instance = wx.SingleInstanceChecker("waifu2x-gui.lock", CONFIG_DIR)
        if self.instance.IsAnotherRunning():
            wx.MessageBox(T("Another instance is running"), T("Error"), style=wx.ICON_ERROR)
            return False
        set_icon_ex(main_frame, path.join(path.dirname(__file__), "icon.ico"), main_frame.GetTitle())
        self.SetAppName(main_frame.GetTitle())
        main_frame.Show()
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
            name="waifu2x-gui",
            title=T("waifu2x-gui") + branch_tag + python_version_tag,
            size=(1000, 740),
            style=(wx.DEFAULT_FRAME_STYLE & ~wx.MAXIMIZE_BOX)
        )
        self.processing = False
        self.start_time = 0
        self.input_type = None
        self.stop_event = threading.Event()
        self.initialize_component()

    def initialize_component(self):
        self.SetFont(wx.Font(10, wx.FONTFAMILY_MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.CreateStatusBar()

        # input output panel

        input_wildcard = (f"Image and Video files|{IMAGE_EXTENSIONS};{VIDEO_EXTENSIONS}"
                          f"|Video files|{VIDEO_EXTENSIONS}"
                          f"|Image files|{IMAGE_EXTENSIONS}"
                          "|All Files|*.*")
        self.pnl_file = IOPathPanel(
            self,
            input_wildcard=input_wildcard,
            default_output_dir_name="waifu2x",
            resolve_output_path=self.resolve_output_path,
            translate_function=T,
        )

        self.pnl_file_option = wx.Panel(self)
        self.chk_resume = wx.CheckBox(self.pnl_file_option, label=T("Resume"), name="chk_resume")
        self.chk_resume.SetToolTip(
            T("What it's for: skips a file entirely if the output it would create already exists.\n"
              "How it helps: lets you stop a big batch job partway (or have it crash/get interrupted) and "
              "restart later without wasting time redoing files you already finished.\n"
              "Con: only checks whether a file with the expected NAME exists, not whether it's actually "
              "complete or correct.\n"
              "Recommended: on for batch folders and long jobs."))
        self.chk_resume.SetValue(True)
        self.chk_recursive = wx.CheckBox(self.pnl_file_option, label=T("Process all subfolders"),
                                         name="chk_recursive")
        self.chk_recursive.SetValue(False)
        self.chk_recursive.SetToolTip(
            T("What it's for: when the input is a folder, also reaches into its subfolders instead of "
              "only converting files sitting directly inside it.\n"
              "Con: if unrelated files live in subfolders you didn't mean to include, they get processed "
              "too — there's no per-folder include/exclude list.\n"
              "Recommended: on if you organize your images/videos into subfolders."))
        self.chk_exif_transpose = wx.CheckBox(self.pnl_file_option, label=T("EXIF Transpose"),
                                              name="chk_exif_transpose")
        self.chk_exif_transpose.SetValue(True)
        self.chk_exif_transpose.SetToolTip(
            T("What it's for: some cameras/phones save a photo already correctly oriented for viewing but "
              "store the actual pixel data sideways/upside-down, with a hidden tag telling viewers how to "
              "rotate it for display. This applies that rotation before upscaling.\n"
              "Recommended: on (default) for virtually all real photos."))

        layout = wx.BoxSizer(wx.HORIZONTAL)
        layout.AddSpacer(4)
        layout.Add(self.chk_resume, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_recursive, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_exif_transpose, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.pnl_file_option.SetSizer(layout)

        # options panel

        self.pnl_options = wx.Panel(self)
        if LAYOUT_DEBUG:
            self.pnl_options.SetBackgroundColour("#cfc")

        # Superresolution settings
        # NOTE: term is translated with webgen locale. See WEBGEN_TERMS in ./locales.py
        self.grp_sr = wx.StaticBox(self.pnl_options, label=T("Superresolution"))
        self.opt_model = wx.RadioBox(
            self.grp_sr, label=T("Model"),
            choices=[T("artwork"), T("artwork") + "/" + T("scan"),
                     T("photo"), "cunet/art", "upconv_7/art", "upconv_7/photo"],
            majorDimension=3, name="opt_model")
        self.model_dirs = [
            DEFAULT_ART_MODEL_DIR, DEFAULT_ART_SCAN_MODEL_DIR, DEFAULT_PHOTO_MODEL_DIR,
            path.join(MODEL_DIR, "cunet", "art"),
            path.join(MODEL_DIR, "upconv_7", "art"),
            path.join(MODEL_DIR, "upconv_7", "photo"),
        ]
        self.model_tta_support = [True, False, False, True, True, True]
        self.model_4x_support = [True, True, True, False, False, False]

        self.opt_model.SetSelection(0)
        self.opt_model.SetItemToolTip(
            0, T("Anime Style Art, Cliparts. Current-generation model family (swin_unet) — best quality "
                 "for hand-drawn/vector-style illustration. Recommended for most anime/illustration content."))
        self.opt_model.SetItemToolTip(
            1, T("Manga, Anime Screencaps, Anime Style Art for more clear results. Same current-generation "
                 "family, tuned specifically for scanned pages/screenshots — handles scan noise/compression "
                 "artifacts better than the plain artwork model. Recommended for manga scans and anime "
                 "screencaps specifically, over the plain \"artwork\" option."))
        self.opt_model.SetItemToolTip(
            2, T("Photograph. Current-generation model family, tuned for real photographic/live-action "
                 "content rather than illustration."))
        self.opt_model.SetItemToolTip(
            3, T("Old version, Art model, fast. A legacy model kept for compatibility/comparison — the "
                 "three current-generation options above generally give better quality at a comparable or "
                 "better speed on any modern GPU. Recommended only if you have a specific reason to match "
                 "old output, or are on very limited/old hardware."))
        self.opt_model.SetItemToolTip(
            4, T("Old version, Art model, veryfast. The fastest but lowest-quality option in this whole "
                 "list — mainly useful on very limited hardware where even the current-generation models "
                 "are too slow."))
        self.opt_model.SetItemToolTip(
            5, T("Old version, Photo model, veryfast. The fastest but lowest-quality photo option — mainly "
                 "useful on very limited hardware."))

        self.opt_noise_level = wx.RadioBox(
            self.grp_sr, label=T("noise_reduction"),
            choices=[T("nr_none"), T("nr_low"), T("nr_medium"), T("nr_high"), T("nr_highest")],
            name="opt_noise_level")
        self.opt_noise_level.SetSelection(1)
        self.opt_noise_level.SetToolTip(
            T("What it's for: cleans up JPEG/compression artifacts and grain while upscaling. Higher = "
              "more aggressive noise removal, but also more risk of softening real fine detail along with "
              "the noise — pick the LOWEST level that actually removes the artifacts you see, not the "
              "highest by default.\n"
              "Recommended: none for a clean, uncompressed source (a fresh screenshot, a lossless scan); "
              "low/medium for typical web images or moderately compressed video; high/highest only for "
              "visibly blocky/artifact-heavy sources."))

        self.opt_upscaling = wx.RadioBox(
            self.grp_sr, label=T("upscaling"),
            choices=[T("up_none"), "2x", "4x"],
            name="opt_upscaling")
        self.opt_upscaling.SetSelection(1)
        self.opt_upscaling.SetToolTip(
            T("What it's for: how much bigger the output image/video becomes (each dimension multiplied "
              "by this). \"none\" runs noise reduction only, at the original size.\n"
              "Con: a larger multiplier costs proportionally more processing time and produces a "
              "proportionally larger file.\n"
              "Recommended: 2x for most everyday upscaling; 4x only if you specifically need a much larger "
              "final image and have the processing time to spare."))

        # Externally-pretrained models (RealESRGAN, BSRGAN -- not this project's own
        # cunet/upconv_7/swin_unet family, no noise-level concept, fixed scale per
        # model). Off by default, and only usable once the corresponding weight file
        # has actually been placed in waifu2x/pretrained_models/external_sr/ --
        # see docs/ai/AI_DECISIONS.md. When enabled, overrides Model/Upscaling/Noise
        # Reduction above entirely rather than combining with them.
        self.chk_external_sr = wx.CheckBox(self.grp_sr, label=T("Use External SR Model"),
                                           name="chk_external_sr")
        self.chk_external_sr.SetValue(False)
        self.chk_external_sr.SetToolTip(
            T("What it's for: uses a separately-obtained pretrained model (RealESRGAN/BSRGAN, or a custom "
              "ONNX model you've supplied) instead of this app's own built-in swin_unet/cunet models.\n"
              "How it works: completely OVERRIDES Model/Upscaling/Noise Reduction above rather than "
              "combining with them — pick the specific model in the dropdown below instead.\n"
              "Con: only works once the model's weight file has actually been placed in "
              "waifu2x/pretrained_models/external_sr/ (not bundled with the app — separately licensed by "
              "their original researchers); each of these models has a fixed scale/noise-handling behavior "
              "of its own, less flexible than the built-in models' separate scale/noise controls.\n"
              "Recommended: off unless you specifically want one of these particular models' characteristics "
              "(e.g. RealESRGAN's general-purpose real-world upscaling) over the built-in options."))
        external_sr_choices = [
            "realesrgan_x4 (Real-ESRGAN, general, 4x)",
            "realesrgan_x2 (Real-ESRGAN, general, 2x)",
            "realesrgan_general_x4 (Real-ESRGAN, lighter/faster, 4x)",
            "bsrgan_x4 (BSRGAN, blind SR, 4x)",
            "bsrgan_x2 (BSRGAN, blind SR, 2x)",
        ]
        # Any *.onnx file the user has dropped in
        # waifu2x/pretrained_models/external_sr/onnx/ (e.g. a custom SPAN-style
        # export) is auto-discovered and appended here by filename -- no code
        # change needed to add a new one, see waifu2x/external_sr.py.
        for onnx_method in available_onnx_sr_methods():
            external_sr_choices.append(f"{onnx_method} (custom ONNX SR model)")
        self.cbo_external_sr_method = wx.ComboBox(
            self.grp_sr, style=wx.CB_READONLY, name="cbo_external_sr_method",
            choices=external_sr_choices)
        self.cbo_external_sr_method.SetSelection(0)
        self.cbo_external_sr_method.Enable(False)
        self.cbo_external_sr_method.SetToolTip(
            T("Which external model to use (only enabled when \"Use External SR Model\" above is "
              "checked). realesrgan_x4/x2: general-purpose real-world upscaling, good all-rounder. "
              "realesrgan_general_x4: a lighter/faster RealESRGAN variant. bsrgan_x4/x2: an alternative "
              "\"blind\" super-resolution model, sometimes better on heavily degraded/unknown-quality "
              "sources. \"(custom ONNX SR model)\" entries are any compatible model you've dropped into "
              "the onnx/ subfolder yourself. Each option only appears usable once its actual weight file "
              "is present on disk."))

        layout = wx.BoxSizer(wx.VERTICAL)
        layout.Add(self.opt_model, 0, wx.ALL | wx.EXPAND, border=4)
        layout.Add(self.opt_upscaling, 0, wx.ALL | wx.EXPAND, border=4)
        layout.Add(self.opt_noise_level, 0, wx.ALL | wx.EXPAND, border=4)
        layout.Add(self.chk_external_sr, 0, wx.ALL | wx.EXPAND, border=4)
        layout.Add(self.cbo_external_sr_method, 0, wx.ALL | wx.EXPAND, border=4)
        sizer_sr = wx.StaticBoxSizer(self.grp_sr, wx.VERTICAL)
        sizer_sr.Add(layout, 1, wx.ALL | wx.EXPAND, 8)

        # video decoding
        # hwaccel
        self.grp_video_dec = VideoDecodingBox(self.pnl_options, translate_function=T)

        # video encoding
        # max-fps, crf, preset, tune
        self.grp_video = VideoEncodingBox(self.pnl_options, translate_function=T,
                                          has_nvenc=has_nvenc(), has_qsv=has_qsv())

        # input video filter
        # deinterlace, rotate, vf
        self.grp_video_filter = wx.StaticBox(self.pnl_options, label=T("Video/Image Filter"))
        self.chk_start_time = wx.CheckBox(self.grp_video_filter, label=T("Start Time"),
                                          name="chk_start_time")
        self.chk_start_time.SetToolTip(
            T("Only process the video from this timestamp onward, instead of from the beginning. Useful "
              "for testing settings on a specific scene, or trimming unwanted intro footage. Video only."))
        self.txt_start_time = TimeCtrl(self.grp_video_filter, value="00:00:00", fmt24hr=True,
                                       name="txt_start_time")
        self.chk_end_time = wx.CheckBox(self.grp_video_filter, label=T("End Time"), name="chk_end_time")
        self.chk_end_time.SetToolTip(T("Stop processing at this timestamp instead of the end of the video. Video only."))
        self.txt_end_time = TimeCtrl(self.grp_video_filter, value="00:00:00", fmt24hr=True,
                                     name="txt_end_time")

        self.lbl_deinterlace = wx.StaticText(self.grp_video_filter, label=T("Deinterlace"))
        self.cbo_deinterlace = wx.ComboBox(self.grp_video_filter, choices=["", "yadif"],
                                           style=wx.CB_READONLY, name="cbo_deinterlace")
        self.cbo_deinterlace.SetSelection(0)
        self.cbo_deinterlace.SetToolTip(
            T("For old interlaced video sources (combed/striped look on motion). \"yadif\" converts it to "
              "normal progressive video before upscaling. Leave blank for modern, already-progressive "
              "sources (most streaming/BluRay video)."))

        self.lbl_vf = wx.StaticText(self.grp_video_filter, label=T("-vf (src)"))
        self.txt_vf = wx.TextCtrl(self.grp_video_filter, name="txt_vf")
        self.txt_vf.SetToolTip(
            T("Advanced: a raw ffmpeg video filter string applied to the source before upscaling (e.g. "
              "cropping, scaling). Leave blank unless you specifically need this."))

        # -- image
        self.lbl_rotate = wx.StaticText(self.grp_video_filter, label=T("Rotate"))
        self.cbo_rotate = wx.ComboBox(self.grp_video_filter, size=self.FromDIP((200, -1)),
                                      style=wx.CB_READONLY, name="cbo_rotate")
        self.cbo_rotate.Append("", "")
        self.cbo_rotate.Append(T("Left 90 (counterclockwise)"), "left")
        self.cbo_rotate.Append(T("Right 90 (clockwise)"), "right")
        self.cbo_rotate.SetSelection(0)
        self.cbo_rotate.SetToolTip(
            T("Rotate the source before upscaling, e.g. for footage/photos shot sideways on a phone."))

        self.chk_grain_noise = wx.CheckBox(self.grp_video_filter,
                                           label=T("Add Noise"), name="chk_grain_noise")
        self.cbo_grain_noise = EditableComboBox(self.grp_video_filter, choices=["0.5", "0.4", "0.3", "0.2", "0.1"],
                                                name="cbo_grain_noise")
        self.chk_grain_noise.SetValue(False)
        self.cbo_grain_noise.SetSelection(3)
        self.chk_grain_noise.SetToolTip(
            T("What it's for: deliberately adds a small amount of grain/noise back into the output, using "
              "the strength set in the box to its right (higher = more noise added).\n"
              "How it helps: for Photo or Generative AI content specifically — a fully clean, "
              "denoised-then-upscaled image can sometimes look artificially smooth/plastic-y, especially "
              "on AI-generated source images; a touch of noise restores a more natural, photographic look.\n"
              "Con: adds a small amount of extra file size/detail that isn't real information, purely "
              "aesthetic.\n"
              "Recommended: off for anime/illustration content; consider on for photo or AI-generated "
              "images if the clean result looks unnaturally smooth."))

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.chk_start_time, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_start_time, (0, 1), flag=wx.EXPAND)
        layout.Add(self.chk_end_time, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_end_time, (1, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_deinterlace, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_deinterlace, (2, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_vf, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_vf, (3, 1), flag=wx.EXPAND)
        layout.Add(wx.StaticLine(self.grp_video_filter), (4, 0), flag=wx.GROW)
        layout.Add(wx.StaticLine(self.grp_video_filter), (4, 1), flag=wx.GROW)
        layout.Add(self.lbl_rotate, (5, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_rotate, (5, 1), flag=wx.EXPAND)
        layout.Add(self.chk_grain_noise, (6, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_grain_noise, (6, 1), flag=wx.EXPAND)

        sizer_video_filter = wx.StaticBoxSizer(self.grp_video_filter, wx.VERTICAL)
        sizer_video_filter.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # processor settings
        # device, batch-size, TTA
        self.grp_processor = wx.StaticBox(self.pnl_options, label=T("Processor"))
        self.lbl_device = wx.StaticText(self.grp_processor, label=T("Device"))
        self.cbo_device = wx.ComboBox(self.grp_processor, size=self.FromDIP((240, -1)), style=wx.CB_READONLY,
                                      name="cbo_device")
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

        self.lbl_tile_size = wx.StaticText(self.grp_processor, label=T("Tile Size"))
        self.cbo_tile_size = wx.ComboBox(self.grp_processor,
                                         choices=["1024", "640", "400", "256", "64"],
                                         style=wx.CB_READONLY, name="cbo_tile_size")
        self.cbo_tile_size.SetSelection(3)
        self.cbo_tile_size.SetToolTip(
            T("What it's for: the model processes the image in square tiles of this size (pixels) rather "
              "than all at once, to keep VRAM use manageable on large images.\n"
              "Con of smaller tiles: more tile-boundary seams to blend, and more overhead — can be slower "
              "overall despite using less memory per step. Con of larger tiles: more VRAM used at once, "
              "risk of running out of memory on a large image.\n"
              "Recommended: 256 as a safe default; raise it if you have VRAM to spare and want fewer "
              "seams/more speed, lower it if you hit out-of-memory errors."))
        self.lbl_batch_size = wx.StaticText(self.grp_processor, label=T("Batch Size"))
        self.cbo_batch_size = wx.ComboBox(self.grp_processor,
                                          choices=["64", "32", "16", "8", "4", "2", "1"],
                                          style=wx.CB_READONLY, name="cbo_batch_size")
        self.cbo_batch_size.SetSelection(4)
        self.cbo_batch_size.SetToolTip(
            T("Video only. How many tiles/frames are processed together at once. Higher = faster overall "
              "but uses more VRAM. Lower it if you run out of memory; raise it if you have VRAM to spare "
              "and want faster processing."))

        self.chk_tta = wx.CheckBox(self.grp_processor, label=T("TTA"), name="chk_tta")
        self.chk_tta.SetToolTip(
            T("What it's for: runs the model on both the normal AND a mirrored version of the image, then "
              "blends the result — often a little cleaner/more accurate.\n"
              "Con: roughly doubles processing time, and some models ignore this setting entirely (no "
              "effect for them either way).\n"
              "Recommended: off for large batches/long videos where the time cost adds up; worth trying "
              "for a single important image where extra quality matters more than speed."))
        self.chk_amp = wx.CheckBox(self.grp_processor, label=T("FP16 (fast)"), name="chk_amp")
        self.chk_amp.SetValue(True)
        self.chk_amp.SetToolTip(
            T("Runs the AI math at lower numeric precision (FP16 instead of FP32), which is significantly "
              "faster and uses less VRAM on modern GPUs, with no visible quality cost in virtually all "
              "cases. Recommended: on."))

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.lbl_device, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_device, (0, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_tile_size, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_tile_size, (1, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_batch_size, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_batch_size, (2, 1), flag=wx.EXPAND)
        layout.Add(self.chk_tta, (3, 0), flag=wx.EXPAND)
        layout.Add(self.chk_amp, (3, 1), flag=wx.EXPAND)

        sizer_processor = wx.StaticBoxSizer(self.grp_processor, wx.VERTICAL)
        sizer_processor.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        sizer_video = wx.BoxSizer(wx.VERTICAL)
        sizer_video.Add(self.grp_video_dec.sizer, 0, wx.ALL | wx.EXPAND, border=4)
        sizer_video.Add(self.grp_video.sizer, 1, wx.ALL | wx.EXPAND, border=4)

        layout = wx.GridBagSizer(wx.HORIZONTAL)
        layout.Add(sizer_sr, (0, 0), (0, 4), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_processor, (1, 0), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_video, (1, 1), flag=wx.ALL | wx.EXPAND, border=0)
        layout.Add(sizer_video_filter, (1, 2), flag=wx.ALL | wx.EXPAND, border=4)
        self.pnl_options.SetSizer(layout)

        # processing panel
        self.pnl_process = wx.Panel(self)
        if LAYOUT_DEBUG:
            self.pnl_process.SetBackgroundColour("#fcc")
        self.prg_tqdm = wx.Gauge(self.pnl_process, style=wx.GA_HORIZONTAL)
        self.btn_start = wx.Button(self.pnl_process, label=T("Start"))
        self.btn_cancel = wx.Button(self.pnl_process, label=T("Cancel"))

        layout = wx.BoxSizer(wx.HORIZONTAL)
        layout.Add(self.prg_tqdm, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        layout.Add(self.btn_start, 0, wx.ALL, 4)
        layout.Add(self.btn_cancel, 0, wx.ALL, 4)
        self.pnl_process.SetSizer(layout)

        # main layout

        layout = wx.BoxSizer(wx.VERTICAL)
        layout.AddSpacer(8)
        layout.Add(self.pnl_file.panel, 0, wx.ALL | wx.EXPAND, 8)
        layout.Add(self.pnl_file_option, 0, wx.ALL | wx.EXPAND, 4)
        layout.Add(self.pnl_options, 1, wx.ALL | wx.EXPAND, 8)
        layout.Add(self.pnl_process, 0, wx.ALL | wx.EXPAND, 8)
        self.SetSizer(layout)

        # bind
        self.pnl_file.bind_input_path_changed(self.on_text_changed_txt_input)
        self.pnl_file.bind_output_path_changed(self.on_text_changed_txt_output)

        self.opt_model.Bind(wx.EVT_RADIOBOX, self.on_selected_index_changed_opt_model)
        self.opt_upscaling.Bind(wx.EVT_RADIOBOX, self.on_selected_index_changed_opt_upscaling)
        self.chk_external_sr.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_external_sr)

        self.btn_start.Bind(wx.EVT_BUTTON, self.on_click_btn_start)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self.on_click_btn_cancel)

        self.Bind(EVT_TQDM, self.on_tqdm)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        editable_comboboxes = [
            *self.grp_video.get_editable_comboboxes(),
            *self.grp_video_dec.get_editable_comboboxes(),
            self.cbo_grain_noise,
        ]

        self.SetDropTarget(FileDropCallback(self.on_drop_files))
        # Disable default drop target
        for control in (self.pnl_file.input_path_widget, self.pnl_file.output_path_widget, self.txt_vf,
                        self.txt_start_time, self.txt_end_time,
                        *editable_comboboxes):
            control.SetDropTarget(FileDropCallback(self.on_drop_files))

        # Fix Frame and Panel background colors are different in windows
        self.SetBackgroundColour(self.pnl_file_option.GetBackgroundColour())

        # state
        self.btn_cancel.Disable()

        self.persistence_manager = persist.PersistenceManager.Get()
        self.persistence_manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
        self.persistence_manager.SetPersistenceFile(CONFIG_PATH)
        persistent_manager_register_all(self.persistence_manager, self)
        for control in editable_comboboxes:
            persistent_manager_register(self.persistence_manager, control, EditableComboBoxPersistentHandler)
        persistent_manager_restore_all(self.persistence_manager)

        self.update_start_button_state()
        self.update_upscaling_state()
        self.update_noise_level_state()
        self.update_external_sr_state()
        self.update_input_option_state()
        self.grp_video.update_controls()

    def on_close(self, event):
        self.persistence_manager.SaveAndUnregister()
        event.Skip()

    def on_drop_files(self, x, y, filenames):
        if filenames:
            self.pnl_file.set_input_path(filenames[0])
        return True

    def update_upscaling_state(self):
        if self.model_4x_support[self.opt_model.GetSelection()]:
            self.opt_upscaling.EnableItem(2, True)
        else:
            if self.opt_upscaling.GetSelection() == 2:
                self.opt_upscaling.SetSelection(1)
            self.opt_upscaling.EnableItem(2, False)

    def update_noise_level_state(self):
        if self.opt_upscaling.GetSelection() == 0:
            # 1x
            if self.opt_noise_level.GetSelection() == 0:
                self.opt_noise_level.SetSelection(1)
            self.opt_noise_level.EnableItem(0, False)
        else:
            self.opt_noise_level.EnableItem(0, True)

    def update_start_button_state(self):
        if not self.processing:
            if self.pnl_file.input_path and self.pnl_file.output_path:
                self.btn_start.Enable()
            else:
                self.btn_start.Disable()

    def update_input_option_state(self):
        input_path = self.pnl_file.input_path
        if path.isdir(input_path) or is_text(input_path):
            self.chk_resume.Enable()
            self.chk_recursive.Enable()
        else:
            self.chk_resume.Disable()
            self.chk_recursive.Disable()
        self.chk_recursive.SetValue(False)

    def reset_time_range(self):
        self.chk_start_time.SetValue(False)
        self.chk_end_time.SetValue(False)
        self.txt_start_time.SetValue("00:00:00")
        self.txt_end_time.SetValue("00:00:00")

    def on_selected_index_changed_opt_model(self, event):
        self.update_upscaling_state()

    def on_selected_index_changed_opt_upscaling(self, event):
        self.update_noise_level_state()

    def on_changed_chk_external_sr(self, event):
        self.update_external_sr_state()

    def update_external_sr_state(self):
        enabled = self.chk_external_sr.GetValue()
        self.cbo_external_sr_method.Enable(enabled)
        # External models have their own fixed scale, no noise-level concept, and
        # no art/photo/scan style variants -- these controls simply don't apply
        # while an external model is selected.
        self.opt_model.Enable(not enabled)
        self.opt_upscaling.Enable(not enabled)
        self.opt_noise_level.Enable(not enabled)

    def resolve_output_path(self, input_path, output_path):
        if is_output_dir(output_path):
            args = self.parse_args()
            video = is_video(input_path)
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

    def confirm_overwrite(self, args):
        input_path = self.pnl_file.input_path
        output_path = self.pnl_file.output_path

        if is_output_dir(output_path):
            output_path = path.join(output_path, make_output_filename(input_path, args, video=is_video(input_path)))
        else:
            output_path = output_path

        if path.exists(output_path) and is_video(output_path):
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

    def parse_args(self):
        if not validate_number(self.grp_video.max_fps, 0.25, 1000.0, allow_empty=False):
            self.show_validation_error_message(T("Max FPS"), 0.25, 1000.0)
            return None
        if not validate_number(self.grp_video.crf, 0, 51, is_int=True):
            self.show_validation_error_message(T("CRF"), 0, 51)
            return None
        if self.chk_grain_noise.GetValue():
            if not validate_number(self.cbo_grain_noise.GetValue(), 0.0, 1.0):
                self.show_validation_error_message(T("Add Noise"), 0.0, 1.0)
                return None
        parser = create_parser(required_true=False)

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

        device = int(self.cbo_device.GetClientData(self.cbo_device.GetSelection()))
        if device == -2:
            # All CUDA
            gpus = list(range(torch.cuda.device_count()))
        else:
            gpus = [device]

        if self.chk_external_sr.GetValue():
            # Externally-pretrained model (RealESRGAN/BSRGAN) -- fixed scale per
            # model, no noise-level concept, no model_dir/style scheme of its own
            # (external_sr.py resolves its own weight-file location). See
            # docs/ai/AI_DECISIONS.md.
            method = self.cbo_external_sr_method.GetValue().split(" ")[0]
            model_dir = None
            noise_level = 0  # unused for this path, but must satisfy the parser's choices
            tta = self.chk_tta.GetValue()
        else:
            noise_level = int(self.opt_noise_level.GetSelection()) - 1
            scale = 2 ** (int(self.opt_upscaling.GetSelection()))
            assert noise_level in {-1, 0, 1, 2, 3}
            assert scale in {1, 2, 4}
            if scale == 1:
                method = "noise"
            else:
                if noise_level >= 0:
                    method = f"noise_scale{scale}x"
                else:
                    method = f"scale{scale}x"
            model_dir = self.model_dirs[self.opt_model.GetSelection()]
            tta = self.chk_tta.GetValue() and self.model_tta_support[self.opt_model.GetSelection()]

        input_path = self.pnl_file.input_path
        resume = (path.isdir(input_path) or is_text(input_path)) and self.chk_resume.GetValue()
        recursive = path.isdir(input_path) and self.chk_recursive.GetValue()
        start_time = self.txt_start_time.GetValue() if self.chk_start_time.GetValue() else None
        end_time = self.txt_end_time.GetValue() if self.chk_end_time.GetValue() else None

        parser.set_defaults(
            input=input_path,
            output=self.pnl_file.output_path,
            model_dir=model_dir,
            noise_level=noise_level,
            method=method,
            yes=True,  # TODO: remove this

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

            rotate_right=rotate_right,
            rotate_left=rotate_left,
            disable_exif_transpose=not self.chk_exif_transpose.GetValue(),
            vf=vf,
            grain=(float(self.cbo_grain_noise.GetValue()) > 0.0 and self.chk_grain_noise.GetValue()),
            grain_strength=float(self.cbo_grain_noise.GetValue()),

            gpu=gpus,
            batch_size=int(self.cbo_batch_size.GetValue()),
            tile_size=int(self.cbo_tile_size.GetValue()),
            tta=tta,
            disable_amp=not self.chk_amp.GetValue(),

            resume=resume,
            recursive=recursive,
            start_time=start_time,
            end_time=end_time,
        )
        args = parser.parse_args()
        set_state_args(
            args,
            stop_event=self.stop_event,
            tqdm_fn=functools.partial(TQDMGUI, self))
        return args

    def on_click_btn_start(self, event):
        args = self.parse_args()
        if args is None:
            return
        if not self.confirm_overwrite(args):
            return
        self.btn_start.Disable()
        self.btn_cancel.Enable()
        self.stop_event.clear()
        self.prg_tqdm.SetValue(0)
        self.SetStatusText("...")

        startWorker(self.on_exit_worker, waifu2x_main, wargs=(args,))
        self.processing = True

    def on_exit_worker(self, result):
        try:
            ret = result.get()  # noqa
            if not self.stop_event.is_set():
                self.prg_tqdm.SetValue(self.prg_tqdm.GetRange())
                self.SetStatusText(T("Finished"))
            else:
                self.SetStatusText(T("Cancelled"))
        except:  # noqa
            self.SetStatusText(T("Error"))
            e_type, e, stacktrace = sys.exc_info()
            message = getattr(e, "message", str(e))
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)

        self.processing = False
        self.btn_cancel.Disable()
        self.update_start_button_state()

        # free vram
        gc.collect()
        torch.cuda.empty_cache()

    def on_click_btn_cancel(self, event):
        self.stop_event.set()

    def on_tqdm(self, event):
        type, value, desc = event.GetValue()
        desc = desc if desc else ""
        if type == 0:
            # initialize
            self.prg_tqdm.SetRange(value)
            self.prg_tqdm.SetValue(0)
            self.start_time = time()
            self.SetStatusText(f"0/{value} {desc}")
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
            fps = pos / (now - self.start_time)
            remaining_time = int((end_pos - pos) / fps)
            h = remaining_time // 3600
            m = (remaining_time - h * 3600) // 60
            s = (remaining_time - h * 3600 - m * 60)
            t = f"{m:02d}:{s:02d}" if h == 0 else f"{h:02d}:{m:02d}:{s:02d}"
            self.SetStatusText(f"{pos}/{end_pos} [ {t}, {fps:.2f}FPS ] {desc}")
        elif type == 2:
            # close
            pass


LOCALE_DICT = LOCALES.get(get_default_locale(), {})
LOCALE_DICT_EN = LOCALES["en_US"]


def T(s):
    if s in LOCALE_DICT:
        return LOCALE_DICT[s]
    if s in LOCALE_DICT_EN:
        return LOCALE_DICT_EN[s]
    return s


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lang", type=str, choices=list(LOCALES.keys()), help="translation language")
    args = parser.parse_args()
    if args.lang:
        global LOCALE_DICT
        LOCALE_DICT = LOCALES.get(args.lang, {})
    sys.argv = [sys.argv[0]]  # clear command arguments

    app = Waifu2xApp()
    app.MainLoop()


if __name__ == "__main__":
    pyav_init_cuda_primary_context()
    init_win32_dpi()
    main()
