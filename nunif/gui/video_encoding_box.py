import wx
from .common import EditableComboBox
import av

LEVEL_LIBX264 = ["3.0", "3.1", "3.2", "4.0", "4.1", "4.2", "5.0", "5.1", "5.2", "6.0", "6.2"]
LEVEL_LIBX265 = ["3.0", "3.1", "4.0", "4.1", "5.0", "5.1", "5.2", "6.0", "6.1", "6.2", "8.5"]
LEVEL_ALL = ["auto"] + sorted(list(set(LEVEL_LIBX264) | set(LEVEL_LIBX265)), key=lambda v: float(v))

TUNE_LIBX264 = ["film", "animation", "grain", "stillimage", "psnr"]
TUNE_LIBX265 = ["grain", "animation", "psnr", "fastdecode", "zerolatency"]
TUNE_NVENC_H264 = ["hq", "ll", "ull", "lossless"]
TUNE_NVENC_HEVC = ["hq", "uhq", "ll", "ull", "lossless"]
TUNE_NVENC = TUNE_NVENC_HEVC
TUNE_ALL = [""] + list(dict.fromkeys(TUNE_LIBX264 + TUNE_LIBX265 + TUNE_NVENC))

PRESET_LIBX264 = ["ultrafast", "superfast", "veryfast", "faster", "fast",
                  "medium", "slow", "slower", "veryslow", "placebo"]
PRESET_NVENC = ["fast", "medium", "slow",
                "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
PRESET_QSV = ["veryfast", "fast", "medium", "slow", "veryslow"]
PRESET_ALL = list(dict.fromkeys(PRESET_LIBX264 + PRESET_NVENC))
PRESET_DEFAULT = "medium"

CODEC_ALL = ["libx264", "libopenh264", "libx265", "utvideo", "ffv1",
             "h264_nvenc", "hevc_nvenc",
             "h264_qsv", "hevc_qsv"]

PIX_FMT_ALL = ["yuv420p", "yuv444p", "yuv420p10le", "rgb24", "gbrp10le", "gbrp16le"]
CODEC_PIX_FMT = {
    "libx264": ["yuv420p", "yuv444p", "yuv420p10le", "rgb24", "gbrp10le"],
    "libx265": ["yuv420p", "yuv444p", "yuv420p10le", "rgb24", "gbrp10le"],
    "h264_nvenc": ["yuv420p", "yuv444p", "yuv420p10le", "rgb24", "gbrp16le"],
    "hevc_nvenc": ["yuv420p", "yuv444p", "yuv420p10le", "rgb24", "gbrp16le"],
    "h264_qsv": ["yuv420p"],
    "hevc_qsv": ["yuv420p", "yuv420p10le"],
    "libopenh264": ["yuv420p"],
    "utvideo": ["yuv420p", "yuv444p", "rgb24"],
    "ffv1": ["yuv420p", "yuv444p", "yuv420p10le", "rgb24", "gbrp16le"],
}


def get_pix_fmt(codec):
    return CODEC_PIX_FMT.get(codec, PIX_FMT_ALL)


def empty_translate_function(s):
    return s


def codecs_available(codecs):
    return [codec for codec in codecs if codec in av.codec.codecs_available]


class VideoEncodingBox():
    def __init__(self, parent, name_prefix="", translate_function=empty_translate_function,
                 has_nvenc=False, has_qsv=False, **kwargs):
        T = translate_function
        prefix = name_prefix + "_" if name_prefix else ""
        self.has_nvenc = has_nvenc
        self.has_qsv = has_qsv

        self.grp_video = wx.StaticBox(parent, label=T("Video Encoding"), **kwargs)

        self.lbl_video_format = wx.StaticText(self.grp_video, label=T("Video Format"))
        self.cbo_video_format = wx.ComboBox(self.grp_video, choices=["mp4", "mkv", "avi"],
                                            name=f"{prefix}cbo_video_format")
        self.cbo_video_format.SetEditable(False)
        self.cbo_video_format.SetSelection(0)
        self.cbo_video_format.SetToolTip(
            T("The output container/file type. mp4 is the most universally compatible. mkv supports more "
              "advanced features (like Dolby Vision/HDR10+ metadata and lossless codecs). avi is mainly "
              "for the lossless utvideo codec. Recommended: mkv if you need HDR/Dolby Vision or lossless "
              "output, mp4 otherwise."))

        self.lbl_video_codec = wx.StaticText(self.grp_video, label=T("Video Codec"))
        self.cbo_video_codec = EditableComboBox(
            self.grp_video, choices=CODEC_ALL,
            name=f"{prefix}cbo_video_codec")
        self.cbo_video_codec.SetSelection(0)
        self.cbo_video_codec.SetToolTip(
            T("The compression format used to encode the output video. libx264/libx265 (CPU-based, "
              "h264/HEVC) give the best quality-per-file-size and work everywhere. h264_nvenc/hevc_nvenc "
              "use your NVIDIA GPU's hardware encoder — much faster, slightly lower quality-per-file-size. "
              "utvideo/ffv1 are lossless (huge files, no quality loss at all) for archival/further editing. "
              "Recommended: libx265 for best quality, hevc_nvenc if encoding speed matters and you have an "
              "NVIDIA GPU."))

        self.lbl_fps = wx.StaticText(self.grp_video, label=T("Max FPS"))
        self.cbo_fps = EditableComboBox(
            self.grp_video, choices=["1000", "60", "59.94", "30", "29.97", "24", "23.976", "15", "1", "0.25"],
            name=f"{prefix}cbo_fps")
        self.cbo_fps.SetSelection(3)
        self.cbo_fps.SetToolTip(
            T("Caps the output frame rate. If the source is already at or below this, nothing changes. "
              "Lowering it reduces processing time roughly proportionally (half the frames = about half "
              "the time), at the cost of less smooth motion. Recommended: leave high (e.g. 1000) to just "
              "keep the source's own frame rate, unless you specifically want to reduce it for speed."))

        self.lbl_pix_fmt = wx.StaticText(self.grp_video, label=T("Pixel Format"))
        self.cbo_pix_fmt = wx.ComboBox(self.grp_video, choices=PIX_FMT_ALL,
                                       name=f"{prefix}cbo_pix_fmt")
        self.cbo_pix_fmt.SetEditable(False)
        self.cbo_pix_fmt.SetSelection(0)
        self.cbo_pix_fmt.SetToolTip(
            T("The color sampling/precision the encoder stores internally. yuv420p (8-bit) is the most "
              "compatible default. yuv420p10le adds the precision-boosting benefit described under Bit "
              "Depth Upgrade (where available, that setting can apply on top of this). yuv444p/rgb24/gbrp* "
              "avoid color-detail loss around sharp red/edge areas, at a larger file size — see this "
              "project's README for when that matters. Recommended: yuv420p10le for most uses."))

        self.lbl_colorspace = wx.StaticText(self.grp_video, label=T("Colorspace"))
        self.cbo_colorspace = wx.ComboBox(
            self.grp_video,
            choices=["auto", "unspecified",
                     "bt709", "bt709-pc", "bt709-tv",
                     "bt601", "bt601-pc", "bt601-tv"],
            name=f"{prefix}cbo_colorspace")
        self.cbo_colorspace.SetEditable(False)
        self.cbo_colorspace.SetSelection(0)
        self.cbo_colorspace.SetToolTip(
            T("How color values are tagged/interpreted (affects color accuracy on playback, not detail). "
              "\"auto\" matches the source's own colorspace and is correct for virtually all sources. "
              "Only change this if you know your source's colorspace is mistagged."))

        self.lbl_crf = wx.StaticText(self.grp_video, label=T("CRF"))
        self.cbo_crf = EditableComboBox(self.grp_video, choices=[str(n) for n in range(16, 28)],
                                        name=f"{prefix}cbo_crf")
        self.cbo_crf.SetSelection(4)
        self.cbo_crf.SetToolTip(
            T("Constant Rate Factor: the main quality/file-size dial for most codecs. Lower number = "
              "higher quality and bigger file; higher number = smaller file and lower quality. "
              "Recommended: 15-18 for near-lossless/archival quality, 20-23 for a good everyday balance."))

        self.lbl_bitrate = wx.StaticText(self.grp_video, label=T("Bitrate"))
        self.cbo_bitrate = EditableComboBox(self.grp_video, choices=["160M", "50M", "16M", "12M", "8M", "4M"],
                                            name=f"{prefix}cbo_bitrate")
        self.cbo_bitrate.SetSelection(4)
        self.cbo_bitrate.SetToolTip(
            T("Only used by libopenh264, which doesn't support CRF. Sets a fixed data rate for the video "
              "(M = megabits/second) — higher = better quality and bigger file. Recommended: 8M-16M for "
              "1080p, higher for 4K."))

        self.lbl_profile_level = wx.StaticText(self.grp_video, label=T("Level"))
        self.cbo_profile_level = EditableComboBox(self.grp_video, choices=LEVEL_ALL, name=f"{prefix}cbo_profile_level")
        self.cbo_profile_level.SetSelection(0)
        self.cbo_profile_level.SetToolTip(
            T("A technical compatibility limit (resolution/bitrate ceiling) some older TVs/players check "
              "before they'll play a file. Recommended: \"auto\" — only set this manually if a specific "
              "playback device of yours refuses to play the output."))

        self.lbl_preset = wx.StaticText(self.grp_video, label=T("Preset"))
        self.cbo_preset = wx.ComboBox(
            self.grp_video, choices=PRESET_ALL,
            name=f"{prefix}cbo_preset")
        self.cbo_preset.SetEditable(False)
        self.cbo_preset.SetSelection(PRESET_ALL.index(PRESET_DEFAULT))
        self.cbo_preset.SetToolTip(
            T("How hard the encoder works to compress efficiently. Slower presets (slow/slower/veryslow) "
              "give a smaller file at the same quality (CRF), but take longer to encode. Faster presets "
              "(fast/veryfast/ultrafast) encode quicker but produce a larger file for the same quality. "
              "Recommended: medium-slow for a good balance; veryfast if encoding time matters more than "
              "file size."))

        self.lbl_tune = wx.StaticText(self.grp_video, label=T("Tune"))
        self.cbo_tune = wx.ComboBox(
            self.grp_video, choices=TUNE_ALL, name=f"{prefix}cbo_tune")
        self.cbo_tune.SetEditable(False)
        self.cbo_tune.SetSelection(0)
        self.cbo_tune.SetToolTip(
            T("For libx264/libx265: nudges the encoder's choices for a specific kind of content — "
              "e.g. \"grain\" preserves film grain/noise better, \"animation\" suits flat-color cartoon "
              "content. Leave blank unless your source clearly matches one of these categories.\n"
              "For hevc_nvenc/h264_nvenc: chooses the GPU encoder's quality/speed tradeoff. \"hq\" = "
              "high quality (good default). \"uhq\" (HEVC only, needs a recent NVIDIA GPU) = an even "
              "higher quality mode than hq, using extra quality heuristics, for a modest speed cost — "
              "worth using if your GPU supports it and you want the best NVENC result. \"lossless\" = "
              "no quality loss at all, but a very large file. \"ll\"/\"ull\" are for low-latency "
              "live-streaming, not useful for a normal conversion."))
        self.chk_tune_fastdecode = wx.CheckBox(self.grp_video, label=T("fastdecode"),
                                               name=f"{prefix}chk_tune_fastdecode")
        self.chk_tune_fastdecode.SetValue(False)
        self.chk_tune_fastdecode.SetToolTip(
            T("Makes the resulting file easier/cheaper to decode during playback (helpful on weaker "
              "playback devices), at a small cost to compression efficiency (slightly bigger file)."))
        self.chk_tune_zerolatency = wx.CheckBox(self.grp_video, label=T("zerolatency"),
                                                name=f"{prefix}chk_tune_zerolatency")
        self.chk_tune_zerolatency.SetValue(False)
        self.chk_tune_zerolatency.SetToolTip(
            T("For live-streaming/real-time use cases — not useful for a normal movie file conversion "
              "like this. Leave off."))

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.SetEmptyCellSize((0, 0))
        layout.Add(self.lbl_fps, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_fps, (0, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_video_format, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_video_format, (1, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_video_codec, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_video_codec, (2, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_pix_fmt, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_pix_fmt, (3, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_colorspace, (4, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_colorspace, (4, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_crf, (5, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_crf, (5, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_bitrate, (6, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_bitrate, (6, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_profile_level, (7, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_profile_level, (7, 1), flag=wx.EXPAND)

        layout.Add(self.lbl_preset, (8, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_preset, (8, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_tune, (9, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_tune, (9, 1), flag=wx.EXPAND)
        layout.Add(self.chk_tune_fastdecode, (10, 1), flag=wx.EXPAND)
        layout.Add(self.chk_tune_zerolatency, (11, 1), flag=wx.EXPAND)

        self.box_sizer = wx.StaticBoxSizer(self.grp_video, wx.VERTICAL)
        self.box_sizer.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        self.cbo_video_format.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_video_format)
        self.cbo_video_codec.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_video_codec)

    def update_controls(self):
        self.update_video_format()
        self.update_video_codec()

    def get_editable_comboboxes(self):
        return [
            self.cbo_fps,
            self.cbo_crf,
            self.cbo_profile_level,
            self.cbo_video_codec,
        ]

    @property
    def sizer(self):
        return self.box_sizer

    @property
    def max_fps(self):
        return float(self.cbo_fps.GetValue())

    @property
    def video_format(self):
        return self.cbo_video_format.GetValue()

    @property
    def video_codec(self):
        return self.cbo_video_codec.GetValue()

    @property
    def pix_fmt(self):
        return self.cbo_pix_fmt.GetValue()

    @property
    def colorspace(self):
        return self.cbo_colorspace.GetValue()

    @property
    def crf(self):
        return int(self.cbo_crf.GetValue())

    @property
    def bitrate(self):
        return self.cbo_bitrate.GetValue()

    @property
    def profile_level(self):
        level = self.cbo_profile_level.GetValue()
        if not level or level == "auto":
            return None
        else:
            return level

    @property
    def preset(self):
        return self.cbo_preset.GetValue()

    @property
    def tune(self):
        tune_set = set()
        if self.chk_tune_zerolatency.GetValue():
            tune_set.add("zerolatency")
        if self.chk_tune_fastdecode.GetValue():
            tune_set.add("fastdecode")
        if self.cbo_tune.GetValue():
            tune_set.add(self.cbo_tune.GetValue())
        return list(tune_set)

    def on_selected_index_changed_cbo_video_format(self, event):
        self.update_video_format()

    def on_selected_index_changed_cbo_video_codec(self, event):
        self.update_video_codec()

    def update_video_format(self):
        name = self.cbo_video_format.GetValue()
        if name == "avi":
            self.cbo_profile_level.Disable()
            self.cbo_crf.Disable()
            self.cbo_preset.Disable()
            self.cbo_tune.Disable()
            self.chk_tune_fastdecode.Disable()
            self.chk_tune_zerolatency.Disable()
        else:
            self.cbo_profile_level.Enable()
            self.cbo_crf.Enable()
            self.cbo_preset.Enable()
            self.cbo_tune.Enable()
            self.chk_tune_fastdecode.Enable()
            self.chk_tune_zerolatency.Enable()

        # codec
        if name == "avi":
            choices = codecs_available(["utvideo"])
        elif name == "mkv":
            choices = codecs_available(["libx264", "libopenh264", "libx265"])
            if self.has_nvenc:
                choices += codecs_available(["h264_nvenc", "hevc_nvenc"])
            if self.has_qsv:
                choices += codecs_available(["h264_qsv", "hevc_qsv"])
            choices += codecs_available(["ffv1"])
        else:
            choices = codecs_available(["libx264", "libopenh264", "libx265"])
            if self.has_nvenc:
                choices += codecs_available(["h264_nvenc", "hevc_nvenc"])
            if self.has_qsv:
                choices += codecs_available(["h264_qsv", "hevc_qsv"])

        user_codec = self.cbo_video_codec.GetValue()
        if user_codec not in CODEC_ALL:
            choices.append(user_codec)
        self.cbo_video_codec.SetItems(choices)
        if user_codec in choices:
            self.cbo_video_codec.SetSelection(choices.index(user_codec))
        else:
            self.cbo_video_codec.SetSelection(0)
        self.update_video_codec()

    def update_video_codec(self):
        container_format = self.cbo_video_format.GetValue()
        codec = self.cbo_video_codec.GetValue()

        # enabel/disable options
        if container_format == "avi" or codec == "libopenh264":
            self.cbo_profile_level.Disable()
            self.cbo_crf.Disable()
            self.cbo_preset.Disable()
            self.cbo_tune.Disable()
            self.chk_tune_fastdecode.Disable()
            self.chk_tune_zerolatency.Disable()
        else:
            self.cbo_profile_level.Enable()
            self.cbo_crf.Enable()
            self.cbo_preset.Enable()
            self.cbo_tune.Enable()
            self.chk_tune_fastdecode.Enable()
            self.chk_tune_zerolatency.Enable()

        # crf
        if codec == "libopenh264":
            self.lbl_bitrate.Show()
            self.cbo_bitrate.Show()
            self.lbl_crf.Hide()
            self.cbo_crf.Hide()
        else:
            self.lbl_bitrate.Hide()
            self.cbo_bitrate.Hide()
            self.lbl_crf.Show()
            self.cbo_crf.Show()

        # pix_fmt
        user_pix_fmt = self.cbo_pix_fmt.GetValue()
        choices = get_pix_fmt(codec)
        self.cbo_pix_fmt.SetItems(choices)
        if user_pix_fmt in choices:
            self.cbo_pix_fmt.SetSelection(choices.index(user_pix_fmt))
        else:
            self.cbo_pix_fmt.SetSelection(0)

        # level
        user_level = self.cbo_profile_level.GetValue()
        if codec in {"libx264", "h264_nvenc"}:
            choices = ["auto"] + LEVEL_LIBX264
        elif codec in {"libx265", "hevc_nvenc"}:
            choices = ["auto"] + LEVEL_LIBX265
        elif codec in {"h264_qsv", "hevc_qsv"}:
            choices = ["auto"]
        else:
            choices = LEVEL_ALL

        self.cbo_profile_level.SetItems(choices)
        if user_level in choices:
            self.cbo_profile_level.SetSelection(choices.index(user_level))
        else:
            self.cbo_profile_level.SetSelection(0)

        # preset
        if container_format in {"mp4", "mkv"}:
            preset = self.cbo_preset.GetValue()
            default_preset = PRESET_DEFAULT
            if codec in {"libx265", "libx264", "libopenh264"}:
                choices = PRESET_LIBX264
            elif codec in {"h264_nvenc", "hevc_nvenc"}:
                choices = PRESET_NVENC
            elif codec in {"h264_qsv", "hevc_qsv"}:
                choices = PRESET_QSV
            else:
                choices = PRESET_ALL

            self.cbo_preset.SetItems(choices)
            if preset in choices:
                self.cbo_preset.SetSelection(choices.index(preset))
            else:
                self.cbo_preset.SetSelection(choices.index(default_preset))
        # tune
        if container_format in {"mp4", "mkv"}:
            if codec == "libx265":
                # tune
                tune = []
                if self.chk_tune_zerolatency.IsChecked():
                    tune.append("zerolatency")
                if self.chk_tune_fastdecode.IsChecked():
                    tune.append("fastdecode")
                tune.append(self.cbo_tune.GetValue())

                choices = [""] + TUNE_LIBX265
                self.cbo_tune.SetItems(choices)
                if tune[0] in choices:
                    self.cbo_tune.SetSelection(choices.index(tune[0]))
                else:
                    self.cbo_tune.SetSelection(0)

                self.chk_tune_fastdecode.SetValue(False)
                self.chk_tune_fastdecode.Disable()
                self.chk_tune_zerolatency.SetValue(False)
                self.chk_tune_zerolatency.Disable()

            elif codec == "libx264":
                tune = []
                tune.append(self.cbo_tune.GetValue())
                if self.chk_tune_zerolatency.IsChecked():
                    tune.append("zerolatency")
                if self.chk_tune_fastdecode.IsChecked():
                    tune.append("fastdecode")

                choices = [""] + TUNE_LIBX264
                self.cbo_tune.SetItems(choices)
                if tune[0] in choices:
                    self.cbo_tune.SetSelection(choices.index(tune[0]))
                else:
                    self.cbo_tune.SetSelection(0)

                self.chk_tune_fastdecode.Enable()
                self.chk_tune_fastdecode.SetValue("fastdecode" in tune)
                self.chk_tune_zerolatency.Enable()
                self.chk_tune_zerolatency.SetValue("zerolatency" in tune)
            elif codec in {"h264_nvenc", "hevc_nvenc"}:
                tune = self.cbo_tune.GetValue()
                choices = [""] + (TUNE_NVENC_HEVC if codec == "hevc_nvenc" else TUNE_NVENC_H264)
                self.cbo_tune.SetItems(choices)
                if tune in choices:
                    self.cbo_tune.SetSelection(choices.index(tune))
                else:
                    self.cbo_tune.SetSelection(0)
                self.chk_tune_fastdecode.SetValue(False)
                self.chk_tune_fastdecode.Disable()
                self.chk_tune_zerolatency.SetValue(False)
                self.chk_tune_zerolatency.Disable()
            elif codec in {"h264_qsv", "hevc_qsv"}:
                self.cbo_tune.SetItems([""])
                self.cbo_tune.SetSelection(0)
                self.cbo_tune.Disable()
                self.chk_tune_fastdecode.SetValue(False)
                self.chk_tune_fastdecode.Disable()
                self.chk_tune_zerolatency.SetValue(False)
                self.chk_tune_zerolatency.Disable()
                self.cbo_profile_level.Disable()

        # update Layout
        self.sizer.Layout()
