from fractions import Fraction
from typing import Callable, Dict

import torch
from av.video.reformatter import ColorPrimaries, ColorRange, Colorspace, ColorTrc


class VideoOutputConfig:
    pix_fmt: str
    fps: int | float | Fraction | None
    output_fps: Fraction | None
    options: Dict[str, str]
    container_options: Dict[str, str]
    output_width: int | None
    output_height: int | None
    colorspace: str
    container_format: str | None
    video_codec: str | None
    state_updated: Callable[["VideoOutputConfig"], None] | None
    device: torch.device | None
    raw_frame_sink: Callable[[bytes, int, int, str, int, int, int, int], None] | None

    # State properties
    output_colorspace: int | None
    output_color_primaries: int | None
    output_color_trc: int | None
    source_color_range: int | None

    def __init__(
        self,
        pix_fmt: str = "yuv420p",
        fps: int | float | Fraction | None = 30,
        options: Dict[str, str] = {},
        container_options: Dict[str, str] = {},
        output_width: int | None = None,
        output_height: int | None = None,
        colorspace: str | None = None,
        container_format: str | None = None,
        video_codec: str | None = None,
        output_fps: Fraction | None = None,
        device: torch.device | None = None,
        output_colorspace: Colorspace | int | None = None,
        output_color_primaries: ColorPrimaries | int | None = None,
        output_color_trc: ColorTrc | int | None = None,
        source_color_range: ColorRange | int | None = None,
        metadata: Dict[str, str] | None = None,
        raw_frame_sink: Callable[[bytes, int, int, str, int, int, int, int], None] | None = None,
    ):
        self.pix_fmt = pix_fmt
        self.fps = fps
        self.output_fps = output_fps
        self.options = options
        self.container_options = container_options
        self.metadata = metadata or {}
        self.output_width = output_width
        self.output_height = output_height
        self.colorspace = colorspace if colorspace is not None else "auto"
        self.container_format = container_format
        self.video_codec = video_codec
        # ADR-283: direct-to-MVC single-pass mode -- when set, _process_video() sends
        # each finished frame's raw bytes straight here instead of opening an output
        # container/temp file at all. None (the default) is a byte-for-byte no-op for
        # every existing caller (this file is shared with waifu2x).
        #
        # A real live test found the first version of this feature (bytes/width/
        # height/pix_fmt only) produced a visibly wrong-colored file. TWO real,
        # separate things were needed, not one:
        # (1) The actual root cause: the caller MUST read the real pix_fmt off the
        #     frame itself (frame.format.name), never assume/pass the merely
        #     CONFIGURED one (e.g. config.pix_fmt). On a CUDA/hwaccel pipeline the
        #     frame's own real format is very often "nv12" (one Y plane + one
        #     INTERLEAVED UV plane) even when config.pix_fmt says "yuv420p" (planar,
        #     separate U-then-V planes) -- .encode() converts this correctly
        #     internally, .to_ndarray() does not, so a caller trusting config.pix_fmt
        #     tells the downstream raw-video reader the wrong byte layout entirely,
        #     scrambling every frame's chroma in a way that still looks like a real
        #     (just badly desaturated/shifted) image -- confirmed by dumping raw
        #     bytes and manually decoding them outside ffmpeg entirely; only
        #     correcting the declared pix_fmt fixed it, confirmed pixel-identical to
        #     the existing two-stage MVC path on the same source frame.
        # (2) Necessary but NOT sufficient on its own: the trailing colorspace/
        #     color_primaries/color_trc/color_range ints (real, per-frame values
        #     output_reformatter already computed) -- raw video (`-f rawvideo`)
        #     carries no embedded color metadata at all, so a downstream decoder has
        #     nothing else to go on. apply_color_settings() communicates this same
        #     information to a real container's codec context in the normal
        #     (non-sink) path -- this is the raw-pipe equivalent.
        self.raw_frame_sink = raw_frame_sink

        self.state_updated = lambda config: None

        self.output_colorspace = int(output_colorspace) if output_colorspace is not None else None
        self.output_color_primaries = int(output_color_primaries) if output_color_primaries is not None else None
        self.output_color_trc = int(output_color_trc) if output_color_trc is not None else None
        self.source_color_range = int(source_color_range) if source_color_range is not None else None

    def __repr__(self):
        return "VideoOutputConfig({!r})".format(self.__dict__)
