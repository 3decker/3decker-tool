"""Settings schema for the web-rendered iw3 GUI (ADR-081, expanded ADR-082).

This is the single source of truth the JS renderer builds its controls from
(see schema_export.py) -- NOT a hand-authored HTML form. Each Field's
`cli_arg` must exactly match a real argument registered in
iw3.utils.create_parser(); schema_export.py's self-test walks the parser's
own registered actions and asserts every field with a non-None cli_arg here
actually exists there, so this can never silently drift from the real CLI
the way a hand-maintained control list could.

ADR-081 (Phase 1) covered 8 fields across 2 tabs, just enough to prove the
architecture end-to-end. ADR-082 (this expansion) covers every remaining
CLI-arg-backed setting across 5 tabs: Stereo Generation (completed),
Dual-Pass Depth Blend (core fields only, see note below), Video Filter,
Video Decoding, Video Encoding, and Processor (completed). Standalone Tools
is deliberately NOT covered here -- those 6 mini-tools shell out to separate
CLI modules (iw3.rife_cli, subtitle_search_cli, ...), a structurally
different backend integration than create_parser()/iw3_main(), and are a
later, separate piece of work.

Two kinds of field deliberately have no 1:1 cli_arg (cli_arg=None) and are
special-cased in worker.py instead of forced into a fully generic mechanism:
- `stereo_format` -- iw3's output format is really several mutually-exclusive
  flags (--half-sbs/--tb/--half-tb/--vr180/--cross-eyed/--rgbd/--half-rgbd/
  --anaglyph), not one.
- `metadata` -- the real --metadata flag is an optional-value flag
  (nargs="?", const="filename"), not a plain boolean; modeled here as a
  checkbox that worker.py turns into "filename" or None.

Known gap, not an oversight: the wx GUI's Dual-Pass Depth Blend tab has
several sub-enhancement controls (Feather Blur, Bilateral Denoise, CLAHE
Contrast, Depth Scale Alignment, Edge Suppression) that are NOT registered
in create_parser() at all -- they're set directly onto the args.Namespace by
gui.py itself (iw3/utils.py reads them via getattr(args, "...", default),
tolerating their absence). Supporting them needs a second, non-argparse-
backed special-casing mechanism this pass doesn't build; the 5 core
Depth Blend fields that ARE real CLI args are covered here, the rest is
deferred.
"""
from dataclasses import dataclass, field as dataclass_field
from typing import Any, List, Optional

from .field_tooltips import FIELD_TOOLTIPS, FIELD_CHOICES


@dataclass
class Rule:
    """A visibility/enablement condition on another field's current value."""
    field: str
    op: str  # "eq" or "in"
    value: Any

    def to_dict(self):
        return {"field": self.field, "op": self.op, "value": self.value}


@dataclass
class Field:
    name: str
    label: str
    widget: str  # "combo_editable" | "select" | "checkbox"
    tab: str
    cli_arg: Optional[str] = None
    value_type: str = "str"  # "str" | "float" | "int" | "bool" | "int_list" | "str_list"
    choices: List[str] = dataclass_field(default_factory=list)
    default: Any = None
    tooltip: str = ""
    visible_if: Optional[Rule] = None
    enabled_if: Optional[Rule] = None
    # marks fields whose real choice list is only known at runtime (e.g. the
    # set of hwaccel devices ffmpeg actually supports on this machine) --
    # schema_export.py fills these in at export time instead of hardcoding
    # a list here that could go stale.
    dynamic_choices: Optional[str] = None

    def to_dict(self):
        return {
            "name": self.name,
            "label": self.label,
            "widget": self.widget,
            "tab": self.tab,
            # Sub-heading within the tab (e.g. "Post-Processing" inside
            # Processor) -- looked up from FIELD_GROUPS below rather than
            # stored per-Field, so grouping ~100 fields doesn't mean editing
            # ~100 individual Field(...) call sites. A field with no entry
            # falls back to the tab's own label (renderer.js's default),
            # rendering as today: one box per tab.
            "group": FIELD_GROUPS.get(self.name),
            "cli_arg": self.cli_arg,
            "value_type": self.value_type,
            # FIELD_CHOICES/FIELD_TOOLTIPS (field_tooltips.py, ADR-088) hold
            # the real suggested values / tooltip text extracted directly
            # from iw3/gui.py's own controls -- preferred over this Field's
            # own short, hand-written fallback when a real one exists.
            "choices": FIELD_CHOICES.get(self.name, self.choices),
            "default": self.default,
            "tooltip": FIELD_TOOLTIPS.get(self.name, self.tooltip),
            "visible_if": self.visible_if.to_dict() if self.visible_if else None,
            "enabled_if": self.enabled_if.to_dict() if self.enabled_if else None,
        }


# Sub-groups within a tab, matching the wx GUI's own visual clustering
# where one exists (Processor/Post-Processing is a real, separate StaticBox
# pair in gui.py -- not invented here) and reasonable semantic clusters
# elsewhere, following the same blank-line groupings visible in the wx
# GUI's own Single Page layout. A tab with no entries for its fields here
# renders as a single box (General, Dual-Pass Depth Blend, Video Decoding
# -- all small enough that sub-grouping would add clutter, not clarity).
FIELD_GROUPS = {
    # Stereo Generation
    "divergence": "Core", "method": "Core", "splat_blend_temperature": "Core",
    "synthetic_view": "Core", "convergence_mode": "Core", "convergence": "Core",
    "convergence_smoothing": "Core", "ipd_offset": "Core",
    "inpaint_model": "Inpainting", "inpaint_overlap_frames": "Inpainting",
    "mask_inner_dilation": "Inpainting", "mask_outer_dilation": "Inpainting",
    "inpaint_max_width": "Inpainting",
    "stereo_width": "Depth & Resolution", "depth_model": "Depth & Resolution",
    "resolution": "Depth & Resolution", "limit_resolution": "Depth & Resolution",
    "foreground_scale": "Edge & Detail", "edge_dilation": "Edge & Detail",
    "depth_aa": "Edge & Detail", "depth_refine": "Edge & Detail",
    "depth_refine_strength": "Edge & Detail", "temporal_stabilize": "Edge & Detail",
    "temporal_stabilize_strength": "Edge & Detail",
    "foreground_pop": "Pop & Divergence Tuning", "foreground_divergence": "Pop & Divergence Tuning",
    "background_pop": "Pop & Divergence Tuning", "background_pop_coverage": "Pop & Divergence Tuning",
    "background_divergence": "Pop & Divergence Tuning", "edge_repair_strength": "Pop & Divergence Tuning",
    "sharpen": "Pop & Divergence Tuning", "sharpen_strength": "Pop & Divergence Tuning",
    "ema_normalize": "Flicker Reduction & Scene Detection",
    "ema_decay": "Flicker Reduction & Scene Detection",
    "ema_buffer": "Flicker Reduction & Scene Detection",
    "ema_motion_adaptive": "Flicker Reduction & Scene Detection",
    "scene_batch_auto_ema": "Flicker Reduction & Scene Detection",
    "scene_batch_auto_ema_model": "Flicker Reduction & Scene Detection",
    "scene_detect": "Flicker Reduction & Scene Detection",
    "disable_scene_cache": "Flicker Reduction & Scene Detection",
    "preserve_screen_border": "Flicker Reduction & Scene Detection",
    "stereo_format": "Output", "anaglyph_method": "Output", "stereo_mode_tag": "Output",

    # Video Filter
    "start_time": "Trim & Filters", "end_time": "Trim & Filters", "vf": "Trim & Filters",
    "rotate_left": "Trim & Filters", "rotate_right": "Trim & Filters", "autocrop": "Trim & Filters",
    "pad": "Trim & Filters", "pad_mode": "Trim & Filters",
    "max_output_width": "Output Sizing", "max_output_height": "Output Sizing",
    "keep_aspect_ratio": "Output Sizing",
    "preserve_dowi": "HDR & Bit Depth", "hdr_to_sdr": "HDR & Bit Depth",
    "upgrade_pix_fmt": "HDR & Bit Depth",
    "auto_resume": "Resume & Quality", "resume_chunk_duration": "Resume & Quality",
    "denoise": "Resume & Quality", "preview": "Resume & Quality", "metadata": "Resume & Quality",
    "scene_batch": "Automated Scene Batch", "scene_batch_crop": "Automated Scene Batch",
    "scene_settings": "Automated Scene Batch", "scene_batch_variant": "Automated Scene Batch",

    # Video Encoding
    "video_format": "Format & Codec", "video_codec": "Format & Codec", "max_fps": "Format & Codec",
    "pix_fmt": "Format & Codec", "colorspace": "Format & Codec",
    "crf": "Rate Control", "video_bitrate": "Rate Control",
    "preset": "Tuning", "tune": "Tuning", "profile_level": "Tuning",

    # Processor -- the one real, confirmed wx-GUI precedent for sub-boxes:
    # grp_processor and grp_postprocess are two separate StaticBoxes there.
    "device": "Processor", "fp16": "Processor", "low_vram": "Processor",
    "batch_size": "Processor", "max_workers": "Processor", "tta": "Processor",
    "cuda_stream": "Processor", "pause_frees_vram": "Processor", "compile": "Processor",
    "waifu2x_upscale": "Post-Processing", "waifu2x_upscale_target": "Post-Processing",
    "rife_interpolate": "Post-Processing", "rife_model": "Post-Processing",
    "rife_multiplier": "Post-Processing", "rife_target_fps": "Post-Processing",
}

TABS = [
    ("general", "General"),
    ("stereo_generation", "Stereo Generation"),
    ("dual_pass_depth_blend", "Dual-Pass Depth Blend"),
    ("video_filter", "Video Filter"),
    ("video_decoding", "Video Decoding"),
    ("video_encoding", "Video Encoding"),
    ("processor", "Processor"),
]

# Real choice list for --method values that support inpaint sub-settings,
# copied from gui.py's own on_selected_index_changed_cbo_method dependency
# logic (the same "which methods enable which sub-controls" source used for
# the splat_blend_temperature enabled_if rule above).
INPAINT_METHODS = ["forward_inpaint", "mlbw_l2_inpaint", "monobw_inpaint"]

# Real choice list, copied verbatim from iw3.utils.create_parser()'s own
# --method registration (utils.py ~line 4861) -- kept here as a literal so
# the coverage self-test can diff it against the parser's real choices and
# catch drift if either side changes without the other.
METHOD_CHOICES = [
    "grid_sample", "backward",
    "monobw", "monobw_inpaint",
    "forward", "forward_fill", "forward_splat_fill", "forward_inpaint",
    "mlbw_l2", "mlbw_l4", "mlbw_l2s", "mlbw_l4s",
    "mask_mlbw_l2", "mlbw_l2_inpaint",
    "row_flow", "row_flow_sym",
    "row_flow_v3", "row_flow_v3_sym",
    "row_flow_v2",
    "NULL",
]

ANAGLYPH_METHOD_CHOICES = ["dubois", "dubois2", "color", "gray", "half-color", "wimmer", "wimmer2"]

STEREO_FORMAT_CHOICES = [
    "full_sbs", "half_sbs", "full_tb", "half_tb", "cross_eyed", "rgbd", "half_rgbd", "vr180", "anaglyph",
]

# Copied verbatim from create_parser()'s --depth-model choices (utils.py ~4934-4949).
DEPTH_MODEL_CHOICES = [
    "ZoeD_N", "ZoeD_K", "ZoeD_NK",
    "Any_S", "Any_B", "Any_L",
    "ZoeD_Any_N", "ZoeD_Any_K",
    "Any_V2_S", "Any_V2_B", "Any_V2_L",
    "Any_V2_N", "Any_V2_K",
    "Any_V2_N_S", "Any_V2_N_B", "Any_V2_N_L",
    "Any_V2_K_S", "Any_V2_K_B", "Any_V2_K_L",
    "Distill_Any_S", "Distill_Any_B", "Distill_Any_L",
    "Any_V3_Mono", "Any_V3_Mono_01",
    "DepthPro", "DepthPro_S",
    "VDA_S", "VDA_B", "VDA_L",
    "VDA_Metric", "VDA_Metric_S", "VDA_Metric_B", "VDA_Metric_L",
    "VDA_Stream_S", "VDA_Stream_B", "VDA_Stream_L",
    "VDA_Stream_Metric_S", "VDA_Stream_Metric_B", "VDA_Stream_Metric_L",
    "NULL",
]

FIELDS: List[Field] = [
    # ---------------- General ----------------
    # Batch/file options -- shown above the tabs entirely in the wx GUI
    # (pnl_file_option), given their own small tab here instead since this
    # renderer doesn't have a separate above-the-tabs region.
    Field(
        name="resume", cli_arg="--resume", label="Resume",
        widget="checkbox", tab="general", value_type="bool", default=True,
        tooltip="Skip processing when the output file already exists. "
                "Matches this GUI's own default (checked) rather than the "
                "bare CLI default (off), since that's what users of this "
                "tool actually expect on by default.",
    ),
    Field(
        name="recursive", cli_arg="--recursive", label="Process all subfolders",
        widget="checkbox", tab="general", value_type="bool", default=False,
        tooltip="When Input is a folder, also processes every subfolder "
                "inside it.",
    ),
    Field(
        name="skip_error", cli_arg="--skip-error", label="Skip Error",
        widget="checkbox", tab="general", value_type="bool", default=False,
        tooltip="Continue processing the rest of a batch even if a "
                "specific file fails.",
    ),
    Field(
        name="exif_transpose", cli_arg=None, label="EXIF Transpose",
        widget="checkbox", tab="general", value_type="bool", default=True,
        tooltip="Rotates an image according to its own EXIF orientation "
                "tag before processing. The real flag is "
                "--disable-exif-transpose (inverted) -- handled specially "
                "in worker.py, same pattern as Stereo Format.",
    ),
    Field(
        name="format", cli_arg="--format", label="Image Format",
        widget="select", tab="general", value_type="str",
        choices=["png", "webp", "jpeg"], default="png",
        tooltip="Output image format, for image (not video) conversions.",
    ),

    # ---------------- Stereo Generation ----------------
    Field(
        name="divergence", cli_arg="--divergence", label="3D Strength",
        widget="combo_editable", tab="stereo_generation", value_type="float",
        choices=["5.0", "4.0", "3.0", "2.5", "2.0", "1.0"], default=2.0,
        tooltip="Strength of the 3D effect. 0-2 is a reasonable range for most "
                "content; higher values push more depth but can increase "
                "artifacts near edges.",
    ),
    Field(
        name="method", cli_arg="--method", label="Method",
        widget="select", tab="stereo_generation", value_type="str",
        choices=METHOD_CHOICES, default="row_flow",
        tooltip="The stereo-generation algorithm used to synthesize the second "
                "eye's view from the depth map. row_flow_v3 is the modern "
                "default for most content.",
    ),
    Field(
        name="splat_blend_temperature", cli_arg="--splat-blend-temperature",
        label="Splat Blend Temperature", widget="combo_editable", tab="stereo_generation",
        value_type="float", default=50.0,
        enabled_if=Rule(field="method", op="eq", value="forward_splat_fill"),
        tooltip="Only used by Method=forward_splat_fill: how sharply the "
                "depth-weighted blend favors the nearer of two colliding "
                "pixels. Higher = sharper cutoff. Default (50.0) matches the "
                "method's original behavior.",
    ),
    Field(
        name="inpaint_model", cli_arg="--inpaint-model", label="Inpainting Model",
        widget="select", tab="stereo_generation", value_type="str",
        choices=[], dynamic_choices="inpaint_models", default=None,
        enabled_if=Rule(field="method", op="in", value=INPAINT_METHODS),
        tooltip="Which inpaint model fills in the disoccluded (revealed) "
                "regions. Only used by inpaint-family Methods.",
    ),
    Field(
        name="inpaint_overlap_frames", cli_arg="--inpaint-overlap-frames",
        label="Inpaint Overlap Frames", widget="combo_editable", tab="stereo_generation",
        value_type="int_list", default=None,
        enabled_if=Rule(field="method", op="in", value=INPAINT_METHODS),
        tooltip="Overlap/padding frames for the video inpaint model, as "
                "\"<frames>\" or \"<pre frames> <post frames>\".",
    ),
    Field(
        name="mask_inner_dilation", cli_arg="--mask-inner-dilation",
        label="Inpaint Mask Inner Dilation", widget="combo_editable", tab="stereo_generation",
        value_type="int", default=0,
        enabled_if=Rule(field="method", op="in", value=INPAINT_METHODS),
        tooltip="Loop count of inner mask dilation for inpaint methods.",
    ),
    Field(
        name="mask_outer_dilation", cli_arg="--mask-outer-dilation",
        label="Inpaint Mask Outer Dilation", widget="combo_editable", tab="stereo_generation",
        value_type="int", default=0,
        enabled_if=Rule(field="method", op="in", value=INPAINT_METHODS),
        tooltip="Loop count of outer mask dilation for inpaint methods.",
    ),
    Field(
        name="inpaint_max_width", cli_arg="--inpaint-max-width", label="Inpaint Max Width",
        widget="combo_editable", tab="stereo_generation", value_type="int", default=None,
        enabled_if=Rule(field="method", op="in", value=INPAINT_METHODS),
        tooltip="Max width of the inpaint result. Leave blank for no limit.",
    ),
    Field(
        name="synthetic_view", cli_arg="--synthetic-view", label="Synthetic View",
        widget="select", tab="stereo_generation", value_type="str",
        choices=["both", "right", "left"], default="both",
        tooltip="Which side generates the synthesized view. 'both' warps both "
                "eyes from the source; 'right'/'left' keeps the original image "
                "as one eye and only synthesizes the other.",
    ),
    Field(
        name="convergence_mode", cli_arg="--convergence-mode", label="Convergence Mode",
        widget="select", tab="stereo_generation", value_type="str",
        choices=["constant", "sod_v1", "face_detect"], default="constant",
        tooltip="How the convergence plane (screen-distance point) is chosen. "
                "'constant' uses the Convergence Plane value directly; "
                "sod_v1/face_detect track it automatically per frame.",
    ),
    Field(
        name="convergence", cli_arg="--convergence", label="Convergence Plane",
        widget="combo_editable", tab="stereo_generation", value_type="float",
        default=0.5,
        tooltip="Normalized depth position (0-1) that appears at screen "
                "distance, with nothing feeling like it's popping out or "
                "sinking in at that point.",
    ),
    Field(
        name="convergence_smoothing", cli_arg="--convergence-smoothing",
        label="Convergence Smoothing", widget="combo_editable", tab="stereo_generation",
        value_type="float", default=0.9,
        enabled_if=Rule(field="convergence_mode", op="in", value=["sod_v1", "face_detect"]),
        tooltip="EMA decay for the auto convergence modes (sod_v1/face_detect). "
                "Higher = smoother but slower to react; 0 = no smoothing.",
    ),
    Field(
        name="ipd_offset", cli_arg="--ipd-offset", label="Your Own Size",
        widget="combo_editable", tab="stereo_generation", value_type="float", default=0,
        tooltip="IPD offset as a width-scale percent. 0-10 is a reasonable "
                "value for Full SBS output.",
    ),
    Field(
        name="stereo_width", cli_arg="--stereo-width", label="Stereo Processing Width",
        widget="combo_editable", tab="stereo_generation", value_type="int", default=None,
        tooltip="Input width for row_flow_v3/row_flow_v2. Leave blank for the "
                "model's default.",
    ),
    Field(
        name="depth_model", cli_arg="--depth-model", label="Depth Model",
        widget="select", tab="stereo_generation", value_type="str",
        choices=DEPTH_MODEL_CHOICES, default="ZoeD_Any_N",
        tooltip="Which neural network estimates depth from the source image. "
                "Different models trade off speed, video-temporal stability, "
                "and fine detail.",
    ),
    Field(
        name="resolution", cli_arg="--resolution", label="Depth Resolution",
        widget="combo_editable", tab="stereo_generation", value_type="int", default=None,
        tooltip="Input resolution (short side) fed to the depth model. Leave "
                "blank for the model's default.",
    ),
    Field(
        name="limit_resolution", cli_arg="--limit-resolution", label="Limit to source",
        widget="checkbox", tab="stereo_generation", value_type="bool", default=False,
        tooltip="If the source resolution is lower than Depth Resolution, cap "
                "the depth resolution at the source's own resolution instead "
                "of upscaling first.",
    ),
    Field(
        name="foreground_scale", cli_arg="--foreground-scale", label="Foreground Scale",
        widget="combo_editable", tab="stereo_generation", value_type="float", default=0,
        tooltip="Foreground scaling level, -3.0 to 3.0. 0 is disabled.",
    ),
    Field(
        name="edge_dilation", cli_arg="--edge-dilation", label="Edge Fix",
        widget="combo_editable", tab="stereo_generation", value_type="int_list", default="2 1",
        tooltip="Loop count of edge dilation, as \"<x> <y>\" or a single "
                "shared value. Space-separated.",
    ),
    Field(
        name="depth_aa", cli_arg="--depth-aa", label="Depth Anti-aliasing",
        widget="checkbox", tab="stereo_generation", value_type="bool", default=False,
        tooltip="Applies depth antialiasing. Ignored for models that don't "
                "support it.",
    ),
    Field(
        name="depth_refine", cli_arg="--depth-refine", label="Depth Detail Refinement",
        widget="checkbox", tab="stereo_generation", value_type="bool", default=False,
        tooltip="Cleans up each depth frame's own internal noise with "
                "edge-preserving smoothing, within that single frame (a "
                "different axis from EMA, which smooths ACROSS frames).",
    ),
    Field(
        name="depth_refine_strength", cli_arg="--depth-refine-strength",
        label="Depth Refine Strength", widget="combo_editable", tab="stereo_generation",
        value_type="float", default=1.0,
        enabled_if=Rule(field="depth_refine", op="eq", value=True),
        tooltip="How strong the Depth Detail Refinement pass is. 1.0 matches "
                "the feature's original fixed behavior.",
    ),
    Field(
        name="temporal_stabilize", cli_arg="--temporal-stabilize",
        label="Object Stability", widget="checkbox", tab="stereo_generation",
        value_type="bool", default=False,
        tooltip="Approximates a video-aware model's frame-to-frame stability "
                "for a single-frame model, using optical flow to reduce "
                "object depth flickering EMA alone can't fix.",
    ),
    Field(
        name="temporal_stabilize_strength", cli_arg="--temporal-stabilize-strength",
        label="Object Stability Strength", widget="combo_editable", tab="stereo_generation",
        value_type="float", default=0.7,
        enabled_if=Rule(field="temporal_stabilize", op="eq", value=True),
        tooltip="How strongly to trust the motion-warped previous frame vs. "
                "the fresh per-frame depth (0-1). Tapers down automatically "
                "during fast motion.",
    ),
    Field(
        name="foreground_pop", cli_arg="--foreground-pop", label="Foreground Pop",
        widget="combo_editable", tab="stereo_generation", value_type="float", default=0.0,
        tooltip="Pushes the nearest pixels even further toward the audience "
                "(0.0=off, 0.5=medium, 1.0=strong).",
    ),
    Field(
        name="foreground_divergence", cli_arg="--foreground-divergence",
        label="Foreground Divergence", widget="combo_editable", tab="stereo_generation",
        value_type="float", default=None,
        tooltip="A separate 3D Strength for the nearest ~15% of pixels only. "
                "Leave blank to use the same value as the rest of the scene.",
    ),
    Field(
        name="background_pop", cli_arg="--background-pop", label="Background Pop",
        widget="combo_editable", tab="stereo_generation", value_type="float", default=0.0,
        tooltip="Pushes the farthest pixels even further from the audience "
                "(0.0=off, 0.5=medium, 1.0=strong).",
    ),
    Field(
        name="background_pop_coverage", cli_arg="--background-pop-coverage",
        label="Background Pop Coverage %", widget="combo_editable", tab="stereo_generation",
        value_type="float", default=0.15,
        tooltip="How much of the scene Background Pop treats as \"background\" "
                "(0.0-1.0). Default 0.15 = farthest 15%.",
    ),
    Field(
        name="background_divergence", cli_arg="--background-divergence",
        label="Background Divergence", widget="combo_editable", tab="stereo_generation",
        value_type="float", default=None,
        tooltip="A separate 3D Strength for the farthest ~15% of pixels only. "
                "Leave blank to use the same value as the rest of the scene.",
    ),
    Field(
        name="edge_repair_strength", cli_arg="--edge-repair-strength", label="Edge Repair",
        widget="combo_editable", tab="stereo_generation", value_type="float", default=0.0,
        tooltip="Final cleanup pass smoothing a thin band right around real "
                "depth edges to reduce hairline fringing. 0.0=off.",
    ),
    Field(
        name="sharpen", cli_arg="--sharpen", label="Sharpen",
        widget="checkbox", tab="stereo_generation", value_type="bool", default=False,
        tooltip="Final edge-aware unsharp-mask pass on the rendered stereo "
                "output. Off by default.",
    ),
    Field(
        name="sharpen_strength", cli_arg="--sharpen-strength", label="Sharpen Strength",
        widget="combo_editable", tab="stereo_generation", value_type="float", default=0.5,
        enabled_if=Rule(field="sharpen", op="eq", value=True),
        tooltip="How strong the Sharpen pass is (0.0=no effect, 1.0=strongest).",
    ),
    Field(
        name="ema_normalize", cli_arg="--ema-normalize", label="Flicker Reduction",
        widget="checkbox", tab="stereo_generation", value_type="bool", default=False,
        tooltip="Uses a moving min/max average to normalize video depth over "
                "time, reducing frame-to-frame flicker.",
    ),
    Field(
        name="ema_decay", cli_arg="--ema-decay", label="EMA Decay",
        widget="combo_editable", tab="stereo_generation", value_type="float", default=0.75,
        enabled_if=Rule(field="ema_normalize", op="eq", value=True),
        tooltip="Smoothing strength for Flicker Reduction (0-1). Larger = "
                "smoother.",
    ),
    Field(
        name="ema_buffer", cli_arg="--ema-buffer", label="EMA Buffer",
        widget="combo_editable", tab="stereo_generation", value_type="int", default=30,
        enabled_if=Rule(field="ema_normalize", op="eq", value=True),
        tooltip="Frame buffer size for Flicker Reduction's moving average.",
    ),
    Field(
        name="ema_motion_adaptive", cli_arg="--ema-motion-adaptive",
        label="Motion-Adaptive Smoothing", widget="checkbox", tab="stereo_generation",
        value_type="bool", default=False,
        enabled_if=Rule(field="ema_normalize", op="eq", value=True),
        tooltip="Automatically eases EMA Decay off during fast motion instead "
                "of one fixed strength for the whole clip. Never smooths more "
                "than EMA Decay itself, only less.",
    ),
    Field(
        name="scene_detect", cli_arg="--scene-detect", label="Scene Boundary Detection",
        widget="checkbox", tab="stereo_generation", value_type="bool", default=False,
        tooltip="Splits processing at real scene cuts (shot boundary "
                "detection); EMA and other running state resets at each cut.",
    ),
    Field(
        name="disable_scene_cache", cli_arg="--disable-scene-cache",
        label="Disable Scene Boundary Cache", widget="checkbox", tab="stereo_generation",
        value_type="bool", default=False,
        enabled_if=Rule(field="scene_detect", op="eq", value=True),
        tooltip="Disables the cache Scene Boundary Detection normally reuses "
                "on a re-run of the same file.",
    ),
    Field(
        name="scene_batch_auto_ema", cli_arg="--scene-batch-auto-ema",
        label="Auto EMA by Scene Length", widget="checkbox", tab="stereo_generation",
        value_type="bool", default=False,
        tooltip="Automatically picks EMA Decay/Buffer per scene based on that "
                "scene's own length, using a built-in table. Requires Scene "
                "Boundary Detection (or Automated Scene Batch, on the Video "
                "Filter tab).",
    ),
    Field(
        name="scene_batch_auto_ema_model", cli_arg="--scene-batch-auto-ema-model",
        label="Auto EMA Model", widget="select", tab="stereo_generation",
        value_type="str", choices=[], dynamic_choices="scene_batch_ema_models",
        default="3DECKER VDA_L",
        enabled_if=Rule(field="scene_batch_auto_ema", op="eq", value=True),
        tooltip="Which built-in EMA-by-duration table to use, matched to the "
                "Depth Model in use.",
    ),
    Field(
        name="preserve_screen_border", cli_arg="--preserve-screen-border",
        label="Preserve Screen Border", widget="checkbox", tab="stereo_generation",
        value_type="bool", default=False,
        tooltip="Forces screen-border parallax to zero.",
    ),
    Field(
        name="stereo_format", cli_arg=None, label="Stereo Format",
        widget="select", tab="stereo_generation", value_type="str",
        choices=STEREO_FORMAT_CHOICES, default="half_sbs",
        tooltip="The output layout. Maps to several mutually-exclusive iw3 CLI "
                "flags under the hood (--half-sbs/--tb/--half-tb/--vr180/"
                "--cross-eyed/--rgbd/--half-rgbd/--anaglyph) -- handled "
                "specially in worker.py, not a direct 1:1 CLI arg.",
    ),
    Field(
        name="anaglyph_method", cli_arg=None, label="Anaglyph Method",
        widget="select", tab="stereo_generation", value_type="str",
        choices=ANAGLYPH_METHOD_CHOICES, default="dubois",
        visible_if=Rule(field="stereo_format", op="eq", value="anaglyph"),
        tooltip="Which anaglyph color-filter recipe to use. Only meaningful "
                "when Stereo Format is Anaglyph.",
    ),
    Field(
        name="stereo_mode_tag", cli_arg="--stereo-mode-tag", label="Tag MKV as 3D",
        widget="checkbox", tab="stereo_generation", value_type="bool", default=False,
        tooltip="Tags the finished .mkv's video track with the Matroska "
                "StereoMode property so 3D-aware players/TVs auto-detect the "
                "3D packing instead of the viewer picking it manually.",
    ),

    # ---------------- Dual-Pass Depth Blend ----------------
    # Core CLI-backed fields only -- see this file's own docstring for the
    # sub-enhancement controls (Feather Blur, Bilateral Denoise, CLAHE,
    # Edge Suppression) deliberately deferred to a later pass.
    Field(
        name="depth_blend", cli_arg="--depth-blend", label="Dual-Pass Depth Blend",
        widget="checkbox", tab="dual_pass_depth_blend", value_type="bool", default=False,
        tooltip="Blends depth from a second model into the primary Depth "
                "Model's output, favoring the second model in areas with "
                "dense fine detail (foliage, hair, close-up texture). Runs "
                "as three full passes over the clip; costs real extra time "
                "and disk space. Requires a single video file input.",
    ),
    Field(
        name="depth_blend_model", cli_arg="--depth-blend-model", label="Secondary Depth Model",
        widget="combo_editable", tab="dual_pass_depth_blend", value_type="str", default="VDA_L",
        enabled_if=Rule(field="depth_blend", op="eq", value=True),
        tooltip="The secondary depth model blended in by Dual-Pass Depth "
                "Blend.",
    ),
    Field(
        name="depth_blend_strength", cli_arg="--depth-blend-strength",
        label="Blend Strength", widget="combo_editable", tab="dual_pass_depth_blend",
        value_type="float", default=1.0,
        enabled_if=Rule(field="depth_blend", op="eq", value=True),
        tooltip="How strongly to favor the secondary model in the selected "
                "region (0-1). 1.0 = fully trust it there.",
    ),
    Field(
        name="depth_blend_region", cli_arg="--depth-blend-region", label="Blend Region",
        widget="select", tab="dual_pass_depth_blend", value_type="str",
        choices=["detail", "foreground", "background"], default="detail",
        enabled_if=Rule(field="depth_blend", op="eq", value=True),
        tooltip="What decides WHERE the secondary model gets blended in: "
                "'detail' (dense fine visual detail), 'foreground' (nearest "
                "X% of the scene), or 'background' (farthest X%).",
    ),
    Field(
        name="depth_blend_region_percent", cli_arg="--depth-blend-region-percent",
        label="Blend Region %", widget="combo_editable", tab="dual_pass_depth_blend",
        value_type="float", default=25.0,
        enabled_if=Rule(field="depth_blend_region", op="in", value=["foreground", "background"]),
        tooltip="For Blend Region foreground/background: what percent of the "
                "scene (by depth) to blend the secondary model into.",
    ),

    # ---------------- Video Filter ----------------
    Field(
        name="start_time", cli_arg="--start-time", label="Start Time",
        widget="combo_editable", tab="video_filter", value_type="str", default=None,
        tooltip="Start time offset for video, hh:mm:ss or mm:ss format.",
    ),
    Field(
        name="end_time", cli_arg="--end-time", label="End Time",
        widget="combo_editable", tab="video_filter", value_type="str", default=None,
        tooltip="End time offset for video, hh:mm:ss or mm:ss format.",
    ),
    Field(
        name="vf", cli_arg="--vf", label="-vf (raw ffmpeg filter)",
        widget="combo_editable", tab="video_filter", value_type="str", default="",
        tooltip="Raw ffmpeg -vf filter string, applied as-is. Advanced use.",
    ),
    Field(
        name="rotate_left", cli_arg="--rotate-left", label="Rotate Left",
        widget="checkbox", tab="video_filter", value_type="bool", default=False,
        tooltip="Rotate 90 degrees counterclockwise.",
    ),
    Field(
        name="rotate_right", cli_arg="--rotate-right", label="Rotate Right",
        widget="checkbox", tab="video_filter", value_type="bool", default=False,
        tooltip="Rotate 90 degrees clockwise.",
    ),
    Field(
        name="autocrop", cli_arg="--autocrop", label="AutoCrop",
        widget="select", tab="video_filter", value_type="str",
        choices=["", "BLACK_TB", "BLACK", "FLAT_TB", "FLAT"], default="",
        tooltip="Automatically removes black/flat-color bars. BLACK_TB/"
                "FLAT_TB: top and bottom only. BLACK/FLAT: all sides. Blank "
                "= off.",
    ),
    Field(
        name="pad", cli_arg="--pad", label="Padding",
        widget="combo_editable", tab="video_filter", value_type="float", default=None,
        tooltip="Pad size = round(width * this) // 2. Leave blank for no "
                "padding.",
    ),
    Field(
        name="pad_mode", cli_arg="--pad-mode", label="Padding Mode",
        widget="select", tab="video_filter", value_type="str",
        choices=["tblr", "tb", "lr", "16:9", "top"], default="tblr",
        enabled_if=Rule(field="pad", op="ne", value=None),
        tooltip="Which sides Padding is applied to.",
    ),
    Field(
        name="max_output_width", cli_arg="--max-output-width", label="Max Output Width",
        widget="combo_editable", tab="video_filter", value_type="int", default=None,
        tooltip="Limits output width, e.g. for cardboard players. Leave blank "
                "for no limit.",
    ),
    Field(
        name="max_output_height", cli_arg="--max-output-height", label="Max Output Height",
        widget="combo_editable", tab="video_filter", value_type="int", default=None,
        tooltip="Limits output height, e.g. for cardboard players. Leave "
                "blank for no limit.",
    ),
    Field(
        name="keep_aspect_ratio", cli_arg="--keep-aspect-ratio", label="Keep Aspect Ratio",
        widget="checkbox", tab="video_filter", value_type="bool", default=False,
        enabled_if=Rule(field="max_output_width", op="ne", value=None),
        tooltip="Keeps the source aspect ratio when Max Output Width/Height "
                "resize the output.",
    ),
    Field(
        name="preserve_dowi", cli_arg="--preserve-dowi", label="Preserve Dolby Vision",
        widget="checkbox", tab="video_filter", value_type="bool", default=False,
        tooltip="Preserves Dolby Vision RPU metadata in HEVC output (requires "
                "dovi_tool, bundled with this app).",
    ),
    Field(
        name="hdr_to_sdr", cli_arg="--hdr-to-sdr", label="HDR to SDR",
        widget="checkbox", tab="video_filter", value_type="bool", default=False,
        tooltip="Tone-maps a PQ/HLG HDR source down to SDR (10-bit retained) "
                "before conversion.",
    ),
    Field(
        name="auto_resume", cli_arg="--auto-resume", label="Auto Resume",
        widget="checkbox", tab="video_filter", value_type="bool", default=False,
        tooltip="Splits video into chunks and resumes from a checkpoint if "
                "the conversion is interrupted.",
    ),
    Field(
        name="resume_chunk_duration", cli_arg="--resume-chunk-duration",
        label="Resume Chunk Duration (sec)", widget="combo_editable", tab="video_filter",
        value_type="int", default=300,
        enabled_if=Rule(field="auto_resume", op="eq", value=True),
        tooltip="Chunk duration in seconds for Auto Resume.",
    ),
    Field(
        name="denoise", cli_arg="--denoise", label="Denoise",
        widget="checkbox", tab="video_filter", value_type="bool", default=False,
        tooltip="Applies temporal denoising (hqdn3d, a one-time ffmpeg "
                "pre-pass) before depth estimation, to reduce film grain "
                "artifacts.",
    ),
    Field(
        name="preview", cli_arg="--preview", label="Preview Mode",
        widget="checkbox", tab="video_filter", value_type="bool", default=False,
        tooltip="Generates a quick 1fps/256p preview of the first 60 seconds "
                "to check 3D settings before a full run.",
    ),
    Field(
        name="upgrade_pix_fmt", cli_arg="--upgrade-pix-fmt", label="Bit Depth Upgrade",
        widget="select", tab="video_filter", value_type="int",
        choices=["", "10", "12"], default=None,
        tooltip="Upgrades an 8-bit source to 10-bit or 12-bit output pixel "
                "format. Blank = no upgrade.",
    ),
    Field(
        name="metadata", cli_arg=None, label="Add Metadata to Filename",
        widget="checkbox", tab="video_filter", value_type="bool", default=False,
        tooltip="Adds a metadata suffix to the output filename. The real "
                "--metadata flag takes an optional value (\"filename\"); this "
                "checkbox is the on/off case, handled specially in worker.py.",
    ),
    Field(
        name="scene_batch", cli_arg="--scene-batch", label="Automated Scene Batch",
        widget="checkbox", tab="video_filter", value_type="bool", default=False,
        tooltip="Fully automated whole-movie pipeline: removes letterbox bars "
                "once, detects scene cuts once, converts each scene "
                "independently, then joins the results back into one "
                "seamless video. Input must be a single video file.",
    ),
    Field(
        name="scene_batch_crop", cli_arg="--scene-batch-crop", label="Scene Batch Crop",
        widget="combo_editable", tab="video_filter", value_type="str", default=None,
        enabled_if=Rule(field="scene_batch", op="eq", value=True),
        tooltip="Explicit crop for Automated Scene Batch, as WxH or "
                "WxH:X:Y. If blank, letterbox bars are auto-detected once "
                "from the whole movie.",
    ),
    Field(
        name="scene_settings", cli_arg="--scene-settings", label="Scene Settings File",
        widget="combo_editable", tab="video_filter", value_type="str", default=None,
        enabled_if=Rule(field="scene_batch", op="eq", value=True),
        tooltip="JSON file giving per-scene setting overrides for Automated "
                "Scene Batch.",
    ),
    Field(
        name="scene_batch_variant", cli_arg="--scene-batch-variant", label="Scene Batch Variant",
        widget="combo_editable", tab="video_filter", value_type="str", default=None,
        enabled_if=Rule(field="scene_batch", op="eq", value=True),
        tooltip="Optional name for this Automated Scene Batch run -- reuses "
                "already-done shared work from a prior run of the same "
                "movie, but writes its own output as <name>_<variant>.<ext> "
                "so trying different settings never overwrites an earlier "
                "attempt.",
    ),

    # ---------------- Video Decoding ----------------
    Field(
        name="hwaccel", cli_arg="--hwaccel", label="HWAccel",
        widget="select", tab="video_decoding", value_type="str",
        choices=[], dynamic_choices="hw_devices", default=None,
        tooltip="Hardware-accelerated video decode.",
    ),
    Field(
        name="disable_software_fallback", cli_arg="--disable-software-fallback",
        label="Disable Software Fallback", widget="checkbox", tab="video_decoding",
        value_type="bool", default=False,
        tooltip="Disables falling back to software (CPU) decoding if "
                "hardware decode fails partway through.",
    ),

    # ---------------- Video Encoding ----------------
    Field(
        name="video_format", cli_arg="--video-format", label="Video Format",
        widget="select", tab="video_encoding", value_type="str",
        choices=["mp4", "mkv", "avi"], default="mp4",
        tooltip="Output video container format.",
    ),
    Field(
        name="video_codec", cli_arg="--video-codec", label="Video Codec",
        widget="combo_editable", tab="video_encoding", value_type="str", default=None,
        choices=["libx264", "libx265", "libopenh264", "utvideo", "ffv1",
                 "h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv"],
        tooltip="Video codec to encode with. Availability depends on Video "
                "Format and this machine's hardware encoders.",
    ),
    Field(
        name="max_fps", cli_arg="--max-fps", label="Max FPS",
        widget="combo_editable", tab="video_encoding", value_type="float", default=30,
        tooltip="Output fps = min(source fps, this value).",
    ),
    Field(
        name="pix_fmt", cli_arg="--pix-fmt", label="Pixel Format",
        widget="select", tab="video_encoding", value_type="str",
        choices=["yuv420p", "yuv444p", "yuv420p10le", "rgb24", "gbrp", "gbrp10le", "gbrp16le"],
        default="yuv420p",
        tooltip="Pixel format for video output.",
    ),
    Field(
        name="colorspace", cli_arg="--colorspace", label="Colorspace",
        widget="select", tab="video_encoding", value_type="str",
        choices=["unspecified", "auto", "bt709", "bt709-pc", "bt709-tv",
                 "bt601", "bt601-pc", "bt601-tv", "bt2020-tv", "bt2020-pq-tv"],
        default="auto",
        tooltip="Video colorspace tag written to the output.",
    ),
    Field(
        name="crf", cli_arg="--crf", label="CRF",
        widget="combo_editable", tab="video_encoding", value_type="int", default=20,
        tooltip="Constant quality value. Smaller = higher quality/larger "
                "file.",
    ),
    Field(
        name="video_bitrate", cli_arg="--video-bitrate", label="Bitrate",
        widget="combo_editable", tab="video_encoding", value_type="str", default="8M",
        enabled_if=Rule(field="video_codec", op="eq", value="libopenh264"),
        tooltip="Bitrate option, only used by the libopenh264 codec.",
    ),
    Field(
        name="preset", cli_arg="--preset", label="Preset",
        widget="select", tab="video_encoding", value_type="str",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                 "medium", "slow", "slower", "veryslow", "placebo",
                 "p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        default="medium",
        tooltip="Encoder speed/quality preset.",
    ),
    Field(
        name="tune", cli_arg="--tune", label="Tune",
        widget="combo_editable", tab="video_encoding", value_type="str_list", default="",
        choices=["film", "animation", "grain", "stillimage", "psnr",
                 "fastdecode", "zerolatency", "hq", "uhq", "ll", "ull", "lossless"],
        tooltip="Encoder tuning(s), space or comma separated. Which values "
                "are valid depends on the chosen codec (this simplified "
                "picker does not filter the list by codec the way the "
                "desktop GUI does).",
    ),
    Field(
        name="profile_level", cli_arg="--profile-level", label="Level",
        widget="combo_editable", tab="video_encoding", value_type="str", default=None,
        tooltip="H.264 profile level, e.g. 4.1. Advanced use.",
    ),

    # ---------------- Processor ----------------
    Field(
        name="device", cli_arg="--gpu", label="Device",
        widget="select", tab="processor", value_type="gpu_id",
        choices=[], dynamic_choices="gpu_devices", default=None,
        tooltip="Which GPU (or CPU) does the work. Populated via nvidia-smi "
                "as a separate process, never torch.cuda.*, so opening this "
                "GUI can never claim a CUDA context before it's needed "
                "(see ADR-034/071/075).",
    ),
    Field(
        name="fp16", cli_arg=None, label="FP16",
        widget="checkbox", tab="processor", value_type="bool", default=True,
        tooltip="Uses half-precision math for faster/lower-VRAM inference. "
                "The real flag is --disable-amp (inverted) -- handled "
                "specially in worker.py, same pattern as Stereo Format.",
    ),
    Field(
        name="low_vram", cli_arg="--low-vram", label="Low VRAM",
        widget="checkbox", tab="processor", value_type="bool", default=False,
        tooltip="Disables batch processing so each frame uses less GPU memory "
                "at a time. Slower, but avoids out-of-memory crashes/freezes "
                "on smaller GPUs or VRAM-heavy methods like forward_splat_fill.",
    ),
    Field(
        name="compile", cli_arg="--compile", label="torch.compile",
        widget="checkbox", tab="processor", value_type="bool", default=False,
        tooltip="Compiles the depth/stereo models for faster inference. First "
                "run after a settings change is slower (compile time); needs "
                "the CUDA-context ordering this app already preserves "
                "(ADR-034/071/075) to combine safely with hardware decode.",
    ),
    Field(
        name="batch_size", cli_arg="--batch-size", label="Depth Batch Size",
        widget="combo_editable", tab="processor", value_type="int", default=2,
        enabled_if=Rule(field="low_vram", op="eq", value=False),
        tooltip="How many frames are processed together per depth-model pass. "
                "Ignored when Low VRAM is on.",
    ),
    Field(
        name="max_workers", cli_arg="--max-workers", label="Worker Threads",
        widget="select", tab="processor", value_type="int",
        choices=["0", "1", "2", "3", "4", "8", "16"], default=0,
        tooltip="Max inference worker threads for video processing. 0 = "
                "disabled.",
    ),
    Field(
        name="tta", cli_arg="--tta", label="TTA",
        widget="checkbox", tab="processor", value_type="bool", default=False,
        tooltip="Uses flip augmentation on the depth model for slightly "
                "higher quality, at extra compute cost.",
    ),
    Field(
        name="cuda_stream", cli_arg="--cuda-stream", label="Stream",
        widget="checkbox", tab="processor", value_type="bool", default=False,
        tooltip="Uses a separate CUDA stream per worker thread/device.",
    ),
    Field(
        name="pause_frees_vram", cli_arg="--pause-frees-vram", label="Free GPU memory while paused",
        widget="checkbox", tab="processor", value_type="bool", default=False,
        tooltip="When Suspend/Resume is used, also moves every loaded model "
                "off the GPU and frees VRAM while paused, reloading on "
                "Resume. Off by default (models stay resident for an instant "
                "Resume). Has no effect with multiple GPUs selected.",
    ),
    Field(
        name="waifu2x_upscale", cli_arg="--waifu2x-upscale", label="Upscale with waifu2x after conversion",
        widget="checkbox", tab="processor", value_type="bool", default=False,
        tooltip="After conversion finishes, runs the output through waifu2x "
                "(a separate bundled AI upscaler) as one extra step, written "
                "to a separate '<name>_w2x<ext>' file.",
    ),
    Field(
        name="waifu2x_upscale_target", cli_arg="--waifu2x-upscale-target",
        label="Upscale Target", widget="select", tab="processor",
        value_type="str", choices=["auto", "4k", "8k"], default="auto",
        enabled_if=Rule(field="waifu2x_upscale", op="eq", value=True),
        tooltip="'auto' uses waifu2x-upscale's plain whole-frame behavior. "
                "'4k'/'8k' use a stereo-aware path that upscales each eye of "
                "a packed 3D output independently.",
    ),
    Field(
        name="rife_interpolate", cli_arg="--rife-interpolate",
        label="Interpolate frames with RIFE after conversion",
        widget="checkbox", tab="processor", value_type="bool", default=False,
        tooltip="After conversion finishes, runs the finished packed stereo "
                "output through RIFE frame interpolation, raising the "
                "effective frame rate. Written to a separate '<name>_rife<ext>' "
                "file. Cannot be combined with Preserve Dolby Vision.",
    ),
    Field(
        name="rife_model", cli_arg="--rife-model", label="RIFE Model",
        widget="select", tab="processor", value_type="str",
        choices=["rife_425", "rife_425_lite"], default="rife_425",
        enabled_if=Rule(field="rife_interpolate", op="eq", value=True),
        tooltip="rife_425 is the recommended full-quality model; "
                "rife_425_lite is a lower-compute-cost variant.",
    ),
    Field(
        name="rife_multiplier", cli_arg="--rife-multiplier", label="RIFE Multiplier",
        widget="select", tab="processor", value_type="int",
        choices=["", "2", "3", "4"], default=None,
        enabled_if=Rule(field="rife_interpolate", op="eq", value=True),
        tooltip="Frame-rate multiplier: N=2 doubles the frame rate, N=3/4 "
                "triple/quadruple it. Mutually exclusive with RIFE Target "
                "FPS. Defaults to 2 if both are left blank.",
    ),
    Field(
        name="rife_target_fps", cli_arg="--rife-target-fps", label="RIFE Target FPS",
        widget="combo_editable", tab="processor", value_type="float", default=None,
        enabled_if=Rule(field="rife_interpolate", op="eq", value=True),
        tooltip="Interpolates to this exact frame rate instead of a simple "
                "multiplier. Must be higher than the source's own frame "
                "rate. Mutually exclusive with RIFE Multiplier.",
    ),
]
