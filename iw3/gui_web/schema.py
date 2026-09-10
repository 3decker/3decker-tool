"""Phase 1 settings schema for the web-rendered iw3 GUI (ADR-081).

This is the single source of truth the JS renderer builds its controls from
(see schema_export.py) -- NOT a hand-authored HTML form. Each Field's
`cli_arg` must exactly match a real argument registered in
iw3.utils.create_parser(); schema_export.py's self-test walks the parser's
own registered actions and asserts every field with a non-None cli_arg here
actually exists there, so this can never silently drift from the real CLI
the way a hand-maintained control list could.

Phase 1 intentionally covers only a narrow slice (Stereo Generation's core
fields + Processor's Low VRAM/torch.compile + the HWAccel field needed to
exercise the ADR-034/071/075 CUDA-context-ordering regression test) -- see
docs/ai/AI_DECISIONS.md ADR-081 and the approved plan this was built from.
Remaining fields are deliberately out of scope for this slice, not missing.

One real field, `stereo_format`, does NOT map 1:1 to a single CLI arg --
iw3's output format is really several mutually-exclusive boolean/optional
flags (--half-sbs, --tb, --half-tb, --vr180, --cross-eyed, --rgbd,
--half-rgbd, --anaglyph). Its `cli_arg` is left None and worker.py special-
cases it (see _apply_stereo_format()) rather than forcing a fully generic
multi-flag-mapping mechanism into the schema for the one field that needs
it today.
"""
from dataclasses import dataclass, field as dataclass_field
from typing import Any, List, Optional


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
    value_type: str = "str"  # "str" | "float" | "bool"
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
            "cli_arg": self.cli_arg,
            "value_type": self.value_type,
            "choices": self.choices,
            "default": self.default,
            "tooltip": self.tooltip,
            "visible_if": self.visible_if.to_dict() if self.visible_if else None,
            "enabled_if": self.enabled_if.to_dict() if self.enabled_if else None,
        }


TABS = [
    ("stereo_generation", "Stereo Generation"),
    ("processor", "Processor"),
]

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
    "half_sbs", "full_tb", "half_tb", "cross_eyed", "rgbd", "half_rgbd", "vr180", "anaglyph",
]

FIELDS: List[Field] = [
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
        name="convergence", cli_arg="--convergence", label="Convergence Plane",
        widget="combo_editable", tab="stereo_generation", value_type="float",
        default=0.5,
        tooltip="Normalized depth position (0-1) that appears at screen "
                "distance, with nothing feeling like it's popping out or "
                "sinking in at that point.",
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
        name="hwaccel", cli_arg="--hwaccel", label="HWAccel",
        widget="select", tab="processor", value_type="str",
        choices=[], dynamic_choices="hw_devices", default=None,
        tooltip="Hardware-accelerated video decode. Included in this slice "
                "specifically to exercise the hwaccel+compile regression test "
                "(ADR-034/071/075) end-to-end through the new GUI.",
    ),
]
