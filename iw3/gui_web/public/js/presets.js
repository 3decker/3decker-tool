// Quick Preset values, copied verbatim from iw3/gui.py's own
// apply_quick_preset() (gui.py:6118) -- not re-derived, the exact same
// field values that function sets. Field names below are this schema's
// names, mapped 1:1 from gui.py's control names where the underlying
// setting is the same.
window.IW3_QUICK_PRESETS = {
  movie: {
    divergence: "2.0",
    convergence: "0.5",
    convergence_mode: "sod_v1",
    foreground_pop: "0.0",
  },
  action: {
    divergence: "3.0",
    convergence: "0.5",
    convergence_mode: "face_detect",
    foreground_pop: "0.5",
  },
  // "3DECKER Preferred" -- gui.py's own confirmed-best combo (ADR-057
  // Amendment 12 / ADR-076), verified via a real full-length movie
  // conversion. temporal_stabilize's flat_boost/edge_protect/max_shift and
  // the "Output Size Limit" combined dropdown have no schema equivalent
  // here (former: not real CLI args, see schema.py's docstring; latter:
  // this GUI exposes max_output_width/height separately) -- width/height
  // are set directly from the same 3840x2160 target instead.
  decker: {
    method: "mlbw_l2_inpaint",
    preserve_screen_border: true,
    depth_model: "Any_V3_Mono_01",
    divergence: "2.5",
    convergence: "0.5",
    background_pop_coverage: "0.0",
    stereo_format: "half_sbs",
    resolution: "512",

    inpaint_model: "light_inpaint_v1",
    inpaint_overlap_frames: "3 3",
    depth_aa: true,

    depth_refine: true,
    depth_refine_strength: "1.0",

    temporal_stabilize: true,
    temporal_stabilize_strength: "0.3",

    scene_detect: true,
    autocrop: "BLACK",

    ema_normalize: true,
    ema_decay: "0.94",
    ema_buffer: "60",

    scene_batch_auto_ema_model: "GEMINI AI",
    scene_batch_auto_ema: true,

    max_output_width: "3840",
    max_output_height: "2160",
    pix_fmt: "yuv420p10le",
    video_format: "mkv",
    video_codec: "hevc_nvenc",
    tune: "uhq",
    max_fps: "1000.0",
    crf: "15",
    hwaccel: "cuda",
    disable_software_fallback: true,

    max_workers: "2",
    metadata: true,
    preserve_dowi: true,
    stereo_mode_tag: true,
    auto_resume: true,
    compile: true,
  },
};

window.IW3_QUICK_PRESET_LABELS = {
  movie: "Movie (subtle 3D)",
  action: "Action (strong pop effects)",
  decker: "3DECKER Preferred",
};
