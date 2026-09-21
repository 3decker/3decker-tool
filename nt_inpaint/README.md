# nt_inpaint -- Rowan's iw3 inpainting extras

Code in `nt_iw3/` was written by Rowan ("iw3 inpainting extras" installer v1.1.1,
shared on the iw3 community Discord). It is bundled here unchanged so it loads
automatically with iw3 (see the small hook at the bottom of `iw3/__init__.py`)
instead of through the standalone installer's site-packages hook, which does not
fire for `python -m iw3`.

What it adds (all applied in memory at run time, no iw3 file is edited):
- a trained video inpainting model (`inpaint.nt_video_inpaint_v2_b`) and the
  mask handling it needs (`iw3_mask.py`)
- low-res inpainting with a full-resolution picture (`lowres.py`)
- one network pass per frame instead of two on video (`window.py`)
- "Preserve Screen Border" working with forward_inpaint (`border.py`)

The 83 MB model file is NOT stored in git. It is downloaded on first use from the
URL in `iw3/inpaint_utils.py` (`OPTIONAL_INPAINT_MODELS_YAML`).

Switches: NT_INPAINT_DISABLE=1 (everything), NT_LOWRES_DISABLE, NT_MASK_DISABLE,
NT_BORDER_DISABLE, NT_WINDOW_DISABLE, NT_INPAINT_VERBOSE=1.
