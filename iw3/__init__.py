import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

# Rowan's inpainting extras (nt_inpaint/): registers the extra model architecture
# and its run-time patches. Loaded here rather than through a site-packages hook
# so it also works for `python -m iw3`. Any failure only prints a note; iw3 starts
# normally without it.
try:
    import sys as _sys
    _nt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nt_inpaint")
    if os.path.isdir(_nt_dir) and not os.environ.get("NT_INPAINT_DISABLE"):
        if _nt_dir not in _sys.path:
            _sys.path.append(_nt_dir)
        from nt_iw3 import boot as _nt_boot
        _nt_boot.install()
except Exception as _e:  # noqa: BLE001
    import sys as _sys
    print(f"iw3: inpainting extras (nt_inpaint) not loaded ({type(_e).__name__}: {_e})", file=_sys.stderr)

# Rowan's Auto 3D Strength (nt_auto3d/): adds the "Auto 3D Strength" control (per-
# scene divergence from shot framing). Same reasoning as nt_inpaint above -- loaded
# here, not through a site-packages hook, so it also works for `python -m iw3`.
# Off by default (--auto-divergence); any failure only prints a note.
try:
    import sys as _sys
    _nt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nt_auto3d")
    if os.path.isdir(_nt_dir) and not os.environ.get("NT_AUTODIV_DISABLE"):
        if _nt_dir not in _sys.path:
            _sys.path.append(_nt_dir)
        from nt_autostrength import autodiv as _nt_autodiv
        # Rowan's own GUI-injection (_patch_nunif_gui/_add_controls) shifts existing rows in the stereo
        # panel's GridBagSizer down by 4 to make room. It was built against vanilla iw3's simpler layout;
        # 3DECKER's stereo panel gives almost every control its own extra slider row, and the shift lands
        # two widgets on the same grid cell -- a real wx assertion (confirmed live), which aborted the
        # insert with one orphaned, overlapping checkbox left on screen. The CLI flags and the
        # apply_divergence patch (_patch_utils) are unaffected and work correctly; 3DECKER adds its own
        # native "Auto 3D Strength" controls in iw3/gui.py instead (see ADR-213), so this half is skipped.
        _nt_autodiv._patch_nunif_gui = lambda *a, **k: None
        _nt_autodiv.install()
except Exception as _e:  # noqa: BLE001
    import sys as _sys
    print(f"iw3: Auto 3D Strength (nt_auto3d) not loaded ({type(_e).__name__}: {_e})", file=_sys.stderr)
