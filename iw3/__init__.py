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
