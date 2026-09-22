# nt_auto3d -- Rowan's Auto 3D Strength

Code in `nt_autostrength/` was written by Rowan (iw3-auto3d v1.0,
https://github.com/Rowan3D/iw3-auto3d, MIT licence). Bundled here unchanged so it
loads automatically with iw3 (see the loader block in `iw3/__init__.py`) instead
of through the standalone installer's site-packages hook, which does not fire for
`python -m iw3` -- the same reason [[nt_inpaint]] is loaded that way.

What it adds: a new "Auto 3D Strength" control under 3D Strength in the iw3
window (and `--auto-divergence` on the command line). It picks the 3D Strength
per scene from how close the shot looks -- wide shots and landscapes get less,
close-ups get more -- using CLIP (ViT-B/32) to judge shot framing, since iw3's
own depth has had its absolute scale normalised away and can't tell a landscape
from a face filling the frame. Off by default; a no-op until you tick the box.

The 176 MB CLIP image model is NOT stored in git. It downloads on first use
(from open_clip's own GitHub release, SHA-256 checked) into `nt_auto3d/cache/`.
Without it, or before it's downloaded, it falls back to a rougher depth-only
estimate and says so.

Switches: `NT_AUTODIV_DISABLE=1` turns it off entirely; `NT_AUTODIV_OVERLAY=1`
or the GUI's "Show strength on video (debug)" burns the strength used into each
frame's corner; `NT_AUTODIV_LOG=<path>` logs one CSV line per frame.
