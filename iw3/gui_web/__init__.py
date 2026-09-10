import os
from os import path

# Must happen before any subprocess call this package (or iw3_main() itself)
# ever makes -- on Windows, torch.compile shells out to the C++ compiler
# (cl.exe), and each invocation flashes a console window unless every
# subprocess.Popen call carries CREATE_NO_WINDOW. Every existing GUI in this
# project (iw3/gui.py, waifu2x/gui.py) imports this as its own first real
# import for the same reason -- gui_web needs the identical fix, applied
# this early since it's a package-wide monkeypatch, not a per-call flag.
import nunif.gui.subprocess_patch  # noqa: E402,F401

from nunif.utils.home_dir import ensure_home_dir

# Same resolution as iw3/gui.py's own CONFIG_DIR (same "iw3" app name, same
# fallback tmp/ folder) so both GUIs' files land in one place -- but every
# file this package writes uses a "-web" suffixed name (iw3-gui-web.cfg,
# iw3-gui-web-crash.log, ...) so the two GUIs' state never collides.
CONFIG_DIR = ensure_home_dir("iw3", path.join(path.dirname(__file__), "..", "tmp"))
os.makedirs(CONFIG_DIR, exist_ok=True)
