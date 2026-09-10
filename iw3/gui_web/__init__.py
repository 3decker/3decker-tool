import os
from os import path

from nunif.utils.home_dir import ensure_home_dir

# Same resolution as iw3/gui.py's own CONFIG_DIR (same "iw3" app name, same
# fallback tmp/ folder) so both GUIs' files land in one place -- but every
# file this package writes uses a "-web" suffixed name (iw3-gui-web.cfg,
# iw3-gui-web-crash.log, ...) so the two GUIs' state never collides.
CONFIG_DIR = ensure_home_dir("iw3", path.join(path.dirname(__file__), "..", "tmp"))
os.makedirs(CONFIG_DIR, exist_ok=True)
