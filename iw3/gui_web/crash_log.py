"""pythonw-safe crash logging.

Launched via pythonw.exe (no console), this process's stdout/stderr are
devnulled by nunif.pythonw_fix -- print()/traceback.print_exc() vanish
silently. The wx GUI already hit this for real (iw3-gui-crash.log); this is
the same fix for the new one, confirmed to work under pythonw in the Phase 0
spike for this rebuild.
"""
import traceback
from os import path


def crash_log_path():
    from iw3.gui_web import CONFIG_DIR
    return path.join(CONFIG_DIR, "iw3-gui-web-crash.log")


def log_exception(context=""):
    """Call from inside an `except:` block. Returns the log file path."""
    log_path = crash_log_path()
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            if context:
                f.write(f"--- {context} ---\n")
            f.write(traceback.format_exc())
            f.write("\n")
    except OSError:
        pass
    return log_path
