"""Window creation, single-instance guard, and top-level crash safety net
for the web-rendered iw3 GUI (ADR-081)."""
import sys
from os import path

import webview
import win32api
import win32event
import winerror

from .api import Api
from .crash_log import log_exception

_MUTEX_NAME = "Global\\3DECKER_iw3_gui_web_single_instance"
PUBLIC_DIR = path.join(path.dirname(__file__), "public")


def _acquire_single_instance_guard():
    """Cross-process equivalent of wx.SingleInstanceChecker, without
    depending on wx (this GUI is deliberately wx-free). Confirmed in the
    Phase 0 spike: a second process racing for the same named mutex
    correctly observes ERROR_ALREADY_EXISTS."""
    mutex = win32event.CreateMutex(None, False, _MUTEX_NAME)
    already_running = (win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS)
    return mutex, already_running


def main():
    mutex, already_running = _acquire_single_instance_guard()
    if already_running:
        # No dialog (yet) for this narrow slice -- just refuse to open a
        # second window, same end result as the wx GUI's default path.
        return

    window_holder = {}

    def get_window():
        return window_holder["window"]

    api = Api(get_window)
    window = webview.create_window(
        "3DECKER — iw3 (web)",
        url=path.join(PUBLIC_DIR, "index.html"),
        js_api=api,
        width=1180,
        height=800,
        min_size=(900, 600),
        background_color="#060a10",
    )
    window_holder["window"] = window

    try:
        webview.start()
    except Exception:
        log_exception("webview.start() crashed")
        raise
    finally:
        win32api.CloseHandle(mutex)


if __name__ == "__main__":
    main()
