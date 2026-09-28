import sys
import os


if sys.executable.endswith("pythonw.exe"):
    """
    python code started from pythonw.exe crashes when accessing stdout/stderr.
    so reopen stdout/stderr with devnull.
    """
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    def _pythonw_startup_excepthook(exc_type, exc_value, exc_tb):
        # Real user report: a startup-time failure (their case: a broken/missing
        # face_detect dependency) produced zero visible output and was hard to
        # diagnose -- because stdout/stderr are devnull'd above, and this covers
        # every one of this project's 4 GUIs (iw3/waifu2x/desktop/player all
        # import this module first, before anything else). The per-job crash
        # dialog (iw3/gui.py's on_exit_worker) already handles exceptions raised
        # while a conversion is running -- this hook covers the gap before that:
        # any exception during this app's own import chain or its main() /
        # App() construction, on the main thread, before any window exists.
        import traceback
        from datetime import datetime
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log_path = None
        try:
            from nunif.utils.home_dir import ensure_home_dir
            # Explicit default_path, matching iw3/gui.py's own CONFIG_DIR convention
            # (ensure_home_dir("iw3", <iw3>/../tmp)) -- without it, ensure_home_dir falls
            # back to writing straight into this package's own source directory instead
            # of the shared nunif/tmp/ folder every other real log/config file in this
            # project already uses (caught by this fix's own real functional test).
            log_dir = ensure_home_dir("nunif", os.path.join(os.path.dirname(__file__), "tmp"))
            log_path = os.path.join(log_dir, "nunif-startup-crash.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n---- {datetime.now().isoformat(timespec='seconds')} ----\n")
                f.write(f"argv[0]: {sys.argv[0]}\n")
                f.write(text)
        except Exception:
            log_path = None

        message = f"{os.path.basename(sys.argv[0])} failed to start.\n\n{exc_type.__name__}: {exc_value}"
        if log_path:
            message += f"\n\n(Full details saved to {log_path})"
        try:
            import wx
            wx.App.Get() or wx.App(False)
            wx.MessageBox(message, "Startup Error", wx.OK | wx.ICON_ERROR)
        except Exception:
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(0, message, "Startup Error", 0x10)
            except Exception:
                pass

    sys.excepthook = _pythonw_startup_excepthook


def _self_test_startup_excepthook():
    """Real functional test (not just import-checked): simulates running under
    pythonw.exe, raises a synthetic exception through the installed sys.excepthook,
    and confirms both the message-box text and the real crash-log file it's supposed
    to produce -- this is the exact class of bug a real user hit (a startup-time
    failure with zero visible output) and the exact log-path bug this test itself
    caught during development (ensure_home_dir with no default_path wrote straight
    into this package's own source directory instead of nunif/tmp/)."""
    import importlib
    import os as _os
    from unittest import mock

    real_executable = sys.executable
    real_stdout = sys.stdout
    sys.executable = r"C:\fake\pythonw.exe"
    try:
        import nunif.pythonw_fix as pf
        importlib.reload(pf)
    finally:
        sys.executable = real_executable
        sys.stdout = real_stdout

    assert sys.excepthook is pf._pythonw_startup_excepthook, "hook was not installed"

    captured = {}

    def fake_message_box(message, title, style):
        captured["message"] = message

    with mock.patch("wx.MessageBox", fake_message_box), mock.patch("wx.App"):
        try:
            raise RuntimeError("self-test synthetic startup failure")
        except RuntimeError:
            exc_type, exc_value, exc_tb = sys.exc_info()
            sys.excepthook(exc_type, exc_value, exc_tb)

    assert "self-test synthetic startup failure" in captured.get("message", ""), \
        f"message missing detail: {captured}"
    assert "Full details saved to" in captured["message"], "log path not mentioned in message"

    from nunif.utils.home_dir import ensure_home_dir
    log_dir = ensure_home_dir("nunif", _os.path.join(_os.path.dirname(pf.__file__), "tmp"))
    log_path = _os.path.join(log_dir, "nunif-startup-crash.log")
    assert _os.path.exists(log_path), "log file was not written"
    with open(log_path, encoding="utf-8") as f:
        content = f.read()
    assert "self-test synthetic startup failure" in content, "log file missing traceback text"
    _os.remove(log_path)  # this test's own entry, not a real crash -- don't leave it behind

    print("_self_test_startup_excepthook: PASS")


if __name__ == "__main__":
    _self_test_startup_excepthook()
