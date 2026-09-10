"""The Api object exposed to the frontend via pywebview's js_api=.

Every method here is called from page JS as `pywebview.api.<name>(...)` and
returns a JSON-serializable value back through a Promise (confirmed working
in the Phase 0 spike, including error propagation -- an exception raised
here reaches JS as a rejected Promise, so methods can just raise normally).
"""
import webview

from .schema_export import build_schema_dict
from .worker import ConversionJob


class Api:
    def __init__(self, window_getter):
        # window isn't available yet when Api() is constructed (pywebview
        # needs the Api instance to create the window) -- window_getter is a
        # zero-arg callable returning it once it exists.
        self._window_getter = window_getter
        self._job = None

    @property
    def _window(self):
        return self._window_getter()

    def get_schema(self):
        return build_schema_dict()

    def browse_input(self):
        # Real native OS file dialog via pywebview -- deliberately NOT an
        # HTML <input type=file>, which typically can't expose a real
        # filesystem path from inside a webview (a known trap, see ADR-081).
        result = self._window.create_file_dialog(webview.FileDialog.OPEN)
        return result[0] if result else None

    def browse_output(self):
        result = self._window.create_file_dialog(webview.FileDialog.SAVE)
        if not result:
            return None
        return result[0] if isinstance(result, (list, tuple)) else result

    def start(self, settings):
        if self._job is None:
            self._job = ConversionJob(self._window)
        self._job.start(settings)
        return {"started": True}

    def cancel(self):
        if self._job is not None:
            self._job.cancel()
        return {"cancelled": True}

    def suspend(self):
        if self._job is not None:
            self._job.suspend()
        return {"suspended": True}

    def resume(self):
        if self._job is not None:
            self._job.resume()
        return {"resumed": True}

    def is_running(self):
        return bool(self._job is not None and self._job.running)
