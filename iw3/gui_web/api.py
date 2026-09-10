"""The Api object exposed to the frontend via pywebview's js_api=.

Every method here is called from page JS as `pywebview.api.<name>(...)` and
returns a JSON-serializable value back through a Promise (confirmed working
in the Phase 0 spike, including error propagation -- an exception raised
here reaches JS as a rejected Promise, so methods can just raise normally).
"""
from os import path

import webview

from .schema_export import build_schema_dict
from .worker import ConversionJob
from .paths_state import (
    get_last_input_dir, get_last_output_dir, set_last_input_dir, set_last_output_dir,
)


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
        # Starts in whichever folder Input was last picked from.
        result = self._window.create_file_dialog(
            webview.FileDialog.OPEN, directory=get_last_input_dir())
        if not result:
            return None
        chosen = result[0]
        set_last_input_dir(path.dirname(chosen))
        return chosen

    def browse_output(self, input_path=None):
        # A native Save dialog with no starting file name leaves its "File
        # name" box empty -- pressing Save on an empty name is a no-op in
        # Windows (the dialog just stays open), which looks exactly like a
        # freeze. Pre-filling it with the input file's own name means
        # pressing Save immediately reuses that name, as intended.
        #
        # The starting FOLDER deliberately does NOT default to the input
        # file's own folder -- real usage here has input and output on
        # different drives/folders entirely (e.g. input on a source movies
        # drive, output on a dedicated 3D-output drive). It starts in
        # whichever folder Output was last saved to instead.
        default_name = path.basename(input_path) if input_path else ""
        result = self._window.create_file_dialog(
            webview.FileDialog.SAVE, directory=get_last_output_dir(), save_filename=default_name)
        if not result:
            return None
        chosen = result[0] if isinstance(result, (list, tuple)) else result
        set_last_output_dir(path.dirname(chosen))
        return chosen

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
