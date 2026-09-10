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
from . import presets_state
from .command_line import build_cli_command, parse_cli_command
from .update_manager import check_for_updates, format_result_message, UpdateJob
from .standalone_tools import TOOLS, tool_schema, StandaloneToolJob


class Api:
    def __init__(self, window_getter):
        # window isn't available yet when Api() is constructed (pywebview
        # needs the Api instance to create the window) -- window_getter is a
        # zero-arg callable returning it once it exists.
        self._window_getter = window_getter
        self._job = None
        self._update_job = None
        self._tool_jobs = {}

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
        if self._update_job is not None and self._update_job.running:
            raise RuntimeError(
                "An update is currently running. Wait for it to finish before starting a "
                "conversion -- packages/models/source may be mid-update.")
        # Auto-save the session on every Start, matching gui.py's own
        # CONFIG_PATH auto-save-on-close behavior closely enough (this GUI
        # has no equivalent "on window close" hook to reuse, but "state as
        # of the last real run" is the more useful moment to capture anyway).
        presets_state.save_session(settings)
        if self._job is None:
            self._job = ConversionJob(self._window)
        self._job.start(settings)
        return {"started": True}

    def get_session(self):
        return presets_state.load_session()

    def list_presets(self):
        return presets_state.list_presets()

    def save_preset(self, name, settings):
        return presets_state.save_preset(name, settings)

    def load_preset(self, name):
        return presets_state.load_preset(name)

    def delete_preset(self, name):
        return presets_state.delete_preset(name)

    def copy_command(self, settings):
        return build_cli_command(settings)

    def import_command(self, text):
        settings, error = parse_cli_command(text)
        return {"settings": settings, "error": error}

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

    def check_for_updates(self):
        # Read-only (git fetch + comparison only, never pull/merge/reset --
        # see iw3/update_check.py's own module docstring). Called
        # synchronously like browse_input/browse_output -- pywebview already
        # runs Api methods off the UI thread, confirmed by native file
        # dialogs (also a blocking call) already working without freezing
        # the window.
        result = check_for_updates()
        return {"status": result["status"], "message": format_result_message(result)}

    def run_update(self):
        if self._update_job is not None and self._update_job.running:
            raise RuntimeError("an update is already running")
        if self._job is not None and self._job.running:
            raise RuntimeError(
                "A conversion is currently running. Wait for it to finish, or cancel it, before "
                "running the updater -- updating packages/models/source while a job is using them "
                "could break that job.")
        self._update_job = UpdateJob(self._window)
        self._update_job.start()
        return {"started": True}

    def is_updating(self):
        return bool(self._update_job is not None and self._update_job.running)

    def get_tool_schemas(self):
        return {t["key"]: tool_schema(t["key"]) for t in TOOLS}

    def run_tool(self, tool_key, values):
        job = self._tool_jobs.get(tool_key)
        if job is not None and job.running:
            raise RuntimeError(f"{tool_key} is already running")
        job = StandaloneToolJob(self._window, tool_key)
        self._tool_jobs[tool_key] = job
        job.start(values)
        return {"started": True}

    def is_tool_running(self, tool_key):
        job = self._tool_jobs.get(tool_key)
        return bool(job is not None and job.running)

    def browse_open_file(self, directory=None):
        result = self._window.create_file_dialog(webview.FileDialog.OPEN, directory=directory or "")
        return result[0] if result else None

    def browse_save_file(self, directory=None, default_name=""):
        # Same fix as browse_output (ADR-082): a starting file name is
        # required or pressing Save on an empty name is a silent no-op that
        # looks like a freeze.
        result = self._window.create_file_dialog(
            webview.FileDialog.SAVE, directory=directory or "", save_filename=default_name or "")
        if not result:
            return None
        return result[0] if isinstance(result, (list, tuple)) else result
