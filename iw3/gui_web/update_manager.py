"""Check for Updates / Run Update for the web GUI (ADR-086).

check_for_updates()/format_result_message() are imported directly from
iw3.update_check -- that module is already wx-free (its own docstring says
so), so no duplication needed there, unlike gpu_query.py's nvidia-smi query
(which had to duplicate gui.py's own function since importing iw3.gui at
all pulls in wx as an import-time side effect).

_find_update_bat()/_git_checkpoint_before_update() DO live directly inside
iw3/gui.py itself, so importing them would still pull in wx even though
neither function touches wx -- these two are duplicated here instead,
verbatim in logic, matching gui.py's own comments/reasoning exactly (see
docs/ai/AI_DECISIONS.md ADR-069 for why the safety checkpoint exists at
all: update.bat's `git pull --ff` falls back to `git reset --hard` on
conflict, which had previously destroyed real uncommitted work).
"""
import os
import subprocess
import threading
from datetime import datetime
from os import path

from iw3.update_check import check_for_updates, format_result_message, _find_git

from .worker import _push


def _find_update_bat():
    nunif_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))  # nunif/
    nunif_windows_root = path.dirname(nunif_dir)
    return path.join(nunif_windows_root, "update.bat"), nunif_windows_root


def _git_checkpoint_before_update(nunif_dir, log_fn):
    """Real, read-only-or-additive git operations only (status/add/commit) --
    never push/reset/checkout/merge/rebase -- so this can only ever ADD
    safety, never risk it. Raises RuntimeError (never silently swallowed) if
    the checkpoint itself fails, so the caller can stop before update.bat is
    ever launched."""
    git_bin = _find_git()
    if git_bin is None:
        raise RuntimeError(
            "Could not locate the bundled git executable (git/cmd/git.exe) -- cannot safety "
            "check-point uncommitted work before updating. Update was NOT started.")

    def _run(args):
        return subprocess.run([git_bin, "-C", nunif_dir] + args,
                               check=True, capture_output=True, text=True)

    try:
        status = _run(["status", "--porcelain"])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "git status failed -- cannot safety check-point uncommitted work before updating. "
            f"Update was NOT started.\n{(e.stderr or '').strip()}")

    changed_files = [line for line in status.stdout.splitlines() if line.strip()]
    if not changed_files:
        log_fn("No uncommitted changes -- nothing to check-point.\n\n")
        return

    try:
        _run(["add", "-A"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _run(["commit", "-m", f"Auto-checkpoint before update ({timestamp})"])
        short_hash = _run(["rev-parse", "--short", "HEAD"]).stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Failed to create the safety check-point commit -- update.bat was NOT started, so "
            f"your uncommitted work is untouched. Fix the error below and try again.\n"
            f"{(e.stderr or '').strip()}")

    log_fn(f"Committed {len(changed_files)} file(s) as a safety checkpoint before updating "
           f"(commit {short_hash}).\n\n")


class UpdateJob:
    """Runs the real update.bat on a background thread, streaming its output
    to the frontend via the same evaluate_js push mechanism worker.py's
    ConversionJob uses for conversion progress -- a genuinely long-running
    job (package/model downloads, a source pull) needs live visibility, not
    a result dumped only at the end."""

    def __init__(self, window):
        self.window = window
        self.running = False

    def start(self):
        if self.running:
            raise RuntimeError("an update is already running")
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            update_bat_path, cwd = _find_update_bat()
            if not path.exists(update_bat_path):
                _push(self.window, "update_log", {"line": f"update.bat was not found at: {update_bat_path}\n"})
                _push(self.window, "update_done", {"ok": False})
                return

            nunif_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
            try:
                _git_checkpoint_before_update(
                    nunif_dir, lambda text: _push(self.window, "update_log", {"line": text}))
            except RuntimeError as e:
                _push(self.window, "update_log", {"line": f"\n{e}\n"})
                _push(self.window, "update_done", {"ok": False})
                return

            comspec = os.environ.get("ComSpec") or r"C:\Windows\System32\cmd.exe"
            cmd = [comspec, "/c", update_bat_path]
            # stdin explicitly closed: update.bat ends with `pause` on both its
            # success and error paths, which would otherwise wait forever for a
            # keypress this non-interactive subprocess can never provide.
            proc = subprocess.Popen(
                cmd, cwd=cwd, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            for line in proc.stdout:
                _push(self.window, "update_log", {"line": line})
            proc.wait()
            _push(self.window, "update_done", {"ok": proc.returncode == 0})
        finally:
            self.running = False
