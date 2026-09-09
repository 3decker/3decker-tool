"""Regression test for the automatic git safety-commit step added to iw3-gui's
"Run Update" button (docs/ai/AI_DECISIONS.md ADR-069 dated amendment).

update.bat's own source-update step runs `git -C "%NUNIF_DIR%" pull --ff`, and if
that fails -- which it reliably does whenever there are uncommitted local changes --
falls back to `git -C "%NUNIF_DIR%" reset --hard`, permanently discarding every
uncommitted change in the working tree before retrying the pull. This nearly
destroyed 21 files of real, uncommitted session work in one real incident (manually
rescued as commit `73adecba`). `iw3.gui._git_checkpoint_before_update` now runs,
on the Run Update background thread, BEFORE update.bat is ever launched: with a
clean working tree it does nothing but log that fact; with uncommitted changes it
makes a real `git add -A` + `git commit` safety checkpoint (never a stash, which can
be lost/forgotten in a way a real commit can't) and logs what it committed; if the
checkpoint itself fails, it raises so the caller stops before update.bat ever runs.

Synthetic/isolated where the codebase's own convention allows it (CS-TEST-001), but
the two primary paths (clean tree / dirty tree) run the REAL bundled git binary
(resolved via the same `iw3.update_check._find_git()` this project always uses --
never assumes git is on PATH) against a REAL, disposable throwaway git repository
created fresh under a temp directory for this test run only -- never the real
project repository, which this test never touches. Only the one realistic failure
case (git commit erroring out) is exercised via a monkeypatch on `subprocess.run`,
matching this file's own CS-TEST-001 "mock only the failure/edge case" convention.

Run directly: python tests/test_iw3_run_update_git_checkpoint.py (from the nunif/
dir, matching this project's other tests/ path conventions), or import and call
main().
"""
import shutil
import subprocess
import sys
import tempfile
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

import iw3.gui as gui_mod  # noqa: E402
import iw3.update_check as update_check  # noqa: E402


def _git(git_bin, repo, *args):
    return subprocess.run([git_bin, "-C", repo] + list(args),
                          check=True, capture_output=True, text=True)


def _make_repo(git_bin, repo):
    _git(git_bin, repo, "init", "-q")
    # Isolate identity from any real ambient/global gitconfig, and match the real
    # project convention of real, identifiable commits (see 73adecba's own
    # "decke <decke@local>" authorship) -- never depend on the machine's real user
    # git config being present.
    _git(git_bin, repo, "config", "user.name", "test-checkpoint")
    _git(git_bin, repo, "config", "user.email", "test-checkpoint@local")
    seed = path.join(repo, "seed.txt")
    with open(seed, "w") as f:
        f.write("seed\n")
    _git(git_bin, repo, "add", "-A")
    _git(git_bin, repo, "commit", "-q", "-m", "seed commit")


def _test_clean_tree_skips_commit():
    git_bin = update_check._find_git()
    assert git_bin is not None, "bundled git binary must resolve for this test to be meaningful"

    tmp = tempfile.mkdtemp(prefix="iw3_checkpoint_test_clean_")
    try:
        _make_repo(git_bin, tmp)
        before_hash = _git(git_bin, tmp, "rev-parse", "HEAD").stdout.strip()

        logged = []
        gui_mod._git_checkpoint_before_update(tmp, logged.append)

        after_hash = _git(git_bin, tmp, "rev-parse", "HEAD").stdout.strip()
        assert after_hash == before_hash, "clean tree must never create a commit"
        assert len(logged) == 1, logged
        assert "nothing to check-point" in logged[0], logged
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("_test_clean_tree_skips_commit: PASS")


def _test_dirty_tree_creates_real_commit():
    git_bin = update_check._find_git()
    assert git_bin is not None, "bundled git binary must resolve for this test to be meaningful"

    tmp = tempfile.mkdtemp(prefix="iw3_checkpoint_test_dirty_")
    try:
        _make_repo(git_bin, tmp)
        before_hash = _git(git_bin, tmp, "rev-parse", "HEAD").stdout.strip()

        # Real uncommitted changes: one modified tracked file, one new untracked file.
        with open(path.join(tmp, "seed.txt"), "a") as f:
            f.write("modified\n")
        with open(path.join(tmp, "new_file.txt"), "w") as f:
            f.write("new\n")

        logged = []
        gui_mod._git_checkpoint_before_update(tmp, logged.append)

        after_hash = _git(git_bin, tmp, "rev-parse", "HEAD").stdout.strip()
        assert after_hash != before_hash, "dirty tree must create a real new commit"

        commit_msg = _git(git_bin, tmp, "log", "-1", "--format=%s").stdout.strip()
        assert commit_msg.startswith("Auto-checkpoint before update ("), commit_msg
        assert commit_msg.endswith(")"), commit_msg

        status_after = _git(git_bin, tmp, "status", "--porcelain").stdout
        assert status_after.strip() == "", "working tree must be clean after the checkpoint commit"

        assert len(logged) == 1, logged
        assert "Committed" in logged[0] and "2" in logged[0], logged
        short_hash = _git(git_bin, tmp, "rev-parse", "--short", "HEAD").stdout.strip()
        assert short_hash in logged[0], (short_hash, logged[0])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("_test_dirty_tree_creates_real_commit: PASS")


def _test_commit_failure_raises_and_leaves_tree_dirty():
    git_bin = update_check._find_git()
    assert git_bin is not None, "bundled git binary must resolve for this test to be meaningful"

    tmp = tempfile.mkdtemp(prefix="iw3_checkpoint_test_fail_")
    try:
        _make_repo(git_bin, tmp)
        with open(path.join(tmp, "seed.txt"), "a") as f:
            f.write("modified\n")
        before_hash = _git(git_bin, tmp, "rev-parse", "HEAD").stdout.strip()

        real_run = subprocess.run

        def _failing_run(cmd, *a, **kw):
            if len(cmd) >= 4 and cmd[3] == "commit":
                raise subprocess.CalledProcessError(
                    1, cmd, output="", stderr="fatal: simulated commit failure for this test\n")
            return real_run(cmd, *a, **kw)

        logged = []
        orig = gui_mod.subprocess.run
        gui_mod.subprocess.run = _failing_run
        try:
            raised = False
            try:
                gui_mod._git_checkpoint_before_update(tmp, logged.append)
            except RuntimeError as e:
                raised = True
                assert "simulated commit failure" in str(e), str(e)
                assert "update.bat was NOT started" in str(e), str(e)
        finally:
            gui_mod.subprocess.run = orig

        assert raised, "a real commit failure must raise, never be silently swallowed"
        after_hash = _git(git_bin, tmp, "rev-parse", "HEAD").stdout.strip()
        assert after_hash == before_hash, "a failed checkpoint commit must not move HEAD"
        status_after = _git(git_bin, tmp, "status", "--porcelain").stdout
        assert status_after.strip() != "", "working tree must remain exactly as dirty as it was"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("_test_commit_failure_raises_and_leaves_tree_dirty: PASS")


def _test_missing_git_binary_raises():
    orig_find_git = gui_mod.update_check._find_git
    gui_mod.update_check._find_git = lambda: None
    try:
        logged = []
        raised = False
        try:
            gui_mod._git_checkpoint_before_update("C:\\does\\not\\matter", logged.append)
        except RuntimeError as e:
            raised = True
            assert "Could not locate the bundled git executable" in str(e), str(e)
        assert raised, "a missing git binary must raise, never proceed silently"
        assert logged == [], logged
    finally:
        gui_mod.update_check._find_git = orig_find_git

    print("_test_missing_git_binary_raises: PASS")


def main():
    _test_clean_tree_skips_commit()
    _test_dirty_tree_creates_real_commit()
    _test_commit_failure_raises_and_leaves_tree_dirty()
    _test_missing_git_binary_raises()
    print("ALL PASS")


if __name__ == "__main__":
    main()
