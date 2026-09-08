"""Read-only "check for updates" helper for the nunif/ repo used by iw3-gui.

This repo (see docs/ai/AI_CONTEXT.md / AI_DECISIONS.md) is a heavily customized fork
on a `my-customizations` branch, tracking the upstream `nagadomi/nunif` project's
`origin/master`. Custom features (RIFE, Z-Splat, HDR reinjection, subtitle muxing,
StereoMode tagging, ...) live in the SAME files upstream also actively develops, so
blindly merging/pulling upstream changes could silently clobber or conflict with
this session's own work.

check_for_updates() therefore ONLY runs `git fetch` (updates remote-tracking refs)
plus read-only comparison commands (`rev-parse`, `rev-list --count`, `log --oneline`)
against the current branch's configured upstream tracking ref. It NEVER runs
`git pull` / `git merge` / `git reset` / `git checkout` or anything else that could
move HEAD or touch the working tree -- see docs/ai/AI_DECISIONS.md for the ADR
covering this design. Actually applying an update is explicitly out of scope.
"""
import subprocess
import sys
from os import path

# How many upstream commit subjects to list in the GUI popup before truncating to a
# "+N more" note.
MAX_LISTED_COMMITS = 20


def _find_git():
    """Resolve the bundled MinGit binary the same way iw3/utils.py resolves ffmpeg/
    dovi_tool/mkvmerge (CS-SUBPROCESS-001: never assume a tool is on the system
    PATH) -- nunif-windows root's `git/cmd/git.exe`, two levels up from `iw3/`."""
    import shutil
    here = path.dirname(path.dirname(path.abspath(__file__)))  # nunif/
    nunif_windows_root = path.dirname(here)
    for candidate in (path.join(nunif_windows_root, "git", "cmd", "git.exe"),):
        if path.exists(candidate):
            return candidate
    return shutil.which("git") or shutil.which("git.exe")


def _get_nunif_repo_root():
    return path.dirname(path.dirname(path.abspath(__file__)))


def _err_text(e):
    if isinstance(e, subprocess.CalledProcessError):
        stderr = (e.stderr or "").strip()
        return stderr if stderr else str(e)
    return str(e)


def check_for_updates(repo_root=None, git_bin=None):
    """Fetch-and-compare only -- see module docstring. Never raises: every failure
    mode returns {"status": "error", "message": ...} so a GUI caller can always show
    a clear message instead of catching a wide range of subprocess exceptions itself.

    Returns one of:
      {"status": "error", "message": str}
      {"status": "up_to_date", "local_branch": str, "upstream_ref": str}
      {"status": "updates_available", "local_branch": str, "upstream_ref": str,
       "count": int, "subjects": [str, ...], "more": int}
    """
    repo_root = repo_root or _get_nunif_repo_root()
    git_bin = git_bin or _find_git()

    if git_bin is None:
        return {"status": "error",
                "message": "Could not locate the bundled git executable (git/cmd/git.exe)."}

    def _run(args):
        return subprocess.run([git_bin, "-C", repo_root] + args,
                              check=True, capture_output=True, text=True)

    try:
        local_branch = _run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    except (subprocess.CalledProcessError, OSError) as e:
        return {"status": "error", "message": f"Failed to read the current branch: {_err_text(e)}"}

    try:
        upstream_ref = _run(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]).stdout.strip()
    except (subprocess.CalledProcessError, OSError) as e:
        return {"status": "error",
                "message": (f"Branch '{local_branch}' has no upstream tracking branch configured "
                            f"-- nothing to compare against. ({_err_text(e)})")}

    remote = upstream_ref.split("/", 1)[0]

    try:
        # Fetch only -- updates remote-tracking refs (e.g. origin/master) only. Never
        # pull/merge/reset/checkout.
        _run(["fetch", remote])
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"git fetch failed (no network?): {_err_text(e)}"}
    except OSError as e:
        return {"status": "error", "message": f"Could not run git: {e}"}

    try:
        count = int(_run(["rev-list", "--count", f"HEAD..{upstream_ref}"]).stdout.strip())
    except (subprocess.CalledProcessError, OSError, ValueError) as e:
        return {"status": "error", "message": f"Failed to compare commits: {_err_text(e)}"}

    if count == 0:
        return {"status": "up_to_date", "local_branch": local_branch, "upstream_ref": upstream_ref}

    try:
        log_output = _run(["log", "--oneline", f"HEAD..{upstream_ref}"]).stdout
        subjects = [(line.split(" ", 1)[1] if " " in line else line)
                    for line in log_output.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, OSError):
        subjects = []

    shown = subjects[:MAX_LISTED_COMMITS]
    return {
        "status": "updates_available",
        "local_branch": local_branch,
        "upstream_ref": upstream_ref,
        "count": count,
        "subjects": shown,
        "more": max(0, count - len(shown)),
    }


def format_result_message(result):
    """Builds the plain-text body for the GUI popup. Kept separate from
    check_for_updates() so the comparison logic and the display text can be tested
    independently."""
    if result["status"] == "error":
        return result["message"]

    if result["status"] == "up_to_date":
        return (f"Already up to date.\n\n"
                f"Local branch '{result['local_branch']}' has all commits from "
                f"'{result['upstream_ref']}'.")

    lines = [
        f"{result['count']} new commit(s) available on '{result['upstream_ref']}' "
        f"not yet in your local '{result['local_branch']}' branch:",
        "",
    ]
    lines.extend(f"  - {subject}" for subject in result["subjects"])
    if result["more"] > 0:
        lines.append(f"  ... +{result['more']} more")
    lines.extend([
        "",
        "These commits come from the original nunif project, NOT from this fork's own "
        "customizations (RIFE, Z-Splat, HDR reinjection, subtitle muxing, StereoMode "
        "tagging, etc.) -- they could differ from or conflict with those customizations.",
        "",
        "This is informational only. Nothing has been changed or applied -- no pull, "
        "merge, or reset was performed. Applying an update is a separate, deliberate "
        "action this button does not perform.",
    ])
    return "\n".join(lines)


def _test_check_for_updates():
    """Synthetic/mocked self-test (CS-TEST-001): no real git repo, network, or GPU
    needed. Mocks subprocess.run per-command so each scenario -- up to date, updates
    available (with truncation), fetch failure, no upstream configured, and an
    unexpected git failure -- is exercised in isolation."""
    import types
    from unittest.mock import patch

    def fake_run_factory(script):
        def fake_run(cmd, check=True, capture_output=True, text=True):
            key = tuple(cmd[3:])
            if key not in script:
                raise AssertionError(f"unexpected git command: {cmd}")
            outcome = script[key]
            return types.SimpleNamespace(stdout=outcome, stderr="", returncode=0)
        return fake_run

    base_script = {
        ("rev-parse", "--abbrev-ref", "HEAD"): "my-customizations\n",
        ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"): "origin/master\n",
        ("fetch", "origin"): "",
    }

    # 1. Already up to date.
    script = dict(base_script)
    script[("rev-list", "--count", "HEAD..origin/master")] = "0\n"
    with patch("subprocess.run", side_effect=fake_run_factory(script)):
        result = check_for_updates(repo_root="dummy_repo", git_bin="git")
    assert result["status"] == "up_to_date", result
    assert "Already up to date" in format_result_message(result)

    # 2. Updates available, no truncation.
    script = dict(base_script)
    script[("rev-list", "--count", "HEAD..origin/master")] = "3\n"
    script[("log", "--oneline", "HEAD..origin/master")] = (
        "abc1230 Fix bug\nabc1231 Add feature\nabc1232 Docs update\n")
    with patch("subprocess.run", side_effect=fake_run_factory(script)):
        result = check_for_updates(repo_root="dummy_repo", git_bin="git")
    assert result["status"] == "updates_available", result
    assert result["count"] == 3
    assert result["subjects"] == ["Fix bug", "Add feature", "Docs update"]
    assert result["more"] == 0
    msg = format_result_message(result)
    assert "3 new commit(s)" in msg and "Fix bug" in msg and "informational only" in msg

    # 3. Updates available with truncation ("+N more").
    script = dict(base_script)
    script[("rev-list", "--count", "HEAD..origin/master")] = "25\n"
    script[("log", "--oneline", "HEAD..origin/master")] = "\n".join(
        f"c{i:04x} subject {i}" for i in range(25))
    with patch("subprocess.run", side_effect=fake_run_factory(script)):
        result = check_for_updates(repo_root="dummy_repo", git_bin="git")
    assert result["status"] == "updates_available"
    assert result["count"] == 25
    assert len(result["subjects"]) == MAX_LISTED_COMMITS
    assert result["more"] == 5
    assert "+5 more" in format_result_message(result)

    # 4. Fetch failure (no network) -- must not crash, must return a clear error.
    def fake_run_fetch_fails(cmd, check=True, capture_output=True, text=True):
        key = tuple(cmd[3:])
        if key == ("fetch", "origin"):
            raise subprocess.CalledProcessError(1, cmd, stderr="fatal: unable to access repository")
        return types.SimpleNamespace(stdout=base_script[key], stderr="", returncode=0)
    with patch("subprocess.run", side_effect=fake_run_fetch_fails):
        result = check_for_updates(repo_root="dummy_repo", git_bin="git")
    assert result["status"] == "error", result
    assert "fetch failed" in result["message"]

    # 5. No upstream tracking branch configured.
    def fake_run_no_upstream(cmd, check=True, capture_output=True, text=True):
        key = tuple(cmd[3:])
        if key == ("rev-parse", "--abbrev-ref", "HEAD"):
            return types.SimpleNamespace(stdout="detached-branch\n", stderr="", returncode=0)
        if key == ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"):
            raise subprocess.CalledProcessError(128, cmd, stderr="fatal: no upstream configured for branch")
        raise AssertionError(f"unexpected git command: {cmd}")
    with patch("subprocess.run", side_effect=fake_run_no_upstream):
        result = check_for_updates(repo_root="dummy_repo", git_bin="git")
    assert result["status"] == "error", result
    assert "upstream tracking branch" in result["message"]

    # 6. Missing git binary entirely (_find_git() returns None).
    def _always_none():
        return None
    with patch(f"{__name__}._find_git", _always_none):
        result = check_for_updates(repo_root="dummy_repo", git_bin=None)
    assert result["status"] == "error"
    assert "Could not locate" in result["message"]

    # 7. Never invokes a write/destructive git subcommand -- confirmed by asserting
    # only the expected read-only verbs appear across every call made in this test.
    allowed_verbs = {"rev-parse", "fetch", "rev-list", "log"}
    forbidden_verbs = {"pull", "merge", "reset", "checkout", "push", "rebase", "clean"}
    seen_verbs = set()

    def fake_run_capture_verbs(cmd, check=True, capture_output=True, text=True):
        key = tuple(cmd[3:])
        seen_verbs.add(key[0])
        if key not in script:
            return types.SimpleNamespace(stdout="0\n", stderr="", returncode=0)
        return types.SimpleNamespace(stdout=script[key], stderr="", returncode=0)
    with patch("subprocess.run", side_effect=fake_run_capture_verbs):
        check_for_updates(repo_root="dummy_repo", git_bin="git")
    assert seen_verbs <= allowed_verbs, seen_verbs
    assert not (seen_verbs & forbidden_verbs), seen_verbs

    print("_test_check_for_updates: PASS")


def _run_self_tests():
    _test_check_for_updates()
    print("All iw3.update_check self-tests PASSED")


def main():
    if "--self-test" in sys.argv[1:]:
        _run_self_tests()
        return
    result = check_for_updates()
    print(format_result_message(result))


if __name__ == "__main__":
    main()
