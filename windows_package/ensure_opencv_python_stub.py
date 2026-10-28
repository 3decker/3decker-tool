"""ADR-401: pre-empts the known mmengine/windows_capture opencv-python conflict
(ADR-396/397/398) BEFORE it can happen, instead of only cleaning up after it.

mmengine and windows_capture each declare their own, unrelated, unpinned
dependency on plain `opencv-python` -- but both only ever `import cv2` at
runtime (neither imports anything literally named `opencv_python`), and
`opencv-contrib-python` (pinned in requirements.txt, ADR-396) already
provides that exact same `cv2` import, plus everything sod_track and
face_detect need on top of it. The real opencv-python package is never
actually wanted here. Pip has no concept of one package "providing" another's
declared name, though, so every `pip install -r requirements.txt` /
`requirements-gui.txt` run tried to download and install it anyway, hit
`OSError: [WinError 5] Access is denied` on cv2.pyd (it's loaded in the
installer's own running process), retried 3 times, and burned a ~44MB
download twice (once per file) for a package that was never going to be
usable anyway -- confirmed real and reproducible 2026-10-28.

This registers an empty, file-less `opencv_python-<version>.dist-info` folder
in site-packages: enough for pip's own installed-package bookkeeping to treat
the `opencv-python>=3` requirement as already satisfied, so it skips the
download/install step entirely. It carries zero real files -- nothing in this
codebase ever imports a module literally named `opencv_python`, and the real
`cv2` module on disk stays exactly whatever opencv-contrib-python installed.

Must run BEFORE requirements.txt/requirements-gui.txt are installed each
time. The forced `opencv-contrib-python` reinstall that already runs after
both of them (ADR-397/398) stays in place unchanged as a safety net, in case
anything else ever reintroduces this conflict a different way.
"""
import shutil
import sysconfig
import pathlib

PINNED_VERSION = "4.10.0.84"


def ensure_stub():
    site_packages = pathlib.Path(sysconfig.get_path("purelib"))
    dist_info = site_packages / f"opencv_python-{PINNED_VERSION}.dist-info"
    if (dist_info / "METADATA").exists():
        return False

    # Never leave more than one opencv_python-*.dist-info around -- a stale
    # one from a previous pinned version (or a real opencv-python that slipped
    # in before this fix existed) would otherwise sit next to the new one.
    for stale in site_packages.glob("opencv_python-*.dist-info"):
        if stale != dist_info:
            shutil.rmtree(stale, ignore_errors=True)

    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: opencv-python\n"
        f"Version: {PINNED_VERSION}\n"
        "Summary: Stub only -- real cv2 is provided by opencv-contrib-python (see ADR-396/401)\n",
        encoding="utf-8")
    (dist_info / "INSTALLER").write_text("pip\n", encoding="utf-8")
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    return True


def main():
    if ensure_stub():
        print(f"[3DECKER] Registered an opencv-python {PINNED_VERSION} stub so pip "
              f"stops trying to install it over opencv-contrib-python (ADR-401).")


if __name__ == "__main__":
    main()
