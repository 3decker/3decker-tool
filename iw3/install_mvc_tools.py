"""python -m iw3.install_mvc_tools -- installs the two tools 3D Blu-ray import needs
into the 3DECKER root folder (ADR-182):

  edge264-mvc\\  the MVC decoder. Copied from nunif\\windows_package\\edge264-mvc\\, a
                 patched build kept in this repo (upstream has no Windows release and
                 its own copy has a 4GB file-size bug -- see that folder's README.txt).
  tsmuxer\\      tsMuxeR (Apache-2.0), downloaded from its official GitHub release --
                 the teaching-droid/tsMuxer fork (ADR-300: migrated off justdan96/tsMuxer
                 2.7.0, which is archived/unmaintained as of 2026-09-30) -- checked against
                 a pinned SHA-256, same pattern as FRIM below.
  frim\\         FRIMEncode (freeware, videofan3d), downloaded from its author's page and
                 checked against a pinned SHA-256 -- only needed by SBS-to-MVC.

Safe to run repeatedly: anything already installed is left alone, except that an
edge264-mvc without the patch marker is replaced with the patched build.
"""
import hashlib
import io
import os
import shutil
import sys
import urllib.request
import zipfile
from os import path

# ADR-300 (2026-09-30): migrated from justdan96/tsMuxer 2.7.0 (archived/unmaintained as of
# 2026-09-30) to teaching-droid/tsMuxer 2.18.14, an actively-maintained fork. The release
# zip's SHA-256 is pinned and verified before extraction -- same integrity pattern
# install_frim() already used below, now applied here too (the old URL had no checksum
# check at all).
TSMUXER_URL = "https://github.com/teaching-droid/tsMuxer/releases/download/v2.18.14/tsMuxeR-2.18.14-windows-x64.zip"
TSMUXER_SHA256 = "366DE3B95442ADF6D272294565B62F20D780CE7697211B80332F1C7C6FB967E6"
PATCH_MARKER = ".3decker-patched-getfilesizeex"
_EDGE264_FILES = ("edge264_test.exe", "edge264.1.dll", "libwinpthread-1.dll", "libgcc_s_seh-1.dll",
                  "LICENSE_BSD.txt", "README.txt")


def _root():
    return path.dirname(path.dirname(path.dirname(path.abspath(__file__))))


def install_edge264(root=None, package_dir=None):
    root = root or _root()
    package_dir = package_dir or path.join(root, "nunif", "windows_package", "edge264-mvc")
    dest = path.join(root, "edge264-mvc")
    if path.exists(path.join(dest, "edge264_test.exe")) and path.exists(path.join(dest, PATCH_MARKER)):
        return "already installed"
    if not path.isdir(package_dir):
        raise RuntimeError(f"patched edge264-mvc build not found at {package_dir}")
    os.makedirs(dest, exist_ok=True)
    for name in _EDGE264_FILES:
        shutil.copy2(path.join(package_dir, name), path.join(dest, name))
    open(path.join(dest, PATCH_MARKER), "w").close()
    return "installed"


def install_tsmuxer(root=None, url=TSMUXER_URL, expected_sha256=TSMUXER_SHA256):
    """Downloads the tsMuxeR release zip and accepts it only if its SHA-256 matches
    expected_sha256 (ADR-300) -- same "verify before trusting a downloaded binary"
    pattern install_frim() uses below, since this file also lands directly in every
    user's install and gets run as a subprocess."""
    root = root or _root()
    dest = path.join(root, "tsmuxer")
    exe = path.join(dest, "tsMuxeR.exe")
    if path.exists(exe):
        return "already installed"
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    if expected_sha256 and hashlib.sha256(data).hexdigest().upper() != expected_sha256.upper():
        raise RuntimeError("downloaded tsMuxeR archive did not match the expected checksum")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        member = next((n for n in z.namelist() if path.basename(n).lower() == "tsmuxer.exe"), None)
        if member is None:
            raise RuntimeError("tsMuxeR.exe not found inside the downloaded archive")
        os.makedirs(dest, exist_ok=True)
        with z.open(member) as src, open(exe, "wb") as out:
            shutil.copyfileobj(src, out)
    return "installed"


def main():
    if sys.platform != "win32":
        print("[mvc-tools] Windows-only bundles; nothing to do on this platform.")
        return 0
    failed = False
    from .sbs_to_mvc_cli import install_frim
    for label, fn in (("edge264-mvc", install_edge264), ("tsMuxeR", install_tsmuxer),
                      ("FRIMEncode", install_frim)):
        try:
            print(f"[mvc-tools] {label}: {fn()}")
        except Exception as e:
            failed = True
            print(f"[mvc-tools] {label}: FAILED -- {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
