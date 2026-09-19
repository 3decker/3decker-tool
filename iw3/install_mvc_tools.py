"""python -m iw3.install_mvc_tools -- installs the two tools 3D Blu-ray import needs
into the 3DECKER root folder (ADR-182):

  edge264-mvc\\  the MVC decoder. Copied from nunif\\windows_package\\edge264-mvc\\, a
                 patched build kept in this repo (upstream has no Windows release and
                 its own copy has a 4GB file-size bug -- see that folder's README.txt).
  tsmuxer\\      tsMuxeR (Apache-2.0), downloaded from its official GitHub release.

Safe to run repeatedly: anything already installed is left alone, except that an
edge264-mvc without the patch marker is replaced with the patched build.
"""
import io
import os
import shutil
import sys
import urllib.request
import zipfile
from os import path

TSMUXER_URL = "https://github.com/justdan96/tsMuxer/releases/download/2.7.0/tsMuxer-2.7.0-win64.zip"
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


def install_tsmuxer(root=None, url=TSMUXER_URL):
    root = root or _root()
    dest = path.join(root, "tsmuxer")
    exe = path.join(dest, "tsMuxeR.exe")
    if path.exists(exe):
        return "already installed"
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
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
    for label, fn in (("edge264-mvc", install_edge264), ("tsMuxeR", install_tsmuxer)):
        try:
            print(f"[mvc-tools] {label}: {fn()}")
        except Exception as e:
            failed = True
            print(f"[mvc-tools] {label}: FAILED -- {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
