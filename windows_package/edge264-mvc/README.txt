edge264-mvc (patched build) -- H.264 / MVC decoder used by 3DECKER's 3D Blu-ray import
==========================================================================================

Upstream: https://github.com/jens-duttke/edge264-mvc  (BSD-3-Clause, see LICENSE_BSD.txt)
Built with MinGW-w64 GCC 16.2.0 (winlibs.com), Windows x64.
Runtime DLLs shipped alongside: libgcc_s_seh-1.dll (GCC runtime library exception),
libwinpthread-1.dll (MIT).

LOCAL PATCH (source/edge264_test.c, function win32_file_size64 + its two call sites in decode_file()):
  Upstream measures the input file with GetFileSize(f, NULL), a 32-bit call that
  silently wraps for files over 4GB. A full 3D Blu-ray stream is ~25GB, so the
  decoder saw only (size mod 2^32) bytes, decoded that slice, and reported success --
  a movie silently cut short (a 90-minute film stopped at 12:44). The patch uses
  GetFileSizeEx (true 64-bit size). 3DECKER also streams the input through stdin, so
  it does not depend on this path, but the patch stays for anyone running the
  decoder directly. See docs/ai/AI_DECISIONS.md ADR-182 UPDATE 3.

Installed to <3DECKER root>\edge264-mvc\ by `python -m iw3.install_mvc_tools`
(run by setup and by every update).
