FRIM 1.31 (x64) -- the two files 3DECKER's "SBS to 3D Blu-ray MVC" tool needs
================================================================================

FRIMEncode64.exe   FRIM Encoder 1.31 (build 2020-03-08) by "videofan3d" -- a free command-line
                   H.264/MVC-3D/HEVC encoder built on Intel Media SDK.
libmfxsw64.dll     Intel Media SDK 2019 R1 software implementation (the part that encodes MVC when
                   there is no supporting Intel graphics hardware -- which is the normal case).

Source:            https://sites.google.com/site/videofan3d/software/frim-encoder (the author's page)
Original archive:  FRIM 1.31 x64, SHA-256 76689784495D53B34889F0EA67C9DB6B9750925DB9D1147F8FD9159E111C0778
Only the two files above are kept; nothing was modified.

Permission:        The author of FRIM has given permission (as reported by this project's owner,
                   2026-09-19) for this project to host and redistribute these files, so they are not
                   lost if the author's download link ever disappears.
Intel's part:      FRIM's own release notes (included here as FRIM_release_notes.txt) state that
                   "Intel Media SDK can be freely distributed and used - see 'Intel Media SDK EULA.rtf' -
                   section Licensed Binaries." libmfxsw64.dll is covered by that Intel licence.
No warranty:       Provided as-is by their respective authors; not part of 3DECKER's own code.

Installed to <3DECKER root>\frim\ by `python -m iw3.install_mvc_tools` (run by setup and by every update).
