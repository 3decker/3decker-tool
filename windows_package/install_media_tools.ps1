# Shared, version-aware ffmpeg/MKVToolNix install-or-update logic, extracted
# out of setup.ps1 so a fresh install and an existing install's update path can never
# drift apart on how these two tools are kept current -- the same reason
# detect_torch_variant.ps1 exists (see that script's own header comment).
#
# Defines two functions, Install-Ffmpeg and Install-MkvToolNix. Each is VERSION-AWARE,
# not just presence-aware: a tool that's already there is only left alone when it's
# actually current; missing or outdated triggers a real checksum-verified
# download and an atomic swap (CS-IO-001) so an interrupted download/replace can
# never leave a half-replaced binary behind.
#
# Usage:
#   Dot-sourced (setup.ps1's own convention -- runs in-process, so it just calls the
#   functions below directly with its own $root/$tmpDir after sourcing this file):
#       . (Join-Path $nunifDir "windows_package\install_media_tools.ps1")
#       Install-Ffmpeg -Root $root -TmpDir $tmpDir
#       Install-MkvToolNix -Root $root -TmpDir $tmpDir
#
#   Run directly (the update path's convention -- same `powershell -File` pattern
#   update-3decker.bat/update.bat already use for detect_torch_variant.ps1): runs
#   both installs itself against the real distribution root (two levels up from this
#   script's own folder, windows_package\, unless -MediaRoot overrides it) and exits
#   with a non-zero code if either failed. That non-zero exit is never allowed to
#   fail the whole update, though -- the .bat callers invoke this with `&`, not
#   `&&`, matching the existing non-fatal convention for `python -m iw3.install_mvc_tools`.
#       powershell -NoProfile -ExecutionPolicy Bypass -File install_media_tools.ps1
#
# NOTE on the -MediaRoot param name: deliberately NOT named -Root/$Root. Dot-sourcing
# a script runs its param block in the CALLER's scope, and PowerShell variable names
# are case-insensitive -- a param named $Root would silently clobber setup.ps1's own
# $root variable the moment this file is dot-sourced, even with no argument passed
# (it would bind to $null). Using a distinct name avoids that collision entirely;
# the Install-Ffmpeg/Install-MkvToolNix FUNCTION parameters below are safe to name
# -Root because function parameters are scoped to the function call, not the caller.
param(
    [string]$MediaRoot
)

# ---------------------------------------------------------------------------
# ffmpeg: ADR-373 -- used to always chase BtbN's GitHub-hosted "latest" GPL build (a
# rotating git-hash+date with no fixed semver), re-verifying its checksums.sha256 on
# every run so a changed build always got re-downloaded. That rolling-nightly design is
# exactly why a real, reproducible ffmpeg crash on certain Dolby Vision HEVC content
# (confirmed live: https://trac.ffmpeg.org/ticket/11468, a regression FFmpeg's own
# developers bisected to libavcodec commits between 2025-01-22 and 2025-02-02, still
# unresolved as of this fix) survived THROUGH an ffmpeg update that was supposed to fix
# it -- every nightly since has carried the same bug, so "grab the newest one" could
# never have helped. Switched to a normal pinned-version-and-checksum design instead
# (matching MKVToolNix's own design just below, and tsMuxeR/FRIM in
# iw3/install_mvc_tools.py/iw3/sbs_to_mvc_cli.py) -- a specific, fixed FFmpeg release
# that predates that regression window, from gyan.dev (one of the Windows builds
# ffmpeg.org's own download page itself links to), whose "full" GPL build already
# matches BtbN's own feature set (nvenc/nvdec/cuvid/libx265/etc., confirmed directly:
# `ffmpeg -version`'s own configure line lists them all). gyan.dev's own site only keeps
# the current + previous release visible, but permanently mirrors every tagged release
# on GitHub (github.com/GyanD/codexffmpeg/releases) -- confirmed live, not guessed, by
# resolving gyan.dev's own "mirror @ github" link. No separate checksums.sha256 asset
# ships on that mirror the way BtbN's release does, so the SHA-256 below was verified
# the same way ADR-301 already established for MKVToolNix: downloaded the real file
# directly and hashed it by hand before pinning. Bump both constants by hand (never
# silently re-chase "latest" again) if a future FFmpeg release is deliberately adopted
# -- re-verify against that regression (or whatever is current) first.
$script:FfmpegVersion = "7.1.1"
$script:FfmpegSha256 = "D760E1B3574402ED18B4865851F87D87E73965A982E6453212DF8621FED1C508"
$script:FfmpegUrl = "https://github.com/GyanD/codexffmpeg/releases/download/$($script:FfmpegVersion)/ffmpeg-$($script:FfmpegVersion)-full_build.zip"

# MKVToolNix, unlike ffmpeg, DOES get a normal pinned-version-and-checksum design
# (matching tsMuxeR/FRIM in iw3/install_mvc_tools.py and iw3/sbs_to_mvc_cli.py) --
# it has real, fixed release versions. Bump these two by hand when adopting a newer
# MKVToolNix (matching setup.ps1's own prior pin-and-comment convention); the SHA-256
# below was verified against this release's own published sha256sums.txt at
# mkvtoolnix.download AND by downloading and hashing the real file directly (see ADR-301,
# which adopted this same 102.0/SHA-256 pin for setup.ps1's own prior, presence-only check).
$script:MkvToolNixVersion = "102.0"
$script:MkvToolNixSha256 = "CD63C42CAEE3D9E5B631E029657B3BD322BEBA77E150CECBFC9AE4923638B049"
$script:MkvToolNixUrl = "https://mkvtoolnix.download/windows/releases/$($script:MkvToolNixVersion)/mkvtoolnix-64-bit-$($script:MkvToolNixVersion).7z"

function Get-File($url, $destination) {
    Write-Host "  Downloading $url"
    Start-BitsTransfer -Source $url -Destination $destination
}

function Install-Ffmpeg {
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$TmpDir
    )

    $ffmpegExe = Join-Path $Root "ffmpeg.exe"
    $ffprobeExe = Join-Path $Root "ffprobe.exe"

    # ADR-373: version-string check, same convention Install-MkvToolNix already uses --
    # no more build-hash marker file (ffmpeg.buildhash.txt, from the old rolling-nightly
    # design) needed now that there's a real, fixed version to compare against directly.
    $installedVersion = $null
    if ((Test-Path $ffmpegExe) -and (Test-Path $ffprobeExe)) {
        try {
            $verLine = (& $ffmpegExe -version 2>$null | Select-Object -First 1)
            if ($verLine -match "ffmpeg version (\S+)") { $installedVersion = $Matches[1] }
        } catch {
            $installedVersion = $null
        }
    }
    if ($installedVersion -eq $script:FfmpegVersion) {
        Write-Host "  ffmpeg is already current (v$installedVersion) -- skipping."
        return
    }
    if ($installedVersion) {
        Write-Host "  ffmpeg v$installedVersion found, expected v$($script:FfmpegVersion) -- updating."
    } else {
        Write-Host "  ffmpeg not found (or its version could not be read) -- installing v$($script:FfmpegVersion)."
    }

    $zipPath = Join-Path $TmpDir "ffmpeg.zip"
    try {
        Get-File $script:FfmpegUrl $zipPath
    } catch {
        if ((Test-Path $ffmpegExe) -and (Test-Path $ffprobeExe)) {
            Write-Host "  Could not download ffmpeg v$($script:FfmpegVersion) ($($_.Exception.Message)) -- keeping the existing one." -ForegroundColor Yellow
            return
        }
        throw "could not download ffmpeg v$($script:FfmpegVersion): $($_.Exception.Message)"
    }
    $actualHash = (Get-FileHash -Path $zipPath -Algorithm SHA256).Hash.ToUpper()
    if ($actualHash -ne $script:FfmpegSha256) {
        throw "downloaded ffmpeg archive did not match the expected checksum (expected $($script:FfmpegSha256), got $actualHash)"
    }

    $extractTmp = Join-Path $TmpDir "ffmpeg-extract"
    if (Test-Path $extractTmp) { Remove-Item $extractTmp -Recurse -Force }
    Expand-Archive -Force $zipPath $extractTmp
    $binDir = Get-ChildItem $extractTmp -Directory | Select-Object -First 1 | ForEach-Object { Join-Path $_.FullName "bin" }
    if (-not $binDir -or -not (Test-Path $binDir)) { throw "unexpected ffmpeg archive layout -- no bin\ folder found under $extractTmp" }

    # Atomic swap (CS-IO-001 / ADR-301 precedent): stage under a .tmp name right next
    # to the real exe, then Move-Item -Force (same-volume rename) over it, so a crash
    # mid-copy can never leave a half-written ffmpeg.exe/ffprobe.exe behind.
    $ffmpegTmp = "$ffmpegExe.tmp"
    $ffprobeTmp = "$ffprobeExe.tmp"
    Copy-Item (Join-Path $binDir "ffmpeg.exe") $ffmpegTmp -Force
    Copy-Item (Join-Path $binDir "ffprobe.exe") $ffprobeTmp -Force
    Move-Item $ffmpegTmp $ffmpegExe -Force
    Move-Item $ffprobeTmp $ffprobeExe -Force
    # Old marker from the pre-ADR-373 rolling-nightly design -- no longer read or
    # written, but remove it so it can't linger and confuse a future reader.
    Remove-Item (Join-Path $Root "ffmpeg.buildhash.txt") -Force -ErrorAction SilentlyContinue

    Write-Host "  ffmpeg + ffprobe v$($script:FfmpegVersion) installed."
}

function Install-MkvToolNix {
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$TmpDir
    )

    $mkvDir = Join-Path $Root "mkvtoolnix"
    $mkvmergeExe = Join-Path $mkvDir "mkvmerge.exe"
    $installedVersion = $null
    if (Test-Path $mkvmergeExe) {
        try {
            $verLine = (& $mkvmergeExe --version 2>$null | Select-Object -First 1)
            if ($verLine -match "mkvmerge v(\S+)") { $installedVersion = $Matches[1] }
        } catch {
            $installedVersion = $null
        }
    }
    if ($installedVersion -eq $script:MkvToolNixVersion) {
        Write-Host "  MKVToolNix is already current (v$installedVersion) -- skipping."
        return
    }
    if ($installedVersion) {
        Write-Host "  MKVToolNix v$installedVersion found, expected v$($script:MkvToolNixVersion) -- updating."
    } else {
        Write-Host "  MKVToolNix not found (or its version could not be read) -- installing v$($script:MkvToolNixVersion)."
    }

    # 7zr.exe is needed to unpack MKVToolNix's official .7z build -- ensured here
    # (downloaded once into $TmpDir if missing) so this function is self-contained
    # and safe to call standalone, not dependent on a caller having fetched it first.
    $sevenZr = Join-Path $TmpDir "7zr.exe"
    if (-not (Test-Path $sevenZr)) {
        Get-File "https://github.com/ip7z/7zip/releases/download/26.03/7zr.exe" $sevenZr
    }

    $mkv7z = Join-Path $TmpDir "mkvtoolnix.7z"
    Get-File $script:MkvToolNixUrl $mkv7z
    $actualHash = (Get-FileHash -Path $mkv7z -Algorithm SHA256).Hash.ToUpper()
    if ($actualHash -ne $script:MkvToolNixSha256) {
        throw "downloaded MKVToolNix archive did not match the expected checksum (expected $($script:MkvToolNixSha256), got $actualHash)"
    }

    $extractTmp = Join-Path $TmpDir "mkvtoolnix-extract"
    if (Test-Path $extractTmp) { Remove-Item $extractTmp -Recurse -Force }
    New-Item -ItemType Directory -Path $extractTmp -Force | Out-Null
    & $sevenZr x $mkv7z "-o$extractTmp" -y | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "7zr.exe failed to extract MKVToolNix (exit code $LASTEXITCODE)" }

    $innerFolder = Get-ChildItem $extractTmp -Directory | Select-Object -First 1
    if (-not $innerFolder) { throw "unexpected MKVToolNix archive layout -- no top-level folder found in $extractTmp" }

    # Preserve any existing cache/ and mkvtoolnix-gui.ini -- runtime-generated /
    # the user's own GUI settings, not part of the distributed package, and would
    # otherwise be lost when the old tree is displaced (ADR-301 precedent).
    if (Test-Path $mkvDir) {
        $cacheDir = Join-Path $mkvDir "cache"
        $guiIni = Join-Path $mkvDir "mkvtoolnix-gui.ini"
        if (Test-Path $cacheDir) { Move-Item $cacheDir (Join-Path $innerFolder.FullName "cache") -Force }
        if (Test-Path $guiIni) { Move-Item $guiIni (Join-Path $innerFolder.FullName "mkvtoolnix-gui.ini") -Force }
    }

    # Atomic-ish swap (CS-IO-001 / ADR-301 precedent): the old tree is moved aside
    # (not deleted -- kept as a manual-delete rollback backup) before the new one is
    # moved into place, rather than a file-by-file copy that could leave a
    # half-replaced multi-file package behind if interrupted partway through.
    if (Test-Path $mkvDir) {
        $oldDir = Join-Path $Root ("mkvtoolnix.old-" + [guid]::NewGuid().ToString("N"))
        Move-Item $mkvDir $oldDir
    }
    Move-Item $innerFolder.FullName $mkvDir

    Write-Host "  MKVToolNix v$($script:MkvToolNixVersion) installed."
}

# When invoked directly (`powershell -File install_media_tools.ps1`), run both
# installs against the real distribution root and exit with a real status code.
# When dot-sourced, $MyInvocation.InvocationName is "." and this block is skipped --
# only the functions above get defined, and the caller (setup.ps1) decides when to
# run them, with its own already-established $root/$tmpDir.
if ($MyInvocation.InvocationName -ne '.') {
    $ErrorActionPreference = "Stop"
    if (-not $MediaRoot) {
        # This script lives at <root>\nunif\windows_package\install_media_tools.ps1 --
        # two levels below the real distribution root, same layout update-3decker.bat's
        # own header comment documents for exactly this reason.
        $MediaRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    }
    $mediaTmpDir = Join-Path $MediaRoot "tmp"
    New-Item -ItemType Directory -Path $mediaTmpDir -Force | Out-Null

    $failed = $false
    Write-Host "==> ffmpeg / ffprobe" -ForegroundColor Cyan
    try {
        Install-Ffmpeg -Root $MediaRoot -TmpDir $mediaTmpDir
    } catch {
        $failed = $true
        Write-Host "  ffmpeg update FAILED -- $($_.Exception.Message)" -ForegroundColor Red
    }

    Write-Host "==> MKVToolNix" -ForegroundColor Cyan
    try {
        Install-MkvToolNix -Root $MediaRoot -TmpDir $mediaTmpDir
    } catch {
        $failed = $true
        Write-Host "  MKVToolNix update FAILED -- $($_.Exception.Message)" -ForegroundColor Red
    }

    if ($failed) { exit 1 } else { exit 0 }
}
