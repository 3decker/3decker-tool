# Bootstrap script for the "3DECKER" nunif fork (https://github.com/3decker/3decker-tool,
# branch my-customizations). Builds a full, self-contained working copy in the folder
# this script is run from: clones the source into nunif\, then downloads everything
# the git repo itself deliberately does NOT contain -- the embedded Python runtime,
# portable Git, ffmpeg, MKVToolNix, dovi_tool, hdr10plus_tool -- installs the pip
# requirements, and downloads the AI model weights. Mirrors the pattern nunif's own
# upstream `nunif\windows_package\update.bat` already uses for Python/Git/pip/models
# (just pointed at this fork's repo instead of upstream's), extended to also cover
# the extra third-party tools this fork's iw3 features depend on (HDR/DV tooling, MKV
# muxing) that upstream's script does not know about.
#
# Usage: save this one file anywhere (an empty folder is recommended -- it will
# create python\, git\, nunif\, mkvtoolnix\, and tools\ as siblings of itself) and
# run it. No prior git clone needed -- this script does that itself. Safe to re-run:
# skips any step whose target already exists, so it can also be used to repair a
# partially-broken environment or pull the latest source update.
#
#   .\setup.ps1
#   .\setup.ps1 -TorchVariant cu126     # default; see requirements-torch-*.txt for alternatives (rocm, xpu)
#   .\setup.ps1 -SkipModels             # skip the (large, slow) AI model download step

param(
    [string]$TorchVariant = "cu126",
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$tmpDir = Join-Path $root "tmp"
$pythonDir = Join-Path $root "python"
$gitDir = Join-Path $root "git"
$nunifDir = Join-Path $root "nunif"
$mkvtoolnixDir = Join-Path $root "mkvtoolnix"

New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Get-File($url, $destination) {
    Write-Host "  Downloading $url"
    Start-BitsTransfer -Source $url -Destination $destination
}



# ---------------------------------------------------------------------------
Write-Step "Embedded Python 3.12.10"

if (Test-Path (Join-Path $pythonDir "python.exe")) {
    Write-Host "  Already present at $pythonDir -- skipping."
} else {
    $pythonUrl = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
    $getPipUrl = "https://bootstrap.pypa.io/get-pip.py"
    $pythonZip = Join-Path $tmpDir "python.zip"

    Get-File $pythonUrl $pythonZip
    Expand-Archive -Force $pythonZip $pythonDir

    # Isolated-mode .pth setup, same as update.bat: makes the embedded interpreter
    # see nunif\ as importable and re-enables `site` (embeddable Python disables it
    # by default, which breaks pip-installed packages).
    $pthPath = Join-Path $pythonDir "python312._pth"
    Add-Content -Path $pthPath -Value "..\nunif"
    Add-Content -Path $pthPath -Value "import site"

    Write-Host "  Installing pip..."
    $getPipPath = Join-Path $pythonDir "get-pip.py"
    Get-File $getPipUrl $getPipPath
    & (Join-Path $pythonDir "python.exe") $getPipPath
    if ($LASTEXITCODE -ne 0) { throw "get-pip.py failed with exit code $LASTEXITCODE" }

    Write-Host "  Python + pip installed."
}
$pythonExe = Join-Path $pythonDir "python.exe"

# ---------------------------------------------------------------------------
Write-Step "MinGit (portable Git)"

if (Test-Path (Join-Path $gitDir "cmd\git.exe")) {
    Write-Host "  Already present at $gitDir -- skipping."
} else {
    $minGitUrl = "https://github.com/git-for-windows/git/releases/download/v2.45.2.windows.1/MinGit-2.45.2-64-bit.zip"
    $minGitZip = Join-Path $tmpDir "mingit.zip"
    Get-File $minGitUrl $minGitZip
    Expand-Archive -Force $minGitZip $gitDir
    Write-Host "  MinGit installed."
}
$gitExe = Join-Path $gitDir "cmd\git.exe"

# ---------------------------------------------------------------------------
Write-Step "3DECKER source (nunif\)"

# This fork's repo root IS the nunif source tree itself (matching upstream
# nagadomi/nunif's own convention) -- mirrors nunif\windows_package\update.bat's
# own `git clone https://github.com/nagadomi/nunif.git "%NUNIF_DIR%"` step,
# just pointed at this fork's URL/branch instead of upstream's.
if (Test-Path (Join-Path $nunifDir ".git")) {
    Write-Host "  Already present at $nunifDir -- pulling latest instead of cloning."
    & $gitExe -C $nunifDir pull --ff
    if ($LASTEXITCODE -ne 0) { throw "git pull failed in $nunifDir (exit $LASTEXITCODE)" }
} elseif (Test-Path $nunifDir) {
    # nunif\ exists but has no .git folder -- e.g. GitHub's "Download ZIP" button
    # strips git history entirely, so a renamed zip extraction lands here. Use it
    # as-is; just can't auto-update via git later without a real clone.
    Write-Host "  Found $nunifDir but it is not a git repository (no .git folder) -- using it as-is."
    Write-Host "  NOTE: without git history, 'Run Update' and future setup.ps1 re-runs cannot pull updates automatically. Delete this nunif\ folder and re-run setup.ps1 in an empty folder for a real git clone if you want that." -ForegroundColor Yellow
} else {
    & $gitExe clone -b my-customizations "https://github.com/3decker/3decker-tool.git" $nunifDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed (exit $LASTEXITCODE)" }
    Write-Host "  Cloned to $nunifDir."
}

# ---------------------------------------------------------------------------
Write-Step "Launcher scripts"

# nunif\windows_package\ holds the templates for these (same convention upstream
# nunif uses) -- copy them out to root so iw3-gui.bat etc. actually exist to run.
# Re-copied every run (even if already present) so an update.bat/update-installer.bat
# change in a newer nunif\ pull is picked up, matching update.bat's own
# `copy /y ... update-installer.bat` behavior.
$launcherFiles = @(
    "install.bat", "update.bat", "update-installer.bat", "setenv.bat",
    "nunif-prompt.bat", "iw3-gui.bat", "iw3-desktop-gui.bat", "iw3-player-gui.bat",
    "waifu2x-gui.bat", "waifu2x-web.bat"
)
$windowsPackageDir = Join-Path $nunifDir "windows_package"
foreach ($f in $launcherFiles) {
    $src = Join-Path $windowsPackageDir $f
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $root $f) -Force
    }
}
Write-Host "  Launcher scripts copied to $root."

# ---------------------------------------------------------------------------
Write-Step "7-Zip console extractor (needed only to unpack MKVToolNix's official .7z build)"

$sevenZr = Join-Path $tmpDir "7zr.exe"
if (-not (Test-Path $sevenZr)) {
    Get-File "https://github.com/ip7z/7zip/releases/download/26.03/7zr.exe" $sevenZr
}

# ---------------------------------------------------------------------------
Write-Step "MKVToolNix (mkvmerge/mkvpropedit)"

if (Test-Path (Join-Path $mkvtoolnixDir "mkvmerge.exe")) {
    Write-Host "  Already present at $mkvtoolnixDir -- skipping."
} else {
    # NOTE: version pinned here (101.0) is the latest confirmed available at the time
    # this script was written, NOT necessarily identical to whatever version was used
    # during this fork's own development (v97.0) -- mkvmerge's CLI surface this project
    # actually uses (-o/--default-duration/--no-video/--no-audio/"+" append, see
    # CS-SUBPROCESS-001 in docs/ai/CODING_STANDARDS.md) has been stable across many
    # releases, so this should work, but if something breaks, try pinning to 97.0 via
    # https://mkvtoolnix.download/windows/releases/97.0/mkvtoolnix-64-bit-97.0.7z instead.
    $mkvtoolnixUrl = "https://mkvtoolnix.download/windows/releases/101.0/mkvtoolnix-64-bit-101.0.7z"
    $mkvtoolnix7z = Join-Path $tmpDir "mkvtoolnix.7z"
    Get-File $mkvtoolnixUrl $mkvtoolnix7z

    $mkvtoolnixExtractTmp = Join-Path $tmpDir "mkvtoolnix-extract"
    New-Item -ItemType Directory -Path $mkvtoolnixExtractTmp -Force | Out-Null
    & $sevenZr x $mkvtoolnix7z "-o$mkvtoolnixExtractTmp" -y | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "7zr.exe failed to extract MKVToolNix (exit code $LASTEXITCODE)" }

    # The .7z contains a single top-level "mkvtoolnix" folder -- flatten it so
    # mkvmerge.exe ends up directly at <root>\mkvtoolnix\mkvmerge.exe, matching
    # _find_mkvmerge()'s expected layout (nunif/iw3/utils.py).
    $innerFolder = Get-ChildItem $mkvtoolnixExtractTmp -Directory | Select-Object -First 1
    if (-not $innerFolder) { throw "Unexpected MKVToolNix archive layout -- no top-level folder found in $mkvtoolnixExtractTmp" }
    Move-Item $innerFolder.FullName $mkvtoolnixDir

    Write-Host "  MKVToolNix installed."
}

# ---------------------------------------------------------------------------
Write-Step "dovi_tool (Dolby Vision RPU extraction)"

$doviToolExe = Join-Path $root "dovi_tool.exe"
if (Test-Path $doviToolExe) {
    Write-Host "  Already present at $doviToolExe -- skipping."
} else {
    $doviToolUrl = "https://github.com/quietvoid/dovi_tool/releases/download/2.3.2/dovi_tool-2.3.2-x86_64-pc-windows-msvc.zip"
    $doviToolZip = Join-Path $tmpDir "dovi_tool.zip"
    Get-File $doviToolUrl $doviToolZip
    Expand-Archive -Force $doviToolZip $tmpDir
    Move-Item (Join-Path $tmpDir "dovi_tool.exe") $doviToolExe
    Write-Host "  dovi_tool installed."
}

# ---------------------------------------------------------------------------
Write-Step "hdr10plus_tool (HDR10+ metadata extraction)"

$hdr10plusToolExe = Join-Path $root "hdr10plus_tool.exe"
if (Test-Path $hdr10plusToolExe) {
    Write-Host "  Already present at $hdr10plusToolExe -- skipping."
} else {
    $hdr10plusToolUrl = "https://github.com/quietvoid/hdr10plus_tool/releases/download/1.7.2/hdr10plus_tool-1.7.2-x86_64-pc-windows-msvc.zip"
    $hdr10plusToolZip = Join-Path $tmpDir "hdr10plus_tool.zip"
    Get-File $hdr10plusToolUrl $hdr10plusToolZip
    Expand-Archive -Force $hdr10plusToolZip $tmpDir
    Move-Item (Join-Path $tmpDir "hdr10plus_tool.exe") $hdr10plusToolExe
    Write-Host "  hdr10plus_tool installed."
}

# ---------------------------------------------------------------------------
Write-Step "ffmpeg / ffprobe"

$ffmpegExe = Join-Path $root "ffmpeg.exe"
if (Test-Path $ffmpegExe) {
    Write-Host "  Already present at $ffmpegExe -- skipping."
} else {
    # BtbN's GitHub-hosted GPL build (.zip -- no 7z needed), not gyan.dev's .7z-only
    # builds this fork's dev environment happened to use. Functionally equivalent
    # (GPL/full build with libx264/libx265/nvenc/nvdec/cuda support) but not a
    # byte-identical build -- if something ffmpeg-specific ever misbehaves, compare
    # against a gyan.dev "release-essentials" build before assuming it's a real bug.
    $ffmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip"
    $ffmpegZip = Join-Path $tmpDir "ffmpeg.zip"
    Get-File $ffmpegUrl $ffmpegZip

    $ffmpegExtractTmp = Join-Path $tmpDir "ffmpeg-extract"
    Expand-Archive -Force $ffmpegZip $ffmpegExtractTmp
    $ffmpegBinDir = Get-ChildItem $ffmpegExtractTmp -Directory | Select-Object -First 1 | ForEach-Object { Join-Path $_.FullName "bin" }
    if (-not $ffmpegBinDir -or -not (Test-Path $ffmpegBinDir)) { throw "Unexpected ffmpeg archive layout -- no bin\ folder found under $ffmpegExtractTmp" }

    Copy-Item (Join-Path $ffmpegBinDir "ffmpeg.exe") $ffmpegExe
    Copy-Item (Join-Path $ffmpegBinDir "ffprobe.exe") (Join-Path $root "ffprobe.exe")
    Write-Host "  ffmpeg + ffprobe installed."
}

# NOTE: Tesseract OCR and BDSup2Sub (+ its bundled JRE, tools\bdsup2sub and
# tools\openjdk) were confirmed unused by any current iw3/waifu2x feature (no
# tesseract.exe, pytesseract dependency, or code reference anywhere in nunif\iw3)
# and removed from the live tree. Deliberately not downloaded here. If a real
# feature ever needs Tesseract, add a step here following the same pattern as
# dovi_tool/hdr10plus_tool above (UB-Mannheim's installer is the usual Windows
# source: https://github.com/UB-Mannheim/tesseract/wiki).

# ---------------------------------------------------------------------------
Write-Step "Python packages (this can take a while)"

$torchReq = Join-Path $nunifDir "requirements-torch-$TorchVariant.txt"
if (-not (Test-Path $torchReq)) {
    throw "No requirements-torch-$TorchVariant.txt found in nunif\ -- valid -TorchVariant values are whatever requirements-torch-*.txt files exist there (cu126, rocm, xpu)."
}

& $pythonExe -m pip install --no-cache-dir --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip self-upgrade failed with exit code $LASTEXITCODE" }

& $pythonExe -m pip install --no-cache-dir --upgrade -r $torchReq
if ($LASTEXITCODE -ne 0) { throw "pip install of $torchReq failed with exit code $LASTEXITCODE" }

& $pythonExe -m pip install --no-cache-dir --upgrade -r (Join-Path $nunifDir "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install of requirements.txt failed with exit code $LASTEXITCODE" }

& $pythonExe -m pip install --no-cache-dir --upgrade -r (Join-Path $nunifDir "requirements-gui.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install of requirements-gui.txt failed with exit code $LASTEXITCODE" }

Write-Host "  All Python packages installed."

# ---------------------------------------------------------------------------
if ($SkipModels) {
    Write-Step "Skipping AI model download (-SkipModels passed)"
} else {
    Write-Step "Downloading AI model weights (large, slow -- this is expected to take a while)"

    Push-Location $nunifDir
    try {
        & $pythonExe -m waifu2x.download_models
        if ($LASTEXITCODE -ne 0) { throw "waifu2x.download_models failed with exit code $LASTEXITCODE" }

        & $pythonExe -m iw3.download_models
        if ($LASTEXITCODE -ne 0) { throw "iw3.download_models failed with exit code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }

    Write-Host "  Models downloaded."
}

# ---------------------------------------------------------------------------
Write-Step "Done"
Write-Host "Setup complete. Launch iw3-gui.bat or waifu2x-gui.bat to get started." -ForegroundColor Green
Write-Host "If you plan to use torch.compile, see nunif\windows_package\docs\torch_compile.md for the additional one-time setup it needs." -ForegroundColor Green
