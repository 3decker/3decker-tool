# Adds the Python "libs\" and "include\" folders to the embedded Python runtime.
#
# Why: the official embeddable Python zip (python-3.12.10-embed-amd64.zip) that
# setup.ps1 and update.bat download does NOT contain either folder. Triton (used by the
# MoGe-3 depth model through flex_gemm on Windows) compiles a small C helper with its
# bundled tcc compiler, which needs Python.h (python\include) and python312.lib
# (python\libs). Without them the compile fails with "Failed to find Python libs".
#
# Source: the official Python 3.12.10 NuGet package (published by the Python Software
# Foundation). Its tools\include and tools\libs folders are copied out. Only those two
# folders are taken from it -- the interpreter itself stays the embeddable zip.
#
# Idempotent: if python\libs\python312.lib and python\include\Python.h both already
# exist, nothing is downloaded. Safe to run on every update.
#
# Usage:
#   Dot-sourced (setup.ps1's convention -- runs in-process, the caller then calls
#   Install-PythonDevFiles with its own $pythonDir/$tmpDir):
#       . (Join-Path $nunifDir "windows_package\install_python_dev_files.ps1")
#       Install-PythonDevFiles -PythonDir $pythonDir -TmpDir $tmpDir
#
#   Run directly (update.bat / update-3decker.bat's convention -- same
#   `powershell -File` pattern as install_media_tools.ps1):
#       powershell -NoProfile -ExecutionPolicy Bypass -File install_python_dev_files.ps1
#   Runs against the real distribution root (two levels above this script's folder,
#   unless -PythonFolder overrides it) and exits 1 on failure. Callers do NOT treat that
#   as fatal -- the MoGe-3 depth model is optional, so a failed download must never fail
#   the whole install or update (same non-fatal convention as install_media_tools.ps1).
#
# NOTE on the -PythonFolder param name: deliberately NOT named $PythonDir/$TmpDir.
# Dot-sourcing runs this param block in the caller's scope and PowerShell variable names
# are case-insensitive, so a matching name would clobber setup.ps1's own $pythonDir.
param(
    [string]$PythonFolder
)

# Python 3.12.10 NuGet package. SHA-512 is the packageHash NuGet publishes for this exact
# version in its catalog (https://api.nuget.org/v3/catalog0/data/2025.04.08.14.01.15/python.3.12.10.json),
# checked against a real download on 2026-10-06. Bump by hand together with setup.ps1's
# and update.bat's embeddable zip version (currently 3.12.10).
# Flat-container URL (direct file, no redirect): BITS fails with HTTP 404 on the
# www.nuget.org/api/v2 redirect URL, and the flat container serves the identical .nupkg.
$script:PythonNuGetUrl = "https://api.nuget.org/v3-flatcontainer/python/3.12.10/python.3.12.10.nupkg"
$script:PythonNuGetSha512 = "BBDA4DCF688A94211B62D50968A91B38F305D0B8D1ECD90269F74A86F8A0A4FCEBB7CA162A0753A47691EB3DF0C964009BD3D8194C6FD19AFAE8D5FD01E1CC0F"

function Install-PythonDevFiles {
    param(
        [string]$PythonDir,
        [string]$TmpDir
    )

    $libDir = Join-Path $PythonDir "libs"
    $includeDir = Join-Path $PythonDir "include"
    if ((Test-Path (Join-Path $libDir "python312.lib")) -and (Test-Path (Join-Path $includeDir "Python.h"))) {
        Write-Host "  Python libs and include folders already present -- skipping."
        return
    }

    New-Item -ItemType Directory -Path $TmpDir -Force | Out-Null
    $nupkg = Join-Path $TmpDir "python-nuget.zip"
    $extractDir = Join-Path $TmpDir "python-nuget-extract"

    Write-Host "  Downloading $($script:PythonNuGetUrl)"
    Start-BitsTransfer -Source $script:PythonNuGetUrl -Destination $nupkg

    $actualSha512 = (Get-FileHash -Path $nupkg -Algorithm SHA512).Hash
    if ($actualSha512 -ne $script:PythonNuGetSha512) {
        Remove-Item $nupkg -Force -ErrorAction SilentlyContinue
        throw "SHA-512 mismatch for the Python NuGet package (does not match the pinned hash in install_python_dev_files.ps1) -- download discarded."
    }

    if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force }
    Expand-Archive -Force $nupkg $extractDir

    $srcInclude = Join-Path $extractDir "tools\include"
    $srcLibs = Join-Path $extractDir "tools\libs"
    if (-not (Test-Path $srcInclude) -or -not (Test-Path $srcLibs)) {
        throw "Unexpected Python NuGet package layout -- tools\include or tools\libs missing under $extractDir"
    }

    New-Item -ItemType Directory -Path $includeDir -Force | Out-Null
    New-Item -ItemType Directory -Path $libDir -Force | Out-Null
    Copy-Item -Path (Join-Path $srcInclude "*") -Destination $includeDir -Recurse -Force
    Copy-Item -Path (Join-Path $srcLibs "*") -Destination $libDir -Recurse -Force

    Remove-Item $extractDir -Recurse -Force
    Remove-Item $nupkg -Force

    if (-not (Test-Path (Join-Path $libDir "python312.lib")) -or -not (Test-Path (Join-Path $includeDir "Python.h"))) {
        throw "Copy finished but python312.lib or include\Python.h is still missing under $PythonDir."
    }
    Write-Host "  Python libs and include folders installed."
}

# When dot-sourced, $MyInvocation.InvocationName is "." and this block is skipped --
# only the function above is defined and the caller decides when to run it.
if ($MyInvocation.InvocationName -ne '.') {
    $ErrorActionPreference = "Stop"
    if (-not $PythonFolder) {
        # This script lives at <root>\nunif\windows_package\ -- two levels below the
        # real distribution root, same layout install_media_tools.ps1 documents.
        $PythonFolder = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) "python"
    }
    $devTmpDir = Join-Path (Split-Path -Parent $PythonFolder) "tmp"

    try {
        Install-PythonDevFiles -PythonDir $PythonFolder -TmpDir $devTmpDir
        exit 0
    } catch {
        Write-Host "  Python libs/include update FAILED -- $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}
