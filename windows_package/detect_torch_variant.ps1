# Prints the correct torch build variant ("cu126" or "cu130") for this machine's
# GPU to stdout, and nothing else -- callers (setup.ps1, and the batch update
# scripts via `powershell -File`) capture that one line directly.
#
# Real bug this exists to prevent (2026-09-22): this exact detection logic
# originally lived ONLY inline in setup.ps1 (fresh installs), so it correctly
# picked cu126 for an older GPU (e.g. a GTX 1060, Pascal, compute capability
# 6.1) on a first-time install -- but update.bat/update-3decker.bat/
# update-nagadomi.bat (the path an EXISTING install actually takes) had no such
# logic at all and just blindly reinstalled the plain, hardcoded
# requirements-torch.txt (pinned to cu130) on every update, regardless of GPU.
# A real user's working GTX 1060 install broke this way: cu130-tagged PyTorch
# wheels don't include Pascal kernel images at all (CUDA 13.0 dropped
# Maxwell/Pascal/Volta support -- Turing and newer only), so it failed
# immediately with "CUDA error: no kernel image is available for execution on
# the device" the next time they ran an update, on a setup that had been
# working fine before. Centralizing the detection here, called from every
# entry point (setup.ps1 AND all three update scripts) instead of duplicated
# inline in just one of them, is what actually prevents that class of drift.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File detect_torch_variant.ps1
# Exit code is always 0 -- this never fails the caller; an undetectable GPU is
# not an error, it's just resolved to the safe cu126 default (see below).

$ErrorActionPreference = "Stop"

# Default: cu126 -- broader minimum-driver-version compatibility, works on
# every older NVIDIA generation (Turing/Ampere/Ada/Hopper) and is the safe
# choice whenever detection is inconclusive (no nvidia-smi, unparsable output,
# non-NVIDIA GPU). Only overridden to cu130 for a confirmed Blackwell-or-newer
# card (compute capability >= 12.0), which cu126-tagged wheels cannot run on
# at all ("CUDA error: no kernel image is available for execution on the
# device" -- confirmed real, see docs/ai/AI_DECISIONS.md).
$variant = "cu126"

$nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    try {
        $computeCap = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Trim()
        if ($computeCap -and ([double]$computeCap -ge 12.0)) {
            $variant = "cu130"
        }
    } catch {
        # Leave $variant at the cu126 default -- unparsable nvidia-smi output
        # is not this script's problem to solve, just something to fall back
        # from safely.
    }
}

Write-Output $variant
