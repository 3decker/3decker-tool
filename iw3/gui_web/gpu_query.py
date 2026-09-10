"""GPU name lookup for the Device field's dynamic choices.

Deliberately duplicates iw3/gui.py's _query_nvidia_smi_gpu_names() rather
than importing it -- that function lives in gui.py alongside the rest of
the wx GUI, and importing it would pull wx into gui_web's dependency chain,
which this package is otherwise deliberately free of (ADR-081). The safety
property that matters is copied exactly, not just the general idea:
querying nvidia-smi as a separate process, NEVER torch.cuda.get_device_
properties(), so populating this dropdown can never claim PyTorch's CUDA
primary context before pyav_init_cuda_primary_context() gets a chance to
run first. Same real bug this avoids: ADR-034/071/075 (a 100%-reproducible
--hwaccel cuda + --compile crash caused by exactly this ordering being
violated). If gui.py's version ever changes, this one may need updating to
match -- there's no single source of truth between the two GUIs here since
gui_web has no dependency on gui.py at all.
"""
import subprocess


def query_nvidia_smi_gpu_names():
    """Returns a list of CUDA GPU names via nvidia-smi, or None on failure."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if names:
                return names
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def device_choices():
    """Matches iw3/gui.py's cbo_device population order exactly: each real
    CUDA device as "<index>:<name>", then "All CUDA Device", then "CPU"
    last. Falls back to just CPU if nvidia-smi is unavailable."""
    names = query_nvidia_smi_gpu_names()
    choices = []
    if names:
        choices.extend(f"{i}:{name}" for i, name in enumerate(names))
        choices.append("All CUDA Device")
    choices.append("CPU")
    return choices


def device_choice_to_gpu_id(choice):
    """Returns a LIST, matching iw3/gui.py's own real convention exactly
    (gui.py:5586-5591: device_id = int(...) then immediately re-wrapped as
    device_id = [device_id], or list(range(device_count)) for "All CUDA
    Device") -- set_state_args() does args.gpu[0] and
    `for gpu_id in args.gpu`, so a bare scalar breaks it (confirmed by
    direct test: 'int' object is not subscriptable/iterable). "All CUDA
    Device"'s device count comes from the same nvidia-smi name list this
    module already queried to populate the dropdown -- not
    torch.cuda.device_count(), to stay consistent with never touching
    torch.cuda.* here (ADR-034/071/075)."""
    if choice == "CPU":
        return [-1]
    if choice == "All CUDA Device":
        names = query_nvidia_smi_gpu_names() or []
        return list(range(len(names))) or [-1]
    return [int(choice.split(":", 1)[0])]
