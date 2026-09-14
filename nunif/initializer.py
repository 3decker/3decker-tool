import os
import torch
import random
import numpy as np
import secrets
import gc


def disable_image_lib_threads():
    # Disable OpenMP
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['OMP_THREAD_LIMIT'] = '1'

    # Disable ImageMagick's Threading
    os.environ['MAGICK_THREAD_LIMIT'] = '1'
    try:
        from wand.resource import limits
        limits["thread"] = 1
    except ImportError:
        pass

    # Disable OpenCV's Threading/OpenCL
    try:
        import cv2
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except ImportError:
        pass


def set_seed(seed):
    if seed is None or seed < 0:
        seed = secrets.randbelow(1_000_000_000)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def gc_collect():
    # TODO: Calling `gc.collect()` without first calling `gc.set_debug(gc.DEBUG_LEAK)` will cause a deadlock.
    #       Most likely a problem with pyav.
    # gc.set_debug(gc.DEBUG_LEAK)
    gc.collect()

    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "reset"):
        torch._dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # ADR-148: torch.cuda.empty_cache() above only clears the DEVICE memory
        # cache. Pinned ("page-locked") CPU memory -- used throughout iw3's video
        # pipeline for fast GPU transfers (VU.OffloadFrame, one pinned buffer per
        # frame) -- goes through PyTorch's own SEPARATE host-memory caching
        # allocator, which this never touched. Freed pinned buffers stayed cached
        # for reuse instead of being returned to the OS, which Windows keeps
        # counting as GPU "shared" memory against the process indefinitely.
        # Confirmed live: releasing every Python reference to ~700 pinned buffers
        # (matching a real EMA-buffered job) left shared memory completely
        # unchanged at ~22GB until this call ran, which dropped it to 0 instantly.
        # No public torch.cuda equivalent exists as of torch 2.12, so this is a
        # private API, guarded defensively in case a future torch build renames
        # or removes it.
        if hasattr(torch._C, "_host_emptyCache"):
            try:
                torch._C._host_emptyCache()
            except Exception:
                pass
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
