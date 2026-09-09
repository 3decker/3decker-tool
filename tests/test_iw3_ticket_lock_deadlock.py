"""Regression test for a real, confirmed deadlock in
iw3.utils.bind_batch_frame_callback's --method forward_fill/forward/
forward_splat_fill thread-serialization, caught live via a real `py-spy dump` on
the user's own running conversion (Any_V3_Mono_01, forward_splat_fill, Divergence
2.5, Auto EMA by Scene Length / Nagadomi_Reference, Scene Boundary Detection,
Depth Batch Size 2, Worker Threads 4 -- stalled at 0/7193 frames). See
docs/ai/AI_DECISIONS.md ADR-067.

**Root cause:** `_postprocess()` and `_batch_infer()` (both closures inside
`bind_batch_frame_callback`) acquired the module's two `nunif.utils.ticket_lock.
TicketLock` instances (`enqueue_ticket_lock`, `dequeue_ticket_lock`) in OPPOSITE
orders:
  - `_postprocess()` already holds `dequeue_ticket_lock`'s raw lock for its ENTIRE
    call (from `with dequeue_ticket_lock(dequeue_ticket_id):` at function entry --
    TicketLock's own `_wait()` leaves the underlying `threading.Condition`'s lock
    held, not released, once it's that ticket's turn), then for
    forward_fill/forward/forward_splat_fill it ALSO raw-acquired
    `enqueue_ticket_lock` (`with enqueue_ticket_lock, dequeue_ticket_lock,
    depth_lock:`) to guard `ordered_index_copy()`'s global
    `torch.use_deterministic_algorithms` flag flip.
  - `_batch_infer()` acquires `enqueue_ticket_lock`'s ticket first (`with
    enqueue_ticket_lock(enqueue_ticket_id):`), then, still holding it, calls
    `dequeue_ticket_lock.new_ticket()` to mint its corresponding dequeue ticket.

Thread A (running `_postprocess` for batch N) holding `dequeue_ticket_lock` and
wanting `enqueue_ticket_lock`, while thread B (running `_batch_infer` for batch
N+1) holds `enqueue_ticket_lock` and wants `dequeue_ticket_lock`, is a classic
AB-BA lock-ordering deadlock -- confirmed live via py-spy (`ticket_lock.py:36`
inside `_postprocess`, `ticket_lock.py:11` inside `_batch_infer`), and
independently reproduced here, in-process, by forcing that exact interleaving
with two timing-only monkeypatches (no locking/business logic changed by them).

**Fix:** `_postprocess()` no longer touches `enqueue_ticket_lock` at all.
`depth_lock` alone is sufficient to guard the deterministic-flag flip: no
concurrent `_postprocess` call is possible anyway (already excluded by holding
`dequeue_ticket_lock` for the whole call), and `_batch_infer()`'s only
CUDA-visible work (`depth_model.infer()`) already requires the SAME `depth_lock`.

Synthetic/isolated (CS-TEST-001): no GPU, no real movie file, no network. The
forced-interleave check below runs the REAL `bind_batch_frame_callback` closures
(`_cuda_stream_wrapper`/`_preprocess`) with a REAL `NullDepthModel` and REAL
`TicketLock` objects, in a subprocess with a bounded timeout -- a regression here
means the subprocess hangs (deadlock), never that it returns wrong data, so it
must run out-of-process to fail cleanly instead of hanging the whole suite.

Run directly: python tests/test_iw3_ticket_lock_deadlock.py
"""
import os
import subprocess
import sys
import threading
import time
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

_WORKER_MARKER = "--worker"


def _worker_main():
    """Runs inside the subprocess. Forces thread A (_postprocess, holding
    dequeue_ticket_lock) and thread B (_batch_infer, holding enqueue_ticket_lock)
    into the exact interleaving from the real py-spy evidence, deterministically
    (not relying on timing luck), then proves both finish."""
    import torch
    from iw3.utils import create_parser, bind_batch_frame_callback
    from iw3.null_depth_model import NullDepthModel
    from nunif.utils.ticket_lock import TicketLock

    args = create_parser(required_true=False).parse_args([])
    args.method = "forward_splat_fill"
    args.batch_size = 1
    args.cuda_stream = False
    args.convergence = 0.5
    args.divergence = 2.0
    args.mapper = "none"
    args.state = {"device": torch.device("cpu"), "convergence_model": None}

    depth_model = NullDepthModel("NULL")
    depth_model.load(gpu=-1, resolution=8)

    frame_callback, preprocess_callback = bind_batch_frame_callback(
        depth_model=depth_model, side_model=None, segment_pts=set(), args=args)

    xa = torch.rand(1, 3, 8, 8)
    xb = torch.rand(1, 3, 8, 8)
    # Mint enqueue tickets in strict submission order, exactly like
    # FrameCallbackPool.submit() calling preprocess_callback synchronously before
    # dispatching each batch to the thread pool.
    args_a = preprocess_callback(xa, [0], False)
    args_b = preprocess_callback(xb, [1], False)

    b_about_to_mint = threading.Event()
    a_ready = threading.Event()

    orig_new_ticket = TicketLock.new_ticket

    def patched_new_ticket(self):
        # Thread B (_batch_infer) is about to try to raw-acquire dequeue_ticket_lock
        # via new_ticket() -- signal thread A, then give it time to actually be
        # blocked on the real acquire below before we call it for real.
        b_about_to_mint.set()
        time.sleep(0.2)
        return orig_new_ticket(self)

    TicketLock.new_ticket = patched_new_ticket

    orig_minmax_normalize = depth_model.minmax_normalize

    def patched_minmax_normalize(depth, reset_ema=None, ema_updates=None):
        # Thread A (_postprocess) already holds dequeue_ticket_lock's raw lock at
        # this point (from function entry) -- wait for thread B to be genuinely
        # blocked trying to mint a dequeue ticket before proceeding into the
        # section that (pre-fix) also grabbed enqueue_ticket_lock.
        a_ready.set()
        b_about_to_mint.wait(timeout=5)
        return orig_minmax_normalize(depth, reset_ema=reset_ema, ema_updates=ema_updates)

    depth_model.minmax_normalize = patched_minmax_normalize

    results = {}

    def run(name, preprocess_args):
        results[name] = frame_callback(preprocess_args)

    thread_a = threading.Thread(target=run, args=("A", args_a), daemon=True)
    thread_b = threading.Thread(target=run, args=("B", args_b), daemon=True)
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=10)
    thread_b.join(timeout=10)

    if thread_a.is_alive() or thread_b.is_alive():
        print("WORKER: DEADLOCK", flush=True)
        os._exit(1)

    assert len(results.get("A", [])) == 1, results.get("A")
    assert len(results.get("B", [])) == 1, results.get("B")
    print("WORKER: PASS", flush=True)
    os._exit(0)


def _test_forced_interleave_no_deadlock():
    proc = subprocess.Popen(
        [sys.executable, path.abspath(__file__), _WORKER_MARKER],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace")
    try:
        out, _ = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=10)
        raise AssertionError(
            "ticket-lock deadlock reproduced: _postprocess()/_batch_infer() "
            "never returned under the forced AB-BA interleaving (see "
            "docs/ai/AI_DECISIONS.md ADR-067)")

    assert proc.returncode == 0, f"worker failed (code={proc.returncode}):\n{out}"
    assert "WORKER: PASS" in out, out

    print("_test_forced_interleave_no_deadlock: PASS")


def main():
    _test_forced_interleave_no_deadlock()
    print("ALL PASS")


if __name__ == "__main__":
    if _WORKER_MARKER in sys.argv:
        _worker_main()
    else:
        main()
