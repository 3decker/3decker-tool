#!/bin/bash -e

echo "**** ${0}"

TEST_IMAGE=tests/data/smoke/sd.png
TEST_VIDEO=tests/data/smoke/sd.mkv
TEST_VIDEO_HDR=tests/data/smoke/hdr.mkv
TEST_DIR=tests/data/smoke/
OUTPUT_DIR=tests/data/smoke/iw3

set -x

# base
python -m iw3.cli -y -i ${TEST_IMAGE} -o ${OUTPUT_DIR} --depth-model Any_S --metadata
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --metadata
python -m iw3 -y -i ${TEST_DIR} -o ${OUTPUT_DIR} --depth-model Any_S --metadata

# EMA
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --metadata --ema-normalize --ema-buffer 10

# Batch
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --metadata --batch-size 4 --max-workers 2 --cuda-stream

# Low VRAM
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --metadata --low-vram

# VDA
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model VDA_S --metadata --ema-normalize --ema-buffer 10 --scene-detect --disable-scene-cache

# VDA Stream
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model VDA_Stream_S --metadata --ema-normalize --ema-buffer 10 --scene-detect

# Auto EMA by Scene Length on a regular (non---scene-batch) conversion (see
# docs/ai/AI_DECISIONS.md ADR-060) -- real end-to-end run through the actual
# --scene-detect + --scene-batch-auto-ema combination on the VDA and VDA Stream
# single-pass paths (bind_vda_frame_callback / bind_single_frame_callback).
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model VDA_S --metadata --ema-normalize --ema-buffer 10 --scene-detect --disable-scene-cache --scene-batch-auto-ema --scene-batch-auto-ema-model VDA_L
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model VDA_Stream_S --metadata --ema-normalize --ema-buffer 10 --scene-detect --scene-batch-auto-ema --scene-batch-auto-ema-model VDA_L

# HDR
python -m iw3.cli -y -i ${TEST_VIDEO_HDR} -o ${OUTPUT_DIR} --depth-model Any_S --colorspace auto --video-codec libx265 --pix-fmt yuv420p10le
python -m iw3.cli -y -i ${TEST_VIDEO_HDR} -o ${OUTPUT_DIR} --depth-model VDA_S --colorspace auto --ema-normalize --ema-buffer 10 --scene-detect  --video-codec libx265 --pix-fmt yuv420p10le

# HDR2SDR
python -m iw3.cli -y -i ${TEST_VIDEO_HDR} -o ${OUTPUT_DIR} --depth-model Any_S --colorspace bt709-tv --video-codec libx265 --pix-fmt yuv420p

# sharpen (post-render unsharp-mask pass, see docs/ai/AI_DECISIONS.md ADR-061)
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --metadata --sharpen --sharpen-strength 0.75

# inpaint
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --method forward_inpaint

# splat blend
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --method forward_splat_fill
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model VDA_S --method mlbw_l2_inpaint --ema-normalize --ema-buffer 10 --scene-detect

# Real ticket-lock concurrency combination that deadlocked live (see
# docs/ai/AI_DECISIONS.md ADR-067): forward_splat_fill batch path with multiple
# worker threads + a real batch size > 1, Scene Boundary Detection, and Auto EMA
# by Scene Length all on together -- the exact settings combination that hit the
# bug. A synthetic, GPU-less, forced-interleaving reproduction of the deadlock
# itself lives in tests/test_iw3_ticket_lock_deadlock.py (run below); this line
# is the real GPU smoke coverage for the same combination.
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --method forward_splat_fill --batch-size 2 --max-workers 4 --ema-normalize --ema-buffer 10 --scene-detect --disable-scene-cache --scene-batch-auto-ema --scene-batch-auto-ema-model Nagadomi_Reference

# export
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --export-disparity --export-depth-only --export-depth-fit
python -m iw3.cli -y -i ${TEST_VIDEO_HDR} -o ${OUTPUT_DIR} --depth-model VDA_S --export --ema-normalize --ema-buffer 10 --scene-detect

# vf
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --vf "scale=-2:320,crop=256:256"

# keyframe
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --keyframe

# subtitle mux (standalone post-processing tool, see docs/ai/AI_DECISIONS.md ADR-032)
# Embedded self-test only (synthetic filenames/SRT content, no GPU, no real mkvmerge
# invocation needed) -- mirrors iw3.reinject_hdr_cli's --self-test convention.
python -m iw3.subtitle_mux_cli --self-test

# audio mux / dub track (standalone post-processing tool, see docs/ai/AI_DECISIONS.md
# ADR-044). Embedded self-test only (mocked ffmpeg trim + language-table lookups, no
# GPU, no real ffmpeg/mkvmerge invocation needed) -- same convention as the subtitle
# mux self-test above, which this module imports its ISO 639-1 language table from.
python -m iw3.audio_mux_cli --self-test

# GUI startup must not grab a CUDA context / probe torch.compile before Start is
# clicked (see docs/ai/AI_DECISIONS.md ADR-033). Embedded self-test only (mocked
# pyav_init_cuda_primary_context/check_compile_support, no GPU needed). Also runs
# _self_test_compile_probe_crash_handled (ADR-068): the torch.compile checkbox must
# never crash the GUI with a raw error popup if the real compile probe fails.
python -m iw3.gui --self-test

# torch.compile checkbox crash fix (see docs/ai/AI_DECISIONS.md ADR-068): a real
# OSError [Errno 129] from check_compile_support()'s real torch.compile() probe used
# to propagate uncaught. Synthetic only (torch.compile monkeypatched to raise, no GPU,
# no real compiler toolchain needed) -- covers OSError/RuntimeError/AssertionError/
# ValueError all being caught cleanly, the success path, and per-device caching.
python tests/test_iw3_gui_torch_compile_crash.py

# Check for Updates: read-only git fetch + compare, never pull/merge/reset (see
# docs/ai/AI_DECISIONS.md ADR-035). Embedded self-test only (mocked subprocess.run,
# no network/real repo needed).
python -m iw3.update_check --self-test

# RIFE frame interpolation (see docs/ai/AI_DECISIONS.md ADR-029 and its
# amendment note). Embedded self-test only (synthetic fixture archives standing
# in for the real Google Drive / GitHub downloads, no network/GPU needed) --
# covers the real "model/ package missing" bug found on first real use, plus
# the RIFE multiplier/target-fps scheduling math (ADR-049) and the RIFE frame
# manifest emission (ADR-051, no GPU -- interpolate_frame/VU.to_tensor mocked).
python -m iw3.rife_cli --self-test

# Retroactive Dolby Vision/HDR RPU reinjection, standalone tool (see
# docs/ai/AI_DECISIONS.md ADR-031's strict exact-frame-count gate and ADR-051's
# RIFE-manifest-aware expansion path). Embedded self-test only (mocked
# subprocess.run/ffprobe/dovi_tool throughout, no GPU/network/real dovi_tool
# invocation needed) -- covers both the original strict gate (confirmed
# unaffected by ADR-051) and the new RIFE-manifest duplicate-ops/expansion
# logic, including the same-offset tie-break and simulated scene-cut-boundary
# cases.
python -m iw3.reinject_hdr_cli --self-test

# Filename/comment metadata tag audit (see docs/ai/AI_DECISIONS.md ADR-050).
# Synthetic only (fake args Namespace, mocked subprocess.run/path.exists, no
# GPU/network/real files) -- covers the stereo-aware 4K/8K waifu2x upscale
# derivative-filename gap this closed, a regression guard on the RIFE
# multiplier/target-fps tags found already correct, and confirms
# --pause-frees-vram is correctly excluded from tagging.
python tests/test_iw3_filename_metadata_tags.py

# Post-render Sharpen filter (see docs/ai/AI_DECISIONS.md ADR-061). Synthetic only
# (no GPU, no real movie file) -- covers the detail-aware weighting math (a real
# synthetic edge is sharpened, a flat/grain-noisy region is not), the exact
# strength=0.0 no-op, clipping to the [0, 1] tensor range, the real hook point
# inside apply_divergence() (after Edge Repair, via create_parser()'s grid_sample
# method), and the "only tagged when strength is non-zero" filename/comment
# metadata convention.
python tests/test_iw3_sharpen.py

# Scene-boundary cache staleness/unit-mismatch investigation (see
# docs/ai/AI_DECISIONS.md ADR-059). Synthetic only (temp-dir cache + dummy
# placeholder video file, no GPU/real video decode) -- covers the cancelled-scan
# cache-poisoning gate (should_save_scene_cache), the scene_batch.py pts-to-
# seconds unit fix (resolve_scene_scan_fps), and the exact short-scan-vs-
# full-range-request scenario that triggered this investigation.
python tests/test_iw3_scene_cache_regression.py

# Auto EMA by Scene Length extended to a regular (non---scene-batch) --scene-detect
# conversion (see docs/ai/AI_DECISIONS.md ADR-060), plus its user-visible reporting
# (live log lines, a saved per-scene CSV report, an end-of-run summary -- ADR-062).
# No GPU, no downloaded checkpoint, no network -- a NullDepthModel + synthetic CPU
# tensors stand in for a real depth model. Covers compute_scene_ema_schedule's
# duration math (first scene measured from --start-time, middle scenes, the last
# scene measured to end-of-clip, zero cuts), BaseDepthModel.minmax_normalize's new
# ema_updates param (an update actually reconfigures the EMA scaler; None is an
# exact no-op), an end-to-end simulation confirming three different-length synthetic
# scenes actually get three different Buffer/Decay values, the ADR-057 override file
# being reused as-is, and the off-by-default filename/comment metadata tagging
# (including --scene-batch's own per-scene tagging staying untouched). Also runs one
# real end-to-end conversion (real ffmpeg-generated video, real decode/encode via
# the actual iw3_main/process_video_full pipeline, only the scene-boundary neural
# scan itself mocked) confirming the real saved report/live log/summary all appear
# correctly against real output.
python tests/test_iw3_scene_auto_ema_regular.py

# Existing scene_boundary_cache.py unittest suite (save/load, out-of-range
# rejection, corrupted JSON, mtime-changed invalidation).
python -m iw3.scene_boundary_cache

# Real, confirmed ticket-lock deadlock in bind_batch_frame_callback's
# forward_fill/forward/forward_splat_fill thread-serialization (see
# docs/ai/AI_DECISIONS.md ADR-067), caught live via a real py-spy dump on the
# user's own conversion. No GPU, no real movie file -- forces the exact AB-BA
# lock-ordering interleaving from the real evidence deterministically (via two
# timing-only monkeypatches) against the REAL bind_batch_frame_callback closures
# and REAL TicketLock objects, in a subprocess with a bounded timeout so a
# regression fails cleanly instead of hanging the suite.
python tests/test_iw3_ticket_lock_deadlock.py

# Standalone Sharpen post-processing tool (see docs/ai/AI_DECISIONS.md ADR-063) --
# applies ADR-061's Sharpen filter to an already-converted video without
# re-running depth/stereo conversion. Embedded self-test only (synthetic
# filenames/tensors, mocked mkvmerge/subprocess.run, no GPU/real mkvmerge
# invocation needed) -- covers format auto-detection, the lossless
# split/rejoin round trip, the exact strength=0.0 no-op for every format,
# rgbd/half_rgbd leaving the depth-map half untouched, anaglyph sharpening
# the whole frame unsplit, two-eye seam independence, and run()'s
# pre-processing refusal gates.
python -m iw3.sharpen_cli --self-test
