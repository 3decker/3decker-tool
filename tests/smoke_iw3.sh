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

# HDR
python -m iw3.cli -y -i ${TEST_VIDEO_HDR} -o ${OUTPUT_DIR} --depth-model Any_S --colorspace auto --video-codec libx265 --pix-fmt yuv420p10le
python -m iw3.cli -y -i ${TEST_VIDEO_HDR} -o ${OUTPUT_DIR} --depth-model VDA_S --colorspace auto --ema-normalize --ema-buffer 10 --scene-detect  --video-codec libx265 --pix-fmt yuv420p10le

# HDR2SDR
python -m iw3.cli -y -i ${TEST_VIDEO_HDR} -o ${OUTPUT_DIR} --depth-model Any_S --colorspace bt709-tv --video-codec libx265 --pix-fmt yuv420p

# inpaint
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --method forward_inpaint

# splat blend
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model Any_S --method forward_splat_fill
python -m iw3.cli -y -i ${TEST_VIDEO} -o ${OUTPUT_DIR} --depth-model VDA_S --method mlbw_l2_inpaint --ema-normalize --ema-buffer 10 --scene-detect

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
# pyav_init_cuda_primary_context/check_compile_support, no GPU needed).
python -m iw3.gui --self-test

# Check for Updates: read-only git fetch + compare, never pull/merge/reset (see
# docs/ai/AI_DECISIONS.md ADR-035). Embedded self-test only (mocked subprocess.run,
# no network/real repo needed).
python -m iw3.update_check --self-test

# RIFE frame interpolation (see docs/ai/AI_DECISIONS.md ADR-029 and its
# amendment note). Embedded self-test only (synthetic fixture archives standing
# in for the real Google Drive / GitHub downloads, no network/GPU needed) --
# covers the real "model/ package missing" bug found on first real use.
python -m iw3.rife_cli --self-test
