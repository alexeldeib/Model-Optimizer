#!/usr/bin/env bash
#
# K2.6 NVFP4 multinode-layerwise PTQ launcher.
#
# Differences from the team's existing single-node modelopt path:
#   * Uses BF16 source (k26-source-bf16) instead of INT4 -- avoids the
#     stack of dequant-on-load patches the team carries.
#   * Uses torchrun + FSDP2 across 4x GB200.
#   * Layerwise calibration with shared-FS checkpoint resume so a
#     preempted job continues from the last fully-calibrated layer.
#
# Env (set by the k8s manifest):
#   TARGET_MODEL      Path to BF16 source weights (typically /source/k26-bf16)
#   RECIPE            K2.6-aware NVFP4 experts-only recipe yaml
#   CHECKPOINT_DIR    Shared dir for layerwise per-layer checkpoints
#   EXPORT_DIR        Final NVFP4 unified-HF safetensors output
#   CALIB_DATASET     Calibration dataset name (default: cnn_dailymail)
#   CALIB_SAMPLES     Calibration sample count (default 2048)
#   CALIB_BATCH       Calibration batch size (default 8)
#   CALIB_SEQ_LEN     Calibration seq length (default 2048)
#   NUM_GPUS          Visible GPUs (default: torch.cuda.device_count())

set -euo pipefail

: "${TARGET_MODEL:?set TARGET_MODEL}"
: "${RECIPE:?set RECIPE}"
: "${CHECKPOINT_DIR:?set CHECKPOINT_DIR}"
: "${EXPORT_DIR:?set EXPORT_DIR}"
: "${CALIB_DATASET:=cnn_dailymail}"
: "${CALIB_SAMPLES:=2048}"
: "${CALIB_BATCH:=8}"
: "${CALIB_SEQ_LEN:=2048}"
: "${NUM_GPUS:=$(python -c 'import torch;print(torch.cuda.device_count())')}"

mkdir -p "${CHECKPOINT_DIR}" "${EXPORT_DIR}"

# Patch the K2.6 modeling bug if the source dir is local.
if [[ -d "${TARGET_MODEL}" ]]; then
    /opt/quanty/quanty/scripts/patch-k26-modeling.sh "${TARGET_MODEL}" || true
fi

cd /opt/quanty

LOG="${EXPORT_DIR}/run.log"

# Layerwise calibration + checkpoint dir are enabled by the recipe yaml
# (``layerwise: true`` and ``layerwise_checkpoint_dir`` under the
# ``algorithm`` section), so we just forward --recipe and let
# mtq.quantize dispatch into layerwise_calibrate.  multinode_ptq.py does
# not expose --calib_seq or --layerwise_checkpoint_dir as flags; sequence
# length comes from the dataset preset and the checkpoint dir is recipe-
# scoped.
# --trust_remote_code is omitted: the Kimi-K2.6-DeepseekV3 source dir
# carries only config.json + safetensors, and transformers 5.x has a
# built-in DeepseekV3 implementation that handles the architecture
# natively.  Re-add the flag only if you switch TARGET_MODEL to the
# Kimi-K2.6-BF16 multimodal wrapper (with custom modeling files).
torchrun \
    --nproc-per-node="${NUM_GPUS}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    examples/llm_ptq/multinode_ptq.py \
    --pyt_ckpt_path "${TARGET_MODEL}" \
    --recipe "${RECIPE}" \
    --calib_size "${CALIB_SAMPLES}" \
    --batch_size "${CALIB_BATCH}" \
    --dataset "${CALIB_DATASET}" \
    --export_path "${EXPORT_DIR}" \
    2>&1 | tee "${LOG}"
