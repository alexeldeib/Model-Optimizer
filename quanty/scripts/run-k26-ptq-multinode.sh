#!/usr/bin/env bash
#
# K2.6 NVFP4 multinode-layerwise PTQ launcher (multi-node).
#
# Differences from run-k26-ptq.sh (single-node):
#   * torchrun rendezvous uses MASTER_ADDR / MASTER_PORT pointing at the
#     index-0 pod via headless service DNS (Indexed Job + subdomain).
#   * NNODES > 1 -- 4 GPUs per GB200 node.  K2.6 generally needs at
#     least 4 nodes for BF16 layerwise PTQ headroom; 5-6 nodes are
#     preferred when available.
#
# Env (set by the k8s manifest):
#   TARGET_MODEL      Path to BF16 source weights (typically /source/k26-bf16/...)
#   QFORMAT           ModelOpt qformat/export identity (default nvfp4_mlp_only)
#   RECIPE            Optional PTQ recipe yaml for the main quantization config.
#   RUN_ID            Unique artifact id shared by all ranks
#   CHECKPOINT_DIR    Shared run workspace for scratch files and diagnostics
#                     (default /work/quanty/checkpoints/$RUN_ID)
#   EXPORT_DIR        Final NVFP4 unified-HF safetensors output
#                     (default /work/quanty/exports/$RUN_ID)
#   CALIB_DATASET     Calibration dataset name
#   CALIB_SAMPLES     Calibration sample count
#   CALIB_BATCH       Calibration batch size
#   CALIB_SEQ_LEN     Calibration seq length
#   KV_CACHE_QFORMAT  KV cache quantization format (default fp8)
#   NUM_GPUS          Per-node GPU count (default 4)
#   NNODES            Total nodes (set by manifest from parallelism)
#   MASTER_ADDR       Headless DNS for index-0 pod
#   MASTER_PORT       Rendezvous port (default 29500)
#   JOB_COMPLETION_INDEX  k8s-injected node rank, set automatically by
#                         Indexed Jobs since 1.24
#
# Required script ConfigMap keys:
#   * run-k26-ptq-multinode.sh
#   * patch-k26-modeling.sh
#   * inspect-nvfp4-export.py

set -euo pipefail

: "${TARGET_MODEL:?set TARGET_MODEL}"
: "${QFORMAT:=nvfp4_mlp_only}"
: "${RECIPE:=}"
: "${RUN_ID:?set RUN_ID to a unique shared artifact id}"
if [[ "${RUN_ID}" == REPLACE_WITH* ]]; then
    echo "RUN_ID must be replaced with a unique artifact id before launching" >&2
    exit 64
fi
: "${CHECKPOINT_DIR:=/work/quanty/checkpoints/${RUN_ID}}"
: "${EXPORT_DIR:=/work/quanty/exports/${RUN_ID}}"
: "${CALIB_DATASET:=cnn_dailymail}"
: "${CALIB_SAMPLES:=2048}"
: "${CALIB_BATCH:=8}"
: "${CALIB_SEQ_LEN:=2048}"
: "${KV_CACHE_QFORMAT:=fp8}"
: "${NUM_GPUS:=4}"
: "${NNODES:?set NNODES (total node count)}"
: "${MASTER_ADDR:?set MASTER_ADDR (index-0 headless DNS)}"
: "${MASTER_PORT:=29500}"
: "${JOB_COMPLETION_INDEX:?set JOB_COMPLETION_INDEX (k8s injects on Indexed Jobs)}"
: "${S3_DEST_PREFIX:=s3://infr/test/ace/quants/${RUN_ID}}"

NODE_RANK="${JOB_COMPLETION_INDEX}"

echo "============================================================"
echo "K2.6 multinode PTQ rank=${NODE_RANK}/${NNODES}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  per-node gpus: ${NUM_GPUS}"
echo "============================================================"

# Wait for master DNS + TCP listener before launching torchrun.
# Headless service publishes pod-0's A record as soon as the pod
# exists (publishNotReadyAddresses=true), but the master's TCPStore
# isn't bound until rank-0's torchrun starts.  Ranks >= 1 retry-
# connect with a budget; once master is reachable, torchrun's own
# TCPStore client takes over.  Caps the total wait at 8 minutes;
# beyond that something is structurally wrong (DNS, firewall,
# image-pull skew) and failing fast surfaces it.
if [[ "${NODE_RANK}" != "0" ]]; then
    echo "rank ${NODE_RANK}: waiting for master TCPStore at ${MASTER_ADDR}:${MASTER_PORT}"
    deadline=$(( SECONDS + 480 ))
    until python -c "import socket,sys; s=socket.create_connection(('${MASTER_ADDR}',${MASTER_PORT}),timeout=5); s.close()" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
            echo "rank ${NODE_RANK}: timed out waiting for master after 480s" >&2
            exit 1
        fi
        sleep 5
    done
    echo "rank ${NODE_RANK}: master reachable, launching torchrun"
fi

mkdir -p "${CHECKPOINT_DIR}" "${EXPORT_DIR}"

# Rank-0 owns the writable scratch dir + modeling patch.  Other ranks
# wait for the patched files to appear, then read them.  This avoids
# concurrent sed mutations of tokenization_kimi.py / modeling_kimi*.py.
SCRATCH="${CHECKPOINT_DIR%/*}/k26-source-scratch-$(basename "${CHECKPOINT_DIR}")"
SCRATCH_READY="${SCRATCH}/.scratch-ready"

if [[ -d "${TARGET_MODEL}" ]]; then
    if [[ "${NODE_RANK}" == "0" ]]; then
        mkdir -p "${SCRATCH}"
        rm -f "${SCRATCH_READY}"
        find "${TARGET_MODEL}" -maxdepth 1 -type f -name "*.safetensors" \
            -exec ln -sf {} "${SCRATCH}/" \;
        find "${TARGET_MODEL}" -maxdepth 1 -type f \
            ! -name "*.safetensors" -exec cp -f {} "${SCRATCH}/" \;
        /opt/quanty/quanty/scripts/patch-k26-modeling.sh "${SCRATCH}"
        SCRATCH_KEY=$(basename "${SCRATCH}" | sed 's/-/_hyphen_/g')
        rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/modules/transformers_modules/${SCRATCH_KEY}"
        tmp_ready="${SCRATCH_READY}.$$"
        : > "${tmp_ready}"
        mv "${tmp_ready}" "${SCRATCH_READY}"
    else
        echo "rank ${NODE_RANK}: waiting for rank-0 to publish patched scratch dir"
        until [[ -f "${SCRATCH_READY}" ]]; do
            sleep 5
        done
        # Rank>=1 still needs to bust its own per-pod transformers module
        # cache because each pod has its own /root/.cache.  HF_HOME on
        # the shared PVC is not used as the modules-cache root in this
        # image.
        SCRATCH_KEY=$(basename "${SCRATCH}" | sed 's/-/_hyphen_/g')
        rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/modules/transformers_modules/${SCRATCH_KEY}"
    fi
    TARGET_MODEL="${SCRATCH}"
fi

# Avoid a transformers trust_remote_code cache race.  Without this,
# the 24 torchrun workers can import tokenization_kimi.py while another
# worker is still populating the shared dynamic-module cache, producing
# intermittent "module ... has no attribute TikTokenTokenizer" failures.
# Use a per-node modules cache and prewarm the tokenizer/config/model code
# once in the parent shell before torchrun forks local workers.
SCRATCH_KEY=$(basename "${TARGET_MODEL}" | sed 's/-/_hyphen_/g')
export TARGET_MODEL
export PYTHONPATH="/opt/quanty:${PYTHONPATH:-}"
export HF_MODULES_CACHE="${CHECKPOINT_DIR%/*}/hf-modules-node-${NODE_RANK}"
rm -rf "${HF_MODULES_CACHE}/transformers_modules/${SCRATCH_KEY}"
echo "rank ${NODE_RANK}: prewarming transformers modules in ${HF_MODULES_CACHE}"
python - <<'PY'
import os
import torch
from accelerate import init_empty_weights
from modelopt.torch.quantization.plugins.huggingface import patch_compressed_linear_loading
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

target = os.environ["TARGET_MODEL"]
config = AutoConfig.from_pretrained(target, trust_remote_code=True)
AutoTokenizer.from_pretrained(target, trust_remote_code=True)
model_dtype = getattr(config, "torch_dtype", None)
if model_dtype is None or model_dtype == "auto":
    model_dtype = torch.bfloat16
elif isinstance(model_dtype, str):
    model_dtype = getattr(torch, model_dtype)
with init_empty_weights():
    with patch_compressed_linear_loading():
        AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            dtype=model_dtype,
        )
PY

cd /opt/quanty

# Persist enough source identity in the rank logs to reproduce or reject an
# artifact produced by ConfigMap overlays.
echo "rank ${NODE_RANK}: runtime source checksums"
for f in \
    /opt/quanty/modelopt/torch/quantization/model_calib.py \
    /opt/quanty/modelopt/torch/export/layer_utils.py \
    /opt/quanty/modelopt/torch/export/unified_export_hf.py \
    /opt/quanty/modelopt/torch/export/quant_utils.py \
    /opt/quanty/modelopt/torch/quantization/plugins/huggingface.py \
    /opt/quanty/modelopt/torch/quantization/nn/modules/tensor_quantizer.py \
    /opt/quanty/modelopt/torch/quantization/qtensor/nvfp4_tensor.py \
    /opt/quanty/modelopt/torch/quantization/utils/core_utils.py \
    /opt/quanty/modelopt/torch/quantization/utils/layerwise_calib.py \
    /opt/quanty/examples/llm_ptq/multinode_ptq.py \
    /opt/quanty/quanty/scripts/run-k26-ptq-multinode.sh \
    /opt/quanty/quanty/scripts/patch-k26-modeling.sh \
    /opt/quanty/quanty/scripts/inspect-nvfp4-export.py; do
    if [[ ! -f "${f}" ]]; then
        echo "missing required runtime file: ${f}" >&2
        exit 1
    fi
    sha256sum "${f}"
done
if [[ -n "${RECIPE}" ]]; then
    if [[ ! -f "${RECIPE}" ]]; then
        echo "missing required recipe file: ${RECIPE}" >&2
        exit 1
    fi
    sha256sum "${RECIPE}"
fi

# Stamp the per-node log so we can grep for which rank failed without
# digging through pod uids.
LOG="${EXPORT_DIR}/run.rank${NODE_RANK}.log"

EXTRA_PTQ_ARGS=()
if [[ "${REQUIRE_MOE_COVERAGE:-0}" == "1" ]]; then
    EXTRA_PTQ_ARGS+=(--require_moe_coverage)
fi

PTQ_CONFIG_ARGS=(--qformat "${QFORMAT}")
if [[ -n "${RECIPE}" ]]; then
    PTQ_CONFIG_ARGS+=(--recipe "${RECIPE}")
fi

# torchrun multi-node *static* rendezvous:
#   * --master-addr / --master-port pin the TCPStore on rank-0
#   * --node-rank from JOB_COMPLETION_INDEX
#   * No --rdzv-id / --rdzv-backend: dynamic c10d rendezvous needed
#     a per-job ID consistent across all pods, but the launcher's
#     ${RANDOM} fallback evaluated independently in each pod's
#     shell, so all 4 ranks landed in different rdzv groups and
#     timed out.  Static rendezvous removes the entire rdzv-ID
#     coordination class of bugs; with Indexed Jobs we already
#     have unambiguous node ranks.
torchrun \
    --nnodes="${NNODES}" \
    --nproc-per-node="${NUM_GPUS}" \
    --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    examples/llm_ptq/multinode_ptq.py \
    --pyt_ckpt_path "${TARGET_MODEL}" \
    "${PTQ_CONFIG_ARGS[@]}" \
    --calib_size "${CALIB_SAMPLES}" \
    --calib_seq "${CALIB_SEQ_LEN}" \
    --batch_size "${CALIB_BATCH}" \
    --kv_cache_qformat "${KV_CACHE_QFORMAT}" \
    --dataset "${CALIB_DATASET}" \
    --export_path "${EXPORT_DIR}" \
    --trust_remote_code \
    "${EXTRA_PTQ_ARGS[@]}" \
    2>&1 | tee "${LOG}"

# Rank-0 only uploads the export to s3://infr/test/ace/quants/<run-id>/.
# All ranks complete torchrun; only rank-0 has the gathered, packed
# weights at ${EXPORT_DIR}.  S3 creds + endpoint are injected via the
# quanty-s3-creds k8s secret.
if [[ "${NODE_RANK}" == "0" && -n "${S3_DEST_PREFIX:-}" && -d "${EXPORT_DIR}" ]]; then
    INSPECTOR="/opt/quanty/quanty/scripts/inspect-nvfp4-export.py"
    if [[ ! -f "${INSPECTOR}" ]]; then
        echo "rank 0: missing ${INSPECTOR}; refusing to upload uninspected export" >&2
        exit 1
    fi

    echo "rank 0: inspecting ${EXPORT_DIR} before upload"
    python3 "${INSPECTOR}" \
        --model-dir "${EXPORT_DIR}" \
        --out-dir "${EXPORT_DIR}/diagnostics/export-inspect" \
        --layers auto \
        --expected-experts 384 \
        --expected-moe-layers 60 \
        --expected-kv-cache "${KV_CACHE_QFORMAT}" \
        --allow-shared-experts \
        --allow-layer0-mlp \
        --fail-on-warnings \
        --model-label "${RUN_ID}"

    echo "rank 0: uploading ${EXPORT_DIR} to ${S3_DEST_PREFIX}"
    pip install --quiet boto3 2>&1 | tail -2 || true
    python3 - <<PYEOF
import os
import pathlib
import boto3
from botocore.config import Config

src = pathlib.Path("${EXPORT_DIR}")
dest = "${S3_DEST_PREFIX}".rstrip("/")
# Parse bucket + key prefix from s3://bucket/prefix
assert dest.startswith("s3://"), f"S3_DEST_PREFIX must be s3://...; got {dest}"
bucket, _, prefix = dest[len("s3://"):].partition("/")

session = boto3.session.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_REGION", "US-EAST-04"),
)
# CW's S3-compatible endpoint (cwlota.com / cwobject.com) requires
# virtual-hosted-style addressing -- "bucket.endpoint/key" -- and rejects
# the path-style "endpoint/bucket/key" form with PathStyleRequestNotAllowed.
# boto3's default for non-AWS endpoints is path-style, so override here.
# Without this we hit:
#   boto3.exceptions.S3UploadFailedError: ... PathStyleRequestNotAllowed:
#   The path style requests are not allowed for this method, please switch
#   to hostname-based requests.
s3 = session.client(
    "s3",
    endpoint_url=os.environ["AWS_ENDPOINT_URL"],
    config=Config(s3={"addressing_style": "virtual"}),
)

uploaded = 0
total_bytes = 0
for path in sorted(src.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(src).as_posix()
    if any(part.startswith(".") for part in pathlib.PurePosixPath(rel).parts):
        print(f"  skipping hidden export file {rel}", flush=True)
        continue
    key = f"{prefix}/{rel}" if prefix else rel
    size = path.stat().st_size
    print(f"  uploading {rel} ({size / 1024**2:.1f} MiB) -> s3://{bucket}/{key}", flush=True)
    s3.upload_file(str(path), bucket, key)
    uploaded += 1
    total_bytes += size
print(f"upload complete: {uploaded} files, {total_bytes / 1024**3:.2f} GiB to s3://{bucket}/{prefix}")
PYEOF
fi
