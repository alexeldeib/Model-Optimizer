#!/usr/bin/env bash
#
# K2.6 NVFP4 multinode-layerwise PTQ launcher (multi-node).
#
# Differences from run-k26-ptq.sh (single-node):
#   * torchrun rendezvous uses MASTER_ADDR / MASTER_PORT pointing at the
#     index-0 pod via headless service DNS (Indexed Job + subdomain).
#   * NNODES > 1 -- 16 GPUs across 4 GB200 nodes -- enough HBM (~3 TB)
#     for K2.6's ~2 TB BF16 model with layerwise sharding headroom.
#
# Env (set by the k8s manifest):
#   TARGET_MODEL      Path to BF16 source weights (typically /source/k26-bf16/...)
#   RECIPE            K2.6-aware NVFP4 experts-only recipe yaml
#   CHECKPOINT_DIR    Shared dir for layerwise per-layer checkpoints
#   EXPORT_DIR        Final NVFP4 unified-HF safetensors output
#   CALIB_DATASET     Calibration dataset name
#   CALIB_SAMPLES     Calibration sample count
#   CALIB_BATCH       Calibration batch size
#   CALIB_SEQ_LEN     Calibration seq length
#   NUM_GPUS          Per-node GPU count (default 4)
#   NNODES            Total nodes (set by manifest from parallelism)
#   MASTER_ADDR       Headless DNS for index-0 pod
#   MASTER_PORT       Rendezvous port (default 29500)
#   JOB_COMPLETION_INDEX  k8s-injected node rank, set automatically by
#                         Indexed Jobs since 1.24

set -euo pipefail

: "${TARGET_MODEL:?set TARGET_MODEL}"
: "${RECIPE:?set RECIPE}"
: "${CHECKPOINT_DIR:?set CHECKPOINT_DIR}"
: "${EXPORT_DIR:?set EXPORT_DIR}"
: "${CALIB_DATASET:=cnn_dailymail}"
: "${CALIB_SAMPLES:=2048}"
: "${CALIB_BATCH:=8}"
: "${CALIB_SEQ_LEN:=2048}"
: "${NUM_GPUS:=4}"
: "${NNODES:?set NNODES (total node count)}"
: "${MASTER_ADDR:?set MASTER_ADDR (index-0 headless DNS)}"
: "${MASTER_PORT:=29500}"
: "${JOB_COMPLETION_INDEX:?set JOB_COMPLETION_INDEX (k8s injects on Indexed Jobs)}"

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
SCRATCH="${CHECKPOINT_DIR%/*}/k26-source-scratch"
SCRATCH_READY="${SCRATCH}/.scratch-ready"

if [[ -d "${TARGET_MODEL}" ]]; then
    if [[ "${NODE_RANK}" == "0" ]]; then
        mkdir -p "${SCRATCH}"
        find "${TARGET_MODEL}" -maxdepth 1 -type f -name "*.safetensors" \
            -exec ln -sf {} "${SCRATCH}/" \;
        find "${TARGET_MODEL}" -maxdepth 1 -type f \
            ! -name "*.safetensors" -exec cp -f {} "${SCRATCH}/" \;
        /opt/quanty/quanty/scripts/patch-k26-modeling.sh "${SCRATCH}" || true
        SCRATCH_KEY=$(basename "${SCRATCH}" | sed 's/-/_hyphen_/g')
        rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/modules/transformers_modules/${SCRATCH_KEY}"
        : > "${SCRATCH_READY}"
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

cd /opt/quanty

# Stamp the per-node log so we can grep for which rank failed without
# digging through pod uids.
LOG="${EXPORT_DIR}/run.rank${NODE_RANK}.log"

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
    --recipe "${RECIPE}" \
    --calib_size "${CALIB_SAMPLES}" \
    --batch_size "${CALIB_BATCH}" \
    --dataset "${CALIB_DATASET}" \
    --export_path "${EXPORT_DIR}" \
    --trust_remote_code \
    2>&1 | tee "${LOG}"
