#!/usr/bin/env bash
#
# Single-node Kimi-K2.6 NVFP4 PTQ via the known dequant/hf_ptq.py path.
#
# This intentionally avoids examples/llm_ptq/multinode_ptq.py/FSDP, but it
# still applies the same K2.6 ModelOpt correctness recipe used by the
# multinode candidate. It bootstraps ModelOpt 0.43.0, applies a narrow
# 0.43-compatible Kimi shim, stages the INT4 source into a writable local
# directory, overlays NVIDIA's patched Kimi modeling file, then runs hf_ptq.py with the K2.6
# MLP-only NVFP4 + FP8 KV recipe. The coherent production baseline quantizes
# routed experts, shared experts, and layer-0 dense MLP; leaving shared/layer-0
# native produced a vLLM-loaded but low-quality repeated-token model.
#
# Required ConfigMap keys under /scripts:
#   * run-k26-ptq-dequant.sh
#   * example_utils.py
#   * patch-modelopt-043-kimi.py
#   * k26-nvfp4_mlp_only-fp8_kv.yaml
#   * inspect-nvfp4-export.py

set -euxo pipefail

: "${MODEL_ID:=moonshotai/Kimi-K2.6}"
: "${MODEL_DIR:=/mnt/local/models/Kimi-K2.6-source-int4}"
: "${SOURCE_DIR:=/source/int4/Kimi-K2.6-source-int4}"
: "${SOURCE_S3:=s3://cwstudio/quantize/kimi-k26-source/}"
: "${S3_OUTPUT_PREFIX:=s3://infr/test/ace/quants}"
: "${RUN_ID:=k26-nvfp4-mlponly-dequant-hfptq-$(date -u +%Y%m%dT%H%M%SZ)}"
: "${OUTPUT_DIR:=/mnt/local/models/${RUN_ID}}"
: "${PVC_OUTPUT_DIR:=/work/quanty/exports/${RUN_ID}}"
: "${S3_OUTPUT:=${S3_OUTPUT_PREFIX%/}/${RUN_ID}/}"
: "${QFORMAT:=nvfp4_mlp_only}"
: "${RECIPE:=/scripts/k26-nvfp4_mlp_only-fp8_kv.yaml}"
: "${MODELOPT043_RECIPE:=/tmp/k26-nvfp4_mlp_only-fp8_kv-modelopt043.yaml}"
: "${KV_QFORMAT:=fp8}"
: "${CALIB_DATASET:=cnn_dailymail}"
: "${CALIB_SIZE:=512}"
: "${CALIB_SEQ:=512}"
: "${BATCH_SIZE:=1}"
: "${INFERENCE_TP:=4}"
: "${UPLOAD_TO_HF:=0}"
: "${NVIDIA_MODEL_REPO:=nvidia/Kimi-K2.5-NVFP4}"
: "${NVIDIA_MODEL_FILE:=/opt/quantize/nvidia_modeling_kimi_k25.py}"
: "${HF_HOME:=/mnt/local/hf_cache}"
: "${NODE_NAME:=unknown}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
: "${USE_SEQ_DEVICE_MAP:=1}"
: "${GPU_MAX_MEM_PERCENTAGE:=0.88}"
: "${MODELOPT043_USE_SEQUENTIAL:=0}"
: "${MODELOPT_MOE_CALIB_TOKENS_PER_EXPERT:=64}"
: "${MODELOPT_MOE_CALIB_TOKEN_CHUNK:=2048}"

export MODEL_ID MODEL_DIR SOURCE_DIR SOURCE_S3 S3_OUTPUT_PREFIX RUN_ID
export OUTPUT_DIR PVC_OUTPUT_DIR S3_OUTPUT QFORMAT RECIPE KV_QFORMAT
export MODELOPT043_RECIPE CALIB_DATASET CALIB_SIZE CALIB_SEQ BATCH_SIZE INFERENCE_TP UPLOAD_TO_HF
export NVIDIA_MODEL_REPO NVIDIA_MODEL_FILE HF_HOME NODE_NAME
export PYTORCH_CUDA_ALLOC_CONF USE_SEQ_DEVICE_MAP GPU_MAX_MEM_PERCENTAGE
export MODELOPT043_USE_SEQUENTIAL MODELOPT_MOE_CALIB_TOKENS_PER_EXPERT MODELOPT_MOE_CALIB_TOKEN_CHUNK
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTHONUNBUFFERED=1

LOG_DIR="/mnt/local/quantize_logs/${RUN_ID}-$(hostname -s)-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR" "$(dirname "$MODEL_DIR")" "$(dirname "$OUTPUT_DIR")" "$PVC_OUTPUT_DIR"
exec > >(tee -a "$LOG_DIR/stdout.log") 2> >(tee -a "$LOG_DIR/stderr.log" >&2)

echo "=== Kimi-K2.6 NVFP4 dequant PTQ on $(hostname) ==="
echo "run_id=$RUN_ID"
echo "logs=$LOG_DIR"
echo "model_dir=$MODEL_DIR"
echo "output_dir=$OUTPUT_DIR"
echo "pvc_output_dir=$PVC_OUTPUT_DIR"
echo "s3_output=$S3_OUTPUT"
echo "node_name=$NODE_NAME"
echo "host_local_note=/mnt/local is node-local; recover unfinished outputs from NODE_NAME"
nvidia-smi -L
df -h /mnt/local /work || true

aws_args=()
if [[ -n "${AWS_ENDPOINT_URL:-}" ]]; then
    aws_args+=(--endpoint-url "$AWS_ENDPOINT_URL")
fi

install_runtime() {
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
        rsync dnsutils curl ca-certificates git
    rm -rf /var/lib/apt/lists/*

    pip install --no-cache-dir --upgrade \
        "nvidia-modelopt[hf]==0.43.0" \
        "transformers==4.57.1" \
        "accelerate>=1.0" \
        "compressed-tensors==0.12.0" \
        "datasets>=3.0" \
        "huggingface_hub[hf_transfer]==0.36.2" \
        tiktoken fire rouge_score transformers_stream_generator zstandard \
        fsspec s3fs awscli

    rm -rf /opt/Model-Optimizer
    git clone --depth=1 --branch 0.43.0 \
        https://github.com/NVIDIA/Model-Optimizer.git /opt/Model-Optimizer
}

apply_modelopt_dequant_fixes() {
    local src dst
    local -a required_files=(
        "example_utils.py:/opt/Model-Optimizer/examples/llm_ptq/example_utils.py"
        "patch-modelopt-043-kimi.py:/tmp/patch-modelopt-043-kimi.py"
    )

    for entry in "${required_files[@]}"; do
        src="/scripts/${entry%%:*}"
        dst="${entry#*:}"
        if [[ ! -f "$src" ]]; then
            echo "missing required dequant ModelOpt 0.43 shim: $src" >&2
            echo "recreate quanty-k26-dequant-ptq-scripts with the dequant launcher files" >&2
            exit 1
        fi
        cp -f "$src" "$dst"
        echo "Applied $(basename "$src") -> $dst"
        sha256sum "$dst"
    done

    if [[ ! -f "$RECIPE" ]]; then
        echo "missing required K2.6 PTQ recipe: $RECIPE" >&2
        exit 1
    fi

    PYTHONPATH="/opt/Model-Optimizer:${PYTHONPATH:-}" python3 /tmp/patch-modelopt-043-kimi.py \
        --modelopt-root /opt/Model-Optimizer \
        --recipe-in "$RECIPE" \
        --recipe-out "$MODELOPT043_RECIPE"
    RECIPE="$MODELOPT043_RECIPE"
    export RECIPE

    sha256sum "$RECIPE"

    PYTHONPATH="/opt/Model-Optimizer:${PYTHONPATH:-}" python3 - <<'PY'
import os
from modelopt.recipe import ModelOptPTQRecipe, load_recipe
from modelopt.torch.quantization.plugins.huggingface import patch_compressed_linear_loading

recipe = load_recipe(os.environ["RECIPE"])
assert isinstance(recipe, ModelOptPTQRecipe), type(recipe)
assert "ptq_cfg" in recipe.model_dump(), recipe.model_dump().keys()
assert callable(patch_compressed_linear_loading)
print("ModelOpt 0.43 Kimi dequant shim smoke passed")
PY
}

stage_source() {
    local shard_count=0
    if [[ -d "$MODEL_DIR" ]]; then
        shard_count=$(find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)
    fi
    if [[ "$shard_count" -ge 60 ]]; then
        echo "Model cached ($shard_count shards in $MODEL_DIR), skipping stage"
        return 0
    fi

    rm -rf "$MODEL_DIR"
    mkdir -p "$MODEL_DIR"

    local source_shards=0
    if [[ -d "$SOURCE_DIR" ]]; then
        source_shards=$(find "$SOURCE_DIR" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)
    fi

    if [[ "$source_shards" -ge 60 ]]; then
        echo "=== Staging $MODEL_ID from read-only INT4 source $SOURCE_DIR ($source_shards shards) ==="
        rsync -a --exclude='*.safetensors' "$SOURCE_DIR"/ "$MODEL_DIR"/
        find "$SOURCE_DIR" -maxdepth 1 -type f -name '*.safetensors' \
            -exec ln -sf {} "$MODEL_DIR"/ \;
    else
        local s3_shards=0
        s3_shards=$(aws "${aws_args[@]}" s3 ls "$SOURCE_S3" 2>/dev/null | grep -c '\.safetensors$' || true)
        if [[ -z "$s3_shards" ]]; then
            s3_shards=0
        fi
        if [[ "$s3_shards" -ge 60 ]]; then
            echo "=== Downloading $MODEL_ID from $SOURCE_S3 (S3 mirror, $s3_shards shards) ==="
            aws "${aws_args[@]}" s3 sync "$SOURCE_S3" "$MODEL_DIR" --size-only --no-progress
        else
            echo "=== Downloading $MODEL_ID from HuggingFace ==="
            huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR" --max-workers 16
            NEEDS_S3_MIRROR=1
        fi
    fi

    echo "Download/stage complete: $(du -sh "$MODEL_DIR")"
    find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' | wc -l
}

validate_source_layout() {
    local index_path="$MODEL_DIR/model.safetensors.index.json"
    [[ -f "$index_path" ]] || return 0

    python3 - "$index_path" <<'PY'
import json
import os
import sys

qformat = os.environ["QFORMAT"]
with open(sys.argv[1]) as f:
    weight_map = json.load(f).get("weight_map", {})

packed = [key for key in weight_map if key.endswith("weight_packed")]

def is_routed_expert_key(key):
    return (
        ".mlp.experts." in key
        or ".block_sparse_moe.experts." in key
        or ".block_sparse_moe.local_experts." in key
    )

packed_not_experts = [key for key in packed if not is_routed_expert_key(key)]
plain_sensitive = [
    key
    for key in weight_map
    if key.endswith(".weight")
    and (
        key == "language_model.lm_head.weight"
        or key.startswith("mm_projector.")
        or key.startswith("vision_tower.")
    )
]
plain_non_routed_mlp = [
    key
    for key in weight_map
    if key.endswith(".weight")
    and not is_routed_expert_key(key)
    and (
        ".mlp." in key
        or ".block_sparse_moe." in key
        or ".shared_expert" in key
    )
]

print(
    "source_layout "
    f"qformat={qformat} packed={len(packed)} "
    f"packed_not_experts={len(packed_not_experts)} "
    f"plain_sensitive={len(plain_sensitive)} "
    f"plain_non_routed_mlp={len(plain_non_routed_mlp)}"
)
if packed_not_experts:
    print("packed_not_experts_preview=" + ",".join(packed_not_experts[:8]))
if plain_sensitive:
    print("plain_sensitive_preview=" + ",".join(plain_sensitive[:8]))
if plain_non_routed_mlp:
    print("plain_non_routed_mlp_preview=" + ",".join(plain_non_routed_mlp[:8]))

if qformat == "nvfp4_mlp_only" and plain_sensitive:
    print(
        "plain_sensitive_note=allowed because the K2.6 mlp_only recipe explicitly "
        "excludes lm_head, vision, and projector modules; post-export inspection "
        "will fail if scales are written for those modules."
    )

if qformat == "nvfp4_experts_only" and packed_not_experts:
    raise SystemExit(
        "Refusing nvfp4_experts_only because the source contains packed tensors "
        "outside language MoE experts. Revisit the recipe before launching PTQ."
    )
PY
}

ensure_nvidia_modeling() {
    mkdir -p "$(dirname "$NVIDIA_MODEL_FILE")"
    if [[ -f "$NVIDIA_MODEL_FILE" ]]; then
        return 0
    fi

    python3 - <<'PY'
import os
import shutil
from huggingface_hub import hf_hub_download

repo = os.environ.get("NVIDIA_MODEL_REPO", "nvidia/Kimi-K2.5-NVFP4")
target = os.environ.get("NVIDIA_MODEL_FILE", "/opt/quantize/nvidia_modeling_kimi_k25.py")
path = hf_hub_download(
    repo_id=repo,
    filename="modeling_kimi_k25.py",
    token=os.environ.get("HF_TOKEN") or None,
)
os.makedirs(os.path.dirname(target), exist_ok=True)
shutil.copy2(path, target)
print(f"Downloaded NVIDIA modeling_kimi_k25.py from {repo} to {target}")
PY
}

overlay_nvidia_modeling() {
    local d="$1"
    [[ -d "$d" ]] || return 0
    local target="$d/modeling_kimi_k25.py"
    cp -f "$NVIDIA_MODEL_FILE" "$target"
    echo "Overlaid NVIDIA modeling_kimi_k25.py -> $target"
}

prepare_modeling_overlay() {
    ensure_nvidia_modeling
    overlay_nvidia_modeling "$MODEL_DIR"

    # Avoid stale trust_remote_code cache entries. The local source path is
    # rewritten per node/run, and transformers sanitizes names differently for
    # local dirs vs repo ids, so clear Kimi-like dynamic modules broadly.
    for base in \
        /root/.cache/huggingface/modules/transformers_modules \
        "$HF_HOME/modules/transformers_modules"; do
        [[ -d "$base" ]] || continue
        find "$base" -maxdepth 1 -type d \( -name '*Kimi*' -o -name '*kimi*' \) -exec rm -rf {} +
    done
}

write_run_info() {
    local path="$1"
    mkdir -p "$path"
    python3 - "$path/run-info.json" <<'PY'
import json
import os
import sys

keys = [
    "RUN_ID",
    "MODEL_ID",
    "MODEL_DIR",
    "SOURCE_DIR",
    "SOURCE_S3",
    "OUTPUT_DIR",
    "PVC_OUTPUT_DIR",
    "S3_OUTPUT",
    "QFORMAT",
    "RECIPE",
    "KV_QFORMAT",
    "CALIB_DATASET",
    "CALIB_SIZE",
    "CALIB_SEQ",
    "BATCH_SIZE",
    "INFERENCE_TP",
    "MODELOPT_MOE_CALIB_TOKENS_PER_EXPERT",
    "MODELOPT_MOE_CALIB_TOKEN_CHUNK",
    "MODELOPT043_USE_SEQUENTIAL",
    "NVIDIA_MODEL_REPO",
    "NVIDIA_MODEL_FILE",
    "NODE_NAME",
]
with open(sys.argv[1], "w") as f:
    json.dump({key: os.environ.get(key) for key in keys}, f, indent=2, sort_keys=True)
PY
}

NEEDS_S3_MIRROR=0

install_runtime
apply_modelopt_dequant_fixes
stage_source
validate_source_layout
prepare_modeling_overlay

export PYTHONPATH="/opt/Model-Optimizer:${PYTHONPATH:-}"
cd /opt/Model-Optimizer/examples/llm_ptq

echo "=== Launching hf_ptq.py ==="
echo "qformat=$QFORMAT kv_cache_qformat=$KV_QFORMAT"
echo "recipe=$RECIPE"
echo "dataset=$CALIB_DATASET calib_size=$CALIB_SIZE calib_seq=$CALIB_SEQ batch_size=$BATCH_SIZE"
echo "export_path=$OUTPUT_DIR inference_tp=$INFERENCE_TP"
echo "use_seq_device_map=$USE_SEQ_DEVICE_MAP gpu_max_mem_percentage=$GPU_MAX_MEM_PERCENTAGE"
echo "moe_tokens_per_expert=$MODELOPT_MOE_CALIB_TOKENS_PER_EXPERT moe_token_chunk=$MODELOPT_MOE_CALIB_TOKEN_CHUNK"
echo "modelopt043_use_sequential=$MODELOPT043_USE_SEQUENTIAL"

extra_ptq_args=()
if [[ "$USE_SEQ_DEVICE_MAP" == "1" ]]; then
    extra_ptq_args+=(--use_seq_device_map --gpu_max_mem_percentage "$GPU_MAX_MEM_PERCENTAGE")
fi
ptq_config_args=(--qformat "$QFORMAT")
if [[ -n "$RECIPE" ]]; then
    ptq_config_args+=(--recipe "$RECIPE")
fi

python3 hf_ptq.py \
    --pyt_ckpt_path "$MODEL_DIR" \
    "${ptq_config_args[@]}" \
    --kv_cache_qformat "$KV_QFORMAT" \
    --dataset "$CALIB_DATASET" \
    --calib_size "$CALIB_SIZE" \
    --calib_seq "$CALIB_SEQ" \
    --batch_size "$BATCH_SIZE" \
    --export_path "$OUTPUT_DIR" \
    --inference_tensor_parallel "$INFERENCE_TP" \
    --trust_remote_code \
    --skip_generate \
    "${extra_ptq_args[@]}" \
    2>&1 | tee /tmp/ptq.log

rc=${PIPESTATUS[0]}
if [[ "$rc" -ne 0 ]]; then
    echo "hf_ptq.py failed with exit code $rc"
    exit "$rc"
fi

if [[ -f /scripts/inspect-nvfp4-export.py ]]; then
    echo "=== Inspecting export before sync/upload ==="
    python3 /scripts/inspect-nvfp4-export.py \
        --model-dir "$OUTPUT_DIR" \
        --out-dir "$OUTPUT_DIR/diagnostics/export-inspect" \
        --layers auto \
        --expected-experts 384 \
        --expected-moe-layers 60 \
        --expected-kv-cache "$KV_QFORMAT" \
        --allow-shared-experts \
        --allow-layer0-mlp \
        --fail-on-warnings \
        --model-label "$RUN_ID"
else
    echo "missing /scripts/inspect-nvfp4-export.py; refusing to upload uninspected export" >&2
    exit 1
fi

echo "=== Output summary ==="
du -sh "$OUTPUT_DIR"
find "$OUTPUT_DIR" -maxdepth 1 -type f | sed -n '1,40p'
write_run_info "$OUTPUT_DIR"

echo "=== Syncing output to PVC $PVC_OUTPUT_DIR ==="
rsync -a --delete "$OUTPUT_DIR"/ "$PVC_OUTPUT_DIR"/
write_run_info "$PVC_OUTPUT_DIR"

echo "=== Uploading output to $S3_OUTPUT ==="
aws "${aws_args[@]}" s3 sync "$OUTPUT_DIR" "$S3_OUTPUT" --size-only --no-progress

if [[ "$NEEDS_S3_MIRROR" == "1" ]]; then
    echo "=== Mirroring source $MODEL_DIR to $SOURCE_S3 for future runs ==="
    aws "${aws_args[@]}" s3 sync "$MODEL_DIR" "$SOURCE_S3" --size-only --no-progress || \
        echo "WARNING: source mirror failed, continuing"
fi

if [[ "$UPLOAD_TO_HF" == "1" ]]; then
    echo "UPLOAD_TO_HF=1 requested, but no upload_to_hf.py is mounted in this verifier image."
    exit 1
fi

echo "=== run-k26-ptq-dequant.sh complete ==="
echo "RUN_ID=$RUN_ID"
echo "PVC_OUTPUT_DIR=$PVC_OUTPUT_DIR"
echo "S3_OUTPUT=$S3_OUTPUT"
