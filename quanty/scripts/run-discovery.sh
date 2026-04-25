#!/usr/bin/env bash
#
# Phase 0 discovery harness.
#
# Runs multinode_ptq.py with the layerwise=true recipe on a small MoE under
# torchrun + FSDP2 across all visible GPUs.  Captures every error, stack
# trace, and memory snapshot.  Produces a structured report at
# ${RUN_DIR}/discovery.json suitable for grep'ing for failure modes.
#
# Env (set by the k8s manifest, overridable):
#   TARGET_MODEL    HF id or local path of the smoke-test model
#   RECIPE          Path to a layerwise=true PTQ recipe yaml
#   NUM_GPUS        Visible GPUs (defaults from CUDA_VISIBLE_DEVICES)
#   RUN_DIR         Output directory (PVC-backed)
#   CALIB_SAMPLES   Calibration sample count (default 64)
#   CALIB_BATCH    Calibration batch size  (default 4)
#   CALIB_SEQ      Calibration seq length  (default 512)

set -euo pipefail

: "${TARGET_MODEL:?set TARGET_MODEL}"
: "${RECIPE:?set RECIPE (layerwise=true yaml)}"
: "${NUM_GPUS:=$(python -c 'import torch;print(torch.cuda.device_count())')}"
: "${RUN_DIR:=/work/quanty/runs/discovery-$(date +%s)}"
: "${CALIB_SAMPLES:=64}"
: "${CALIB_BATCH:=4}"
: "${CALIB_SEQ:=512}"

mkdir -p "${RUN_DIR}"
cd /opt/quanty

# Shared FS for layerwise checkpoints + output.
CKPT="${RUN_DIR}/layerwise_ckpts"
EXPORT="${RUN_DIR}/export"
LOG="${RUN_DIR}/run.log"

# Apply the K2.6 modeling patch if the source directory is a K2.6 repo.
# No-op for any non-K2.6 model.
if [[ -d "${TARGET_MODEL}" ]]; then
    /opt/quanty/quanty/scripts/patch-k26-modeling.sh "${TARGET_MODEL}" || true
fi

cat > "${RUN_DIR}/inputs.json" <<EOF
{
  "target_model":    "${TARGET_MODEL}",
  "recipe":          "${RECIPE}",
  "num_gpus":        ${NUM_GPUS},
  "calib_samples":   ${CALIB_SAMPLES},
  "calib_batch":     ${CALIB_BATCH},
  "calib_seq":       ${CALIB_SEQ},
  "checkpoint_dir":  "${CKPT}",
  "export_dir":      "${EXPORT}",
  "image":           "${HOSTNAME}",
  "started_at":      "$(date -u +%FT%TZ)"
}
EOF

set +e
torchrun \
    --nproc-per-node="${NUM_GPUS}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    examples/llm_ptq/multinode_ptq.py \
    --pyt_ckpt_path "${TARGET_MODEL}" \
    --recipe "${RECIPE}" \
    --calib_size "${CALIB_SAMPLES}" \
    --batch_size "${CALIB_BATCH}" \
    --dataset cnn_dailymail \
    --export_path "${EXPORT}" \
    --trust_remote_code \
    2>&1 | tee "${LOG}"
RC=${PIPESTATUS[0]}
set -e

# Synthesise a tiny structured report so post-run inspection is grep-able.
python - "${RUN_DIR}" "${LOG}" "${RC}" <<'PY'
import json, os, re, sys

run_dir, log_path, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
log = open(log_path, errors="replace").read() if os.path.exists(log_path) else ""

# Heuristics for known failure-mode signals.
SIGNALS = [
    ("argparse_error",        r"unrecognized arguments|error: argument"),
    ("checkpoint_state_raise", r"Layerwise calibration checkpointing is not supported"),
    ("oom",                   r"CUDA out of memory|OutOfMemoryError"),
    ("nccl_timeout",          r"NCCL .* timed out|Watchdog caught collective"),
    ("dtype_mismatch",        r"FSDP expects uniform .* dtype"),
    ("fsdp_state_attr",       r"_get_fsdp_state|_fsdp_param_group"),
    ("hf_hook_clash",         r"We dont support FSDP2 with HF accelerate hooks"),
    ("decoder_layers_none",   r"Could not find transformer layers in model"),
    ("compressed_linear",     r"compressed_tensors|patch_compressed_linear_loading"),
    ("kimi_modeling_bug",     r"use_deterministic_attn"),
    ("gptq_hessian",          r"GPTQHelper|computing Hessian"),
    ("expert_routing",        r"expert.*not.*calibrated|amax is None"),
    ("hf_download_fail",      r"Cannot resolve|repository not found|gated repo|401 Client Error"),
    # Attention/cache replay bug surfaced by GPTQ layerwise on transformers 5.x:
    # mask k_len mismatches actual KV k_len because DynamicCache state leaks
    # across captured-input replays.  ``cache.reset()`` in model_calib.py only
    # covers kwargs_input["past_key_values"]; layer-internal cache mutation
    # across replays is not reset.
    ("attn_cache_shape",      r"scaled_dot_product_attention.*\n.*expanded size of the tensor|sdpa_attention_forward.*RuntimeError"),
    ("transformers_5_compat", r"is_torch_fx_available|cannot import name '\w+' from 'transformers"),
]
hits = {name: bool(re.search(pat, log)) for name, pat in SIGNALS}

# First exception traceback if any.
m = re.search(r"Traceback \(most recent call last\):.*?(?=\n\S|\Z)", log, re.DOTALL)
first_traceback = m.group(0)[-4000:] if m else None

report = {
    "exit_code": rc,
    "ok": rc == 0,
    "log_path": log_path,
    "signals": hits,
    "first_traceback": first_traceback,
}
out = os.path.join(run_dir, "discovery.json")
with open(out, "w") as f:
    json.dump(report, f, indent=2, sort_keys=True)
print(f"=== discovery report -> {out} ===")
print(json.dumps(report["signals"], indent=2, sort_keys=True))
PY

exit "${RC}"
