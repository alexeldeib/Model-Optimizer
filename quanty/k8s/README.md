# Quanty k8s manifests — ace-inference / cw4637-dev-us-e-01a

Five job profiles covering the full PTQ -> validation pipeline:

| File                                       | Purpose                                                        | Hardware          |
| ------------------------------------------ | -------------------------------------------------------------- | ----------------- |
| `job-rtxpro6000-discovery.yaml`            | Phase 0 discovery on small MoE                                 | 8× RTX Pro 6000   |
| `job-gb200-qwen3-moe-ptq.yaml`             | Phase 2 e2e: Qwen3-30B-A3B NVFP4 experts-only (max-calibrate)  | 4× GB200          |
| `job-gb200-qwen3-moe-checkpoint.yaml`      | Phase 2 e2e: Qwen3-30B-A3B GPTQ + distributed-checkpoint upstream commit | 4× GB200 |
| `job-gb200-qwen3-moe-eval.yaml`            | Phase 2 e2e: lm-eval regression vs BF16 baseline (<1% delta gate) | 4× GB200       |
| `job-gb200-k26-ptq-fast.yaml`              | K2.6 NVFP4 single-node (OOMs; documented in `quanty_strategy.md`) | 4× GB200       |
| `job-gb200-k26-multinode-ptq.yaml`         | K2.6 NVFP4 multinode FSDP2 PTQ (Phase 3); defaults to 5 nodes, raise to 6 for extra headroom | 4× GB200 per node |
| `job-gb200-k26-dequant-ptq.yaml`           | K2.6 NVFP4 single-node dequant verifier using ModelOpt 0.43.0 `hf_ptq.py` | 4× GB200 |

The team's existing `k26-quantize-nvfp4` Job (single-node, INT4 source,
patched-up modelopt main) remains the source of truth.  These manifests
use distinct names (`-fast` / `-disco`) so they don't collide.

## Cluster conventions used here

* **Pull secret**: `inference-backends-image-pull-secret` (covers all
  `docker.cloudsmith.io/coreweave/*` repos)
* **HF token secret**: `k26-hf-secret` (key: `HF_TOKEN`)
* **Scheduler**: `binpack-scheduler`
* **Priority class**: `inference-partial-node`
* **GB200 nodes**: `node.coreweave.cloud/type=gb200-4x-l` (4 GPUs, 744 Gi
  HBM/node, 16 nodes available cluster-wide)
* **RTX Pro 6000 nodes**: `gpu.nvidia.com/class=rtxp6000-8x` (8 GPUs,
  768 Gi HBM/node, 2 nodes available)

## PVCs (already provisioned)

| PVC                          | Size     | Used for                                         |
| ---------------------------- | -------- | ------------------------------------------------ |
| `k26-source-bf16`            | 2400 Gi  | K2.6 BF16 source weights (preferred over INT4)   |
| `k26-source-int4`            | 600 Gi   | K2.6 INT4 source (used by team's modelopt path)  |
| `k26-nvfp4-ckpt`             | cluster configured size; verify with `df -h /work` in pod | NVFP4 export targets + layerwise checkpoint dirs |
| `k26-eagle3-ckpt`            | 50 Gi    | EAGLE-3 draft checkpoints                        |
| `k26-eagle3-hidden-states`   | 4 Ti     | Hidden-state dumps for offline EAGLE-3 training  |

We use the **BF16** source PVC for the fast-multinode path because the
INT4 source requires the team's accumulated dequant-on-load patches in
modelopt's accelerate plugin (`unpack_weight`, `requantize_resmooth_fused_llm_layers`,
`remove_hook_from_module`).  Avoiding those patches keeps the upstream-bound
fixes in this branch clean.

## Run

```bash
# Phase 0: discovery on RTX Pro 6000
kubectl -n ace-inference apply -f job-rtxpro6000-discovery.yaml
kubectl -n ace-inference logs -l app=quanty-rtxpro6000-discovery -f --max-log-requests=8

# Phase 2: e2e validation on Qwen3-30B-A3B (small MoE, fits 4x GB200)
kubectl -n ace-inference apply -f job-gb200-qwen3-moe-ptq.yaml          # NVFP4 max-calibrate
kubectl -n ace-inference apply -f job-gb200-qwen3-moe-checkpoint.yaml   # GPTQ + dist-checkpoint
kubectl -n ace-inference apply -f job-gb200-qwen3-moe-eval.yaml         # lm-eval regression

# ConfigMap overlays for K2.6 jobs. Delete/create avoids kubectl apply's
# last-applied annotation size limit on larger script/model overlays.
kubectl -n ace-inference delete configmap modelopt-k26-minimal-fixes --ignore-not-found
kubectl -n ace-inference create configmap modelopt-k26-minimal-fixes \
  --from-file=model_calib.py=../../modelopt/torch/quantization/model_calib.py \
  --from-file=layer_utils.py=../../modelopt/torch/export/layer_utils.py \
  --from-file=unified_export_hf.py=../../modelopt/torch/export/unified_export_hf.py \
  --from-file=quant_utils.py=../../modelopt/torch/export/quant_utils.py \
  --from-file=huggingface.py=../../modelopt/torch/quantization/plugins/huggingface.py \
  --from-file=tensor_quantizer.py=../../modelopt/torch/quantization/nn/modules/tensor_quantizer.py \
  --from-file=nvfp4_tensor.py=../../modelopt/torch/quantization/qtensor/nvfp4_tensor.py \
  --from-file=core_utils.py=../../modelopt/torch/quantization/utils/core_utils.py \
  --from-file=layerwise_calib.py=../../modelopt/torch/quantization/utils/layerwise_calib.py \
  --from-file=multinode_ptq.py=../../examples/llm_ptq/multinode_ptq.py

kubectl -n ace-inference delete configmap quanty-k26-nvfp4-mlponly-recipe --ignore-not-found
kubectl -n ace-inference create configmap quanty-k26-nvfp4-mlponly-recipe \
  --from-file=k26-nvfp4_mlp_only-fp8_kv.yaml=../recipes/k26-nvfp4_mlp_only-fp8_kv.yaml

kubectl -n ace-inference delete configmap quanty-k26-ptq-scripts --ignore-not-found
kubectl -n ace-inference create configmap quanty-k26-ptq-scripts \
  --from-file=run-k26-ptq-multinode.sh=../scripts/run-k26-ptq-multinode.sh \
  --from-file=patch-k26-modeling.sh=../scripts/patch-k26-modeling.sh \
  --from-file=inspect-nvfp4-export.py=../scripts/inspect-nvfp4-export.py

kubectl -n ace-inference delete configmap quanty-k26-dequant-ptq-scripts --ignore-not-found
kubectl -n ace-inference create configmap quanty-k26-dequant-ptq-scripts \
  --from-file=run-k26-ptq-dequant.sh=../scripts/run-k26-ptq-dequant.sh \
  --from-file=example_utils.py=../../examples/llm_ptq/example_utils.py \
  --from-file=patch-modelopt-043-kimi.py=../scripts/patch-modelopt-043-kimi.py \
  --from-file=k26-nvfp4_mlp_only-fp8_kv.yaml=../recipes/k26-nvfp4_mlp_only-fp8_kv.yaml \
  --from-file=inspect-nvfp4-export.py=../scripts/inspect-nvfp4-export.py

# Phase 3: K2.6 multinode FSDP2 PTQ on 5x GB200 (20 GPUs)
kubectl -n ace-inference apply -f job-gb200-k26-multinode-ptq.yaml
kubectl -n ace-inference logs -l app=quanty-gb200-k26-mn-ptq -f --max-log-requests=8
```

Outputs land on `k26-nvfp4-ckpt` PVC under
`/work/quanty/{checkpoints,exports}/<run-id>/`.

The eval Job exits non-zero if any candidate's gsm8k / arc_challenge /
hellaswag accuracy regresses by more than `REGRESSION_THRESHOLD_PCT`
(default 1.0%) versus the BF16 baseline.  Override on apply:

```bash
kubectl -n ace-inference set env job/quanty-gb200-qwen3-moe-eval \
    EVAL_LIMIT=0 REGRESSION_THRESHOLD_PCT=0.5     # full eval, tighter gate
```

The K2.6 multinode Job uses an Indexed completion mode + headless
service so each pod's hostname (`quanty-gb200-k26-mn-ptq-<index>`)
resolves under the `quanty-gb200-k26-mn-rdzv` subdomain.  Pod-0 is
the torchrun rendezvous master at `:29500`; the launcher reads
`JOB_COMPLETION_INDEX` for `--node-rank`.
