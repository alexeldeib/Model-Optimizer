# Quanty k8s manifests — ace-inference / cw4637-dev-us-e-01a

Two job profiles for FSDP2 layerwise NVFP4 PTQ work:

| File                                     | Purpose                                | Hardware          |
| ---------------------------------------- | -------------------------------------- | ----------------- |
| `job-rtxpro6000-discovery.yaml`          | Phase 0 discovery on small MoE         | 8× RTX Pro 6000   |
| `job-gb200-k26-ptq-fast.yaml`            | K2.6 NVFP4 multinode-layerwise PTQ     | 4× GB200          |

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
| `k26-nvfp4-ckpt`             | 800 Gi   | NVFP4 export targets + layerwise checkpoint dirs |
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

# Phase 3+: full K2.6 NVFP4 PTQ on 4x GB200
kubectl -n ace-inference apply -f job-gb200-k26-ptq-fast.yaml
kubectl -n ace-inference logs -l app=quanty-gb200-k26-ptq-fast -f
```

Outputs land on `k26-nvfp4-ckpt` PVC under
`/work/quanty/{checkpoints,exports}/<run-id>/`.
