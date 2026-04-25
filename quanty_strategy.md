# Strategy: K2.6 NVFP4 fast multinode PTQ — landing approach

**Branch:** `feat/quanty/fast-multinode-layerwise-ptq`
**Owner:** aeldeib@coreweave.com
**Decision date:** 2026-04-25

## Landing strategy: HYBRID — stacked commits for upstreaming

Each commit is atomic and independently mergeable. Subjects use prefixes that
identify upstream candidates:

- `upstream:` — generic fix, target NVIDIA/Model-Optimizer main
- `quanty:`   — CoreWeave infra/opinion, stays on this branch indefinitely

`git log --no-merges --grep="^upstream:" --reverse` yields the upstream PR set.

## Upstream boundary

| Path                                          | Lands               |
| --------------------------------------------- | ------------------- |
| `modelopt/torch/quantization/**`              | upstream            |
| `modelopt/torch/quantization/utils/**`        | upstream            |
| `examples/llm_ptq/multinode_ptq.py`           | upstream            |
| `examples/llm_ptq/example_utils.py`           | upstream            |
| `examples/speculative_decoding/**`            | upstream            |
| `tests/unit/torch/quantization/**`            | upstream            |
| `tests/gpu/torch/quantization/test_fsdp2.py`  | upstream            |
| `modelopt_recipes/**`                         | upstream            |
| `Dockerfile.quanty`                           | quanty (carry)      |
| `quanty/**`                                   | quanty (carry)      |
| `k8s/**`                                      | quanty (carry)      |
| `quanty_strategy.md`                          | quanty (carry)      |

## Stack (in PR order)

| # | Subject                                                                       | Lands     |
| - | ----------------------------------------------------------------------------- | --------- |
| 0 | quanty: carry local PTQ + speculator patches on top of main                  | carry     |
| 1 | upstream: forward --recipe flag through multinode_ptq.py (parity with hf_ptq) | upstream  |
| 2 | upstream: fix: _SkipLayer proxies FSDP2 internal attrs cleanly                | upstream  |
| 3 | upstream: feat: distributed-safe layerwise checkpoint save/resume             | upstream  |
| 4 | upstream: feat: register Kimi K2.5/K2.6 decoder layer for layerwise calib     | upstream  |
| 5 | upstream: test: tests/gpu test_fsdp2_layerwise.py covers MoE FSDP2 layerwise  | upstream  |
| 6 | quanty: Dockerfile + depot + k8s for ace-inference RTX Pro 6000 / GB200       | carry     |
| 7 | quanty: discovery harness for FSDP2 layerwise on small MoE                    | carry     |
| 8 | quanty: K2.6-specific recipe overlay (NVFP4 experts-only + FP8 MLA-aware KV)  | carry     |

## K2.6 test sequencing: de-risk with small MoE first

Order of attempts (each gates the next):

1. DeepSeek-V2-Lite (16B, 64 experts) on 2× RTX Pro 6000 — plumbing
2. Mixtral-8x7B on 2× RTX Pro 6000 — different MoE topology
3. Qwen3-MoE-30B-A3B on 8× GB200 — bigger, MoE NVFP4 kernel validation on SM100a
4. **Kimi K2.5** on 8× GB200 — closest analog with known modelopt support
5. **Kimi K2.6** on 8× GB200 — target

Rationale: K2.6 1T at NVFP4-experts-only is ~250 GB; fits one node with
layerwise (per-layer materialization). No multi-node TP needed for v1.
Smaller models give faster iteration on FSDP2-layerwise bugs that are
architecture-class-independent.

## Hardware + cluster

Cluster: `cw4637-dev-us-e-01a`, namespace `ace-inference`.

| Node type        | Label selector                            | GPUs/node | HBM/node | Count |
| ---------------- | ----------------------------------------- | --------- | -------- | ----- |
| GB200 (SM100a)   | `node.coreweave.cloud/type=gb200-4x-l`    | 4         | 744 Gi   | 16    |
| RTX Pro 6000 (SM120) | `gpu.nvidia.com/class=rtxp6000-8x`    | 8         | 768 Gi   | 2     |

Conventions:

- **Pull secret**: `inference-backends-image-pull-secret` (covers all
  `docker.cloudsmith.io/coreweave/*` repos, including `infr-dev`)
- **HF token**: `k26-hf-secret` / key `HF_TOKEN`
- **Scheduler**: `binpack-scheduler`, priority `inference-partial-node`
- **Object store**: bucket `infr`, prefix `test/ace/quants/`,
  endpoint `http://cwlota.com` (in-cluster) / `https://cwobject.com` (laptop)
- **Existing PVCs to use**:
  - `k26-source-bf16` (2400 Gi) — preferred source; BF16 weights
  - `k26-source-int4` (600 Gi) — INT4 packed; team's modelopt path uses this
  - `k26-nvfp4-ckpt` (800 Gi) — output target + layerwise checkpoint dirs
  - `k26-eagle3-ckpt` (50 Gi) — speculator
  - `k26-eagle3-hidden-states` (4 Ti) — speculator hidden states

Hardware caveats:

- **RTX Pro 6000 (SM120)**: FlashInfer #2723 / CUTLASS #3096 — NVFP4 MoE
  grouped GEMM produces garbage output. **Use only for FSDP2 plumbing, dense
  NVFP4, and correctness checks.** Final K2.6 NVFP4 export validates on
  GB200.
- **GB200 (SM100a)**: production target; NVFP4 MoE works correctly.

Coexistence with team's existing path:

- Team's `k26-quantize-nvfp4` Job is the source of truth for working
  conventions; my Jobs use `-fast` / `-disco` suffixes to avoid collision.
- Team uses INT4 source + extensive dequant-on-load patches; quanty path
  uses BF16 source so the upstream-bound fixes stay clean.

## K2.6 modeling bug to bake in

`MoonViT3dEncoder.__init__` references `self.use_deterministic_attn` before
assignment.  Fix shipped at `quanty/scripts/patch-k26-modeling.sh` (sed-based,
idempotent).  Both the discovery and K2.6 PTQ launchers run it
automatically when the source dir is local.

## What NOT to do (recorded so we don't drift)

- Do not skip the Mixtral / Qwen3-MoE smoke tests; cluster hours saved by
  going straight to K2.6 are dwarfed by debugging hours when something
  architecture-independent breaks.
- Do not implement multi-node TP for PTQ until we have a measured signal
  that single-node 8×GB200 + layerwise is insufficient. Likely it is enough.
- Do not amend `40ab13d95` (carry commit). Stack new commits on top so the
  upstream-candidate set remains rebaseable.
