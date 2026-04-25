# Strategy: K2.6 NVFP4 fast multinode PTQ

**Owner:** aeldeib@coreweave.com  ·  **Started:** 2026-04-25
**Cluster:** `cw4637-dev-us-e-01a`  ·  **Namespace:** `ace-inference`

## Branches

| Branch                                          | Off              | Pushed to (alexeldeib fork) | Purpose                          |
| ----------------------------------------------- | ---------------- | --------------------------- | -------------------------------- |
| `feat/vlm-decoder-discovery`                    | `origin/main`    | yes                         | Upstream PR — VLM decoder discovery |
| `feat/distributed-layerwise-checkpoint`         | `origin/main`    | yes                         | Upstream PR — distributed checkpoint |
| `feat/quanty/fast-multinode-layerwise-ptq`      | `origin/main`    | yes                         | Internal dev: cherry-picks both upstream commits + carry (Dockerfile, k8s, recipe overlay, harnesses, K2.6 modeling fix) |

Subjects on the upstream branches read like upstream code (no prefix);
internal-only commits on the dev branch carry a `quanty:` prefix.

## Upstream boundary

| Path                                          | Lands               |
| --------------------------------------------- | ------------------- |
| `modelopt/torch/quantization/**`              | upstream            |
| `examples/llm_ptq/**`                         | upstream (via `feat/quanty/...` carry until ready) |
| `examples/speculative_decoding/**`            | upstream (via carry) |
| `tests/unit/torch/quantization/**`            | upstream            |
| `modelopt_recipes/**`                         | upstream            |
| `quanty/**`                                   | quanty (carry)      |
| `quanty_strategy.md`                          | quanty (carry)      |

## K2.6 test sequencing: de-risk before K2.6

Each step gates the next:

1. **DeepSeek-V2-Lite** (16B, 64 experts) on 8× RTX Pro 6000 — plumbing only
2. **Mixtral-8x7B** on 8× RTX Pro 6000 — different MoE topology
3. **Qwen3-MoE-30B-A3B** on 4× GB200 — MoE NVFP4 kernel validation on SM100a
4. **Kimi K2.5** on 4× GB200 — closest known-good analog
5. **Kimi K2.6** on 4× GB200 — target

K2.6 1T at NVFP4 experts-only is ~250 GB; layerwise calibration
materialises one decoder layer at a time, so a single 4× GB200 node
(744 Gi HBM) fits with headroom.  No multi-node TP planned for v1.

## Cluster

| Node type            | Label selector                          | GPUs/node | HBM/node | Count |
| -------------------- | --------------------------------------- | --------- | -------- | ----- |
| GB200 (SM100a)       | `node.coreweave.cloud/type=gb200-4x-l`  | 4         | 744 Gi   | 16    |
| RTX Pro 6000 (SM120) | `gpu.nvidia.com/class=rtxp6000-8x`      | 8         | 768 Gi   | 2     |

- **Pull secret**: `inference-backends-image-pull-secret`
- **HF token secret**: `k26-hf-secret` (key `HF_TOKEN`)
- **Scheduler**: `binpack-scheduler`, priority `inference-partial-node`
- **PVCs**: `k26-source-bf16` (2400 Gi, preferred source),
  `k26-source-int4` (600 Gi, team's path), `k26-nvfp4-ckpt` (800 Gi,
  output + layerwise checkpoint dirs), `k26-eagle3-ckpt` (50 Gi),
  `k26-eagle3-hidden-states` (4 Ti)
- **Object store**: bucket `infr`, prefix `test/ace/quants/`;
  endpoints `http://cwlota.com` (in-cluster) / `https://cwobject.com` (laptop)

## Hardware caveats

- **RTX Pro 6000 (SM120)**: FlashInfer #2723 / CUTLASS #3096 cause garbage
  output for NVFP4 MoE grouped GEMM.  Use only for FSDP2 plumbing, dense
  NVFP4, and correctness checks.  Final K2.6 NVFP4 export validates on GB200.
- **GB200 (SM100a)**: production target; NVFP4 MoE works correctly.

## K2.6 modeling bug

`MoonViT3dEncoder.__init__` references `self.use_deterministic_attn`
before assignment.  Fix shipped at `quanty/scripts/patch-k26-modeling.sh`
(idempotent sed).  Both launchers run it on local source dirs.

## Coexistence with team's existing path

- Team's `k26-quantize-nvfp4` Job is the source of truth for working
  conventions; new Jobs here use `-fast` / `-disco` suffixes.
- Team uses INT4 source + extensive dequant-on-load patches; this path
  uses BF16 source to keep the upstream-bound fixes clean.

## Phase 0 discovery findings (2026-04-25)

Iter 1-3 exposed three image / launcher gaps (all fixed):
- `multinode_ptq.py` does not accept `--calib_seq` / `--layerwise_checkpoint_dir`
- `compressed_tensors` was missing from the depot image
- `deepseek-ai/DeepSeek-V2-Lite` is not viable with transformers 5.x (its
  `modeling_deepseek.py` imports `is_torch_fx_available`, removed upstream)

Iter 4 reached **GPTQ Hessian collection on Qwen2.5-7B-Instruct** under
FSDP2 + layerwise, then crashed in transformers' `sdpa_attention_forward`:

    RuntimeError: The expanded size of the tensor (1024) must match the
    existing size (512) at non-singleton dimension 3.
    Target sizes: [4, 28, 512, 1024].  Tensor sizes: [4, 1, 512, 512]

This is a **transformers 5.x ``DynamicCache`` × layerwise GPTQ × replayed
forward** bug that lives below our upstream commits' scope: the captured
``kwargs_input`` for a layer carries an attention mask sized for the
first forward (k_len=512), but the cache mutates in place across replays,
so the second replay sees k_len=1024 KV with a 512-wide mask.  The reset
in `model_calib.py:1657-1666` only handles ``kwargs_input["past_key_values"]``,
not the in-place state mutation that transformers 5.x's `DynamicCache`
performs across a layer's forward.

Logged as a Phase 1 follow-up; **not in our upstream PR scope**.  Both
upstream commits (VLM decoder discovery + distributed checkpoint)
executed cleanly past their contracts.

K2.6 uses MLA attention, not standard SDPA; the bug should not reproduce
on the GB200 K2.6 path.  Validate that hypothesis there before
committing engineering time to a Phase 1 fix.

## Ground rules

- Do not skip the small-MoE smoke tests before K2.6.  Cluster hours saved
  going straight to K2.6 are dwarfed by debugging when something
  architecture-independent breaks.
- Do not implement multi-node TP until we have measured signal that
  single-node 4× GB200 + layerwise is insufficient.
- Do not amend the carry commit on `feat/quanty/...` once it's pushed;
  stack new commits on top so the upstream-cherry-picks rebase cleanly.
