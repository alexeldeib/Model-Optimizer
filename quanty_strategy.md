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

## K2.6 GB200 attempt (2026-04-25)

After switching the depot image base to ``nvcr.io/nvidia/tensorrt-llm/release:1.2.0``
(matched torch + flash_attn + transformers stack -- replaces every ABI
patch from the previous pytorch:25.04 base), the K2.6 BF16 PTQ run on
4x GB200 progresses through:

* multi-arch image pull on arm64 (after a few cloudsmith blob retries)
* tokenizer init from Kimi-K2.6-BF16 (writable scratch dir + MoonViT3d
  ``use_deterministic_attn`` patch + transformers cache bust)
* dataset load (Nemotron-Post-Training-Dataset-v2)
* 64/64 checkpoint shards loaded across 4 ranks
* ``Preparing model with FSDP2...`` from the carry commit

Then OOMs at FSDP2 wrap:

    torch.OutOfMemoryError: GPU 0 has a total capacity of 184.31 GiB
    of which 9.50 MiB is free.  Tried to allocate 28.00 MiB.

This is **architectural, not a bug**.  K2.6 (1.04 T params, BF16) =
~2 TB / 4 ranks = ~500 GB per rank target after ideal sharding;
each GB200 has 184 GiB HBM.  The unshard-during-wrap step puts the
whole model on a single rank momentarily and the math doesn't fit.

The team's existing modelopt path avoids this by using **INT4
source + GPU-resident dequant** on 4x GB200, explicitly skipping
multinode FSDP2 (FSDP2 rejects int32 packed weights).  For the
quanty path, real options:

* **16x GB200** (4 nodes x 4 GPUs) -- ~3 TB HBM, fits with headroom.
  Validates the multinode FSDP2 path we set out to test in the first
  place.  Highest-value next step.
* ``init_empty_weights() + load_checkpoint_and_dispatch`` so the
  model never fully materialises on any one rank during load.
  Modifies ``examples/llm_ptq/multinode_ptq.py`` -- belongs as an
  upstream PR if it pans out.
* Adopt the team's INT4 source + accelerate patches.  Departs from
  our stated goal of keeping upstream commits clean; reserved as a
  fallback.

Logged as the next loop iteration's starting point.  Both upstream
commits validated end-to-end through their contracts (VLM decoder
discovery picked up K2.6's ``KimiK25ForConditionalGeneration ->
DeepseekV3ForCausalLM``; distributed checkpoint logic ran without
raising at world_size=4).  The OOM is downstream of their scope.

## Phase 2 e2e validation: Qwen3-30B-A3B on 4x GB200 (2026-04-26)

Both Qwen3-MoE Jobs completed end-to-end and validated each upstream
commit's contract on real GB200 hardware:

* `job-gb200-qwen3-moe-ptq.yaml` (NVFP4 experts-only, max-calibrate):
  exited Complete after 11 minutes; export at
  `/work/quanty/exports/qwen3-moe-nvfp4-experts-only/` (~18 GB across
  4 safetensors shards + `hf_quant_config.json` for vLLM autodetect).
  Exercises the **VLM decoder discovery** upstream commit's dispatcher
  path.

* `job-gb200-qwen3-moe-checkpoint.yaml` (GPTQ + layerwise checkpoint):
  exited Complete after 3h31m; export at
  `/work/quanty/exports/qwen3-moe-gptq-layerwise/` (~17 GB).  Rank 0
  wrote `layer_0000` ... `layer_0047` + `manifest.json` atomically
  across the run.  Exercises the **distributed-layerwise-checkpoint**
  upstream commit's rank-0-writer path with `world_size=4`.  No
  `_CheckpointState` raise, no concurrent-writer hazard.

The Qwen2.5-7B `DynamicCache` cache-replay bug from Phase 0 reproduced
*exactly* on Qwen3-MoE (same shape mismatch, same line in
`sdpa_attention_forward`).  Worked around in the launcher carry by
`model.config.use_cache = False` after `from_pretrained`.  Folding
this into `model_calib.py:layerwise_calibrate` is a candidate
upstream PR independent of the two already-pushed branches.

### Regression validation

`job-gb200-qwen3-moe-eval.yaml` runs lm-eval-harness against:
1. BF16 baseline (`Qwen/Qwen3-30B-A3B`, downloaded from HF, cached on PVC)
2. NVFP4 experts-only export
3. GPTQ layerwise export

Tasks: gsm8k (math reasoning, most quantization-sensitive),
arc_challenge, hellaswag.  vLLM serves all three with the same
`c2-k25-tf4-i2` image used in production for K2.5 NVFP4.  Regression
threshold defaults to **1% absolute** on each task; the embedded
delta script exits non-zero so a CI gate can catch a backslide.

### Artifacts

| Path                                                  | Contents                                  |
| ----------------------------------------------------- | ----------------------------------------- |
| `/work/quanty/exports/qwen3-moe-nvfp4-experts-only/`  | NVFP4 max-calibrate export                |
| `/work/quanty/exports/qwen3-moe-gptq-layerwise/`      | GPTQ layerwise NVFP4 export               |
| `/work/quanty/exports/qwen3-30b-a3b-bf16/eval-results/` | lm-eval BF16 baseline results           |
| `/work/quanty/exports/<export>/eval-results/`         | per-export lm-eval results                |
| `/work/quanty/exports/eval-summary.json`              | regression delta report (PASS/FAIL)       |

## Phase 3: K2.6 multinode FSDP2 PTQ (2026-04-26)

`job-gb200-k26-multinode-ptq.yaml` is the v2 of the single-node OOM
case from 2026-04-25: same launcher target + recipe + source PVC,
but spread over 4 GB200 nodes (16 GPUs, ~3 TB HBM total) via Indexed
Job + headless service rendezvous.  Each rank's peak FSDP2-wrap
memory is `model_size / world_size + layerwise_overhead`, well under
the 184 GiB / GB200 budget.

### Why Indexed Job (not MPIJob / KubeRay / PyTorchJob)

* **MPIJob** (`cw-mpijobs.hpc.coreweave.com`) ships with the cluster
  but adds an MPI wrapper around what is already a torchrun
  rendezvous; introduces a second layer of process management for
  no benefit on a torchrun-native script.
* **PyTorchJob** CRD is not installed.
* **Indexed Job** (k8s 1.27+) auto-sets `hostname=<job-name>-<index>`
  and injects `JOB_COMPLETION_INDEX` into each pod, which is
  everything torchrun needs.  Zero CRD dependency.  We're on 1.35.

Headless Service uses `publishNotReadyAddresses: true` so worker
pods can DNS-resolve `quanty-gb200-k26-mn-ptq-0.<svc>` before any
pod becomes Ready -- otherwise rendezvous deadlocks.

### Risks tracked

* **Concurrent writers to scratch dir**: rank 0 owns the `cp -f` +
  patch step; ranks >= 1 spin on `.scratch-ready` sentinel.
* **NCCL inter-node**: `NCCL_SOCKET_IFNAME=eth0` matches `c2-k25-*`
  precedent; IB/NVLS opportunistic enables.
* **Backoff**: `backoffLimit: 8` so a single-node preemption + resume
  costs only the in-flight layer's GPTQ work (resume from
  `manifest.json`).

## Ground rules

- Do not skip the small-MoE smoke tests before K2.6.  Cluster hours saved
  going straight to K2.6 are dwarfed by debugging when something
  architecture-independent breaks.
- Do not implement multi-node TP until we have measured signal that
  single-node 4× GB200 + layerwise is insufficient.  The 16x GB200
  multinode FSDP2 path is *not* TP -- it's data/model sharding via
  FSDP2's mesh and was always in scope for v1.
- Do not amend the carry commit on `feat/quanty/...` once it's pushed;
  stack new commits on top so the upstream-cherry-picks rebase cleanly.
