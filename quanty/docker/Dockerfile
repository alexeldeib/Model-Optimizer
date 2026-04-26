# syntax=docker/dockerfile:1.7
#
# quanty / modelopt dev image
# Built via depot, pushed to docker.cloudsmith.io/coreweave/infr-dev/quanty-modelopt:<tag>
# Run on ace-inference namespace in cw4637-dev-us-e-01a
#
# Targets:
#   * RTX Pro 6000 (SM120) for FSDP2/layerwise plumbing tests + dense NVFP4
#   * GB200 (SM100a) for NVFP4 MoE end-to-end validation
#
# Tag convention:
#   :dev          -> mainline of feat/quanty/fast-multinode-layerwise-ptq
#   :sha-<short>  -> immutable per-commit
#   :pr-<n>       -> PR builds
#
# Base: TRT-LLM 1.2.0 (the team's known-working stack for K2.6).  Ships
# matched torch + flash_attn + transformer_engine + transformers 5.x;
# avoids the C-ABI mismatches that show up when you upgrade torch on
# top of the NGC PyTorch base.

ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.2.0
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    HF_HOME=/work/hf-cache \
    TRANSFORMERS_CACHE=/work/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1

# OS deps: git for editable installs, tini as PID-1, jq for json
# wrangling in launchers, openssh-client for git+ssh fetches.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git tini jq openssh-client ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/quanty

# Install modelopt + recipes editable so the image picks up branch
# changes at build time without re-baking the wheel on every iteration.
COPY pyproject.toml noxfile.py uv.lock ./
COPY modelopt/ ./modelopt/
COPY modelopt_recipes/ ./modelopt_recipes/
COPY examples/ ./examples/
COPY tests/ ./tests/

# TRT-LLM 1.2.0 ships transformers 5.x, accelerate 1.x, datasets,
# flash_attn matched to its bundled torch, etc.  We only add the
# small set of extras the discovery harness post-processor needs
# (jsonlines / lm-eval / pyyaml / hf_transfer) and modelopt itself.
#
# Critically we do NOT upgrade torch / torchvision / flash_attn here:
# the C extensions ship matched against the bundled torch ABI in the
# TRT-LLM base, and any pip --upgrade of torch silently breaks them.
RUN pip install --upgrade pip wheel setuptools \
 && pip install \
        "compressed_tensors>=0.13.0" \
        "hf_transfer>=0.1.6" \
        "jsonlines>=4.0.0" \
        "lm-eval>=0.4.5" \
        "pyyaml>=6.0" \
 && pip install -e ".[hf,torch]"

# Sanity probes: a busted install must fail the build, not the cluster job.
RUN python <<'PY'
import modelopt
import modelopt.torch.quantization as mtq  # noqa: F401
print('modelopt', getattr(modelopt, '__version__', 'editable'))
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
import transformers
print('transformers', transformers.__version__)
import accelerate
print('accelerate', accelerate.__version__)
try:
    import flash_attn
    print('flash_attn', flash_attn.__version__)
except Exception as e:
    print('flash_attn ImportError:', e)
PY

# Quanty-specific harnesses + recipe overlays.
COPY quanty/scripts/ /opt/quanty/quanty/scripts/
COPY quanty/recipes/ /opt/quanty/quanty/recipes/

# Make scripts executable here (writeable host filesystems sometimes lose
# the +x bit on copy).
RUN chmod +x /opt/quanty/quanty/scripts/*.sh /opt/quanty/quanty/scripts/*.py 2>/dev/null || true

# Base image already owns its default uid; we run as root in the
# container.  k8s `securityContext.runAsNonRoot` / `runAsUser` is the
# correct place to enforce non-root in production deployments.
WORKDIR /work

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-l"]
