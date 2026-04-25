# syntax=docker/dockerfile:1.7
#
# quanty / modelopt dev image
# Built via depot, pushed to docker.cloudsmith.io/coreweave/infr-dev/quanty-modelopt:<tag>
# Run on ace-inference namespace in CW4637-DEV-USE01A
#
# Targets:
#   * RTX Pro 6000 (SM120) for FSDP2/layerwise plumbing tests + dense NVFP4
#   * GB200 (SM100a) for NVFP4 MoE end-to-end validation
#
# Tag convention:
#   :dev          -> mainline of feat/quanty/fast-multinode-layerwise-ptq
#   :sha-<short>  -> immutable per-commit
#   :pr-<n>       -> PR builds

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3
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

# OS deps: git for editable installs, tini as PID-1, jq for json wrangling in launchers,
# unzip for downloading kernel wheels, openssh-client for git+ssh model fetches.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git tini jq unzip openssh-client ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/quanty

# Install modelopt + recipes editable so the image picks up branch changes
# at build time without re-baking the wheel on every iteration.
COPY pyproject.toml noxfile.py uv.lock ./
COPY modelopt/ ./modelopt/
COPY modelopt_recipes/ ./modelopt_recipes/
COPY examples/ ./examples/
COPY tests/ ./tests/

# transformers 5.x for K2.6 / DeepseekV3 / Qwen3-MoE modeling code;
# accelerate 1.6+ for FSDP2 layerwise composition.  Other deps under
# ``modelopt[hf,torch]`` are left to modelopt's pyproject to resolve.
#
# The NGC PyTorch image ships a global ``PIP_CONSTRAINT`` file that pins
# every base-image package (e.g. ``dill==0.3.9``, ``pyarrow==19.0.1``).
# Those pins fight modern ``datasets`` (which wants ``dill<0.3.9``).
# We disable the constraint file for this layer only; everything else
# in the image keeps the base-image pins.
#
# FlashInfer / vLLM are intentionally NOT installed: PTQ does not import
# them, and serving images install matched versions themselves.
RUN PIP_CONSTRAINT= pip install --upgrade pip wheel setuptools \
 && PIP_CONSTRAINT= pip install --upgrade \
        "transformers>=5.0.0" \
        "accelerate>=1.6.0" \
        "datasets>=3.0.0,<4" \
        "torchvision" \
 && PIP_CONSTRAINT= pip install \
        "compressed_tensors>=0.13.0" \
        "hf_transfer>=0.1.6" \
        "numpy<2.0" \
        "pyyaml>=6.0" \
        "jsonlines>=4.0.0" \
        "lm-eval>=0.4.5" \
 && PIP_CONSTRAINT= pip install -e ".[hf,torch]"

# Sanity probes: a busted install must fail the build, not the cluster job.
RUN python -c "import modelopt; import modelopt.torch.quantization as mtq; \
print('modelopt', getattr(modelopt, '__version__', 'editable')); \
import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); \
import transformers; print('transformers', transformers.__version__); \
import accelerate; print('accelerate', accelerate.__version__)"

# Quanty-specific harnesses + recipe overlays.
COPY quanty/scripts/ /opt/quanty/quanty/scripts/
COPY quanty/recipes/ /opt/quanty/quanty/recipes/

# Make scripts executable here (writeable host filesystems sometimes lose
# the +x bit on copy).
RUN chmod +x /opt/quanty/quanty/scripts/*.sh /opt/quanty/quanty/scripts/*.py 2>/dev/null || true

# NGC base image already owns UID 1000 (ubuntu user); we run as root in
# the container.  k8s `securityContext.runAsNonRoot` / `runAsUser` is the
# correct place to enforce non-root in production deployments.
WORKDIR /work

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-l"]
