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

# Pin to known-good versions for SM100a + SM120 NVFP4 paths.
# transformers 5.x for K2.6 / DeepSeek V3.2 / V4 support.
# accelerate 1.6+ for FSDP2 + recipe support.
# Pre-install FlashInfer wheel that includes the SM120 grouped-block-scaled
# GEMM patches (FlashInfer #2723 / CUTLASS #3096 workaround).  GB200 (SM100a)
# uses the standard wheel; the SM120 patched wheel is gated by build arg.
ARG FLASHINFER_WHEEL=flashinfer-python==0.6.0
RUN pip install --upgrade pip wheel setuptools \
 && pip install \
        "transformers>=5.0.0" \
        "accelerate>=1.6.0" \
        "datasets>=3.6.0" \
        "tokenizers>=0.21.0" \
        "huggingface_hub>=0.30.0" \
        "hf_transfer>=0.1.6" \
        "numpy<2.0" \
        "pyyaml>=6.0" \
        "jsonlines>=4.0.0" \
        "lm-eval>=0.4.5" \
        ${FLASHINFER_WHEEL} \
 && pip install -e ".[hf,torch]"

# Sanity probes baked at build time so a busted wheel fails the build,
# not the cluster job.  Each probe is small + fast.
RUN python -c "import modelopt; import modelopt.torch.quantization as mtq; \
print('modelopt', getattr(modelopt, '__version__', 'editable')); \
import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# vLLM compat: install AFTER modelopt because vLLM pins compressed_tensors.
# Used only by post-quant validation scripts; not load-bearing for PTQ itself.
ARG INSTALL_VLLM=1
RUN if [ "${INSTALL_VLLM}" = "1" ]; then \
        pip install "vllm>=0.10.0" "compressed_tensors>=0.13.0" || \
        echo "WARN: vLLM install failed; PTQ still works without it"; \
    fi

# Quanty-specific harnesses + recipe overlays.
COPY quanty/scripts/ /opt/quanty/quanty/scripts/
COPY quanty/recipes/ /opt/quanty/quanty/recipes/

# Make scripts executable here (writeable host filesystems sometimes lose
# the +x bit on copy).
RUN chmod +x /opt/quanty/quanty/scripts/*.sh /opt/quanty/quanty/scripts/*.py 2>/dev/null || true

# Default to a non-root entrypoint that's friendly to k8s readiness probes.
RUN useradd --create-home --uid 1000 --shell /bin/bash quanty \
 && chown -R quanty:quanty /opt/quanty
USER quanty
WORKDIR /work

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-l"]
