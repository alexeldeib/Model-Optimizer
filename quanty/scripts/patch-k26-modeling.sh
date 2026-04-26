#!/usr/bin/env bash
#
# Patch the K2.6 trust_remote_code modeling files in-place.
#
# Bug:  MoonViT3dEncoder.__init__ references self.use_deterministic_attn
#       before it's assigned, raising AttributeError on first instantiation.
# Fix:  rewrite the offending kwarg to a literal False.  Equivalent to the
#       team's existing runtime patch in the modelopt path.
#
# Earlier iterations of this script also patched ``modeling_deepseek.py``
# (``is_torch_fx_available`` removed in transformers 5.x, ``flash_attn``
# import unconditional, etc.) and ``tokenization_kimi.py``
# (``bytes_to_unicode`` re-export dropped).  Those issues all stemmed
# from upgrading torch + transformers on top of the NGC PyTorch base
# image, which silently invalidated the bundled C extensions.  The
# current image uses ``nvcr.io/nvidia/tensorrt-llm/release:1.2.0`` as
# base instead -- it ships the matched torch / flash_attn /
# transformer_engine / transformers stack, so those compat patches
# are no longer needed.  Only this real model bug remains.
#
# Usage:
#   patch-k26-modeling.sh <path-to-K2.6-source-dir>
#
# Idempotent: re-running on already-patched files is a no-op.

set -euo pipefail

SRC="${1:-}"
if [[ -z "${SRC}" ]]; then
    echo "usage: $0 <path-to-K2.6-source-dir>" >&2
    exit 2
fi

if [[ ! -d "${SRC}" ]]; then
    echo "no such directory: ${SRC}" >&2
    exit 2
fi

KIMI="${SRC}/modeling_kimi_k25.py"
if [[ -f "${KIMI}" ]]; then
    if grep -q "use_deterministic_attn=self\.use_deterministic_attn" "${KIMI}"; then
        sed -i 's/use_deterministic_attn=self\.use_deterministic_attn/use_deterministic_attn=False/g' "${KIMI}"
        echo "patched: ${KIMI}"
    else
        echo "already patched: ${KIMI}"
    fi
else
    echo "skip: ${KIMI} not present" >&2
fi
