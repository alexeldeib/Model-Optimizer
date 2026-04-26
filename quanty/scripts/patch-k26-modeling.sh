#!/usr/bin/env bash
#
# Patch the K2.6 trust_remote_code modeling files in-place.
#
# Bug:  MoonViT3dEncoder.__init__ references self.use_deterministic_attn
#       before it's assigned, raising AttributeError on first instantiation.
# Fix:  rewrite the offending kwarg to a literal False.  Equivalent to the
#       team's existing runtime patch in the modelopt path.
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

# Bug 1: MoonViT3dEncoder.__init__ uses ``self.use_deterministic_attn``
# before assignment.
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

# Bug 2: K2.6's ``modeling_deepseek.py`` imports
# ``is_torch_fx_available`` from ``transformers.utils.import_utils``.
# That symbol was removed in transformers 5.x (torch.fx is always
# available in torch 2.x, the gate is gone).  Replace the import with
# a constant True stub so callsites continue to work without the
# upstream symbol.
DEEPSEEK="${SRC}/modeling_deepseek.py"
if [[ -f "${DEEPSEEK}" ]]; then
    if grep -q "from transformers.utils.import_utils import is_torch_fx_available" "${DEEPSEEK}"; then
        sed -i 's|^from transformers\.utils\.import_utils import is_torch_fx_available$|def is_torch_fx_available(): return True  # quanty: transformers 5.x removed this symbol|' "${DEEPSEEK}"
        echo "patched: ${DEEPSEEK}"
    else
        echo "already patched: ${DEEPSEEK}"
    fi
else
    echo "skip: ${DEEPSEEK} not present" >&2
fi
