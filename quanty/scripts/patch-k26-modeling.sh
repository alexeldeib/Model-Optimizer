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

TARGET="${SRC}/modeling_kimi_k25.py"
if [[ ! -f "${TARGET}" ]]; then
    echo "skip: ${TARGET} not present" >&2
    exit 0
fi

# Only mutate when the buggy substring is still present.
if grep -q "use_deterministic_attn=self\.use_deterministic_attn" "${TARGET}"; then
    sed -i 's/use_deterministic_attn=self\.use_deterministic_attn/use_deterministic_attn=False/g' "${TARGET}"
    echo "patched: ${TARGET}"
else
    echo "already patched: ${TARGET}"
fi
