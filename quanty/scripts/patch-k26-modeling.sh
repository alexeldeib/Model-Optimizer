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

# Bug 3 (deepseek): the bundled ``modeling_deepseek.py`` does an
# unconditional ``from flash_attn import flash_attn_func, flash_attn_varlen_func``.
# Our depot image pre-installs flash_attn, but its C extension was
# built against a different torch ABI (the warning at startup is
# ``undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...``).
# Module-load fails before the runtime ``HAS_FLASH_ATTN`` gate ever
# fires.  Wrap the import in try/except + None fallback so the file
# loads cleanly and downstream gates degrade to native attention.
if [[ -f "${DEEPSEEK}" ]]; then
    if grep -q "^from flash_attn import flash_attn_func, flash_attn_varlen_func" "${DEEPSEEK}"; then
        python3 - "${DEEPSEEK}" <<'PY'
import sys, pathlib
path = pathlib.Path(sys.argv[1])
src = path.read_text()
old = "from flash_attn import flash_attn_func, flash_attn_varlen_func"
new = (
    "try:  # quanty: flash_attn C ext can be ABI-incompatible with the bundled torch\n"
    "    from flash_attn import flash_attn_func, flash_attn_varlen_func\n"
    "except ImportError:\n"
    "    flash_attn_func = None\n"
    "    flash_attn_varlen_func = None\n"
)
if old in src:
    path.write_text(src.replace(old, new))
PY
        echo "patched flash_attn guard: ${DEEPSEEK}"
    fi
fi

# Bug 4: K2.6's ``tokenization_kimi.py`` imports
# ``bytes_to_unicode`` from ``transformers.convert_slow_tokenizer``.
# That re-export was dropped in transformers 5.x (it now lives only
# under ``transformers.models.gpt2.tokenization_gpt2``).  Inline the
# canonical implementation so the module loads cleanly.
TOKENIZER="${SRC}/tokenization_kimi.py"
if [[ -f "${TOKENIZER}" ]]; then
    if grep -q "from transformers.convert_slow_tokenizer import bytes_to_unicode" "${TOKENIZER}"; then
        # Inline polyfill -- the canonical GPT-2 bytes-to-unicode mapping.
        python3 - "${TOKENIZER}" <<'PY'
import sys, pathlib
path = pathlib.Path(sys.argv[1])
src = path.read_text()
old = "from transformers.convert_slow_tokenizer import bytes_to_unicode"
new = (
    "# quanty: transformers 5.x dropped the convert_slow_tokenizer re-export.\n"
    "def bytes_to_unicode():\n"
    "    bs = (list(range(ord('!'), ord('~') + 1))\n"
    "          + list(range(ord('¡'), ord('¬') + 1))\n"
    "          + list(range(ord('®'), ord('ÿ') + 1)))\n"
    "    cs = bs[:]\n"
    "    n = 0\n"
    "    for b in range(2 ** 8):\n"
    "        if b not in bs:\n"
    "            bs.append(b)\n"
    "            cs.append(2 ** 8 + n)\n"
    "            n += 1\n"
    "    return dict(zip(bs, [chr(c) for c in cs]))\n"
)
if old in src:
    path.write_text(src.replace(old, new))
PY
        echo "patched: ${TOKENIZER}"
    else
        echo "already patched: ${TOKENIZER}"
    fi
else
    echo "skip: ${TOKENIZER} not present" >&2
fi
