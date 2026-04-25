#!/usr/bin/env bash
#
# Targeted pytest runner for the upstream-bound commits in this branch.
#
# Runs only the unit tests we touched, not the full modelopt suite.  These
# tests use stubbed distributed primitives + tiny CPU-only Llama configs
# so they finish in seconds and do not require a GPU (despite living
# inside the GPU-targeted depot image).
#
# Use this from the depot image OR from a host with the modelopt editable
# install + pytest available.

set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo /opt/quanty)"

TESTS=(
    "tests/unit/torch/quantization/plugins/test_huggingface.py::test_get_homogeneous_hf_decoder_layers_vlm"
    "tests/unit/torch/quantization/plugins/test_huggingface.py::test_get_homogeneous_hf_decoder_layers_returns_none_for_unwrapped"
    "tests/unit/torch/quantization/plugins/test_huggingface.py::test_hf_decoder_discoverer_registration_path"
    "tests/unit/torch/quantization/test_sequential_checkpoint.py::test_distributed_save_only_rank0_writes"
    "tests/unit/torch/quantization/test_sequential_checkpoint.py::test_distributed_save_rank0_writes"
    "tests/unit/torch/quantization/test_sequential_checkpoint.py::test_checkpoint_state_init_does_not_raise_in_distributed"
    "tests/unit/torch/quantization/test_sequential_checkpoint.py::test_full_run_creates_checkpoints"
    "tests/unit/torch/quantization/test_sequential_checkpoint.py::test_resume_matches_full_run"
    "tests/unit/torch/quantization/test_sequential_checkpoint.py::test_no_checkpoint_unchanged"
)

# Override modelopt's pyproject ``addopts`` (they pull in pytest-instafail
# / pytest-cov which the depot image intentionally does not install).
exec python -m pytest -x -v --no-header -o "addopts=" "${TESTS[@]}" "$@"
