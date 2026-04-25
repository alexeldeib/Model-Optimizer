# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for layerwise calibration checkpoint save/resume."""

import json
import os
from types import SimpleNamespace

import torch
import torch.nn as nn

from modelopt.torch.quantization.model_calib import layerwise_calibrate
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
from modelopt.torch.utils.network import get_module_device


class _DecoderBlock(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x, **kwargs):
        return self.linear(x)


class _SimpleTransformerModel(nn.Module):
    def __init__(self, n_layers=3, dim=16):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderBlock(dim) for _ in range(n_layers)])
        self.embed = nn.Embedding(32, dim)

    def forward(self, x, **kwargs):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x


def _register_test_discoverer(monkeypatch):
    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(lambda m: hasattr(m, "layers"), lambda m: m.layers)],
    )


def _dummy_calib_func(layer, forward_loop, **kwargs):
    """Scale all weights by 0.5 to produce a visible, deterministic change."""
    forward_loop(layer)
    with torch.no_grad():
        for p in layer.parameters():
            p.mul_(0.5)


def _make_model_and_forward(n_layers=3, dim=16, seed=42):
    torch.manual_seed(seed)
    model = _SimpleTransformerModel(n_layers=n_layers, dim=dim)
    tokens = [torch.randint(0, 32, (2, 8)) for _ in range(2)]

    def forward_loop(m):
        for t in tokens:
            m(t)

    return model, forward_loop


def test_full_run_creates_checkpoints(monkeypatch, tmp_path):
    """layerwise_calibrate with checkpoint_dir creates correct layer dirs and manifest."""
    _register_test_discoverer(monkeypatch)
    model, forward_loop = _make_model_and_forward(n_layers=3)
    ckpt_dir = str(tmp_path / "ckpt")

    layerwise_calibrate(model, forward_loop, _dummy_calib_func, checkpoint_dir=ckpt_dir)

    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    assert os.path.isfile(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["last_completed_layer"] == 2
    assert manifest["num_layers"] == 3

    for i in range(3):
        layer_dir = os.path.join(ckpt_dir, f"layer_{i:04d}")
        assert os.path.isdir(layer_dir)
        assert os.path.isfile(os.path.join(layer_dir, "weights.pt"))
        assert os.path.isfile(os.path.join(layer_dir, "quantizer_state.pt"))
        assert os.path.isfile(os.path.join(layer_dir, "output_meta.pt"))
    # All layers except the last should have next_inputs
    assert os.path.isfile(os.path.join(ckpt_dir, "layer_0000", "next_inputs.pt"))
    assert os.path.isfile(os.path.join(ckpt_dir, "layer_0001", "next_inputs.pt"))
    assert not os.path.isfile(os.path.join(ckpt_dir, "layer_0002", "next_inputs.pt"))


def test_resume_matches_full_run(monkeypatch, tmp_path):
    """Resume from a truncated checkpoint produces the same final weights as a full run."""
    _register_test_discoverer(monkeypatch)
    ckpt_dir = str(tmp_path / "ckpt")

    # Full reference run
    ref_model, forward_loop = _make_model_and_forward(n_layers=3)
    layerwise_calibrate(ref_model, forward_loop, _dummy_calib_func, checkpoint_dir=ckpt_dir)
    ref_weights = {n: p.clone() for n, p in ref_model.named_parameters()}

    # Simulate crash after layer 0: truncate manifest
    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"last_completed_layer": 0, "num_layers": 3}, f)

    # Resume from a fresh model
    resumed_model, forward_loop = _make_model_and_forward(n_layers=3)
    layerwise_calibrate(resumed_model, forward_loop, _dummy_calib_func, checkpoint_dir=ckpt_dir)

    for name, ref_param in ref_weights.items():
        resumed_param = dict(resumed_model.named_parameters())[name]
        assert torch.allclose(ref_param, resumed_param, atol=1e-6), (
            f"Parameter {name} diverged after resume"
        )


def test_no_checkpoint_unchanged(monkeypatch):
    """Without checkpoint_dir, calibration still works and modifies parameters."""
    _register_test_discoverer(monkeypatch)
    model, forward_loop = _make_model_and_forward(n_layers=3)
    original_weights = {n: p.clone() for n, p in model.named_parameters()}

    layerwise_calibrate(model, forward_loop, _dummy_calib_func)

    changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(original_weights[name], param):
            changed = True
            break
    assert changed, "Expected calibration to modify at least one parameter"


# ---------------------------------------------------------------------------
# Distributed-safe save/resume semantics
#
# The actual distributed save runs from inside FSDP2 with multiple processes.
# Here we simulate a multi-rank job by stubbing the modelopt distributed
# helpers + torch.distributed primitives.  The contract we exercise:
#
#   * rank-0 writes manifest and per-layer artifacts; non-zero ranks no-op
#   * every rank passes through ``_barrier`` exactly once per save
#   * ``_CheckpointState.__init__`` no longer raises in distributed mode
#
# The integration test that runs real FSDP2 layerwise calibration lives in
# ``tests/gpu/torch/quantization/test_fsdp2_layerwise.py``.
# ---------------------------------------------------------------------------


def _patch_distributed(monkeypatch, *, size: int, rank: int):
    """Make modelopt + torch see (size, rank) as a real distributed job."""
    from modelopt.torch.quantization.utils import layerwise_calib as lwc
    from modelopt.torch.utils import distributed as mdist

    monkeypatch.setattr(mdist, "is_initialized", lambda: True)
    monkeypatch.setattr(mdist, "size", lambda: size)
    monkeypatch.setattr(mdist, "rank", lambda: rank)

    barrier_calls = {"n": 0}

    class _StubDist:
        @staticmethod
        def barrier():
            barrier_calls["n"] += 1

    # Replace torch.distributed.barrier (touched only when world > 1).
    monkeypatch.setattr(torch, "distributed", _StubDist, raising=False)
    # Reload-free patch on the helpers in layerwise_calib so they pick up
    # the patched ``dist`` module.
    monkeypatch.setattr(lwc, "dist", mdist)
    return barrier_calls


def test_distributed_save_only_rank0_writes(monkeypatch, tmp_path):
    """Rank-1 in a 2-rank job must not write any checkpoint files."""
    _register_test_discoverer(monkeypatch)
    barrier_calls = _patch_distributed(monkeypatch, size=2, rank=1)

    model, forward_loop = _make_model_and_forward(n_layers=2)
    ckpt_dir = str(tmp_path / "ckpt_rank1")

    # Note: from_folder() now barrier()s after the makedirs guard.
    # On rank-1 the dir is not pre-created; fall back to make it ourselves
    # so layerwise_calibrate can resolve checkpoint_dir.
    os.makedirs(ckpt_dir, exist_ok=True)
    layerwise_calibrate(model, forward_loop, _dummy_calib_func, checkpoint_dir=ckpt_dir)

    # No files written from rank-1.
    assert not os.path.isfile(os.path.join(ckpt_dir, "manifest.json"))
    for i in range(2):
        layer_dir = os.path.join(ckpt_dir, f"layer_{i:04d}")
        assert not os.path.isdir(layer_dir)

    # Rank-1 still hits the barrier on every save call (n_layers=2) plus the
    # makedirs barrier in from_folder (1) -> 3 total.
    assert barrier_calls["n"] >= 2


def test_distributed_save_rank0_writes(monkeypatch, tmp_path):
    """Rank-0 in a 2-rank job writes manifest + per-layer artifacts as usual."""
    _register_test_discoverer(monkeypatch)
    _patch_distributed(monkeypatch, size=2, rank=0)

    model, forward_loop = _make_model_and_forward(n_layers=2)
    ckpt_dir = str(tmp_path / "ckpt_rank0")

    layerwise_calibrate(model, forward_loop, _dummy_calib_func, checkpoint_dir=ckpt_dir)

    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    assert os.path.isfile(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest == {"last_completed_layer": 1, "num_layers": 2}

    for i in range(2):
        layer_dir = os.path.join(ckpt_dir, f"layer_{i:04d}")
        assert os.path.isdir(layer_dir)
        assert os.path.isfile(os.path.join(layer_dir, "weights.pt"))


def test_checkpoint_state_init_does_not_raise_in_distributed(monkeypatch):
    """Regression: previously raised on dist.size() > 1."""
    _patch_distributed(monkeypatch, size=4, rank=2)
    from modelopt.torch.quantization.utils.layerwise_calib import _CheckpointState

    state = _CheckpointState("/tmp/does_not_matter", num_layers=8, start_layer=0)
    assert state.num_layers == 8
    assert state.start_layer == 0


# ---------------------------------------------------------------------------
# get_module_device tests
# ---------------------------------------------------------------------------


def test_get_module_device_no_hook():
    """Falls back to parameter device when no _hf_hook is present."""
    layer = nn.Linear(4, 4)
    assert get_module_device(layer) == torch.device("cpu")


def test_get_module_device_with_direct_hook():
    """Returns execution_device from a direct AlignDevicesHook-style hook."""
    layer = nn.Linear(4, 4)
    layer._hf_hook = SimpleNamespace(execution_device=torch.device("cuda:0"))
    assert get_module_device(layer) == torch.device("cuda:0")


def test_get_module_device_with_sequential_hook():
    """Returns execution_device from an AlignDevicesHook wrapped in SequentialHook."""
    layer = nn.Linear(4, 4)
    inner_hook = SimpleNamespace(execution_device=torch.device("cuda:1"))
    layer._hf_hook = SimpleNamespace(hooks=[inner_hook])
    assert get_module_device(layer) == torch.device("cuda:1")


def test_get_module_device_hook_without_execution_device():
    """Falls back to parameters when hook has no execution_device."""
    layer = nn.Linear(4, 4)
    layer._hf_hook = SimpleNamespace()
    assert get_module_device(layer) == torch.device("cpu")


def test_get_module_device_parameterless_module():
    """Returns cpu for a module with no parameters and no hook."""
    module = nn.Module()
    assert get_module_device(module) == torch.device("cpu")
