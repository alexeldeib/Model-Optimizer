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

"""Tests for _is_sparse_sequaential_moe_block and _QuantSparseSequentialMoe."""

import copy

import pytest
import torch
import torch.nn as nn
from packaging.version import Version

pytest.importorskip("transformers")

TORCH_VERSION_LT_2_9 = Version(torch.__version__) < Version("2.9")

from _test_utils.torch.transformers_models import get_tiny_qwen3_moe

import modelopt.torch.quantization as mtq
from modelopt.torch.export.layer_utils import (
    count_moe_gate_up_export_buffer_mismatches,
    get_expert_linear_names,
    is_moe,
    sync_direct_gate_up_amax,
    sync_moe_gate_up_amax,
)
from modelopt.torch.quantization.model_calib import enable_stats_collection, finish_stats_collection
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer, QuantModuleRegistry, TensorQuantizer
from modelopt.torch.quantization.plugins.huggingface import (
    TRANSFORMERS_VERSION_GE_5_0,
    _is_sparse_sequaential_moe_block,
    register_sparse_moe_on_the_fly,
)


# ---------------------------------------------------------------------------
# Helpers: lightweight mock modules for _is_sparse_sequaential_moe_block
# ---------------------------------------------------------------------------
class _FakeGateWithRouter(nn.Module):
    """Mimics a v5.x TopKRouter gate with top_k and num_experts."""

    def __init__(self, top_k=2, num_experts=4):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.linear = nn.Linear(8, num_experts)

    def forward(self, x):
        return self.linear(x)


class _FakeExperts(nn.ModuleList):
    def __init__(self, n=4):
        super().__init__([nn.Linear(8, 8) for _ in range(n)])
        self.num_experts = n


class _MoEBlockWithGateRouter(nn.Module):
    """Matches the primary detection path: gate.top_k + gate.num_experts."""

    def __init__(self, num_experts=4, top_k=2):
        super().__init__()
        self.gate = _FakeGateWithRouter(top_k=top_k, num_experts=num_experts)
        self.experts = _FakeExperts(num_experts)

    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        routing_weights, selected = torch.topk(logits, self.gate.top_k, dim=-1)
        out = torch.zeros_like(hidden_states)
        for i in range(self.gate.num_experts):
            mask = (selected == i).any(dim=-1)
            if mask.any():
                out[mask] += self.experts[i](hidden_states[mask])
        return out


class _MoEBlockFallback(nn.Module):
    """Matches the fallback path: top_k + num_experts on the block itself."""

    def __init__(self, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(8, num_experts)
        self.experts = _FakeExperts(num_experts)

    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        routing_weights, selected = torch.topk(logits, self.top_k, dim=-1)
        out = torch.zeros_like(hidden_states)
        for i in range(self.num_experts):
            mask = (selected == i).any(dim=-1)
            if mask.any():
                out[mask] += self.experts[i](hidden_states[mask])
        return out


class _CollapsedKimiGate(nn.Module):
    """Kimi/DeepSeek-style gate that always chooses the same experts."""

    def __init__(self, hidden_size=8, num_experts=16, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.n_routed_experts = num_experts
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states):
        num_tokens = hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0]
        idx = torch.arange(self.top_k, device=hidden_states.device).expand(num_tokens, -1)
        weight = torch.ones(num_tokens, self.top_k, device=hidden_states.device)
        return idx, weight


class _CountingLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.forward_calls = 0

    def forward(self, input):
        self.forward_calls += 1
        return super().forward(input)


class _CollapsedKimiMoE(nn.Module):
    """Small DeepSeek/Kimi-style sequential MoE with a ``moe_infer`` method."""

    def __init__(self, hidden_size=8, num_experts=16, top_k=2):
        super().__init__()
        self.gate = _CollapsedKimiGate(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList(
            [_CountingLinear(hidden_size, hidden_size) for _ in range(num_experts)]
        )
        self.moe_infer_topks = []

    def moe_infer(self, x, topk_ids, topk_weight):
        self.moe_infer_topks.append(topk_ids.shape[-1])
        out = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, topk_idx = torch.where(topk_ids == expert_idx)
            if token_idx.numel():
                out.index_add_(0, token_idx, expert(x[token_idx]) * topk_weight[token_idx, topk_idx, None])
        return out

    def forward(self, hidden_states):
        original_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        return self.moe_infer(
            hidden_states.reshape(-1, hidden_states.shape[-1]), topk_idx, topk_weight
        ).view(original_shape)


class _DeepseekExpert(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, hidden_size)
        self.up_proj = nn.Linear(hidden_size, hidden_size)


class DeepseekV3MoE(nn.Module):
    """Minimal class-name/structure match for Kimi K2.6 routed MoE blocks."""

    def __init__(self, num_experts=4):
        super().__init__()
        self.gate = _FakeGateWithRouter(top_k=2, num_experts=num_experts)
        self.experts = nn.ModuleList([_DeepseekExpert() for _ in range(num_experts)])


class _FakeWeightQuantizer:
    is_enabled = True

    def __init__(self, amax: float, global_amax: float):
        self.amax = torch.tensor(amax)
        self.global_amax = torch.tensor(global_amax)


def _attach_fake_weight_quantizer(module: nn.Module, amax: float, global_amax: float) -> None:
    module.weight_quantizer = _FakeWeightQuantizer(amax, global_amax)


class _GatedMlp(nn.Module):
    def __init__(self, gate_amax: float, up_amax: float):
        super().__init__()
        self.gate_proj = nn.Linear(8, 8)
        self.up_proj = nn.Linear(8, 8)
        _attach_fake_weight_quantizer(self.gate_proj, gate_amax, gate_amax)
        _attach_fake_weight_quantizer(self.up_proj, up_amax, up_amax)


# ---------------------------------------------------------------------------
# Tests for _is_sparse_sequaential_moe_block
# ---------------------------------------------------------------------------
class TestIsSparseBlock:
    def test_no_experts_returns_false(self):
        module = nn.Linear(8, 8)
        assert _is_sparse_sequaential_moe_block(module) is False

    def test_experts_but_no_gate_or_topk_returns_false(self):
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        assert _is_sparse_sequaential_moe_block(module) is False

    def test_gate_with_router_attrs_returns_true(self):
        block = _MoEBlockWithGateRouter(num_experts=4, top_k=2)
        assert _is_sparse_sequaential_moe_block(block) is True

    def test_fallback_block_level_attrs_returns_true(self):
        block = _MoEBlockFallback(num_experts=4, top_k=2)
        assert _is_sparse_sequaential_moe_block(block) is True

    def test_gate_missing_num_experts_returns_false(self):
        """gate.top_k present but gate.num_experts absent -> primary path fails."""
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        gate = nn.Module()
        gate.top_k = 2
        module.gate = gate
        assert _is_sparse_sequaential_moe_block(module) is False

    def test_gate_missing_top_k_returns_false(self):
        """gate.num_experts present but gate.top_k absent -> primary path fails."""
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        gate = nn.Module()
        gate.num_experts = 4
        module.gate = gate
        assert _is_sparse_sequaential_moe_block(module) is False

    def test_block_level_top_k_infers_num_experts(self):
        """top_k on block + experts with __len__ -> num_experts is inferred, returns True."""
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        module.top_k = 2
        assert _is_sparse_sequaential_moe_block(module) is True
        assert module.num_experts == 1

    def test_block_level_top_k_no_len_returns_false(self):
        """top_k on block but experts has no __len__ -> cannot infer num_experts, returns False."""
        module = nn.Module()
        module.experts = nn.Module()
        module.top_k = 2
        assert _is_sparse_sequaential_moe_block(module) is False

    def test_block_level_only_num_experts_returns_false(self):
        """Only num_experts on block (no top_k) -> fallback fails."""
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        module.num_experts = 4
        assert _is_sparse_sequaential_moe_block(module) is False

    def test_n_routed_experts_accepted(self):
        """A module with n_routed_experts (NemotronH-style) should be accepted."""
        module = nn.Module()
        module.experts = nn.ModuleList([nn.Linear(8, 8)])
        gate = nn.Module()
        gate.top_k = 2
        gate.n_routed_experts = 4
        module.gate = gate
        assert _is_sparse_sequaential_moe_block(module) is True


def test_export_utils_detect_deepseek_v3_moe_structurally():
    """Kimi K2.6 uses DeepseekV3MoE, not the older DeepseekMoE class name."""

    module = DeepseekV3MoE()

    assert is_moe(module) is True
    assert get_expert_linear_names(module) == ["gate_proj", "down_proj", "up_proj"]


def test_gate_up_amax_sync_covers_dense_shared_and_routed_pairs():
    """Serving fusion needs shared gate/up scale-2 beyond routed experts."""

    model = nn.Module()
    model.dense_mlp = _GatedMlp(gate_amax=4.0, up_amax=3.0)
    model.shared_experts = _GatedMlp(gate_amax=2.0, up_amax=5.0)
    model.routed_expert = _GatedMlp(gate_amax=7.0, up_amax=1.0)

    assert sync_moe_gate_up_amax(model) == 3

    for module, expected in [
        (model.dense_mlp, 4.0),
        (model.shared_experts, 5.0),
        (model.routed_expert, 7.0),
    ]:
        gate_wq = module.gate_proj.weight_quantizer
        up_wq = module.up_proj.weight_quantizer
        assert torch.equal(gate_wq.amax, torch.tensor(expected))
        assert torch.equal(up_wq.amax, torch.tensor(expected))
        assert torch.equal(gate_wq.global_amax, torch.tensor(expected))
        assert torch.equal(up_wq.global_amax, torch.tensor(expected))


def test_gate_up_export_buffer_sync_covers_dense_shared_and_routed_pairs():
    """Post-export gate/up scale mismatch must be detected, not mutated."""

    model = nn.Module()
    model.dense_mlp = _GatedMlp(gate_amax=4.0, up_amax=3.0)
    model.shared_experts = _GatedMlp(gate_amax=2.0, up_amax=5.0)
    model.routed_expert = _GatedMlp(gate_amax=7.0, up_amax=1.0)

    model.dense_mlp.gate_proj.register_buffer("weight_scale_2", torch.tensor(0.4))
    model.dense_mlp.up_proj.register_buffer("weight_scale_2", torch.tensor(0.3))
    model.shared_experts.gate_proj.register_buffer("weight_scale_2", torch.tensor(0.2))
    model.shared_experts.up_proj.register_buffer("weight_scale_2", torch.tensor(0.5))
    model.routed_expert.gate_proj.register_buffer("weight_scale_2", torch.tensor(0.7))
    model.routed_expert.up_proj.register_buffer("weight_scale_2", torch.tensor(0.1))

    assert count_moe_gate_up_export_buffer_mismatches(model) == 3

    assert torch.equal(model.dense_mlp.gate_proj.weight_scale_2, torch.tensor(0.4))
    assert torch.equal(model.dense_mlp.up_proj.weight_scale_2, torch.tensor(0.3))
    assert torch.equal(model.shared_experts.gate_proj.weight_scale_2, torch.tensor(0.2))
    assert torch.equal(model.shared_experts.up_proj.weight_scale_2, torch.tensor(0.5))
    assert torch.equal(model.routed_expert.gate_proj.weight_scale_2, torch.tensor(0.7))
    assert torch.equal(model.routed_expert.up_proj.weight_scale_2, torch.tensor(0.1))


def _nvfp4_static_quantizer(global_amax: float | None = None, amax: torch.Tensor | None = None):
    quantizer = TensorQuantizer()
    quantizer.block_sizes = {-1: 4, "type": "dynamic", "scale_bits": (4, 3)}
    quantizer = NVFP4StaticQuantizer.from_tensor_quantizer(quantizer)
    if amax is not None:
        quantizer.amax = amax
    if global_amax is not None:
        quantizer.global_amax = torch.tensor(global_amax)
    return quantizer


def test_nvfp4_static_gate_up_sync_only_shares_global_amax():
    """NVFP4 static fused serving needs shared global scale, not shared block scales."""

    module = _GatedMlp(gate_amax=1.0, up_amax=1.0)
    gate_amax = torch.arange(1, 17, dtype=torch.float32).reshape(8, 2)
    up_amax = torch.arange(17, 33, dtype=torch.float32).reshape(8, 2)
    module.gate_proj.weight_quantizer = _nvfp4_static_quantizer(4.0, gate_amax)
    module.up_proj.weight_quantizer = _nvfp4_static_quantizer(7.0, up_amax)

    assert sync_direct_gate_up_amax(module, context="mlp") == 1

    assert torch.equal(module.gate_proj.weight_quantizer.amax, gate_amax)
    assert torch.equal(module.up_proj.weight_quantizer.amax, up_amax)
    assert torch.equal(module.gate_proj.weight_quantizer.global_amax, torch.tensor(7.0))
    assert torch.equal(module.up_proj.weight_quantizer.global_amax, torch.tensor(7.0))


def test_deepseek_moe_calibration_counts_kimi_gate_tuple():
    """Kimi/DeepSeek gates return ``(topk_idx, topk_weight)`` instead of logits."""

    module = _CollapsedKimiMoE(num_experts=16, top_k=2)
    if QuantModuleRegistry.get(type(module)) is None:
        register_sparse_moe_on_the_fly(module)

    converted = QuantModuleRegistry.convert(module)
    converted._moe_calib_experts_ratio = 0.5
    converted.experts[0]._if_calib = True

    x = torch.randn(1, 16, 8)
    with torch.no_grad():
        out = converted(x)

    assert out.shape == x.shape
    assert converted.gate.top_k == 2
    assert hasattr(converted, "expert_token_count")
    assert converted.expert_token_count[:8].tolist() == [16] * 8
    assert converted.expert_token_count[8:].sum().item() == 0


def test_deepseek_moe_full_ratio_still_counts_coverage():
    """Coverage diagnostics should work for ratio=1.0 full expert calibration."""

    module = _CollapsedKimiMoE(num_experts=16, top_k=2)
    if QuantModuleRegistry.get(type(module)) is None:
        register_sparse_moe_on_the_fly(module)

    converted = QuantModuleRegistry.convert(module)
    converted._moe_calib_experts_ratio = 1.0
    converted.experts[0]._if_calib = True

    x = torch.randn(1, 16, 8)
    with torch.no_grad():
        out = converted(x)

    assert out.shape == x.shape
    assert converted.gate.top_k == 2
    assert hasattr(converted, "expert_token_count")
    assert converted.expert_token_count.tolist() == [16] * 16
    assert max(converted.moe_infer_topks) == 2
    assert all(expert.forward_calls > 0 for expert in converted.experts)


def test_deepseek_moe_full_ratio_uses_bounded_gate_candidate_tokens(monkeypatch):
    """Full coverage should not force every expert to calibrate on every token."""

    monkeypatch.setenv("MODELOPT_MOE_CALIB_TOKENS_PER_EXPERT", "3")
    module = _CollapsedKimiMoE(num_experts=16, top_k=2)
    if QuantModuleRegistry.get(type(module)) is None:
        register_sparse_moe_on_the_fly(module)

    converted = QuantModuleRegistry.convert(module)
    converted._moe_calib_experts_ratio = 1.0
    converted.experts[0]._if_calib = True

    x = torch.randn(1, 16, 8)
    with torch.no_grad():
        out = converted(x)

    assert out.shape == x.shape
    assert converted.expert_token_count.tolist() == [3] * 16
    assert converted._expert_token_count_source == "gate_top_tokens"
    assert max(converted.moe_infer_topks) == 2


def test_deepseek_moe_full_ratio_uses_model_calibration_flag(monkeypatch):
    """MoE forcing should not depend on expert quantizer _if_calib internals."""

    monkeypatch.setenv("MODELOPT_MOE_CALIB_TOKENS_PER_EXPERT", "3")
    module = _CollapsedKimiMoE(num_experts=16, top_k=2)
    if QuantModuleRegistry.get(type(module)) is None:
        register_sparse_moe_on_the_fly(module)

    converted = QuantModuleRegistry.convert(module)
    converted._moe_calib_experts_ratio = 1.0

    x = torch.randn(1, 16, 8)
    with torch.no_grad():
        enable_stats_collection(converted)
        try:
            out = converted(x)
        finally:
            finish_stats_collection(converted)

    assert out.shape == x.shape
    assert converted.expert_token_count.tolist() == [3] * 16
    assert converted._expert_token_count_source == "gate_top_tokens"
    assert converted._modelopt_moe_calibrating is False
    assert max(converted.moe_infer_topks) == 2


# ---------------------------------------------------------------------------
# Tests for _QuantSparseSequentialMoe
# ---------------------------------------------------------------------------
@pytest.mark.skipif(TORCH_VERSION_LT_2_9, reason="torch 2.8 grouped_mm is CUDA-only")
@pytest.mark.skipif(TRANSFORMERS_VERSION_GE_5_0, reason="Transformers v5 has stacked MoE")
class TestQuantSparseSequentialMoe:
    """Tests for _QuantSparseSequentialMoe using a real tiny Qwen3Moe model."""

    @staticmethod
    def _get_moe_block(model):
        """Return the first MoE block from the model."""
        for module in model.modules():
            if _is_sparse_sequaential_moe_block(module):
                return module
        raise RuntimeError("No MoE block found in model")

    def test_register_sparse_moe_on_the_fly(self):
        model = get_tiny_qwen3_moe()
        moe_block = self._get_moe_block(model)
        moe_type = type(moe_block)

        if QuantModuleRegistry.get(moe_type) is not None:
            pytest.skip("MoE type already registered (upstream change)")

        register_sparse_moe_on_the_fly(model)
        assert QuantModuleRegistry.get(moe_type) is not None

    def test_setup_config_knobs_default(self):
        """_setup should only initialize config knobs, no buffer or hook."""
        model = get_tiny_qwen3_moe()
        moe_block = self._get_moe_block(model)
        if QuantModuleRegistry.get(type(moe_block)) is None:
            register_sparse_moe_on_the_fly(model)

        converted = QuantModuleRegistry.convert(moe_block)
        assert converted._moe_calib_experts_ratio is None
        assert not hasattr(converted, "expert_token_count")

    def test_forward_default_config_passthrough(self):
        """With default config (both features off), forward should be a direct pass-through."""
        model = get_tiny_qwen3_moe()
        moe_block = self._get_moe_block(model)
        if QuantModuleRegistry.get(type(moe_block)) is None:
            register_sparse_moe_on_the_fly(model)

        ref_block = self._get_moe_block(get_tiny_qwen3_moe())
        ref_block.load_state_dict(moe_block.state_dict())
        converted = QuantModuleRegistry.convert(moe_block)

        x = torch.randn(1, 4, 32, dtype=ref_block.gate.weight.dtype)
        with torch.no_grad():
            out_ref = ref_block(x)
            out_test = converted(x)

        if isinstance(out_ref, tuple):
            out_ref = out_ref[0]
        if isinstance(out_test, tuple):
            out_test = out_test[0]
        assert torch.allclose(out_ref, out_test, atol=1e-5)
        assert not hasattr(converted, "expert_token_count")

    def test_forward_calib_restores_top_k(self):
        """After calibration forward with moe_calib_experts_ratio, top_k should be restored."""
        model = get_tiny_qwen3_moe()
        moe_block = self._get_moe_block(model)
        if QuantModuleRegistry.get(type(moe_block)) is None:
            register_sparse_moe_on_the_fly(model)

        if TRANSFORMERS_VERSION_GE_5_0:
            original_top_k = moe_block.gate.top_k
        else:
            original_top_k = moe_block.top_k

        converted = QuantModuleRegistry.convert(moe_block)
        converted._moe_calib_experts_ratio = 1.0

        # Simulate calibration mode
        for m in converted.experts.modules():
            if hasattr(m, "_if_calib"):
                m._if_calib = True
                break

        x = torch.randn(1, 4, 32, dtype=converted.gate.weight.dtype)
        with torch.no_grad():
            converted(x)

        if TRANSFORMERS_VERSION_GE_5_0:
            assert converted.gate.top_k == original_top_k
        else:
            assert converted.top_k == original_top_k

    def test_token_counting_lazy_init(self):
        """When moe_calib_experts_ratio > 0, token counting infra is lazy-inited."""
        model = get_tiny_qwen3_moe()
        moe_block = self._get_moe_block(model)
        if QuantModuleRegistry.get(type(moe_block)) is None:
            register_sparse_moe_on_the_fly(model)

        converted = QuantModuleRegistry.convert(moe_block)
        converted._moe_calib_experts_ratio = 0.5

        assert not hasattr(converted, "expert_token_count")

        # Simulate calibration mode so lazy-init triggers during forward
        # Set _if_calib on an expert sub-module (not set by default since only the MoE
        # block was converted, not the full model).
        next(converted.experts.modules())._if_calib = True

        x = torch.randn(1, 4, 32, dtype=converted.gate.weight.dtype)
        with torch.no_grad():
            converted(x)

        # Buffer and hook should now exist
        assert hasattr(converted, "expert_token_count")
        assert converted.expert_token_count.numel() > 0

        # Manually enable counting and call gate to verify hook works
        converted._count_expert_tokens = True
        if TRANSFORMERS_VERSION_GE_5_0:
            hidden_size = converted.gate.weight.shape[1]
            top_k = converted.gate.top_k
        else:
            hidden_size = converted.gate.in_features
            top_k = converted.top_k if hasattr(converted, "top_k") else converted.gate.top_k

        converted.expert_token_count.zero_()
        tokens = torch.randn(8, hidden_size, dtype=converted.gate.weight.dtype)
        with torch.no_grad():
            converted.gate(tokens)
        assert converted.expert_token_count.sum().item() == 8 * top_k


@pytest.mark.skipif(TORCH_VERSION_LT_2_9, reason="torch 2.8 grouped_mm is CUDA-only")
@pytest.mark.skipif(TRANSFORMERS_VERSION_GE_5_0, reason="Transformers v5 has stacked MoE")
def test_qwen3_sequential_moe_quantize_with_token_forcing_and_counting():
    """End-to-end: mtq.quantize a Qwen3MoE with INT8 + moe_calib_experts_ratio + token counting."""
    model = get_tiny_qwen3_moe()

    # Verify detection
    moe_found = any(_is_sparse_sequaential_moe_block(m) for m in model.modules())
    assert moe_found, "Qwen3MoE should be detected as a sparse MoE block"

    quant_cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    quant_cfg["algorithm"] = {
        "method": "max",
        "moe_calib_experts_ratio": 0.5,
    }

    def calib_fn(model):
        x = model.dummy_inputs["input_ids"]
        for _ in range(2):
            model(x)

    mtq.quantize(model, quant_cfg, calib_fn)

    # Verify token counting worked
    for name, module in model.named_modules():
        if hasattr(module, "expert_token_count") and module.expert_token_count.numel() > 0:
            assert (module.expert_token_count > 0).all(), (
                f"Not all experts received tokens in {name}: {module.expert_token_count}"
            )

    # Verify model still runs
    with torch.no_grad():
        out = model(model.dummy_inputs["input_ids"])
    assert out.logits is not None
