import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def _load_inspector():
    path = Path(__file__).parents[3] / "quanty" / "scripts" / "inspect-nvfp4-export.py"
    spec = importlib.util.spec_from_file_location("inspect_nvfp4_export", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mlp_only_allows_shared_expert_and_layer0_scale_tensors():
    inspector = _load_inspector()
    keys = [
        "language_model.model.layers.1.mlp.shared_experts.gate_proj.input_scale",
        "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight_scale",
        "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight_scale_2",
        "language_model.model.layers.0.mlp.gate_proj.input_scale",
        "language_model.model.layers.0.mlp.gate_proj.weight_scale",
        "language_model.model.layers.0.mlp.gate_proj.weight_scale_2",
    ]

    warnings = inspector.build_warnings(
        groups=[],
        coverage=inspector.count_coverage(keys),
        expected_experts=None,
        expected_moe_layers=None,
        allow_shared_experts=True,
        allow_layer0_mlp=True,
        expected_kv_cache="",
    )

    assert warnings == []


def test_unallowed_non_expert_scale_tensor_still_warns():
    inspector = _load_inspector()
    keys = ["language_model.model.layers.2.mlp.gate_proj.weight_scale"]

    warnings = inspector.build_warnings(
        groups=[],
        coverage=inspector.count_coverage(keys),
        expected_experts=None,
        expected_moe_layers=None,
        allow_shared_experts=True,
        allow_layer0_mlp=True,
        expected_kv_cache="",
    )

    assert any("outside the allowed K2.6 MLP set" in warning for warning in warnings)


def test_gate_up_pair_scan_covers_dense_and_shared_mlp(tmp_path):
    inspector = _load_inspector()
    shard = tmp_path / "model-00001-of-00001.safetensors"
    tensors = {
        "language_model.model.layers.0.mlp.gate_proj.weight_scale_2": torch.tensor(4.0),
        "language_model.model.layers.0.mlp.up_proj.weight_scale_2": torch.tensor(3.0),
        "language_model.model.layers.1.mlp.shared_experts.gate_proj.weight_scale_2": torch.tensor(2.0),
        "language_model.model.layers.1.mlp.shared_experts.up_proj.weight_scale_2": torch.tensor(5.0),
        "language_model.model.layers.1.mlp.experts.0.gate_proj.weight_scale_2": torch.tensor(7.0),
        "language_model.model.layers.1.mlp.experts.0.up_proj.weight_scale_2": torch.tensor(7.0),
    }
    save_file(tensors, shard)
    weight_map = {key: shard.name for key in tensors}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )

    scan = inspector.scan_gate_up_scalar_pairs(
        tmp_path,
        weight_map,
        {"weight_scale_2"},
    )

    assert scan["weight_scale_2"]["compared_pairs"] == 3
    assert scan["weight_scale_2"]["mismatched_pairs"] == 2
    groups = {example["group"] for example in scan["weight_scale_2"]["examples"]}
    assert groups == {"dense_mlp", "shared_experts"}
