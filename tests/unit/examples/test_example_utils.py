# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import types
from enum import Enum
from pathlib import Path

import torch
from safetensors.torch import save_file


def _load_example_utils():
    path = Path(__file__).parents[3] / "examples" / "llm_ptq" / "example_utils.py"
    spec = importlib.util.spec_from_file_location("example_utils_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_pack_quantized_config_ignores_plain_modules(tmp_path):
    example_utils = _load_example_utils()

    config = types.SimpleNamespace(
        text_config=types.SimpleNamespace(
            quantization_config={
                "format": "pack-quantized",
                "ignore": ["lm_head", "re:.*self_attn.*"],
            }
        )
    )
    weight_map = {
        "language_model.model.layers.1.mlp.experts.0.down_proj.weight_packed": "model.safetensors",
        "language_model.model.layers.1.mlp.experts.0.down_proj.weight_scale": "model.safetensors",
        "language_model.model.layers.0.mlp.down_proj.weight": "model.safetensors",
        "vision_tower.encoder.blocks.0.mlp.fc0.weight": "model.safetensors",
        "language_model.lm_head.weight": "model.safetensors",
    }
    with (tmp_path / "model.safetensors.index.json").open("w") as f:
        json.dump({"weight_map": weight_map}, f)

    added = example_utils._normalize_pack_quantized_config_for_mixed_checkpoint(
        config, str(tmp_path)
    )

    ignore = config.text_config.quantization_config["ignore"]
    assert added == 3
    assert "language_model.model.layers.1.mlp.experts.0.down_proj" not in ignore
    assert "language_model.model.layers.0.mlp.down_proj" in ignore
    assert "vision_tower.encoder.blocks.0.mlp.fc0" in ignore
    assert "language_model.lm_head" in ignore
    assert ignore[:2] == ["lm_head", "re:.*self_attn.*"]


def test_unpack_compressed_linear_weights_removes_stale_packed_plain_weights(
    monkeypatch, tmp_path
):
    class QuantizationStatus(Enum):
        FROZEN = "frozen"
        COMPRESSED = "compressed"

    class CompressedLinear(torch.nn.Linear):
        pass

    compressed_tensors = types.ModuleType("compressed_tensors")
    linear_pkg = types.ModuleType("compressed_tensors.linear")
    compressed_linear = types.ModuleType("compressed_tensors.linear.compressed_linear")
    quantization = types.ModuleType("compressed_tensors.quantization")
    compressed_linear.CompressedLinear = CompressedLinear
    quantization.QuantizationStatus = QuantizationStatus

    monkeypatch.setitem(sys.modules, "compressed_tensors", compressed_tensors)
    monkeypatch.setitem(sys.modules, "compressed_tensors.linear", linear_pkg)
    monkeypatch.setitem(
        sys.modules,
        "compressed_tensors.linear.compressed_linear",
        compressed_linear,
    )
    monkeypatch.setitem(sys.modules, "compressed_tensors.quantization", quantization)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.plain = CompressedLinear(2, 2, bias=False)
            self.packed = CompressedLinear(2, 2, bias=False)

    model = ToyModel()
    model.config = types.SimpleNamespace(_name_or_path=str(tmp_path))

    model.plain.register_parameter(
        "weight_packed",
        torch.nn.Parameter(torch.full((1,), 123, dtype=torch.int32), requires_grad=False),
    )
    model.plain.register_parameter(
        "weight_shape",
        torch.nn.Parameter(torch.tensor([2, 2], dtype=torch.int32), requires_grad=False),
    )
    model.plain.register_buffer("weight_scale", torch.ones(1))
    model.plain.quantization_scheme = object()
    model.plain.quantization_status = QuantizationStatus.COMPRESSED

    model.packed.register_parameter(
        "weight_packed",
        torch.nn.Parameter(torch.full((1,), -99, dtype=torch.int32), requires_grad=False),
    )
    model.packed.register_parameter(
        "weight_shape",
        torch.nn.Parameter(torch.tensor([9, 9], dtype=torch.int64), requires_grad=False),
    )
    model.packed.register_buffer("weight_scale", torch.full((1,), -1.0))
    model.packed.quantization_status = QuantizationStatus.COMPRESSED

    tensors = {
        "plain.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "packed.weight_packed": torch.full((1,), 7, dtype=torch.int32),
        "packed.weight_shape": torch.tensor([2, 2], dtype=torch.int32),
        "packed.weight_scale": torch.ones(1),
    }
    save_file(tensors, tmp_path / "model-00001-of-00001.safetensors")
    with (tmp_path / "model.safetensors.index.json").open("w") as f:
        json.dump(
            {
                "weight_map": {
                    key: "model-00001-of-00001.safetensors" for key in tensors
                }
            },
            f,
        )

    example_utils = _load_example_utils()
    example_utils._unpack_compressed_linear_weights(model, str(tmp_path))

    assert torch.equal(model.plain.weight, tensors["plain.weight"])
    assert "weight_packed" not in model.plain._parameters
    assert "weight_shape" not in model.plain._parameters
    assert "weight_scale" not in model.plain._buffers
    assert model.plain.quantization_scheme is None
    assert model.plain.quantization_status == QuantizationStatus.FROZEN

    assert "weight" not in model.packed._parameters
    assert torch.equal(model.packed.weight_packed, tensors["packed.weight_packed"])
    assert torch.equal(model.packed.weight_shape, tensors["packed.weight_shape"])
    assert model.packed.weight_shape.dtype == torch.int32
    assert torch.equal(model.packed.weight_scale, tensors["packed.weight_scale"])
    assert model.packed.quantization_status == QuantizationStatus.COMPRESSED


def test_unpack_compressed_linear_weights_uses_buffer_device_when_no_parameters(
    monkeypatch, tmp_path
):
    class QuantizationStatus(Enum):
        FROZEN = "frozen"
        COMPRESSED = "compressed"

    class CompressedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight_scale", torch.ones(1))

    compressed_tensors = types.ModuleType("compressed_tensors")
    linear_pkg = types.ModuleType("compressed_tensors.linear")
    compressed_linear = types.ModuleType("compressed_tensors.linear.compressed_linear")
    quantization = types.ModuleType("compressed_tensors.quantization")
    compressed_linear.CompressedLinear = CompressedLinear
    quantization.QuantizationStatus = QuantizationStatus

    monkeypatch.setitem(sys.modules, "compressed_tensors", compressed_tensors)
    monkeypatch.setitem(sys.modules, "compressed_tensors.linear", linear_pkg)
    monkeypatch.setitem(
        sys.modules,
        "compressed_tensors.linear.compressed_linear",
        compressed_linear,
    )
    monkeypatch.setitem(sys.modules, "compressed_tensors.quantization", quantization)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.plain = CompressedLinear()

    model = ToyModel()
    model.config = types.SimpleNamespace(_name_or_path=str(tmp_path))

    tensors = {"plain.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)}
    save_file(tensors, tmp_path / "model-00001-of-00001.safetensors")
    with (tmp_path / "model.safetensors.index.json").open("w") as f:
        json.dump(
            {
                "weight_map": {
                    key: "model-00001-of-00001.safetensors" for key in tensors
                }
            },
            f,
        )

    example_utils = _load_example_utils()
    example_utils._unpack_compressed_linear_weights(model, str(tmp_path))

    assert torch.equal(model.plain.weight, tensors["plain.weight"])
    assert "weight_scale" not in model.plain._buffers
    assert model.plain.quantization_status == QuantizationStatus.FROZEN
