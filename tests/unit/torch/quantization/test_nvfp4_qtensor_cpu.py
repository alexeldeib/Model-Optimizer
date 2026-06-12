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

import socket
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from modelopt.torch.export.quant_utils import get_scaling_factor_from_weight
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

try:
    from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
except ImportError:
    DeviceMesh = Shard = distribute_tensor = None


@contextmanager
def _cpu_dtensor_mesh():
    if DeviceMesh is None:
        pytest.skip("DTensor is not available")
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    initialized_before = dist.is_initialized()
    if not initialized_before:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        dist.init_process_group(
            "gloo",
            rank=0,
            world_size=1,
            init_method=f"tcp://127.0.0.1:{port}",
        )
    elif dist.get_world_size() != 1:
        pytest.skip("DTensor unit coverage expects a single-rank process group")

    try:
        yield DeviceMesh("cpu", [0])
    finally:
        if not initialized_before and dist.is_initialized():
            dist.destroy_process_group()


def test_nvfp4_dynamic_weight_scale_replaces_zero_blocks():
    weight = torch.tensor(
        [
            [0.0, 0.0, 1.0, -2.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    scale, scale_2 = NVFP4QTensor.get_weights_scaling_factor(
        weight,
        block_size=2,
        weights_scaling_factor_2=torch.tensor(1.0),
        keep_high_precision=True,
    )

    assert torch.allclose(scale_2, torch.tensor(1.0))
    assert torch.allclose(scale, torch.tensor([[1.0, 2.0 / 6.0], [1.0, 1.0]]))


def test_nvfp4_dynamic_weight_scale_accepts_dtensor_zero_blocks():
    with _cpu_dtensor_mesh() as mesh:
        weight = distribute_tensor(
            torch.tensor(
                [
                    [0.0, 0.0, 1.0, -2.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            mesh,
            [Shard(0)],
        )

        scale, scale_2 = NVFP4QTensor.get_weights_scaling_factor(
            weight,
            block_size=2,
            weights_scaling_factor_2=torch.tensor(1.0),
            keep_high_precision=True,
        )

        assert hasattr(scale, "to_local")
        assert torch.allclose(scale_2, torch.tensor(1.0))
        assert torch.allclose(scale.to_local(), torch.tensor([[1.0, 2.0 / 6.0], [1.0, 1.0]]))


def test_nvfp4_cast_fp4_matches_searchsorted_reference():
    values = torch.tensor(
        [
            -6.0,
            -5.0,
            -3.5,
            -2.5,
            -1.75,
            -1.25,
            -0.75,
            -0.25,
            0.0,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            6.0,
        ]
    )
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    weight_abs = values.abs()
    reference = (
        ((values < 0).to(torch.uint8) << 3)
        + torch.searchsorted(bounds, weight_abs, out_int32=True).to(torch.uint8)
        + torch.any(
            weight_abs.unsqueeze(-1) == bounds[[1, 3, 5]],
            dim=-1,
        ).to(torch.uint8)
    )

    assert torch.equal(NVFP4QTensor._cast_fp4(values.clone()), reference)


def test_nvfp4_cast_fp4_accepts_dtensor():
    with _cpu_dtensor_mesh() as mesh:
        values = torch.tensor(
            [
                -6.0,
                -5.0,
                -3.5,
                -2.5,
                -1.75,
                -1.25,
                -0.75,
                -0.25,
                0.0,
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5.0,
                6.0,
            ]
        )
        expected = NVFP4QTensor._cast_fp4(values.clone())
        dtensor_values = distribute_tensor(values.clone(), mesh, [Shard(0)])

        cast = NVFP4QTensor._cast_fp4(dtensor_values.clone())

        assert hasattr(cast, "to_local")
        assert torch.equal(cast.to_local(), expected)


def test_nvfp4_static_weight_scale_replaces_zero_blocks():
    quantizer = SimpleNamespace(
        block_sizes={-1: 2},
        global_amax=torch.tensor(12.0),
        _amax=torch.tensor([[0.0, 6.0], [3.0, 0.0]]),
    )

    scale, scale_2 = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(
        quantizer,
        weight=torch.empty(2, 4),
        keep_high_precision=True,
    )

    assert torch.allclose(scale_2, torch.tensor(12.0 / (6.0 * 448.0)))
    assert torch.allclose(scale, torch.tensor([[1.0, 1.0], [0.5, 1.0]]))


def test_export_weight_scale_replaces_zero_groups():
    weight = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 7.0, 0.0, 0.0],
        ]
    )

    scale = get_scaling_factor_from_weight(weight, group_size=2)

    assert torch.allclose(scale, torch.tensor([[1.0, 1.0], [1.0, 1.0]]))


def test_tensor_quantizer_export_amax_replaces_zeros_without_inplace_index_put():
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
    quantizer.amax = torch.tensor([0.0, float("nan"), 2.0])

    exported = quantizer.export_amax()

    assert torch.allclose(exported, torch.tensor([127.0, 127.0, 2.0]))
