#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect NVFP4 safetensors scale coverage for a Kimi/DeepSeek-style MoE export."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


SCALE_ATTRS = {"input_scale", "weight_scale", "weight_scale_2"}
COMPRESSED_WEIGHT_ATTRS = {"weight_packed", "weight_scale", "weight_zero_point", "weight_g_idx", "weight_shape"}
PROJECTIONS = {"gate_proj", "up_proj", "down_proj"}

PER_EXPERT_RE = re.compile(
    r"(?:^|.*\.)layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<attr>input_scale|weight_scale|weight_scale_2)$"
)
PACKED_EXPERT_RE = re.compile(
    r"(?:^|.*\.)layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<attr>input_scale|weight_scale|weight_scale_2)$"
)
GATED_MLP_PAIR_RE = re.compile(
    r"(?:^|.*\.)layers\.(?P<layer>\d+)\.mlp\."
    r"(?:(?P<shared>shared_experts)\.)?"
    r"(?P<proj>gate_proj|up_proj)\."
    r"(?P<attr>input_scale|weight_scale_2)$"
)


@dataclass
class TensorRecord:
    key: str
    shard: str
    layer: int
    expert: int | None
    proj: str
    attr: str
    shape: list[int]
    dtype: str
    loaded: bool
    numel: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    zeros: int | None = None
    nans: int | None = None
    infs: int | None = None
    load_error: str | None = None


@dataclass
class GroupSummary:
    layer: int
    proj: str
    attr: str
    tensor_count: int = 0
    loaded_count: int = 0
    skipped_count: int = 0
    total_elements_loaded: int = 0
    experts_seen: set[int] = field(default_factory=set)
    packed_tensor_count: int = 0
    tensor_shapes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tensor_dtypes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    values: list[torch.Tensor] = field(default_factory=list)
    tensor_means: list[float] = field(default_factory=list)
    load_errors: list[str] = field(default_factory=list)

    def add_record(self, rec: TensorRecord, max_group_elements: int) -> None:
        self.tensor_count += 1
        self.tensor_shapes[str(rec.shape)] += 1
        self.tensor_dtypes[rec.dtype] += 1
        if rec.expert is None:
            self.packed_tensor_count += 1
        else:
            self.experts_seen.add(rec.expert)
        if rec.load_error:
            self.load_errors.append(f"{rec.key}: {rec.load_error}")
        if not rec.loaded or rec.mean is None:
            self.skipped_count += 1
            return
        self.loaded_count += 1
        self.total_elements_loaded += rec.numel
        self.tensor_means.append(rec.mean)

    def add_values(self, vals: torch.Tensor, max_group_elements: int) -> None:
        if vals.numel() == 0:
            return
        current = sum(v.numel() for v in self.values)
        remaining = max_group_elements - current
        if remaining <= 0:
            return
        flat = vals.flatten().float().cpu()
        if flat.numel() > remaining:
            stride = max(1, math.ceil(flat.numel() / remaining))
            flat = flat[::stride][:remaining]
        self.values.append(flat)

    def to_json(self, expected_experts: int | None) -> dict[str, Any]:
        values = torch.cat(self.values) if self.values else torch.empty(0, dtype=torch.float32)
        stats = tensor_stats(values)
        tensor_mean_values = torch.tensor(self.tensor_means, dtype=torch.float32)
        tensor_mean_stats = tensor_stats(tensor_mean_values)
        experts = sorted(self.experts_seen)
        expert_count = len(experts)
        missing_experts: list[int] = []
        if expected_experts is not None and self.packed_tensor_count == 0:
            missing_experts = [i for i in range(expected_experts) if i not in self.experts_seen]
        all_equal = False
        if values.numel() > 1 and stats.get("min") is not None and stats.get("max") is not None:
            all_equal = bool(stats["min"] == stats["max"])
        return {
            "layer": self.layer,
            "projection": self.proj,
            "attribute": self.attr,
            "tensor_count": self.tensor_count,
            "loaded_count": self.loaded_count,
            "skipped_count": self.skipped_count,
            "packed_tensor_count": self.packed_tensor_count,
            "expert_count": expert_count,
            "missing_expert_count": len(missing_experts),
            "missing_experts_preview": missing_experts[:32],
            "total_elements_loaded": self.total_elements_loaded,
            "tensor_shapes": dict(sorted(self.tensor_shapes.items())),
            "tensor_dtypes": dict(sorted(self.tensor_dtypes.items())),
            "value_stats": stats,
            "tensor_mean_stats": tensor_mean_stats,
            "all_loaded_values_equal": all_equal,
            "load_errors": self.load_errors[:20],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--layers",
        default="1,2,30,60",
        help="Comma-separated layer ids to inspect, or 'auto' for first/middle/last MoE layers.",
    )
    parser.add_argument(
        "--attrs",
        default="input_scale,weight_scale_2",
        help="Comma-separated scale attributes to load for selected layers.",
    )
    parser.add_argument("--expected-experts", default=384, type=int)
    parser.add_argument("--expected-moe-layers", default=60, type=int)
    parser.add_argument(
        "--expected-kv-cache",
        default="",
        help="Expected kv_cache_quant_algo value, e.g. fp8. Empty disables the check.",
    )
    parser.add_argument(
        "--allow-shared-experts",
        action="store_true",
        help="Account for K2.5-style quantized shared_experts scale tensors.",
    )
    parser.add_argument(
        "--allow-layer0-mlp",
        action="store_true",
        help="Account for K2.5-style quantized dense layer-0 MLP scale tensors.",
    )
    parser.add_argument("--fail-on-warnings", action="store_true")
    parser.add_argument(
        "--max-tensor-elements",
        default=2_000_000,
        type=int,
        help="Skip value loading for individual scale tensors larger than this.",
    )
    parser.add_argument(
        "--max-group-elements",
        default=2_000_000,
        type=int,
        help="Maximum sampled values retained per layer/projection/attribute group.",
    )
    parser.add_argument("--model-label", default="")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def read_index(model_dir: Path) -> tuple[dict[str, str], list[Path]]:
    index = load_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if isinstance(weight_map, dict) and weight_map:
        shards = sorted({model_dir / shard for shard in weight_map.values()})
        return {str(k): str(v) for k, v in weight_map.items()}, shards

    shards = sorted(model_dir.glob("*.safetensors"))
    discovered: dict[str, str] = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                discovered[key] = shard.name
    return discovered, shards


def parse_layers(raw: str, all_keys: list[str]) -> list[int]:
    if raw.strip().lower() != "auto":
        return sorted({int(part.strip()) for part in raw.split(",") if part.strip()})

    layers = sorted(
        {
            int(match.group("layer"))
            for key in all_keys
            for match in (PER_EXPERT_RE.match(key), PACKED_EXPERT_RE.match(key))
            if match is not None
        }
    )
    if not layers:
        return []
    candidates = {layers[0], layers[min(1, len(layers) - 1)], layers[len(layers) // 2], layers[-1]}
    return sorted(candidates)


def match_scale_key(key: str) -> tuple[int, int | None, str, str] | None:
    match = PER_EXPERT_RE.match(key)
    if match:
        return (
            int(match.group("layer")),
            int(match.group("expert")),
            match.group("proj"),
            match.group("attr"),
        )
    match = PACKED_EXPERT_RE.match(key)
    if match:
        return (
            int(match.group("layer")),
            None,
            match.group("proj"),
            match.group("attr"),
        )
    return None


def num_elements(shape: list[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def float_or_none(value: torch.Tensor) -> float | None:
    if value.numel() == 0:
        return None
    out = float(value.item())
    if math.isnan(out) or math.isinf(out):
        return out
    return out


def tensor_stats(values: torch.Tensor) -> dict[str, Any]:
    if values.numel() == 0:
        return {
            "count": 0,
            "min": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "max": None,
            "mean": None,
            "std": None,
            "zero_count": 0,
            "nan_count": 0,
            "inf_count": 0,
            "unique_count": 0,
        }
    vals = values.flatten().float().cpu()
    finite = vals[torch.isfinite(vals)]
    unique_count = int(torch.unique(vals).numel()) if vals.numel() <= 1_000_000 else None
    out = {
        "count": int(vals.numel()),
        "zero_count": int((vals == 0).sum().item()),
        "nan_count": int(torch.isnan(vals).sum().item()),
        "inf_count": int(torch.isinf(vals).sum().item()),
        "unique_count": unique_count,
    }
    if finite.numel() == 0:
        out.update({"min": None, "p01": None, "p50": None, "p99": None, "max": None, "mean": None, "std": None})
        return out
    q = torch.quantile(finite, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float32))
    out.update(
        {
            "min": float_or_none(finite.min()),
            "p01": float(q[0].item()),
            "p50": float(q[1].item()),
            "p99": float(q[2].item()),
            "max": float_or_none(finite.max()),
            "mean": float_or_none(finite.mean()),
            "std": float_or_none(finite.std(unbiased=False)),
        }
    )
    return out


def get_shape_dtype(handle: Any, key: str) -> tuple[list[int], str]:
    try:
        tensor_slice = handle.get_slice(key)
        shape = [int(dim) for dim in tensor_slice.get_shape()]
        dtype = str(tensor_slice.get_dtype())
        return shape, dtype
    except Exception:
        tensor = handle.get_tensor(key)
        return [int(dim) for dim in tensor.shape], str(tensor.dtype)


def load_record(
    handle: Any,
    key: str,
    shard: str,
    layer: int,
    expert: int | None,
    proj: str,
    attr: str,
    max_tensor_elements: int,
) -> tuple[TensorRecord, torch.Tensor | None]:
    try:
        shape, dtype = get_shape_dtype(handle, key)
        count = num_elements(shape)
        rec = TensorRecord(
            key=key,
            shard=shard,
            layer=layer,
            expert=expert,
            proj=proj,
            attr=attr,
            shape=shape,
            dtype=dtype,
            loaded=False,
            numel=count,
        )
        if count > max_tensor_elements:
            rec.load_error = f"skipped: {count} elements exceeds max-tensor-elements={max_tensor_elements}"
            return rec, None

        values = handle.get_tensor(key).detach().cpu().float().flatten()
        rec.loaded = True
        rec.min = float_or_none(values.min())
        rec.max = float_or_none(values.max())
        rec.mean = float_or_none(values.mean())
        rec.std = float_or_none(values.std(unbiased=False))
        rec.zeros = int((values == 0).sum().item())
        rec.nans = int(torch.isnan(values).sum().item())
        rec.infs = int(torch.isinf(values).sum().item())
        return rec, values
    except Exception as exc:  # noqa: BLE001 - diagnostic script should report and continue.
        rec = TensorRecord(
            key=key,
            shard=shard,
            layer=layer,
            expert=expert,
            proj=proj,
            attr=attr,
            shape=[],
            dtype="unknown",
            loaded=False,
            numel=0,
            load_error=f"{type(exc).__name__}: {exc}",
        )
        return rec, None


def count_coverage(keys: list[str]) -> dict[str, Any]:
    scale_keys = [key for key in keys if key.rsplit(".", 1)[-1] in SCALE_ATTRS]
    compressed_weight_keys = [
        key for key in keys if key.rsplit(".", 1)[-1] in COMPRESSED_WEIGHT_ATTRS
    ]
    non_expert_compressed_weight_keys = [
        key
        for key in compressed_weight_keys
        if ".mlp.experts." not in key and ".block_sparse_moe." not in key
    ]
    by_attr = {attr: sum(key.endswith(f".{attr}") for key in keys) for attr in sorted(SCALE_ATTRS)}
    return {
        "total_tensors": len(keys),
        "total_scale_tensors": len(scale_keys),
        "scale_tensors_by_attr": by_attr,
        "compressed_weight_tensors": len(compressed_weight_keys),
        "non_expert_compressed_weight_tensors": len(non_expert_compressed_weight_keys),
        "non_expert_compressed_weight_keys": non_expert_compressed_weight_keys,
        "non_expert_compressed_weight_preview": non_expert_compressed_weight_keys[:32],
        "lm_head_compressed_weight_tensors": sum(
            ".lm_head." in key and key.rsplit(".", 1)[-1] in COMPRESSED_WEIGHT_ATTRS
            for key in keys
        ),
        "vision_compressed_weight_tensors": sum(
            (".vision." in key or ".vit." in key or ".moon_vit." in key or key.startswith("vision_tower."))
            and key.rsplit(".", 1)[-1] in COMPRESSED_WEIGHT_ATTRS
            for key in keys
        ),
        "projector_compressed_weight_tensors": sum(
            (key.startswith("mm_projector.") or ".mm_projector." in key)
            and key.rsplit(".", 1)[-1] in COMPRESSED_WEIGHT_ATTRS
            for key in keys
        ),
        "attention_scale_tensors": sum(".self_attn." in key and key.rsplit(".", 1)[-1] in SCALE_ATTRS for key in keys),
        "shared_expert_scale_tensors": sum(
            ".mlp.shared_experts." in key and key.rsplit(".", 1)[-1] in SCALE_ATTRS for key in keys
        ),
        "lm_head_scale_tensors": sum(".lm_head." in key and key.rsplit(".", 1)[-1] in SCALE_ATTRS for key in keys),
        "embed_scale_tensors": sum(".embed_tokens." in key and key.rsplit(".", 1)[-1] in SCALE_ATTRS for key in keys),
        "router_scale_tensors": sum(".router." in key and key.rsplit(".", 1)[-1] in SCALE_ATTRS for key in keys),
        "vision_scale_tensors": sum(
            (".vision." in key or ".vit." in key or ".moon_vit." in key or key.startswith("vision_tower."))
            and key.rsplit(".", 1)[-1] in SCALE_ATTRS
            for key in keys
        ),
        "moe_gate_scale_tensors": sum(
            ".mlp.gate." in key and key.rsplit(".", 1)[-1] in SCALE_ATTRS for key in keys
        ),
        "shared_expert_gate_scale_tensors": sum(
            ".shared_expert_gate." in key and key.rsplit(".", 1)[-1] in SCALE_ATTRS for key in keys
        ),
        "layer0_scale_tensors": sum(
            ".layers.0." in key and key.rsplit(".", 1)[-1] in SCALE_ATTRS for key in keys
        ),
        "expert_scale_tensors_matched": sum(match_scale_key(key) is not None for key in keys),
    }


def scan_all_scalar_scale_groups(
    model_dir: Path,
    weight_map: dict[str, str],
    attrs: set[str],
    max_group_elements: int,
) -> list[dict[str, Any]]:
    """Scan scalar per-expert scale tensors for every routed MoE layer."""

    scalar_attrs = attrs & {"input_scale", "weight_scale_2"}
    if not scalar_attrs:
        return []

    groups: dict[tuple[int, str, str], GroupSummary] = {}
    keys_by_shard: dict[str, list[tuple[str, int, int | None, str, str]]] = defaultdict(list)
    for key, shard_name in weight_map.items():
        match = match_scale_key(key)
        if match is None:
            continue
        layer, expert, proj, attr = match
        if attr in scalar_attrs:
            keys_by_shard[shard_name].append((key, layer, expert, proj, attr))

    for shard_name, entries in sorted(keys_by_shard.items()):
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            for key, layer, expert, proj, attr in entries:
                rec = TensorRecord(
                    key=key,
                    shard=shard_name,
                    layer=layer,
                    expert=expert,
                    proj=proj,
                    attr=attr,
                    shape=[],
                    dtype="unknown",
                    loaded=False,
                    numel=0,
                    load_error=f"missing shard {shard_path}",
                )
                group = groups.setdefault((layer, proj, attr), GroupSummary(layer, proj, attr))
                group.add_record(rec, max_group_elements)
            continue

        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key, layer, expert, proj, attr in sorted(entries):
                rec, values = load_record(
                    handle,
                    key,
                    shard_name,
                    layer,
                    expert,
                    proj,
                    attr,
                    max_tensor_elements=1024,
                )
                group = groups.setdefault((layer, proj, attr), GroupSummary(layer, proj, attr))
                group.add_record(rec, max_group_elements)
                if values is not None:
                    group.add_values(values, max_group_elements)

    return [
        groups[key].to_json(expected_experts=None)
        for key in sorted(groups, key=lambda item: (item[0], item[1], item[2]))
    ]


def scan_gate_up_scalar_pairs(
    model_dir: Path,
    weight_map: dict[str, str],
    attrs: set[str],
) -> dict[str, Any]:
    """Check scalar gate/up scale parity for fused-serving compatibility."""

    scalar_attrs = attrs & {"input_scale", "weight_scale_2"}
    values: dict[tuple[int, str, str], dict[str, tuple[float, str]]] = defaultdict(dict)
    summary: dict[str, Any] = {
        attr: {
            "compared_pairs": 0,
            "mismatched_pairs": 0,
            "max_abs_delta": 0.0,
            "examples": [],
            "load_errors": [],
        }
        for attr in sorted(scalar_attrs)
    }
    if not scalar_attrs:
        return summary

    keys_by_shard: dict[str, list[tuple[str, int, str, str, str]]] = defaultdict(list)
    for key, shard_name in weight_map.items():
        match = PER_EXPERT_RE.match(key)
        if match is not None:
            proj = match.group("proj")
            attr = match.group("attr")
            group = f"expert.{int(match.group('expert'))}"
        else:
            match = GATED_MLP_PAIR_RE.match(key)
            if match is None:
                continue
            proj = match.group("proj")
            attr = match.group("attr")
            group = match.group("shared") or "dense_mlp"

        if proj in {"gate_proj", "up_proj"} and attr in scalar_attrs:
            keys_by_shard[shard_name].append(
                (
                    key,
                    int(match.group("layer")),
                    group,
                    proj,
                    attr,
                )
            )

    for shard_name, entries in sorted(keys_by_shard.items()):
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            for key, _, _, _, attr in entries:
                summary[attr]["load_errors"].append(f"{key}: missing shard {shard_path}")
            continue
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key, layer, group, proj, attr in sorted(entries):
                rec, tensor = load_record(
                    handle,
                    key,
                    shard_name,
                    layer,
                    None if group in {"dense_mlp", "shared_experts"} else int(group.split(".", 1)[1]),
                    proj,
                    attr,
                    max_tensor_elements=1,
                )
                if rec.load_error or tensor is None or tensor.numel() != 1:
                    summary[attr]["load_errors"].append(
                        f"{key}: {rec.load_error or f'expected scalar, got {rec.shape}'}"
                    )
                    continue
                values[(layer, group, attr)][proj] = (float(tensor.item()), key)

    for (layer, group, attr), by_proj in sorted(values.items()):
        if "gate_proj" not in by_proj or "up_proj" not in by_proj:
            continue
        gate_value, gate_key = by_proj["gate_proj"]
        up_value, up_key = by_proj["up_proj"]
        entry = summary[attr]
        entry["compared_pairs"] += 1
        delta = abs(gate_value - up_value)
        entry["max_abs_delta"] = max(entry["max_abs_delta"], delta)
        if not math.isclose(gate_value, up_value, rel_tol=1e-6, abs_tol=1e-12):
            entry["mismatched_pairs"] += 1
            if len(entry["examples"]) < 20:
                entry["examples"].append(
                    {
                        "layer": layer,
                        "group": group,
                        "gate": gate_value,
                        "up": up_value,
                        "abs_delta": delta,
                        "gate_key": gate_key,
                        "up_key": up_key,
                    }
                )

    return summary


def config_summary(model_dir: Path) -> dict[str, Any]:
    config = load_json(model_dir / "config.json")
    hf_quant_config = load_json(model_dir / "hf_quant_config.json")
    config_quantization = config.get("quantization_config", {})
    quant_config = hf_quant_config or config_quantization
    quantization = quant_config.get("quantization") if isinstance(quant_config, dict) else None
    if quantization is None and isinstance(config_quantization, dict):
        quantization = config_quantization.get("quantization", config_quantization)
    index = load_json(model_dir / "model.safetensors.index.json")
    files = sorted(model_dir.glob("*.safetensors"))
    return {
        "model_dir": str(model_dir),
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_experts": config.get("n_routed_experts") or config.get("num_experts"),
        "num_experts_per_tok": config.get("num_experts_per_tok"),
        "quant_config_source": "hf_quant_config.json" if hf_quant_config else "config.json.quantization_config",
        "quant_config_keys": sorted(quant_config.keys()) if isinstance(quant_config, dict) else [],
        "quantization": quantization,
        "quant_algo": (quantization or {}).get("quant_algo") if isinstance(quantization, dict) else None,
        "kv_cache_quant_algo": (quantization or {}).get("kv_cache_quant_algo")
        if isinstance(quantization, dict)
        else None,
        "weight_map_entries": len(index.get("weight_map", {})) if isinstance(index.get("weight_map"), dict) else None,
        "safetensors_shards": len(files),
        "safetensors_size_gib": round(sum(path.stat().st_size for path in files) / (1024**3), 3),
    }


def format_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        if value == 0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.4e}"
        return f"{value:.6g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    meta = report["metadata"]
    coverage = report["coverage"]
    lines.append(f"# NVFP4 Export Inspection: {report['model_label']}")
    lines.append("")
    lines.append(f"- Model dir: `{meta['model_dir']}`")
    lines.append(f"- Architectures: `{meta['architectures']}`")
    lines.append(f"- Model type: `{meta['model_type']}`")
    lines.append(f"- Hidden layers: `{meta['num_hidden_layers']}`")
    lines.append(f"- Experts: `{meta['num_experts']}`, experts/token: `{meta['num_experts_per_tok']}`")
    lines.append(f"- KV cache quant: `{meta['kv_cache_quant_algo']}`")
    lines.append(f"- Shards: `{meta['safetensors_shards']}`, size GiB: `{meta['safetensors_size_gib']}`")
    lines.append(f"- Scale tensors: `{coverage['total_scale_tensors']}` / total tensors `{coverage['total_tensors']}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append("| Check | Count |")
    lines.append("| --- | ---: |")
    for key in (
        "expert_scale_tensors_matched",
        "compressed_weight_tensors",
        "non_expert_compressed_weight_tensors",
        "lm_head_compressed_weight_tensors",
        "vision_compressed_weight_tensors",
        "projector_compressed_weight_tensors",
        "attention_scale_tensors",
        "shared_expert_scale_tensors",
        "lm_head_scale_tensors",
        "embed_scale_tensors",
        "router_scale_tensors",
        "vision_scale_tensors",
        "moe_gate_scale_tensors",
        "shared_expert_gate_scale_tensors",
        "layer0_scale_tensors",
    ):
        lines.append(f"| {key} | {coverage[key]} |")
    for attr, count in coverage["scale_tensors_by_attr"].items():
        lines.append(f"| scale_tensors_by_attr.{attr} | {count} |")
    lines.append("")

    if report["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Selected Scale Groups")
    lines.append("")
    lines.append(
        "| Layer | Projection | Attr | Tensors | Experts | Missing | Loaded elems | "
        "Min | P01 | P50 | P99 | Max | Mean | Std | Unique | Equal | Shapes |"
    )
    lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for group in report["groups"]:
        stats = group["value_stats"]
        shapes = ", ".join(f"{shape} x{count}" for shape, count in group["tensor_shapes"].items())
        lines.append(
            "| {layer} | {proj} | {attr} | {tensors} | {experts} | {missing} | {elems} | "
            "{minv} | {p01} | {p50} | {p99} | {maxv} | {mean} | {std} | {unique} | {equal} | {shapes} |".format(
                layer=group["layer"],
                proj=group["projection"],
                attr=group["attribute"],
                tensors=group["tensor_count"],
                experts=group["expert_count"] or group["packed_tensor_count"],
                missing=group["missing_expert_count"],
                elems=group["total_elements_loaded"],
                minv=format_number(stats["min"]),
                p01=format_number(stats["p01"]),
                p50=format_number(stats["p50"]),
                p99=format_number(stats["p99"]),
                maxv=format_number(stats["max"]),
                mean=format_number(stats["mean"]),
                std=format_number(stats["std"]),
                unique=format_number(stats["unique_count"]),
                equal=format_number(group["all_loaded_values_equal"]),
                shapes=shapes or "-",
            )
        )
    lines.append("")
    if report.get("all_layer_scalar_scan"):
        lines.append("## All-Layer Scalar Scan")
        lines.append("")
        lines.append(
            f"- Groups scanned: `{len(report['all_layer_scalar_scan'])}` "
            "(`input_scale` / `weight_scale_2` across routed MoE layers)"
        )
        constant_groups = [
            group
            for group in report["all_layer_scalar_scan"]
            if group["attribute"] == "input_scale"
            and group["expert_count"] > 1
            and group["all_loaded_values_equal"]
        ]
        lines.append(f"- Constant input-scale groups: `{len(constant_groups)}`")
        lines.append("")
    if report.get("gate_up_pair_scan"):
        lines.append("## Gate/Up Pair Scan")
        lines.append("")
        lines.append("| Attr | Compared | Mismatched | Max Abs Delta | Load Errors |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for attr, entry in report["gate_up_pair_scan"].items():
            lines.append(
                f"| {attr} | {entry['compared_pairs']} | {entry['mismatched_pairs']} | "
                f"{format_number(entry['max_abs_delta'])} | {len(entry['load_errors'])} |"
            )
        lines.append("")
    lines.append(f"JSON report: `{report['json_path']}`")
    return "\n".join(lines) + "\n"


def build_warnings(
    groups: list[dict[str, Any]],
    coverage: dict[str, Any],
    expected_experts: int | None,
    expected_moe_layers: int | None,
    allow_shared_experts: bool = False,
    allow_layer0_mlp: bool = False,
    expected_kv_cache: str = "",
    all_layer_scalar_scan: list[dict[str, Any]] | None = None,
    gate_up_pair_scan: dict[str, Any] | None = None,
) -> list[str]:
    warnings: list[str] = []
    if expected_kv_cache:
        actual_kv_value = coverage.get("_kv_cache_quant_algo")
        actual_kv = str(actual_kv_value or "").lower()
        if actual_kv != expected_kv_cache.lower():
            warnings.append(
                f"Expected kv_cache_quant_algo={expected_kv_cache}, found {actual_kv_value!r}."
            )
    if expected_experts is not None and expected_moe_layers is not None:
        expected_routed_per_attr = expected_experts * expected_moe_layers * len(PROJECTIONS)
        expected_extra_per_attr = 0
        if allow_shared_experts:
            expected_extra_per_attr += expected_moe_layers * len(PROJECTIONS)
        if allow_layer0_mlp:
            expected_extra_per_attr += len(PROJECTIONS)
        expected_per_attr = expected_routed_per_attr + expected_extra_per_attr
        expected_total = expected_per_attr * len(SCALE_ATTRS)
        if coverage["total_scale_tensors"] != expected_total:
            warnings.append(
                f"Expected {expected_total} total scale tensors "
                f"({expected_routed_per_attr} routed expert tensors/attr"
                f" + {expected_extra_per_attr} dense/shared MLP tensors/attr"
                f") * {len(SCALE_ATTRS)} attrs, "
                f"found {coverage['total_scale_tensors']}."
            )
        for attr in sorted(SCALE_ATTRS):
            count = coverage["scale_tensors_by_attr"].get(attr, 0)
            if count != expected_per_attr:
                warnings.append(
                    f"Expected {expected_per_attr} {attr} tensors, found {count}."
                )
        expected_unmatched = expected_extra_per_attr * len(SCALE_ATTRS)
        actual_unmatched = coverage["total_scale_tensors"] - coverage["expert_scale_tensors_matched"]
        if actual_unmatched != expected_unmatched:
            warnings.append(
                "Unexpected count of non-routed-expert scale tensors: "
                f"expected {expected_unmatched}, found {actual_unmatched}."
            )
    if coverage["attention_scale_tensors"]:
        warnings.append("Attention scale tensors are present; this K2.6 NVFP4 recipe should not quantize attention.")

    unexpected_non_expert_compressed = []
    for key in coverage.get("non_expert_compressed_weight_keys", []):
        if allow_shared_experts and ".mlp.shared_experts." in key:
            continue
        if allow_layer0_mlp and ".layers.0.mlp." in key and ".mlp.experts." not in key:
            continue
        unexpected_non_expert_compressed.append(key)
    if unexpected_non_expert_compressed:
        preview = ", ".join(unexpected_non_expert_compressed[:8])
        warnings.append(
            "Unexpected compressed weight metadata or scale tensors are present outside the allowed K2.6 MLP set; "
            f"this indicates plain modules were packed or left with stale packed placeholders. Preview: {preview}"
        )
    if (
        coverage["lm_head_compressed_weight_tensors"]
        or coverage["vision_compressed_weight_tensors"]
        or coverage["projector_compressed_weight_tensors"]
    ):
        warnings.append(
            "lm_head, vision, or projector compressed weight tensors are present; "
            "these modules must remain native for K2.6 NVFP4."
        )
    if coverage["shared_expert_scale_tensors"] and not allow_shared_experts:
        warnings.append("Shared expert scale tensors are present; confirm whether the recipe intended routed experts only.")
    if coverage["lm_head_scale_tensors"] or coverage["embed_scale_tensors"]:
        warnings.append("Embedding or lm_head scale tensors are present; this is unexpected for K2.6 NVFP4 quantization.")
    if coverage["router_scale_tensors"] or coverage["moe_gate_scale_tensors"]:
        warnings.append("Router/gate scale tensors are present; this K2.6 NVFP4 recipe should not quantize routing.")
    if coverage["vision_scale_tensors"]:
        warnings.append("Vision scale tensors are present; this K2.6 NVFP4 recipe should not quantize vision modules.")
    if coverage["shared_expert_gate_scale_tensors"]:
        warnings.append("shared_expert_gate scale tensors are present; only shared_experts projections should be quantized.")
    if coverage["layer0_scale_tensors"] and not allow_layer0_mlp:
        warnings.append("Layer 0 scale tensors are present; Kimi/DeepSeek MoE usually starts routed experts after layer 0.")

    for group in groups:
        label = f"layer {group['layer']} {group['projection']}.{group['attribute']}"
        if group["tensor_count"] == 0:
            warnings.append(f"No tensors found for {label}.")
            continue
        if expected_experts is not None and group["packed_tensor_count"] == 0 and group["missing_expert_count"]:
            warnings.append(f"{label} is missing {group['missing_expert_count']} expected per-expert tensors.")
        stats = group["value_stats"]
        if stats["nan_count"] or stats["inf_count"]:
            warnings.append(f"{label} contains NaN/Inf scale values.")
        if stats["zero_count"] and stats["zero_count"] == stats["count"] and stats["count"]:
            warnings.append(f"{label} is entirely zero.")
        if (
            group["attribute"] == "input_scale"
            and group["projection"] == "down_proj"
            and group["expert_count"] > 1
            and group["all_loaded_values_equal"]
        ):
            warnings.append(f"{label} is constant across loaded experts.")
    for group in all_layer_scalar_scan or []:
        label = f"all-layer scan: layer {group['layer']} {group['projection']}.{group['attribute']}"
        stats = group["value_stats"]
        if group["load_errors"]:
            warnings.append(f"{label} has load errors: {group['load_errors'][:3]}.")
        if stats["nan_count"] or stats["inf_count"]:
            warnings.append(f"{label} contains NaN/Inf scale values.")
        if stats["zero_count"] and stats["zero_count"] == stats["count"] and stats["count"]:
            warnings.append(f"{label} is entirely zero.")
        if (
            group["attribute"] == "input_scale"
            and group["projection"] == "down_proj"
            and group["expert_count"] > 1
            and group["all_loaded_values_equal"]
        ):
            warnings.append(f"{label} is constant across loaded experts.")
    for attr, entry in (gate_up_pair_scan or {}).items():
        if entry.get("load_errors"):
            warnings.append(
                f"Gate/up {attr} pair scan had load errors: {entry['load_errors'][:3]}."
            )
        if entry.get("mismatched_pairs", 0):
            warnings.append(
                f"Gate/up {attr} mismatch: {entry['mismatched_pairs']}/"
                f"{entry.get('compared_pairs', 0)} pairs differ "
                f"(max_abs_delta={entry.get('max_abs_delta')})."
            )
    return warnings


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    attrs = {part.strip() for part in args.attrs.split(",") if part.strip()}
    unknown_attrs = attrs - SCALE_ATTRS
    if unknown_attrs:
        raise SystemExit(f"unknown attrs: {sorted(unknown_attrs)}")

    weight_map, shards = read_index(model_dir)
    all_keys = sorted(weight_map)
    selected_layers = set(parse_layers(args.layers, all_keys))
    selected_records: list[TensorRecord] = []
    selected_key_lines: list[str] = []
    groups: dict[tuple[int, str, str], GroupSummary] = {}

    keys_by_shard: dict[str, list[tuple[str, int, int | None, str, str]]] = defaultdict(list)
    for key in all_keys:
        match = match_scale_key(key)
        if match is None:
            continue
        layer, expert, proj, attr = match
        if layer not in selected_layers or attr not in attrs:
            continue
        keys_by_shard[weight_map[key]].append((key, layer, expert, proj, attr))

    for shard_name, entries in sorted(keys_by_shard.items()):
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            for key, layer, expert, proj, attr in entries:
                rec = TensorRecord(
                    key=key,
                    shard=shard_name,
                    layer=layer,
                    expert=expert,
                    proj=proj,
                    attr=attr,
                    shape=[],
                    dtype="unknown",
                    loaded=False,
                    numel=0,
                    load_error=f"missing shard {shard_path}",
                )
                selected_records.append(rec)
                group = groups.setdefault((layer, proj, attr), GroupSummary(layer, proj, attr))
                group.add_record(rec, args.max_group_elements)
            continue

        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key, layer, expert, proj, attr in sorted(entries):
                rec, values = load_record(
                    handle,
                    key,
                    shard_name,
                    layer,
                    expert,
                    proj,
                    attr,
                    args.max_tensor_elements,
                )
                selected_records.append(rec)
                selected_key_lines.append(
                    f"{key}\tshard={shard_name}\tshape={rec.shape}\tdtype={rec.dtype}\tloaded={rec.loaded}"
                )
                group = groups.setdefault((layer, proj, attr), GroupSummary(layer, proj, attr))
                group.add_record(rec, args.max_group_elements)
                if values is not None:
                    group.add_values(values, args.max_group_elements)

    for layer in selected_layers:
        for proj in sorted(PROJECTIONS):
            for attr in sorted(attrs):
                groups.setdefault((layer, proj, attr), GroupSummary(layer, proj, attr))

    expected_experts = args.expected_experts if args.expected_experts > 0 else None
    expected_moe_layers = args.expected_moe_layers if args.expected_moe_layers > 0 else None
    group_json = [
        groups[key].to_json(expected_experts)
        for key in sorted(groups, key=lambda item: (item[0], item[1], item[2]))
    ]
    all_layer_scalar_scan = scan_all_scalar_scale_groups(
        model_dir,
        weight_map,
        attrs,
        args.max_group_elements,
    )
    gate_up_pair_scan = scan_gate_up_scalar_pairs(model_dir, weight_map, attrs)
    metadata = config_summary(model_dir)
    coverage = count_coverage(all_keys)
    coverage["_kv_cache_quant_algo"] = metadata["kv_cache_quant_algo"]
    report: dict[str, Any] = {
        "model_label": args.model_label or model_dir.name,
        "metadata": metadata,
        "coverage": coverage,
        "selected_layers": sorted(selected_layers),
        "selected_attrs": sorted(attrs),
        "selected_tensor_count": len(selected_records),
        "shards": [str(path) for path in shards],
        "groups": group_json,
        "all_layer_scalar_scan": all_layer_scalar_scan,
        "gate_up_pair_scan": gate_up_pair_scan,
        "warnings": [],
        "json_path": str(out_dir / "summary.json"),
    }
    report["warnings"] = build_warnings(
        group_json,
        report["coverage"],
        expected_experts,
        expected_moe_layers,
        allow_shared_experts=args.allow_shared_experts,
        allow_layer0_mlp=args.allow_layer0_mlp,
        expected_kv_cache=args.expected_kv_cache,
        all_layer_scalar_scan=all_layer_scalar_scan,
        gate_up_pair_scan=gate_up_pair_scan,
    )

    (out_dir / "selected_keys.txt").write_text("\n".join(selected_key_lines) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(report)
    (out_dir / "summary.md").write_text(markdown)
    print(markdown)
    if args.fail_on_warnings and report["warnings"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
