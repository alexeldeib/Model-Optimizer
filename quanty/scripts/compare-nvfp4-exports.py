#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare routed-expert NVFP4 tensors between two HF safetensors exports."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from modelopt.torch.quantization.qtensor import NVFP4QTensor


EXPERT_WEIGHT_RE = re.compile(
    r"(?:^|.*\.)layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
NVFP4_BLOCK_SIZES = {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}


def _load_index(export_dir: Path) -> dict[str, str]:
    index_path = export_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")
    with index_path.open() as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} does not contain a weight_map")
    return weight_map


def _load_tensor(export_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard = weight_map.get(key)
    if shard is None:
        raise KeyError(key)
    with safe_open(export_dir / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _parse_csv_ints(value: str | None) -> set[int] | None:
    if not value:
        return None
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.update(range(int(start), int(end) + 1))
        else:
            out.add(int(part))
    return out


def _even_sample(items: list[str], limit: int | None) -> list[str]:
    if limit is None or limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    indexes = sorted({round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)})
    return [items[i] for i in indexes]


def _select_keys(
    reference_map: dict[str, str],
    candidate_map: dict[str, str],
    layers: set[int] | None,
    experts: set[int] | None,
    projections: set[str] | None,
    limit: int | None,
) -> list[str]:
    keys: list[tuple[int, int, str, str]] = []
    for key in reference_map:
        if key not in candidate_map:
            continue
        match = EXPERT_WEIGHT_RE.match(key)
        if not match:
            continue
        layer = int(match.group("layer"))
        expert = int(match.group("expert"))
        proj = match.group("proj")
        if layers is not None and layer not in layers:
            continue
        if experts is not None and expert not in experts:
            continue
        if projections is not None and proj not in projections:
            continue
        keys.append((layer, expert, proj, key))
    keys.sort()
    return _even_sample([key for *_prefix, key in keys], limit)


def _dequantize_nvfp4(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
) -> torch.Tensor:
    if weight.dtype != torch.uint8:
        raise TypeError(f"expected packed NVFP4 weight dtype uint8, found {weight.dtype}")
    original_shape = torch.Size((*weight.shape[:-1], weight.shape[-1] * 2))
    qtensor = NVFP4QTensor(original_shape, torch.bfloat16, weight)
    return qtensor.dequantize(
        dtype=torch.float32,
        scale=weight_scale,
        double_scale=weight_scale_2,
        block_sizes=NVFP4_BLOCK_SIZES,
    )


def _scalar_or_none(tensor: torch.Tensor | None) -> float | None:
    if tensor is None:
        return None
    if tensor.numel() != 1:
        return None
    return float(tensor.float().item())


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float().reshape(-1)
    cand = candidate.float().reshape(-1)
    diff = cand - ref
    ref_rms = float(torch.sqrt(torch.mean(ref * ref)).item())
    cand_rms = float(torch.sqrt(torch.mean(cand * cand)).item())
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    mean_abs = float(diff.abs().mean().item())
    max_abs = float(diff.abs().max().item())
    denom = max(ref_rms, 1.0e-12)
    dot = float(torch.dot(ref, cand).item())
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    cand_norm = float(torch.linalg.vector_norm(cand).item())
    cosine = dot / max(ref_norm * cand_norm, 1.0e-12)
    return {
        "ref_rms": ref_rms,
        "candidate_rms": cand_rms,
        "rmse": rmse,
        "rel_rmse": rmse / denom,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "cosine": cosine,
    }


def _compare_key(
    reference_dir: Path,
    reference_map: dict[str, str],
    candidate_dir: Path,
    candidate_map: dict[str, str],
    key: str,
) -> dict[str, Any]:
    prefix = key[: -len("weight")]
    companion_names = {
        "weight_scale": prefix + "weight_scale",
        "weight_scale_2": prefix + "weight_scale_2",
        "input_scale": prefix + "input_scale",
    }
    required = ["weight_scale", "weight_scale_2"]
    missing: list[str] = []
    for name in required:
        if companion_names[name] not in reference_map:
            missing.append(f"reference:{companion_names[name]}")
        if companion_names[name] not in candidate_map:
            missing.append(f"candidate:{companion_names[name]}")
    if missing:
        return {"key": key, "missing": missing}

    ref_weight = _load_tensor(reference_dir, reference_map, key)
    cand_weight = _load_tensor(candidate_dir, candidate_map, key)
    ref_wscale = _load_tensor(reference_dir, reference_map, companion_names["weight_scale"])
    cand_wscale = _load_tensor(candidate_dir, candidate_map, companion_names["weight_scale"])
    ref_wscale2 = _load_tensor(reference_dir, reference_map, companion_names["weight_scale_2"])
    cand_wscale2 = _load_tensor(candidate_dir, candidate_map, companion_names["weight_scale_2"])

    input_scale_key = companion_names["input_scale"]
    ref_input_scale = (
        _load_tensor(reference_dir, reference_map, input_scale_key)
        if input_scale_key in reference_map
        else None
    )
    cand_input_scale = (
        _load_tensor(candidate_dir, candidate_map, input_scale_key)
        if input_scale_key in candidate_map
        else None
    )

    result: dict[str, Any] = {
        "key": key,
        "weight_shape": list(ref_weight.shape),
        "candidate_weight_shape": list(cand_weight.shape),
        "weight_dtype": str(ref_weight.dtype),
        "candidate_weight_dtype": str(cand_weight.dtype),
        "packed_equal": torch.equal(ref_weight, cand_weight),
        "weight_scale_equal": torch.equal(ref_wscale, cand_wscale),
        "weight_scale_2_equal": torch.equal(ref_wscale2, cand_wscale2),
        "input_scale_equal": (
            ref_input_scale is not None
            and cand_input_scale is not None
            and torch.equal(ref_input_scale, cand_input_scale)
        ),
        "reference_weight_scale_2": _scalar_or_none(ref_wscale2),
        "candidate_weight_scale_2": _scalar_or_none(cand_wscale2),
        "reference_input_scale": _scalar_or_none(ref_input_scale),
        "candidate_input_scale": _scalar_or_none(cand_input_scale),
    }

    if ref_weight.shape != cand_weight.shape:
        result["error"] = "weight shape mismatch"
        return result
    if ref_wscale.shape != cand_wscale.shape:
        result["error"] = "weight_scale shape mismatch"
        return result

    ref_deq = _dequantize_nvfp4(ref_weight, ref_wscale, ref_wscale2)
    cand_deq = _dequantize_nvfp4(cand_weight, cand_wscale, cand_wscale2)
    result.update(_metrics(ref_deq, cand_deq))
    return result


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [item for item in results if "rel_rmse" in item]
    if not complete:
        return {"compared": 0, "missing_or_failed": len(results)}
    rel_rmse = [float(item["rel_rmse"]) for item in complete]
    cosine = [float(item["cosine"]) for item in complete]
    packed_equal = sum(1 for item in complete if item.get("packed_equal"))
    scale_equal = sum(1 for item in complete if item.get("weight_scale_equal"))
    input_scale_equal = sum(1 for item in complete if item.get("input_scale_equal"))
    return {
        "compared": len(complete),
        "missing_or_failed": len(results) - len(complete),
        "packed_equal": packed_equal,
        "weight_scale_equal": scale_equal,
        "input_scale_equal": input_scale_equal,
        "max_rel_rmse": max(rel_rmse),
        "mean_rel_rmse": sum(rel_rmse) / len(rel_rmse),
        "min_cosine": min(cosine),
        "mean_cosine": sum(cosine) / len(cosine),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--layers", help="Comma-separated layer ids or ranges, e.g. 1,30,59")
    parser.add_argument("--experts", help="Comma-separated expert ids or ranges")
    parser.add_argument("--projections", help="Comma-separated projection names")
    parser.add_argument("--limit", type=int, default=18, help="Evenly sample this many matching tensors")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-rel-rmse", type=float)
    parser.add_argument("--fail-min-cosine", type=float)
    args = parser.parse_args()

    reference_dir = args.reference.resolve()
    candidate_dir = args.candidate.resolve()
    reference_map = _load_index(reference_dir)
    candidate_map = _load_index(candidate_dir)
    projections = set(args.projections.split(",")) if args.projections else None

    keys = _select_keys(
        reference_map,
        candidate_map,
        layers=_parse_csv_ints(args.layers),
        experts=_parse_csv_ints(args.experts),
        projections=projections,
        limit=args.limit,
    )
    if not keys:
        raise SystemExit("no shared routed expert NVFP4 weight keys matched the filters")

    results = [
        _compare_key(reference_dir, reference_map, candidate_dir, candidate_map, key) for key in keys
    ]
    summary = _summarize(results)
    report = {
        "reference": str(reference_dir),
        "candidate": str(candidate_dir),
        "selected_keys": len(keys),
        "summary": summary,
        "results": results,
    }

    print(json.dumps({"summary": summary}, indent=2, sort_keys=True))
    for item in results[: min(len(results), 12)]:
        if "rel_rmse" in item:
            print(
                f"{item['key']}: rel_rmse={item['rel_rmse']:.6g} "
                f"cosine={item['cosine']:.8f} packed_equal={item['packed_equal']} "
                f"scale_equal={item['weight_scale_equal']}"
            )
        else:
            print(f"{item['key']}: {item.get('error') or item.get('missing')}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    if summary.get("missing_or_failed", 0):
        return 2
    if args.fail_rel_rmse is not None and summary.get("max_rel_rmse", math.inf) > args.fail_rel_rmse:
        return 3
    if args.fail_min_cosine is not None and summary.get("min_cosine", -math.inf) < args.fail_min_cosine:
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
