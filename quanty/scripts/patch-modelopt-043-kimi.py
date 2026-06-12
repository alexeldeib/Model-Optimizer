#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply the minimal Kimi K2.6 compatibility shim to a ModelOpt 0.43 tree."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import yaml


def _patch_once(text: str, old: str, new: str, description: str) -> str:
    if new in text:
        print(f"{description}: already patched")
        return text
    if old not in text:
        raise RuntimeError(f"Could not find patch anchor for {description}")
    print(f"{description}: patched")
    return text.replace(old, new, 1)


def patch_model_calib(modelopt_root: Path) -> None:
    path = modelopt_root / "modelopt/torch/quantization/model_calib.py"
    text = path.read_text()

    text = _patch_once(
        text,
        '''def enable_stats_collection(model: nn.Module):
    """Enable stats collection for all quantizers in the model."""
    for name, module in model.named_modules():
''',
        '''def enable_stats_collection(model: nn.Module):
    """Enable stats collection for all quantizers in the model."""
    for _, module in model.named_modules():
        if getattr(module, "_moe_calib_experts_ratio", None) is not None:
            module._modelopt_moe_calibrating = True

    for name, module in model.named_modules():
''',
        "model_calib enable_stats_collection MoE calibration flag",
    )

    text = _patch_once(
        text,
        '''def finish_stats_collection(model: nn.Module, method: str | None = None, **kwargs):
    """Finish stats collection for all quantizers in the model."""
    for _, module in model.named_modules():
''',
        '''def finish_stats_collection(model: nn.Module, method: str | None = None, **kwargs):
    """Finish stats collection for all quantizers in the model."""
    for _, module in model.named_modules():
        if hasattr(module, "_modelopt_moe_calibrating"):
            module._modelopt_moe_calibrating = False

    for _, module in model.named_modules():
''',
        "model_calib finish_stats_collection MoE calibration flag",
    )

    path.write_text(text)


def patch_huggingface_moe(modelopt_root: Path) -> None:
    path = modelopt_root / "modelopt/torch/quantization/plugins/huggingface.py"
    text = path.read_text()

    text = _patch_once(
        text,
        "import logging\nimport warnings\n",
        "import logging\nimport os\nimport warnings\n",
        "huggingface os import",
    )

    class_marker = "class _QuantSparseMoe(QuantModule):"
    if class_marker not in text:
        raise RuntimeError("Could not find ModelOpt 0.43 _QuantSparseMoe class")
    class_start = text.index(class_marker)
    block_start = text.index("    def _setup(self):", class_start)
    block_end = text.index("    def layer_sync_moe_local_experts_amax", block_start)

    new_block = '''    def _setup(self):
        self._moe_calib_experts_ratio = None
        self._token_counting_initialized = False
        self._count_expert_tokens = False

    def _resolve_num_experts(self) -> int:
        for obj in [getattr(self, "gate", None), self, getattr(self, "experts", None)]:
            if obj is None:
                continue
            for attr in ("num_experts", "n_routed_experts"):
                if hasattr(obj, attr):
                    return int(getattr(obj, attr))
        experts = getattr(self, "experts", None)
        if isinstance(experts, nn.ModuleList):
            return len(experts)
        return 0

    def _init_token_counting(self):
        """Lazy-init token counting infra (buffer + gate hook). Called once from forward."""
        self._token_counting_initialized = True
        num_experts = self._resolve_num_experts()

        if num_experts == 0:
            warnings.warn(
                f"{self.__class__.__name__}: could not resolve num_experts; "
                "expert routing will not be tracked for this layer."
            )
            return

        self.register_buffer(
            "expert_token_count",
            torch.zeros(num_experts, dtype=torch.long, device=next(self.parameters()).device),
            persistent=False,
        )
        if hasattr(self, "gate"):
            self.gate.register_forward_hook(self._gate_forward_hook)

    def _gate_forward_hook(self, module, input, output):
        if not self._count_expert_tokens:
            return
        with torch.no_grad():
            if isinstance(output, tuple) and len(output) >= 3:
                # v5.x TopKRouter: returns (logits, scores, indices)
                indices = output[2]
            elif (
                isinstance(output, tuple)
                and len(output) >= 1
                and isinstance(output[0], torch.Tensor)
                and not torch.is_floating_point(output[0])
            ):
                # Kimi/DeepSeek-style gates return (topk_idx, topk_weight).
                indices = output[0]
            else:
                # v4.x nn.Linear gate: returns logits tensor
                logits = output if not isinstance(output, tuple) else output[0]
                top_k = self.gate.top_k if hasattr(self.gate, "top_k") else self.top_k
                _, indices = torch.topk(logits.float(), top_k, dim=-1)
            counts = torch.bincount(indices.reshape(-1), minlength=self.expert_token_count.shape[0])
            self.expert_token_count += counts.to(self.expert_token_count.device)

    def _moe_calib_token_chunk_size(self) -> int:
        value = os.getenv("MODELOPT_MOE_CALIB_TOKEN_CHUNK", "2048")
        try:
            return max(1, int(value))
        except ValueError:
            warnings.warn(
                "MODELOPT_MOE_CALIB_TOKEN_CHUNK must be an integer; using 2048.",
                stacklevel=2,
            )
            return 2048

    def _moe_calib_tokens_per_expert(self) -> int:
        value = os.getenv("MODELOPT_MOE_CALIB_TOKENS_PER_EXPERT", "64")
        try:
            return max(1, int(value))
        except ValueError:
            warnings.warn(
                "MODELOPT_MOE_CALIB_TOKENS_PER_EXPERT must be an integer; using 64.",
                stacklevel=2,
            )
            return 64

    def _expert_candidate_token_indices(
        self, flat_hidden_states: torch.Tensor, num_experts: int
    ) -> list[torch.Tensor] | None:
        """Return top gate-score token indices per expert for bounded forced calibration."""

        gate = getattr(self, "gate", None)
        gate_weight = getattr(gate, "weight", None)
        if gate is None or gate_weight is None or gate_weight.ndim != 2:
            return None
        if gate_weight.shape[0] != num_experts or gate_weight.shape[1] != flat_hidden_states.shape[-1]:
            return None

        with torch.no_grad():
            scores = linear(
                flat_hidden_states.float(),
                gate_weight.detach().float().to(flat_hidden_states.device),
                None,
            )
            scoring_func = getattr(gate, "scoring_func", None)
            if scoring_func == "sigmoid":
                scores = scores.sigmoid()
            elif scoring_func == "softmax":
                scores = scores.softmax(dim=-1)

            correction_bias = getattr(gate, "e_score_correction_bias", None)
            if correction_bias is not None:
                scores = scores + correction_bias.detach().float().to(scores.device).unsqueeze(0)
            raw_scores = scores

            n_group = getattr(gate, "n_group", getattr(gate, "n_groups", None))
            topk_group = getattr(gate, "topk_group", getattr(gate, "topk_groups", None))
            if n_group is not None and topk_group is not None and num_experts % n_group == 0:
                group_size = num_experts // n_group
                group_scores = scores.view(-1, n_group, group_size).topk(
                    min(2, group_size), dim=-1
                )[0].sum(dim=-1)
                group_idx = torch.topk(
                    group_scores, k=min(topk_group, n_group), dim=-1, sorted=False
                )[1]
                group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
                group_mask.scatter_(1, group_idx, True)
                score_mask = group_mask.unsqueeze(-1).expand(-1, n_group, group_size).reshape(
                    -1, num_experts
                )
                scores = scores.masked_fill(~score_mask, float("-inf"))

            tokens_per_expert = min(self._moe_calib_tokens_per_expert(), flat_hidden_states.shape[0])
            if tokens_per_expert <= 0:
                return None

            candidate_indices = []
            for expert_idx in range(num_experts):
                expert_scores = scores[:, expert_idx]
                valid_token_indices = torch.nonzero(
                    torch.isfinite(expert_scores), as_tuple=False
                ).flatten()
                if valid_token_indices.numel() == 0:
                    expert_scores = raw_scores[:, expert_idx]
                    valid_token_indices = torch.arange(
                        expert_scores.shape[0], device=expert_scores.device
                    )
                expert_scores = expert_scores.index_select(0, valid_token_indices)
                k = min(tokens_per_expert, expert_scores.numel())
                selected = torch.topk(expert_scores, k=k, dim=0, sorted=False).indices
                candidate_indices.append(valid_token_indices.index_select(0, selected))
            return candidate_indices

    def _calibrate_all_sequential_experts(self, hidden_states: torch.Tensor) -> bool:
        """Calibrate every sequential expert without materializing all routed outputs."""

        experts = getattr(self, "experts", None)
        if not hasattr(experts, "__iter__"):
            return False

        if not self._token_counting_initialized:
            self._init_token_counting()

        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        num_experts = self._resolve_num_experts()
        candidate_indices = self._expert_candidate_token_indices(flat_hidden_states, num_experts)
        chunk_size = self._moe_calib_token_chunk_size()
        count_source = "gate_top_tokens" if candidate_indices is not None else "all_tokens"

        with torch.no_grad():
            for expert_idx, expert in enumerate(experts):
                if expert is None:
                    continue
                if candidate_indices is None:
                    expert_inputs = flat_hidden_states
                    token_count = flat_hidden_states.shape[0]
                else:
                    expert_inputs = flat_hidden_states.index_select(0, candidate_indices[expert_idx])
                    token_count = expert_inputs.shape[0]
                for chunk in expert_inputs.split(chunk_size, dim=0):
                    expert_output = expert(chunk)
                    del expert_output
                if hasattr(self, "expert_token_count") and expert_idx < self.expert_token_count.numel():
                    self.expert_token_count[expert_idx] += token_count
            self._expert_token_count_source = count_source

        return True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._moe_calib_experts_ratio is None:
            return super().forward(hidden_states)

        is_calib = getattr(self, "_modelopt_moe_calibrating", False) or any(
            getattr(m, "_if_calib", False) for m in self.experts.modules()
        )

        # During calibration, forward all tokens to a larger fraction of experts to improve
        # calibration coverage, then re-run with the original top_k for actual outputs.
        if is_calib:
            self._count_expert_tokens = True
            try:
                if self._count_expert_tokens and not self._token_counting_initialized:
                    self._init_token_counting()
                if TRANSFORMERS_VERSION_GE_5_0:
                    assert hasattr(self, "gate") and hasattr(self.gate, "top_k")
                    top_k_owner = self.gate
                else:
                    top_k_owner = (
                        self.gate if hasattr(self, "gate") and hasattr(self.gate, "top_k") else self
                    )
                original_top_k = top_k_owner.top_k
                num_experts = self._resolve_num_experts()
                if not num_experts:
                    raise ValueError(f"Could not find num_experts in module {self}")
                target_top_k = max(original_top_k, round(num_experts * self._moe_calib_experts_ratio))
                if target_top_k >= num_experts and self._calibrate_all_sequential_experts(hidden_states):
                    pass
                else:
                    top_k_owner.top_k = target_top_k
                    try:
                        super().forward(hidden_states)
                    finally:
                        top_k_owner.top_k = original_top_k
            finally:
                self._count_expert_tokens = False

        output = super().forward(hidden_states)
        self._count_expert_tokens = False
        return output

'''

    if "_calibrate_all_sequential_experts" in text[class_start:block_end]:
        print("huggingface _QuantSparseMoe bounded calibration: already patched")
    else:
        text = text[:block_start] + new_block + text[block_end:]
        print("huggingface _QuantSparseMoe bounded calibration: patched")

    path.write_text(text)


def _selector_entry_to_stock(entry: dict) -> tuple[str, dict] | tuple[str, str, dict]:
    entry = copy.deepcopy(entry)
    quantizer_name = entry.pop("quantizer_name", None)
    parent_class = entry.pop("parent_class", None)
    cfg = entry.pop("cfg", {}) or {}

    if quantizer_name is None:
        raise ValueError(f"quant_cfg entry missing quantizer_name: {entry}")

    stock_entry = dict(cfg)
    stock_entry.update(entry)

    if parent_class:
        return "parent", str(parent_class).strip("'\""), str(quantizer_name), stock_entry

    key = "default" if quantizer_name == "*" else str(quantizer_name)
    return "plain", key, stock_entry


def translate_recipe_for_043(recipe_in: Path, recipe_out: Path) -> None:
    data = yaml.safe_load(recipe_in.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Recipe {recipe_in} did not parse as a YAML mapping")

    metadata = data.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("recipe_type") != "ptq":
        raise ValueError(f"Recipe {recipe_in} must contain metadata.recipe_type=ptq")

    if "ptq_cfg" in data:
        ptq_cfg = copy.deepcopy(data["ptq_cfg"])
    else:
        quantize = data.get("quantize")
        if not isinstance(quantize, dict):
            raise ValueError(f"Recipe {recipe_in} must contain quantize or ptq_cfg")

        algorithm = copy.deepcopy(quantize.get("algorithm", "max"))
        if isinstance(algorithm, dict):
            if "layerwise" in algorithm:
                layerwise = bool(algorithm.pop("layerwise"))
                if layerwise and os.getenv("MODELOPT043_USE_SEQUENTIAL", "0") == "1":
                    algorithm["use_sequential"] = True
                    print("Translated recipe layerwise=True to ModelOpt 0.43 use_sequential=True")
                elif layerwise:
                    print(
                        "Dropping recipe layerwise=True for ModelOpt 0.43 dequant; "
                        "set MODELOPT043_USE_SEQUENTIAL=1 to force old sequential calibration."
                    )
            algorithm.pop("layerwise_checkpoint_dir", None)

        stock_quant_cfg = {}
        for raw_entry in quantize.get("quant_cfg", []):
            converted = _selector_entry_to_stock(raw_entry)
            if converted[0] == "parent":
                _, parent_class, quantizer_name, stock_entry = converted
                stock_quant_cfg.setdefault(parent_class, {})[quantizer_name] = stock_entry
            else:
                _, key, stock_entry = converted
                stock_quant_cfg[key] = stock_entry

        ptq_cfg = {"algorithm": algorithm, "quant_cfg": stock_quant_cfg}

    if isinstance(ptq_cfg.get("algorithm"), dict):
        algorithm = copy.deepcopy(ptq_cfg["algorithm"])
        if "layerwise" in algorithm:
            layerwise = bool(algorithm.pop("layerwise"))
            if layerwise and os.getenv("MODELOPT043_USE_SEQUENTIAL", "0") == "1":
                algorithm["use_sequential"] = True
                print("Translated ptq_cfg layerwise=True to ModelOpt 0.43 use_sequential=True")
            elif layerwise:
                print(
                    "Dropping ptq_cfg layerwise=True for ModelOpt 0.43 dequant; "
                    "set MODELOPT043_USE_SEQUENTIAL=1 to force old sequential calibration."
                )
        algorithm.pop("layerwise_checkpoint_dir", None)
        ptq_cfg["algorithm"] = algorithm

    output = {
        "metadata": {
            "recipe_type": "ptq",
            "description": metadata.get("description", "Kimi K2.6 PTQ recipe for ModelOpt 0.43."),
        },
        "ptq_cfg": ptq_cfg,
    }
    recipe_out.parent.mkdir(parents=True, exist_ok=True)
    recipe_out.write_text(yaml.safe_dump(output, sort_keys=False))
    print(f"Translated recipe for ModelOpt 0.43: {recipe_in} -> {recipe_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelopt-root", required=True, type=Path)
    parser.add_argument("--recipe-in", required=True, type=Path)
    parser.add_argument("--recipe-out", required=True, type=Path)
    args = parser.parse_args()

    patch_model_calib(args.modelopt_root)
    patch_huggingface_moe(args.modelopt_root)
    translate_recipe_for_043(args.recipe_in, args.recipe_out)


if __name__ == "__main__":
    main()
