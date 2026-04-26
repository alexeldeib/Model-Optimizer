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

"""Multi-node PTQ (Post-Training Quantization) with FSDP2 support."""

import argparse
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import safetensors
import torch
import torch.nn as nn
from accelerate import Accelerator, init_empty_weights
from example_utils import build_quant_cfg, get_tokenizer
from modelopt.recipe import ModelOptPTQRecipe, load_recipe
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedTokenizerFast

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export import get_model_type
from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.unified_export_hf import _export_transformers_checkpoint
from modelopt.torch.quantization.config import need_calibration
from modelopt.torch.quantization.plugins.huggingface import patch_compressed_linear_loading
from modelopt.torch.quantization.utils import patch_fsdp_mp_dtypes
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader, get_supported_datasets

# Constants
RAND_SEED = 1234

QUANT_CFG_CHOICES: dict[str, dict[str, Any]] = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,
    "w4a8_mxfp4_fp8": mtq.W4A8_MXFP4_FP8_CFG,
    "nvfp4_mlp_only": mtq.NVFP4_MLP_ONLY_CFG,
    "nvfp4_experts_only": mtq.NVFP4_EXPERTS_ONLY_CFG,
    "nvfp4_omlp_only": mtq.NVFP4_OMLP_ONLY_CFG,
}

KV_QUANT_CFG_CHOICES = {
    "none": "none",
    "fp8": "FP8_KV_CFG",
    "nvfp4": "NVFP4_KV_CFG",
    "nvfp4_affine": "NVFP4_AFFINE_KV_CFG",
}


# Enable HuggingFace checkpointing
mto.enable_huggingface_checkpointing()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-node post-training quantization with FSDP2")

    parser.add_argument(
        "--pyt_ckpt_path",
        required=True,
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--qformat",
        default="fp8",
        choices=QUANT_CFG_CHOICES.keys(),
        help="Quantization format",
    )
    parser.add_argument(
        "--recipe",
        default=None,
        help=(
            "PTQ recipe YAML file or recipe name. When set, this is used instead of --qformat "
            "for the main quantization config."
        ),
    )
    parser.add_argument(
        "--kv_cache_qformat",
        default="fp8",
        choices=list(KV_QUANT_CFG_CHOICES.keys()),
        help="KV cache quantization format",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for calibration",
    )
    parser.add_argument(
        "--calib_size",
        type=str,
        default="512",
        help="Comma-separated list of calibration sizes per dataset",
    )
    parser.add_argument(
        "--dataset",
        help=(
            f"name of a dataset, or a comma separated list of datasets. "
            f"dataset choices are {get_supported_datasets()}"
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--export_path",
        default="exported_model",
        help="Directory to export the quantized model",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for HuggingFace models",
    )
    parser.add_argument("--awq_block_size", default=0, type=int)

    args = parser.parse_args()

    # Parse comma-separated lists
    args.dataset = args.dataset.split(",") if args.dataset else None
    args.calib_size = [int(x) for x in args.calib_size.split(",")]

    return args


def _find_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Locate the homogeneous decoder layers for FSDP2 wrap.

    Walks past common multimodal containers (``language_model`` /
    ``text_model``) so VLM checkpoints like Kimi-K2.x's
    ``KimiK25ForConditionalGeneration -> DeepseekV3ForCausalLM`` resolve
    to their LLM core.  This mirrors the ``feat/vlm-decoder-discovery``
    upstream commit's dispatcher policy.
    """
    cursors = [model]
    for attr in ("language_model", "text_model"):
        if hasattr(cursors[-1], attr):
            cursors.append(getattr(cursors[-1], attr))
    for cursor in reversed(cursors):
        if hasattr(cursor, "model") and hasattr(cursor.model, "layers"):
            return list(cursor.model.layers)
        if hasattr(cursor, "layers"):
            return list(cursor.layers)
    raise ValueError("Could not locate decoder layers for FSDP2 wrap")


def _load_sharded_weights_from_safetensors(
    model: nn.Module,
    model_path: str,
    accelerator: Accelerator,
) -> None:
    """Per-rank sharded read from HF safetensors into FSDP2-wrapped model.

    Each rank reads each tensor on demand and uses ``distribute_tensor`` to
    slice it into its FSDP2 shard placement, then frees the full copy.
    Peak memory per rank is one tensor at a time -- never the full model
    on any rank.

    Required because for trillion-parameter MoE models like Kimi K2.x
    (~2 TB BF16) the standard HF ``from_pretrained`` materializes the
    full model on every pod's HBM (or every rank-0 CPU with
    ``cpu_ram_efficient_loading=True``), exceeding both the per-pod
    HBM (4 GPUs * 184 GiB = 736 GiB) and the per-node CPU RAM (~1 TB).

    Args:
        model: Model with structure already set up (e.g. via
            ``init_empty_weights`` + ``from_config``) and FSDP2-wrapped
            via ``fully_shard``.  Parameters should be DTensors on real
            (non-meta) device storage; call ``model.to_empty(device=...)``
            before this function.
        model_path: Directory containing ``*.safetensors`` shards and
            either ``model.safetensors.index.json`` (sharded) or a single
            ``model.safetensors`` file.
        accelerator: Accelerator instance for rank/device info.
    """
    device = accelerator.device
    model_dir = Path(model_path)

    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as fh:
            weight_map: dict[str, str] = json.load(fh)["weight_map"]
    else:
        single = model_dir / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(
                f"No safetensors checkpoint at {model_dir} "
                "(expected model.safetensors.index.json or model.safetensors)"
            )
        with safetensors.safe_open(str(single), framework="pt", device="cpu") as fh:
            weight_map = {k: "model.safetensors" for k in fh.keys()}

    tensors_by_shard: dict[str, list[str]] = {}
    for tname, shard in weight_map.items():
        tensors_by_shard.setdefault(shard, []).append(tname)

    name_to_target: dict[str, tuple[torch.Tensor, str]] = {
        n: (p, "param") for n, p in model.named_parameters()
    }
    for n, b in model.named_buffers():
        name_to_target.setdefault(n, (b, "buffer"))

    n_shards = len(tensors_by_shard)
    if accelerator.is_main_process:
        print(
            f"Sharded load: {n_shards} safetensors files, "
            f"world_size={accelerator.num_processes}"
        )

    n_loaded = 0
    n_skipped = 0
    for i, shard_file in enumerate(sorted(tensors_by_shard.keys())):
        shard_path = str(model_dir / shard_file)
        with safetensors.safe_open(shard_path, framework="pt", device="cpu") as fh:
            for tname in tensors_by_shard[shard_file]:
                target = name_to_target.get(tname)
                if target is None:
                    n_skipped += 1
                    continue
                tensor, kind = target
                full = fh.get_tensor(tname).to(device, dtype=tensor.dtype, non_blocking=True)
                if isinstance(tensor, DTensor):
                    distributed = distribute_tensor(
                        full, tensor.device_mesh, tensor.placements
                    )
                    with torch.no_grad():
                        tensor.copy_(distributed)
                    del distributed
                elif kind == "param":
                    with torch.no_grad():
                        tensor.data.copy_(full)
                else:
                    with torch.no_grad():
                        tensor.copy_(full)
                del full
                n_loaded += 1
        if accelerator.is_main_process and ((i + 1) % 8 == 0 or (i + 1) == n_shards):
            print(f"  shard {i + 1}/{n_shards}: {shard_file}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"Sharded load complete: {n_loaded} tensors loaded, {n_skipped} skipped")


def load_and_prepare_model(
    model_path: str,
    calib_dataloader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    trust_remote_code: bool = False,
) -> tuple[nn.Module, str, list[str], torch.utils.data.DataLoader]:
    """Load model with meta-init + per-rank sharded load + FSDP2 wrap.

    Avoids any rank materializing the full model.  Sequence:
      1. ``init_empty_weights`` + ``from_config``: structure on meta tensors
         (no real storage anywhere).
      2. ``fully_shard`` per decoder layer + top-level: FSDP2 wraps the
         meta tensors into DTensors with sharded placements.
      3. ``model.to_empty(device=...)``: allocates real (uninitialized)
         storage on each rank's HBM for its FSDP2 shards only.
      4. ``_load_sharded_weights_from_safetensors``: each rank reads each
         tensor from the safetensors shards and ``distribute_tensor``
         slices it into the rank's local DTensor shard; the full copy
         is released immediately.

    The standard ``from_pretrained`` path puts the full model on each
    pod's HBM before FSDP2 has a chance to shard cross-node, which
    OOMs at trillion-parameter scale (e.g. Kimi K2.x BF16 = ~2 TB
    >> 4 * 184 GiB per-pod HBM).  ``cpu_ram_efficient_loading=True``
    helps for models that fit in rank-0's CPU RAM but is also
    inadequate at K2.x scale (~1 TB CPU per node).

    Args:
        model_path: HF safetensors directory.
        calib_dataloader: Calibration dataloader to shard for calibration.
        accelerator: Accelerate's Accelerator instance.
        trust_remote_code: Whether to trust remote code.

    Returns:
        Tuple of (prepared_model, model_type, original_architectures,
        calibration_dataloader).
    """
    if accelerator.is_main_process:
        print(f"Meta-initializing model from {model_path}...")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    with init_empty_weights():
        with patch_compressed_linear_loading():
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=trust_remote_code, dtype="auto"
            )

    model.eval()
    model.requires_grad_(False)
    # Disable KV cache during calibration: layerwise_calibrate replays
    # captured (args, kwargs_input) tuples through each layer multiple
    # times to collect GPTQ Hessians.  transformers 5.x's DynamicCache
    # mutates in place across forwards, so the second replay sees
    # k_len=2*q_len while the captured attention_mask is still q_len-wide,
    # raising ``RuntimeError: The expanded size of the tensor (...) must
    # match the existing size (...) at non-singleton dimension 3``.  The
    # cache reset in model_calib.py:1657-1666 only zeroes the kwargs
    # reference; the layer-internal mutation isn't covered.  PTQ never
    # needs the cache, so disabling it at the config level is the
    # conservative correct fix.
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model_type = get_model_type(model)
    original_architectures = model.config.architectures

    if accelerator.is_main_process:
        print("Applying FSDP2 fully_shard per decoder layer...")
    decoder_layers = _find_decoder_layers(model)
    for layer in decoder_layers:
        fully_shard(layer)
    fully_shard(model)

    # Allocate real (uninitialized) storage for each rank's FSDP2 shard.
    # Without this, ``param._local_tensor`` remains on the meta device
    # and copy_ into it is a no-op.
    if accelerator.is_main_process:
        print(f"Allocating per-rank shard storage on {accelerator.device}...")
    model.to_empty(device=accelerator.device)

    _load_sharded_weights_from_safetensors(model, model_path, accelerator)

    calibration_dataloader = accelerator.prepare(calib_dataloader)

    return model, model_type, original_architectures, calibration_dataloader


def create_calibration_dataloader(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataset_names: list[str],
    calib_sizes: list[int],
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Create calibration dataloader from dataset.

    Args:
        tokenizer: HuggingFace tokenizer
        dataset_names: List of dataset names (defaults to cnn_dailymail)
        calib_sizes: Number of samples for each dataset
        batch_size: Batch size for calibration

    Returns:
        DataLoader for calibration
    """

    return get_dataset_dataloader(
        dataset_name=dataset_names,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=calib_sizes,
        device=None,  # Keep data on CPU, calibration loop handles device transfer
        include_labels=False,
    )


def create_fsdp2_calibration_loop(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
):
    """Create calibration loop compatible with FSDP2.

    For FSDP2, we need to use the outer FSDP-wrapped model instead of
    the parameter passed by mtq.quantize to properly handle DTensor.

    Args:
        model: FSDP2-wrapped model
        dataloader: Calibration dataloader
        accelerator: Accelerator instance for device management

    Returns:
        Calibration function compatible with mtq.quantize
    """

    def calibrate(unwrapped_model):
        """Calibration loop that uses the FSDP-wrapped model."""
        for batch in tqdm(dataloader, desc="Calibrating"):
            if isinstance(batch, dict):
                batch = {
                    k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            # Use outer model (FSDP-wrapped), not the parameter
            # Important: We should forward pass using the unwrapped model
            # mtq.quantize will unwrap the model & pass to the forward_loop
            model(**batch)

    return calibrate


def export_model(
    model: nn.Module,
    accelerator: Accelerator,
    export_path: str | Path,
    architectures: list[str],
):
    """Export quantized model to HuggingFace format.

    Args:
        model: Quantized model
        accelerator: Accelerator instance for state dict gathering
        export_path: Directory to export model to
    """
    export_dir = Path(export_path)
    export_dir.mkdir(parents=True, exist_ok=True)

    post_state_dict, hf_quant_config = _export_transformers_checkpoint(
        model, torch.bfloat16, accelerator=accelerator
    )

    if accelerator.is_main_process:
        # Save hf_quant_config.json for backward compatibility
        with open(f"{export_dir}/hf_quant_config.json", "w") as file:
            json.dump(hf_quant_config, file, indent=4)

        hf_quant_config = convert_hf_quant_config_format(hf_quant_config)

        # Save model
        model.save_pretrained(export_dir, state_dict=post_state_dict, save_modelopt_state=False)

        original_config = f"{export_dir}/config.json"
        config_data = {}

        with open(original_config) as file:
            config_data = json.load(file)

        config_data["quantization_config"] = hf_quant_config
        # Update config architectures to use original architectures that does not have FSDP prefix
        config_data["architectures"] = architectures

        with open(original_config, "w") as file:
            json.dump(config_data, file, indent=4)


def main(args):
    """Main quantization workflow."""
    # Validate GPU availability
    if not torch.cuda.is_available():
        raise OSError("GPU is required for quantization.")

    # Validate quantization format
    if args.qformat not in QUANT_CFG_CHOICES:
        raise ValueError(
            f"Quantization format {args.qformat} not supported. Choose from: {QUANT_CFG_CHOICES.keys()}"
        )

    # Set random seeds
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)

    # Initialize accelerator
    accelerator = Accelerator()

    print(f"Rank: {os.environ.get('RANK', 'Not set')}")
    print(f"World Size: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"Local Rank: {os.environ.get('LOCAL_RANK', 'Not set')}")

    # Load tokenizer
    tokenizer = get_tokenizer(args.pyt_ckpt_path, trust_remote_code=args.trust_remote_code)
    default_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # Left padding for better calibration

    # Set default dataset if not provided
    if args.dataset is None:
        args.dataset = ["cnn_dailymail", "nemotron-post-training-dataset-v2"]
        warnings.warn(
            "No dataset specified. Defaulting to cnn_dailymail and nemotron-post-training-dataset-v2."
        )
        # Adjust calib_size to match dataset length by extending or truncating as needed
        args.calib_size = (args.calib_size + [args.calib_size[-1]] * len(args.dataset))[
            : len(args.dataset)
        ]

    # Create calibration dataloader with max batch size
    calib_dataloader = create_calibration_dataloader(
        tokenizer=tokenizer,
        dataset_names=args.dataset,
        calib_sizes=args.calib_size,
        batch_size=args.batch_size,
    )

    # Load and prepare model
    model, model_type, original_architectures, calib_dataloader = load_and_prepare_model(
        model_path=args.pyt_ckpt_path,
        calib_dataloader=calib_dataloader,
        accelerator=accelerator,
        trust_remote_code=args.trust_remote_code,
    )

    if args.recipe is not None:
        recipe = load_recipe(args.recipe)
        assert isinstance(recipe, ModelOptPTQRecipe), (
            f"Expected PTQ recipe, but got {type(recipe).__name__} from {args.recipe}"
        )
        quant_cfg = recipe.quantize.model_dump()
        enable_quant_kv_cache = False
        if args.kv_cache_qformat != "none" and accelerator.is_main_process:
            warnings.warn("--kv_cache_qformat is ignored when --recipe is used.")
    else:
        quant_cfg = QUANT_CFG_CHOICES[args.qformat]

        quant_cfg = build_quant_cfg(
            args.qformat,
            quant_cfg,
            args.awq_block_size,
            model_type,
        )
        enable_quant_kv_cache = args.kv_cache_qformat != "none"

    print(f"{'Enable' if enable_quant_kv_cache else 'Disable'} KV cache quantization")

    # Check if any bmm_quantizer is in the quant_cfg. If so, we need to enable the bmm_quantizer.
    if enable_quant_kv_cache:
        quant_cfg = mtq.update_quant_cfg_with_kv_cache_quant(
            quant_cfg,
            getattr(mtq, KV_QUANT_CFG_CHOICES[args.kv_cache_qformat])["quant_cfg"],
        )

    # Quantize the model
    if accelerator.is_main_process:
        print("Starting quantization...")

    start_time = time.time()

    if need_calibration(quant_cfg):
        calibrate_fn = create_fsdp2_calibration_loop(model, calib_dataloader, accelerator)
    else:
        calibrate_fn = None
        warnings.warn("Dynamic quantization. Calibration skipped.")

    with torch.no_grad():
        model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_fn)

    elapsed = time.time() - start_time

    if accelerator.is_main_process:
        print(f"Quantization completed in {elapsed:.2f}s")
        mtq.print_quant_summary(model)

    start_time = time.time()
    export_model(model, accelerator, args.export_path, original_architectures)
    elapsed = time.time() - start_time

    if accelerator.is_main_process:
        # Restore default padding and export the tokenizer as well.
        if tokenizer is not None:
            tokenizer.padding_side = default_padding_side
            tokenizer.save_pretrained(args.export_path)
        # Export the model
        print(f"Export completed in {elapsed:.2f}s")
        print(f"Model exported to {args.export_path}")

    print("Unpatching FSDP2 MP dtypes")


if __name__ == "__main__":
    args = parse_args()
    # This context manager can be removed once the update to FSDP2 function is reflected in torch
    with patch_fsdp_mp_dtypes():
        main(args)
