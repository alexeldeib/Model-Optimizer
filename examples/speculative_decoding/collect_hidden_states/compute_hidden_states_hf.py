# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Extract hidden states from an HF-compatible LLM."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm import tqdm as tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Collect hidden states from conversations
        by running full conversations through a Hugging Face model."""
    )

    ## Model & Generation Parameters ##
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the served model.",
    )

    ## Client Parameters ##
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=3072,
        help="""Maximum number of tokens in a conversation. Longer conversations will be skipped.
        Defaults to 3072 tokens.""",
    )
    parser.add_argument(
        "--model-class",
        choices=["auto", "causal-lm"],
        default="auto",
        help="Which Hugging Face auto class to use for model loading.",
    )
    parser.add_argument(
        "--enable-modelopt-checkpointing",
        action="store_true",
        help="Enable ModelOpt checkpoint hooks before loading a ModelOpt-exported checkpoint.",
    )

    ## I/O Parameters ##
    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="""Path to the `jsonl` file or directory containing `jsonl` files.""",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="""Root directory in which to save the hidden states.
        The data will be saved as a torch (`.pt`) dump file for each conversation.""",
    )
    parser.add_argument(
        "--debug-max-num-conversations",
        type=int,
        default=None,
        help="""For debugging purposes, limit the number of conversations processed.
        Default is None, meaning no limit.""",
    )
    parser.add_argument(
        "--dp-rank",
        type=int,
        default=0,
        help="""Data parallel rank. TASK_ID on SLURM.""",
    )
    parser.add_argument(
        "--dp-world-size",
        type=int,
        default=1,
        help="""Data parallel world size. Number of tasks on SLURM.""",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Set trust_remote_code for Huggingface models and tokenizers",
    )

    return parser.parse_args()


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_dataset(path: Path) -> list[dict]:
    if path.is_file() and str(path).endswith(".jsonl"):
        return list(_iter_jsonl(path))
    if path.is_dir():
        entries = []
        for jsonl in sorted(path.glob("*.jsonl")):
            entries.extend(_iter_jsonl(jsonl))
        return entries
    raise ValueError(
        f"input_data must be a .jsonl file or directory containing .jsonl files, got: {path}"
    )


def _enable_modelopt_checkpointing():
    import modelopt.torch.opt as mto

    mto.enable_huggingface_checkpointing()
    try:
        from modelopt.torch.quantization.plugins.huggingface import (
            patch_compressed_linear_loading,
        )
    except ImportError:
        return nullcontext()
    return patch_compressed_linear_loading()


def _get_hidden_states(outputs):
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states
    language_model_outputs = getattr(outputs, "language_model_outputs", None)
    if language_model_outputs is not None:
        hidden_states = getattr(language_model_outputs, "hidden_states", None)
        if hidden_states is not None:
            return hidden_states
    raise RuntimeError("Model output does not contain hidden_states.")


def _get_input_device(model):
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def main(args: argparse.Namespace) -> None:
    # Load conversations
    dataset = _load_dataset(args.input_data)
    print(f"Loaded {len(dataset)} conversations from {args.input_data}")

    # Shard data
    if args.dp_world_size > 1:
        dataset = [
            entry
            for idx, entry in enumerate(dataset)
            if idx % args.dp_world_size == args.dp_rank
        ]
    print(
        f"Sharded dataset to {len(dataset)} conversations for DP#{args.dp_rank}/{args.dp_world_size}"
    )

    # Remove already dumped conversations
    def keep_conversation(entry):
        conversation_id = entry.get("conversation_id", entry.get("uuid", None))
        assert conversation_id is not None, "conversation_id is required"
        output_file = args.output_dir / f"{conversation_id}.pt"
        return not output_file.exists()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    original_num = len(dataset)
    dataset = [entry for entry in dataset if keep_conversation(entry)]
    print(
        "Removed",
        original_num - len(dataset),
        "conversations due to existing output files",
    )

    # For debugging
    if args.debug_max_num_conversations is not None:
        dataset = dataset[: args.debug_max_num_conversations]

    model_cls = AutoModelForCausalLM if args.model_class == "causal-lm" else AutoModel
    checkpoint_context = (
        _enable_modelopt_checkpointing() if args.enable_modelopt_checkpointing else nullcontext()
    )
    with checkpoint_context:
        model = model_cls.from_pretrained(
            args.model, dtype="auto", device_map="auto", trust_remote_code=args.trust_remote_code
        )
    model.eval()
    text_config = getattr(model.config, "text_config", None)
    num_hidden_layers = getattr(model.config, "num_hidden_layers", None) or getattr(
        text_config, "num_hidden_layers", None
    )
    input_device = _get_input_device(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is not None:
        tokenizer.chat_template = tokenizer.chat_template.replace(REMOVE_THINK_CHAT_TEMPLATE, "")

    num_skipped_too_long = 0
    num_invalid = 0
    num_success = 0
    pbar = tqdm(total=len(dataset), desc=f"DP#{args.dp_rank} Processing conversations")

    def dump_hidden_states(conversation_id: int, input_ids: torch.Tensor):
        nonlocal num_success
        nonlocal num_hidden_layers

        # Get hidden states
        with torch.inference_mode():
            outputs = model(input_ids=input_ids.to(input_device), output_hidden_states=True)
            hidden_states = _get_hidden_states(outputs)
            if num_hidden_layers is None:
                num_hidden_layers = len(hidden_states) - 1
            else:
                assert num_hidden_layers + 1 == len(hidden_states), (
                    f"Expected {num_hidden_layers}+1 layers of hidden states, but got {len(hidden_states)}."
                )
            # Extract hidden states from layers with index (2, N/2, N-3), and the output hidden states
            selected_layer_indices = [
                2,
                max(0, num_hidden_layers // 2),
                max(1, num_hidden_layers - 3),
            ]
            selected_layer_indices = sorted(set(selected_layer_indices))
            aux_hidden_states = torch.cat(
                [hidden_states[i].squeeze(0).cpu() for i in selected_layer_indices], dim=-1
            )
            output_hidden_states = hidden_states[-1].squeeze(0).cpu()
        output_file = output_dir / f"{conversation_id}.pt"
        tmp_output_file = output_file.with_suffix(".pt.tmp")

        with open(tmp_output_file, "wb") as f:
            torch.save(
                {
                    "input_ids": input_ids.squeeze(0).cpu(),
                    "hidden_states": output_hidden_states,
                    "aux_hidden_states": aux_hidden_states,
                    "conversation_id": conversation_id,
                },
                f,
            )
        tmp_output_file.replace(output_file)

        num_success += 1
        pbar.update(1)
        del outputs, hidden_states, aux_hidden_states, output_hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for entry in dataset:
        conversation_id = entry.get("conversation_id", entry.get("uuid"))

        conversations = entry.get("messages") or entry["conversations"]
        if not conversations or not isinstance(conversations, list):
            num_invalid += 1
            continue

        # return_dict=True ensures BatchEncoding is returned on all transformers versions.
        input_ids = tokenizer.apply_chat_template(
            conversations, return_tensors="pt", return_dict=True, add_generation_template=False
        )["input_ids"]
        num_input_tokens = input_ids.shape[1]
        if num_input_tokens <= 10 or num_input_tokens > args.max_seq_len:
            num_skipped_too_long += 1
            continue

        dump_hidden_states(conversation_id, input_ids)

    if num_skipped_too_long > 0:
        print(f"Skipped {num_skipped_too_long} conversations due to length constraints.")
    if num_invalid > 0:
        print(f"Skipped {num_invalid} invalid conversations without proper fields.")

    if num_success == len(dataset):
        print(f"Successfully processed all {num_success} conversations.")
    else:
        print(f"Successfully processed {num_success} out of {len(dataset)} conversations.")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
