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

"""Unit tests for modelopt.recipe.loader and modelopt.recipe.loader.load_config."""

import re

import pytest

from modelopt.recipe.config import (
    ModelOptDFlashRecipe,
    ModelOptEagleRecipe,
    ModelOptPTQRecipe,
    RecipeType,
)
from modelopt.recipe.loader import load_config, load_recipe, load_recipe_from_dict

# ---------------------------------------------------------------------------
# Static YAML fixtures
# ---------------------------------------------------------------------------

CFG_AB = """\
a: 1
b: 2
"""

CFG_KEY_VAL = """\
key: val
"""

CFG_RECIPE_MISSING_TYPE = """\
metadata:
  description: Missing recipe_type.
quantize: {}
"""

CFG_RECIPE_MISSING_quantize = """\
metadata:
  recipe_type: ptq
"""

CFG_RECIPE_UNSUPPORTED_TYPE = """\
metadata:
  recipe_type: unknown_type
"""

# ---------------------------------------------------------------------------
# Directory-format YAML fixtures
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# load_config — basic behaviour
# ---------------------------------------------------------------------------


def test_load_config_plain(tmp_path):
    """A plain config is returned as-is."""
    (tmp_path / "cfg.yml").write_text(CFG_AB)
    assert load_config(tmp_path / "cfg.yml") == {"a": 1, "b": 2}


def test_load_config_suffix_probe(tmp_path):
    """load_config finds a .yml file when suffix is omitted from a string path."""
    (tmp_path / "mycfg.yml").write_text(CFG_KEY_VAL)
    assert load_config(str(tmp_path / "mycfg")) == {"key": "val"}


def test_load_config_missing_file_raises(tmp_path):
    """load_config raises ValueError for a path that does not exist."""
    with pytest.raises(ValueError, match="Cannot find config file"):
        load_config(str(tmp_path / "nonexistent"))


# ---------------------------------------------------------------------------
# load_recipe — built-in PTQ recipes
# ---------------------------------------------------------------------------


def test_load_recipe_builtin_with_suffix():
    """load_recipe loads a built-in PTQ recipe given the full YAML path."""
    recipe = load_recipe("general/ptq/fp8_default-fp8_kv.yaml")
    assert recipe.recipe_type == RecipeType.PTQ
    assert isinstance(recipe, ModelOptPTQRecipe)
    assert recipe.quantize


def test_load_recipe_builtin_without_suffix():
    """load_recipe resolves the .yaml suffix automatically."""
    recipe = load_recipe("general/ptq/fp8_default-fp8_kv")
    assert recipe.recipe_type == RecipeType.PTQ


def test_load_recipe_builtin_description():
    """The description field is loaded from the YAML metadata."""
    recipe = load_recipe("general/ptq/fp8_default-fp8_kv.yaml")
    assert isinstance(recipe.description, str)
    assert len(recipe.description) > 0


_BUILTIN_PTQ_RECIPES = [
    "general/ptq/fp8_default-fp8_kv",
    "general/ptq/nvfp4_default-fp8_kv",
    "general/ptq/nvfp4_mlp_only-fp8_kv",
    "general/ptq/nvfp4_omlp_only-fp8_kv",
]


@pytest.mark.parametrize("recipe_path", _BUILTIN_PTQ_RECIPES)
def test_load_recipe_all_builtins(recipe_path):
    """Smoke-test: every built-in PTQ recipe loads without error and has quantize."""
    recipe = load_recipe(recipe_path)
    assert recipe.recipe_type == RecipeType.PTQ
    assert isinstance(recipe, ModelOptPTQRecipe)
    assert recipe.quantize


# ---------------------------------------------------------------------------
# load_recipe — error cases
# ---------------------------------------------------------------------------


def test_load_recipe_missing_raises(tmp_path):
    """load_recipe raises ValueError for a path that doesn't exist."""
    with pytest.raises(ValueError):
        load_recipe(str(tmp_path / "does_not_exist.yml"))


def test_load_recipe_missing_recipe_type_raises(tmp_path):
    """load_recipe raises ValueError when metadata.recipe_type is absent."""
    bad = tmp_path / "bad.yml"
    bad.write_text(CFG_RECIPE_MISSING_TYPE)
    with pytest.raises(ValueError, match="recipe_type"):
        load_recipe(bad)


def test_load_recipe_missing_quantize_raises(tmp_path):
    """load_recipe raises ValueError when quantize is absent for a PTQ recipe."""
    bad = tmp_path / "bad.yml"
    bad.write_text(CFG_RECIPE_MISSING_quantize)
    with pytest.raises(ValueError, match="quantize"):
        load_recipe(bad)


def test_load_recipe_unsupported_type_raises(tmp_path):
    """load_recipe raises ValueError for an unknown recipe_type."""
    bad = tmp_path / "bad.yml"
    bad.write_text(CFG_RECIPE_UNSUPPORTED_TYPE)
    with pytest.raises(ValueError, match="Unsupported recipe type"):
        load_recipe(bad)


# ---------------------------------------------------------------------------
# load_recipe — directory format
# ---------------------------------------------------------------------------


def test_load_recipe_dir(tmp_path):
    """load_recipe loads a recipe from a directory with recipe.yml + quantize.yml."""
    (tmp_path / "recipe.yml").write_text(
        "metadata:\n  recipe_type: ptq\n  description: Dir test.\n"
    )
    (tmp_path / "quantize.yml").write_text("algorithm: max\nquant_cfg: []\n")
    recipe = load_recipe(tmp_path)
    assert recipe.recipe_type == RecipeType.PTQ
    assert recipe.description == "Dir test."
    assert recipe.quantize.algorithm == "max"
    assert recipe.quantize.quant_cfg == []


def test_load_recipe_dir_missing_recipe_raises(tmp_path):
    """load_recipe raises ValueError when recipe.yml is absent from the directory."""
    (tmp_path / "quantize.yml").write_text("algorithm: max\nquant_cfg: {}\n")
    with pytest.raises(ValueError, match="recipe descriptor"):
        load_recipe(tmp_path)


def test_load_recipe_dir_missing_quantize_raises(tmp_path):
    """load_recipe raises ValueError when quantize.yml is absent from the directory."""
    (tmp_path / "recipe.yml").write_text("metadata:\n  recipe_type: ptq\n")
    with pytest.raises(ValueError, match="quantize"):
        load_recipe(tmp_path)


# ---------------------------------------------------------------------------
# load_recipe — EAGLE speculative decoding
# ---------------------------------------------------------------------------


def test_load_recipe_eagle_builtin():
    """load_recipe loads the built-in EAGLE recipe and returns a ModelOptEagleRecipe."""
    recipe = load_recipe("general/speculative_decoding/eagle3")
    assert recipe.recipe_type == RecipeType.SPECULATIVE_EAGLE
    assert isinstance(recipe, ModelOptEagleRecipe)
    assert recipe.eagle.eagle_decoder_type == "llama"
    assert recipe.eagle.eagle_ttt_steps == 3
    # Full-pipeline recipe also carries HF trainer sections.
    assert "mode" in recipe.training
    assert recipe.training["mode"] == "eagle3"


def test_load_recipe_eagle_dir(tmp_path):
    """load_recipe loads an EAGLE recipe from a directory with recipe.yml + eagle.yml."""
    (tmp_path / "recipe.yml").write_text(
        "metadata:\n  recipe_type: speculative_eagle\n  description: Dir eagle test.\n"
    )
    (tmp_path / "eagle.yml").write_text(
        "eagle_decoder_type: llama\neagle_ttt_steps: 5\neagle_use_torch_compile: false\n"
    )
    recipe = load_recipe(tmp_path)
    assert recipe.recipe_type == RecipeType.SPECULATIVE_EAGLE
    assert isinstance(recipe, ModelOptEagleRecipe)
    assert recipe.description == "Dir eagle test."
    assert recipe.eagle.eagle_ttt_steps == 5
    assert recipe.eagle.eagle_use_torch_compile is False


def test_load_recipe_eagle_missing_section_raises(tmp_path):
    """load_recipe raises ValueError when 'eagle' is absent for a SPECULATIVE_EAGLE recipe."""
    bad = tmp_path / "bad.yml"
    bad.write_text("metadata:\n  recipe_type: speculative_eagle\n")
    with pytest.raises(ValueError, match="eagle"):
        load_recipe(bad)


def test_load_recipe_eagle_field_validation_raises(tmp_path):
    """Invalid EAGLE field values must fail Pydantic validation at load time."""
    bad = tmp_path / "bad.yml"
    bad.write_text(
        "metadata:\n  recipe_type: speculative_eagle\neagle:\n  eagle_ttt_steps: not_an_int\n"
    )
    with pytest.raises(Exception):  # pydantic.ValidationError
        load_recipe(bad)


# ---------------------------------------------------------------------------
# load_recipe — DFlash speculative decoding
# ---------------------------------------------------------------------------


def test_load_recipe_dflash_builtin():
    """load_recipe loads the built-in DFlash recipe and returns a ModelOptDFlashRecipe."""
    recipe = load_recipe("general/speculative_decoding/dflash")
    assert recipe.recipe_type == RecipeType.SPECULATIVE_DFLASH
    assert isinstance(recipe, ModelOptDFlashRecipe)
    assert recipe.dflash.dflash_block_size == 8
    assert recipe.dflash.dflash_num_anchors == 512
    # Full-pipeline recipe also carries HF trainer sections.
    assert "mode" in recipe.training
    assert recipe.training["mode"] == "dflash"


def test_load_recipe_dflash_dir(tmp_path):
    """load_recipe loads a DFlash recipe from a directory with recipe.yml + dflash.yml."""
    (tmp_path / "recipe.yml").write_text(
        "metadata:\n  recipe_type: speculative_dflash\n  description: Dir dflash test.\n"
    )
    (tmp_path / "dflash.yml").write_text(
        "dflash_block_size: 16\ndflash_loss_decay_factor: 7.0\ndflash_use_torch_compile: false\n"
    )
    recipe = load_recipe(tmp_path)
    assert recipe.recipe_type == RecipeType.SPECULATIVE_DFLASH
    assert isinstance(recipe, ModelOptDFlashRecipe)
    assert recipe.description == "Dir dflash test."
    assert recipe.dflash.dflash_block_size == 16
    assert recipe.dflash.dflash_loss_decay_factor == 7.0


def test_load_recipe_dflash_missing_section_raises(tmp_path):
    """load_recipe raises ValueError when 'dflash' is absent for a SPECULATIVE_DFLASH recipe."""
    bad = tmp_path / "bad.yml"
    bad.write_text("metadata:\n  recipe_type: speculative_dflash\n")
    with pytest.raises(ValueError, match="dflash"):
        load_recipe(bad)


def test_load_recipe_from_dict_eagle_with_training_sections():
    """load_recipe_from_dict accepts a pre-merged dict and populates HF trainer sections."""
    data = {
        "metadata": {"recipe_type": "speculative_eagle"},
        "model": {"model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
        "data": {"data_path": "train.jsonl"},
        "training": {"mode": "eagle3", "output_dir": "ckpts/test"},
        "eagle": {"eagle_decoder_type": "llama", "eagle_ttt_steps": 2},
    }
    recipe = load_recipe_from_dict(data)
    assert isinstance(recipe, ModelOptEagleRecipe)
    assert recipe.model["model_name_or_path"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert recipe.data["data_path"] == "train.jsonl"
    assert recipe.training["output_dir"] == "ckpts/test"
    assert recipe.eagle.eagle_ttt_steps == 2


def test_load_recipe_dflash_field_validation_raises(tmp_path):
    """Invalid DFlash field values must fail Pydantic validation at load time."""
    bad = tmp_path / "bad.yml"
    bad.write_text(
        "metadata:\n  recipe_type: speculative_dflash\ndflash:\n  dflash_block_size: not_an_int\n"
    )
    with pytest.raises(Exception):  # pydantic.ValidationError
        load_recipe(bad)


# ---------------------------------------------------------------------------
# YAML recipe consistency — built-in general/ptq files match config.py dicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("yaml_path", "model_cfg_name", "kv_cfg_name"),
    [
        ("general/ptq/fp8_default-fp8_kv.yaml", "FP8_DEFAULT_CFG", "FP8_KV_CFG"),
        ("general/ptq/nvfp4_default-fp8_kv.yaml", "NVFP4_DEFAULT_CFG", "FP8_KV_CFG"),
        ("general/ptq/nvfp4_mlp_only-fp8_kv.yaml", "NVFP4_MLP_ONLY_CFG", "FP8_KV_CFG"),
        ("general/ptq/nvfp4_omlp_only-fp8_kv.yaml", "NVFP4_OMLP_ONLY_CFG", "FP8_KV_CFG"),
    ],
)
def test_general_ptq_yaml_matches_config_dicts(yaml_path, model_cfg_name, kv_cfg_name):
    """Each general/ptq YAML's quant_cfg list matches the merged Python config dicts."""
    import json

    import modelopt.torch.quantization.config as qcfg
    from modelopt.torch.quantization.config import normalize_quant_cfg_list

    model_cfg = getattr(qcfg, model_cfg_name)
    kv_cfg = getattr(qcfg, kv_cfg_name)
    yaml_data = load_config(yaml_path)

    def _normalize_fpx(val):
        """Normalize FPx representations to a canonical ``[E, M]`` list.

        Python configs may use tuple form ``(E, M)`` or string alias ``"eEmM"``;
        YAML always uses the string form.  Both are converted to ``[E, M]`` so the
        comparison is representation-agnostic.
        """
        if isinstance(val, str):
            m = re.fullmatch(r"e(\d+)m(\d+)", val)
            if m:
                return [int(m.group(1)), int(m.group(2))]
        if isinstance(val, tuple) and len(val) == 2 and all(isinstance(x, int) for x in val):
            return list(val)
        if isinstance(val, dict):
            return {str(k): _normalize_fpx(v) for k, v in val.items()}
        return val

    def _normalize_entries(raw_entries):
        """Normalize a raw quant_cfg list to a canonical, JSON-serialisable form."""
        entries = normalize_quant_cfg_list(list(raw_entries))
        result = []
        for entry in entries:
            e = {k: v for k, v in entry.items() if v is not None}
            if "cfg" in e and e["cfg"] is not None:
                e["cfg"] = _normalize_fpx(e["cfg"])
            result.append(e)
        return result

    def _sort_key(entry):
        return json.dumps(entry, sort_keys=True, default=str)

    python_entries = _normalize_entries(model_cfg["quant_cfg"] + kv_cfg["quant_cfg"])
    yaml_entries = _normalize_entries(yaml_data["quantize"]["quant_cfg"])

    assert sorted(python_entries, key=_sort_key) == sorted(yaml_entries, key=_sort_key)
    assert model_cfg["algorithm"] == yaml_data["quantize"]["algorithm"]
