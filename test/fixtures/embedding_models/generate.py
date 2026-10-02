#!/usr/bin/env python3
"""Generate deterministic embedding-model parity fixtures.

Requires Python 3.10+, torch 2.7.1 and Transformers commit
3693f8d26311305e914735a6373fb03468d6aaa0. Install it with:

    pip install torch==2.7.1 \
      git+https://github.com/huggingface/transformers.git@3693f8d26311305e914735a6373fb03468d6aaa0

The generated directories are ordinary local Hugging Face repositories, so
they can be loaded by Bumblebee with {:local, path}. `reference.json` contains
float32 hidden states from the same checkpoint and is intentionally kept
separate from the checkpoint to make an update to either reviewable.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
import transformers
from transformers import (
    Gemma3TextConfig,
    Gemma3TextModel,
    LlamaConfig,
    LlamaModel,
    MistralConfig,
    MistralModel,
    Qwen3Config,
    Qwen3Model,
)


torch.manual_seed(0)
torch.set_printoptions(precision=9)

INPUT_IDS = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
CHANGED_FUTURE_IDS = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 9]])
PADDED_IDS = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0]])
FULL_MASK = torch.ones_like(INPUT_IDS)
PADDED_MASK = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])


def common_kwargs():
    return {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 20,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 4,
        "rms_norm_eps": 1.0e-6,
        "pad_token_id": 0,
        "tie_word_embeddings": False,
    }


def model_definitions():
    common = common_kwargs()

    return {
        "gemma3_text": (
            Gemma3TextConfig,
            Gemma3TextModel,
            {
                **common,
                "hidden_activation": "gelu_pytorch_tanh",
                "query_pre_attn_scalar": 8,
                "sliding_window": 4,
                "layer_types": ["sliding_attention", "full_attention"],
                "rope_parameters": {
                    "full_attention": {
                        "rope_type": "yarn",
                        "factor": 2.0,
                        "original_max_position_embeddings": 4,
                        "beta_fast": 32.0,
                        "beta_slow": 1.0,
                    },
                    "sliding_attention": {"rope_theta": 10_000.0},
                },
            },
        ),
        "llama": (LlamaConfig, LlamaModel, {**common, "rope_theta": 10_000.0}),
        "mistral": (
            MistralConfig,
            MistralModel,
            {**common, "rope_theta": 10_000.0, "sliding_window": 4},
        ),
        "qwen3": (
            Qwen3Config,
            Qwen3Model,
            {
                **common,
                "rope_theta": 10_000.0,
                "use_sliding_window": True,
                "sliding_window": 4,
                "max_window_layers": 1,
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": 2.0,
                    "original_max_position_embeddings": 4,
                    "beta_fast": 32.0,
                    "beta_slow": 1.0,
                },
            },
        ),
        # Exercise both accepted serialized spellings. These inputs are longer
        # than max_position_embeddings, so changing the scaling cannot be a
        # no-op in the generated reference.
        "qwen3_linear": (
            Qwen3Config,
            Qwen3Model,
            {
                **common,
                "rope_theta": 10_000.0,
                "rope_scaling": {"type": "linear", "factor": 2.0},
            },
        ),
        "qwen3_dynamic": (
            Qwen3Config,
            Qwen3Model,
            {
                **common,
                "rope_theta": 10_000.0,
                "rope_scaling": {"rope_type": "dynamic", "factor": 2.0},
            },
        ),
    }


def tensor(value):
    return value.detach().to(torch.float32).cpu().tolist()


def evaluate(model, input_ids, attention_mask, position_ids=None):
    # Dynamic RoPE caches its frequency state. Each reference represents an
    # independent inference so prior long inputs must not affect short ones.
    model = copy.deepcopy(model)
    with torch.no_grad():
        return tensor(model(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids).last_hidden_state)


def write_fixture(output_dir, name, config_class, model_class, kwargs, is_causal):
    directory = output_dir / f"tiny-random-{name}-{'causal' if is_causal else 'bidirectional'}"
    # Gemma normalizes its local window in __post_init__, so these flags must
    # be present during construction rather than assigned afterward.
    attention_mode = (
        {"use_bidirectional_attention": not is_causal}
        if name == "gemma3_text"
        else {"is_causal": is_causal}
    )
    config = config_class(**kwargs, **attention_mode, _attn_implementation="eager")

    model = model_class(config).eval()
    model.save_pretrained(directory, safe_serialization=True)

    # Compare against exactly the serialized checkpoint configuration. This
    # catches model-specific config normalization (notably Gemma's window).
    config = config_class.from_pretrained(directory)
    model = model_class.from_pretrained(directory, attn_implementation="eager").eval()

    reference = {
        "transformers_version": transformers.__version__,
        "transformers_commit": "3693f8d26311305e914735a6373fb03468d6aaa0",
        "is_causal": is_causal,
        "input_ids": INPUT_IDS.tolist(),
        "attention_mask": FULL_MASK.tolist(),
        "hidden_state": evaluate(model, INPUT_IDS, FULL_MASK),
        "changed_future_hidden_state": evaluate(model, CHANGED_FUTURE_IDS, FULL_MASK),
        "padded_input_ids": PADDED_IDS.tolist(),
        "padded_attention_mask": PADDED_MASK.tolist(),
        "padded_position_ids": [[0, 1, 2, 3, 0, 0, 0, 0]],
        "padded_hidden_state": evaluate(model, PADDED_IDS, PADDED_MASK,
                                       torch.tensor([[0, 1, 2, 3, 0, 0, 0, 0]])),
        "compact_hidden_state": evaluate(model, INPUT_IDS[:, :4], FULL_MASK[:, :4]),
    }

    (directory / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for name, (config_class, model_class, kwargs) in model_definitions().items():
        for is_causal in (True, False):
            write_fixture(args.output, name, config_class, model_class, kwargs, is_causal)


if __name__ == "__main__":
    main()
