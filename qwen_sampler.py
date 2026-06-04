"""Simple Qwen2.5-7B sampler on TPU using the tunix JAX-native stack."""

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx
from transformers import AutoTokenizer

from tunix.generate import sampler as sampler_lib
from tunix.models.qwen2 import model as model_lib
from tunix.models.qwen2 import params as params_lib

# ---------------------------------------------------------------------------
# Config -- replace with your local path
# ---------------------------------------------------------------------------
MODEL_PATH = "/path/to/qwen2.5-7b"  # local safetensors directory

# Tensor-parallel degree: set to number of TPU chips you want to use.
# For a single TPU v4-8 slice use TP_SIZE=8; for a single chip use 1.
TP_SIZE = 8

MAX_CACHE_LEN = 4096   # KV-cache slots (prompt + generation)
MAX_NEW_TOKENS = 256


def build_mesh(tp_size: int) -> Mesh:
    devices = jax.devices()
    mesh_devices = np.array(devices[:tp_size]).reshape(1, tp_size)
    return Mesh(mesh_devices, axis_names=("fsdp", "tp"))


def load_model_and_sampler(
    model_path: str,
    mesh: Mesh,
) -> tuple[nnx.Module, sampler_lib.Sampler]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model_config = model_lib.ModelConfig.qwen2p5_7b()
    model = params_lib.create_model_from_safe_tensors(
        file_dir=model_path,
        config=model_config,
        mesh=mesh,
        dtype=jnp.bfloat16,
    )

    sampler = sampler_lib.Sampler(
        model,
        tokenizer,
        sampler_lib.CacheConfig(
            cache_size=MAX_CACHE_LEN,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )
    return model, sampler


def generate(
    sampler: sampler_lib.Sampler,
    prompts: list[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[str]:
    out = sampler(
        input_strings=prompts,
        max_generation_steps=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return out.text


if __name__ == "__main__":
    mesh = build_mesh(TP_SIZE)
    model, sampler = load_model_and_sampler(MODEL_PATH, mesh)

    prompts = [
        "Explain the difference between supervised and unsupervised learning.",
        "Write a short poem about the ocean.",
    ]

    responses = generate(sampler, prompts)
    for prompt, response in zip(prompts, responses):
        print(f"Prompt:   {prompt}")
        print(f"Response: {response}")
        print("-" * 60)
