"""Load the model with the settings of exp9 (eager attention, no KV cache)."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(cfg):
    """Load model and tokenizer.

    The dtype comes from the config so that the same edits can be measured in
    bfloat16 (the arithmetic of the paper) and in float32 (near-exact arithmetic).
    """
    model_name = cfg.model.name
    dtype = getattr(torch, cfg.model.dtype)

    print(f"Loading model: {model_name}")
    print(f"  dtype={cfg.model.dtype}, attn={cfg.model.attn_implementation}, cache={cfg.model.use_cache}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
        use_cache=cfg.model.use_cache,
        device_map=cfg.model.device,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model, tokenizer
