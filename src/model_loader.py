"""Load transformer model with settings compatible with forward-mode AD."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(cfg):
    """Load model and tokenizer with JVP-compatible settings.

    Key choices:
    - bfloat16: same exponent range as float32, avoids JVP overflow
    - eager attention: flash attention doesn't support forward-mode AD
    - use_cache=False: KV cache is mutable state, breaks torch.func.jvp
    """
    model_name = cfg.model.name
    dtype = getattr(torch, cfg.model.dtype)
    device = cfg.model.device

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
        device_map=device,
    )
    model.eval()

    # Disable gradient computation globally — we use torch.func for JVP
    for param in model.parameters():
        param.requires_grad_(False)

    return model, tokenizer


def get_transformer_layers(model):
    """Return the list of transformer layer modules.

    Supports LlamaForCausalLM and similar architectures where
    layers live at model.model.layers.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError(
        f"Unsupported architecture: {type(model).__name__}. "
        "Expected model.model.layers to exist."
    )


def resolve_layer_indices(cfg, model):
    """Resolve layer indices from config.

    If cfg.experiment.layers == "all", returns all layer indices.
    Otherwise returns the explicit list from config.
    """
    all_layers = get_transformer_layers(model)
    n_layers = len(all_layers)

    if cfg.experiment.layers == "all":
        return list(range(n_layers))

    indices = list(cfg.experiment.layers)
    # Validate
    for idx in indices:
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"Layer index {idx} out of range [0, {n_layers})")
    return indices
