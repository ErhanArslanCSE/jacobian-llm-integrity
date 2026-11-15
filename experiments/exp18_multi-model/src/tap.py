"""Tap test of Algorithm 1 at one decoder layer (same computation as exp9/src/jvp_engine.py).

The tap response of a prompt is J(x) V: the Jacobian of the decoder-layer output at the last
token with respect to the layer input at the last token (earlier positions held fixed), along
the k probe columns of V, computed with k JVPs.
"""

import torch
from torch.func import jvp, vmap

_SKIP_KEYS = {"past_key_values", "use_cache", "cache_position"}


def generate_random_probes(d, k, seed):
    """k orthonormal probe directions: first k columns of a QR-orthonormalized Gaussian matrix."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    V, _ = torch.linalg.qr(torch.randn(d, k, generator=rng))
    return V


def _layer_input_and_kwargs(model, prompt, layer_idx, device):
    """Input hidden states of a decoder layer and the kwargs the model passes to it."""
    captured = {}

    def pre_hook(module, args, kwargs):
        captured["h"] = args[0].detach().clone()
        captured["kwargs"] = {k: v for k, v in kwargs.items() if k not in _SKIP_KEYS}

    handle = model.model.layers[layer_idx].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(input_ids=prompt["input_ids"].to(device),
                  attention_mask=prompt["attention_mask"].to(device))
    finally:
        handle.remove()
    return captured["h"], captured["kwargs"]


def tap_response(model, prompt, layer_idx, probes, device):
    """J(x) V at the last token of the prompt; (d, k) float32 tensor on CPU."""
    layer = model.model.layers[layer_idx]
    hidden, kwargs = _layer_input_and_kwargs(model, prompt, layer_idx, device)
    last = prompt["attention_mask"].sum(dim=1).long().item() - 1
    dtype = hidden.dtype

    def layer_fn(h):
        out = layer(h, **kwargs)
        out = out[0] if isinstance(out, tuple) else out
        return out if out.dim() == 3 else out.unsqueeze(0)

    pos_mask = torch.zeros(1, hidden.shape[1], 1, device=device, dtype=dtype)
    pos_mask[0, last, 0] = 1.0

    def single_jvp(v):
        _, out = jvp(layer_fn, (hidden,), (v.view(1, 1, -1) * pos_mask,))
        return out[0, last, :]

    probes_T = probes.T.to(device=device, dtype=dtype)
    return vmap(single_jvp)(probes_T).T.detach().float().cpu()


def tap_ratio(response, reference):
    """Per-prompt tap ratio s_k of the paper: ||R - R_ref||_F / ||R_ref||_F."""
    return float((response - reference).norm() / reference.norm().clamp_min(1e-10))
