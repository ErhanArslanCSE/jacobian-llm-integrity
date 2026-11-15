#!/usr/bin/env python3
"""Experiment 16, step 2 (every audit machine): record what every monitor reads.

For the clean model and every model of models/manifest.json (weights loaded bitwise, SHA-256
checked):
  acts   last-token output of every decoder layer on the calibration, extra known and secret
         prompts (single-prompt passes)
  synth  output of each target decoder layer on n_synthetic secret synthetic hidden states
         (one-token inputs, fixed random directions, scaled to the stored median input norm)
  taps   tap responses J(x) V (k = 8) at each target layer on the calibration prompts and on
         the first n_cal secret prompts
Tensors are stored in the model dtype in records/<tag>.pt.

Usage: python record_exp16.py --config configs/benchmark.yaml --tag H200
"""

import argparse
import json
import os
import time

import torch
from omegaconf import OmegaConf

from build_exp16 import split_prompts, tensor_sha256
from src.subspaces import get_sublayer
from src.tap import generate_random_probes, tap_response

_SKIP_KEYS = {"past_key_values", "use_cache", "cache_position"}


def last_token_outputs(model, prompts, device):
    """(n_prompts, L, d) last-token outputs of every decoder layer."""
    layers = model.model.layers
    rows = []
    for p in prompts:
        outs = [None] * len(layers)

        def make_hook(idx):
            def hook(module, args, output):
                out = output[0] if isinstance(output, tuple) else output
                outs[idx] = (out[0] if out.dim() == 3 else out)[-1].detach()
            return hook

        handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
        try:
            with torch.no_grad():
                model(input_ids=p["input_ids"].to(device), attention_mask=p["attention_mask"].to(device))
        finally:
            for h in handles:
                h.remove()
        rows.append(torch.stack(outs).cpu())
    return torch.stack(rows)


def one_token_kwargs(model, layer_idx, device):
    """Kwargs (rotary embeddings, mask) that the model passes to a layer for a one-token input."""
    captured = {}

    def pre_hook(module, args, kwargs):
        captured.update({k: v for k, v in kwargs.items() if k not in _SKIP_KEYS})

    handle = model.model.layers[layer_idx].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        with torch.no_grad():
            ids = torch.tensor([[model.config.bos_token_id or 0]], device=device)
            model(input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        handle.remove()
    return captured


def synthetic_states(d, n, seed):
    """n fixed unit directions (CPU generator), identical on every machine."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=1, keepdim=True)


def synthetic_outputs(model, layer_idx, directions, scale, device):
    """(n, d) output of one decoder layer on one-token inputs scale * direction."""
    layer = model.model.layers[layer_idx]
    kwargs = one_token_kwargs(model, layer_idx, device)
    dtype = next(model.parameters()).dtype
    x = (directions * scale).to(device=device, dtype=dtype).unsqueeze(1)
    with torch.no_grad():
        out = layer(x, **kwargs)
    out = out[0] if isinstance(out, tuple) else out
    return out[:, -1].detach().cpu()


def record_all(model, cal, known_extra, secret, manifest, exp, models_dir, device, tag):
    t0 = time.time()
    layers = model.model.layers
    sub = exp.target_sublayer
    for l, sha in manifest["clean"].items():
        if tensor_sha256(get_sublayer(layers[int(l)], sub).weight.data) != sha:
            raise RuntimeError(f"Clean weights of layer {l} differ from the build machine")
    d = layers[0].self_attn.q_proj.weight.shape[1]
    probes = generate_random_probes(d, exp.tap_k, exp.probe_seed)
    directions = synthetic_states(d, exp.n_synthetic, exp.synthetic_seed)
    act_prompts = cal + known_extra + secret
    tap_sets = {"cal": cal, "secret": secret[:len(cal)]}
    target = sorted(exp.backdoor_layers)

    dtype = next(model.parameters()).dtype

    def record():
        return {"acts": last_token_outputs(model, act_prompts, device),
                "synth": {l: synthetic_outputs(model, l, directions, manifest["synthetic_scale"][str(l)], device)
                          for l in target},
                "taps": {s: {l: torch.stack([tap_response(model, p, l, probes, device).to(dtype) for p in ps])
                             for l in target}
                         for s, ps in tap_sets.items()}}

    rec = {"tag": tag, "torch": torch.__version__, "clean": record(), "clean_repeat": record(), "models": {}}
    print(f"  clean recorded ({time.time() - t0:.0f}s)")
    for name, info in manifest["models"].items():
        saved = {}
        for l, f in info["files"].items():
            weight = get_sublayer(layers[int(l)], sub).weight
            W = torch.load(os.path.join(models_dir, f["file"])).to(weight.device)
            if tensor_sha256(W) != f["sha256"]:
                raise RuntimeError(f"{name} layer {l} does not match the manifest")
            saved[l] = weight.data.clone()
            weight.data.copy_(W)
        rec["models"][name] = record()
        for l, W0 in saved.items():
            get_sublayer(layers[int(l)], sub).weight.data.copy_(W0)
        print(f"  {name} recorded ({time.time() - t0:.0f}s)")
    return rec


def main():
    from src.data_loader import load_prompts
    from src.model_loader import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    exp = cfg.experiment
    t0 = time.time()
    with open(os.path.join(exp.models_dir, "manifest.json")) as f:
        manifest = json.load(f)
    model, tokenizer = load_model(cfg)
    cal, known_extra, secret, _ = split_prompts(load_prompts(cfg, tokenizer), exp)
    rec = record_all(model, cal, known_extra, secret, manifest, exp, exp.models_dir, cfg.model.device, args.tag)
    rec["gpu"] = torch.cuda.get_device_name(0)
    os.makedirs(exp.records_dir, exist_ok=True)
    torch.save(rec, os.path.join(exp.records_dir, f"{args.tag}.pt"))
    print(f"RECORD_DONE {args.tag} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
