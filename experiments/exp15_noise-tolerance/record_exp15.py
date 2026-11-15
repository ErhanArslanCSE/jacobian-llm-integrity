#!/usr/bin/env python3
"""Experiment 15, step 2 (every audit machine): record what both monitors read.

For the clean model and for every edit of edits/manifest.json (weights loaded bitwise from
edits/, SHA-256 checked), records on the calibration and held-out prompts:
  - last-token output of every decoder layer, prompts processed one at a time ("single"),
  - the same with the prompts processed in right-padded batches ("batched"; serving-like),
  - tap responses J(x) V (k = 8, probe seed 42): at every target layer for the clean model,
    at the edited layer for an edit.
The clean model is recorded twice (single) to check same-machine determinism. All tensors are
stored in the model dtype (they are produced in it) in records/<tag>.pt.

Usage: python record_exp15.py --config configs/noise.yaml --tag H200
"""

import argparse
import json
import os
import time

import torch
from omegaconf import OmegaConf

from make_edits_exp15 import tensor_sha256
from src.perturbations import get_sublayer
from src.tap import generate_random_probes, tap_response


def last_token_outputs_single(model, prompts, device):
    """(n_prompts, L, d): output of every decoder layer at the last token, one prompt at a time."""
    rows = []
    for p in prompts:
        rows.append(_last_token_outputs(model, p["input_ids"].to(device), p["attention_mask"].to(device)))
    return torch.cat(rows, dim=0)


def last_token_outputs_batched(model, prompts, batch_size, pad_id, device):
    """Same, with right-padded batches of batch_size prompts."""
    rows = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        T = max(p["input_ids"].shape[1] for p in chunk)
        ids = torch.full((len(chunk), T), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), T), dtype=torch.long)
        for j, p in enumerate(chunk):
            n = p["input_ids"].shape[1]
            ids[j, :n] = p["input_ids"][0]
            mask[j, :n] = 1
        rows.append(_last_token_outputs(model, ids.to(device), mask.to(device)))
    return torch.cat(rows, dim=0)


def _last_token_outputs(model, input_ids, attention_mask):
    layers = model.model.layers
    outs = [None] * len(layers)
    last = attention_mask.sum(dim=1).long() - 1
    rows = torch.arange(input_ids.shape[0], device=input_ids.device)

    def make_hook(idx):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            out = out if out.dim() == 3 else out.unsqueeze(0)
            outs[idx] = out[rows, last].detach()
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return torch.stack(outs, dim=1).cpu()


def taps(model, prompts, layer_idx, probes, device):
    """(n_prompts, d, k) tap responses, in the model dtype."""
    dtype = next(model.parameters()).dtype
    return torch.stack([tap_response(model, p, layer_idx, probes, device).to(dtype) for p in prompts])


def record_all(model, prompts, pad_id, exp, device, tag):
    """Clean and edited recordings of one machine (see the module docstring)."""
    t0 = time.time()
    with open(os.path.join(exp.edits_dir, "manifest.json")) as f:
        manifest = json.load(f)
    layers = model.model.layers
    sub = exp.target_sublayer
    for l, sha in manifest["clean"].items():
        if tensor_sha256(get_sublayer(layers[int(l)], sub).weight.data) != sha:
            raise RuntimeError(f"Clean weights of layer {l} differ from the reference machine")

    d = layers[0].self_attn.q_proj.weight.shape[1]
    probes = generate_random_probes(d, exp.tap_k, exp.probe_seed)

    def record_activations():
        return {"single": last_token_outputs_single(model, prompts, device),
                "batched": last_token_outputs_batched(model, prompts, exp.batch_size, pad_id, device)}

    record = {"tag": tag, "torch": torch.__version__,
              "clean": record_activations(),
              "clean_repeat_single": last_token_outputs_single(model, prompts, device),
              "clean_taps": {l: taps(model, prompts, l, probes, device) for l in exp.target_layers},
              "edits": {}}
    print(f"  clean recorded ({time.time() - t0:.0f}s)")

    for name, info in manifest["edits"].items():
        l = info["layer"]
        weight = get_sublayer(layers[l], sub).weight
        clean_W = weight.data.clone()
        W = torch.load(os.path.join(exp.edits_dir, f"{name}.pt")).to(weight.device)
        if tensor_sha256(W) != info["sha256"]:
            raise RuntimeError(f"Edit {name} does not match the manifest")
        weight.data.copy_(W)
        record["edits"][name] = {**record_activations(), "taps": taps(model, prompts, l, probes, device)}
        weight.data.copy_(clean_W)
        print(f"  {name} recorded ({time.time() - t0:.0f}s)")
    return record


def main():
    from src.data_loader import load_prompts
    from src.model_loader import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", required=True, help="name of this machine, e.g. H200")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    exp = cfg.experiment
    device = cfg.model.device
    t0 = time.time()

    model, tokenizer = load_model(cfg)
    prompts = load_prompts(cfg, tokenizer)[:exp.n_cal + exp.n_test]
    record = record_all(model, prompts, tokenizer.pad_token_id, exp, device, args.tag)
    record["gpu"] = torch.cuda.get_device_name(0)
    os.makedirs(exp.records_dir, exist_ok=True)
    torch.save(record, os.path.join(exp.records_dir, f"{args.tag}.pt"))
    print(f"RECORD_DONE {args.tag} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
