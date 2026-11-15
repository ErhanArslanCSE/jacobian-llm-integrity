#!/usr/bin/env python3
"""Experiment 15, step 1 (reference machine): build the edits once and save the edited weights.

The adversary builds each edit once; every audit machine then serves bitwise the same
weights. Edits (seed 42, as exp12/exp13): Gaussian control, the paper's L3 edit (ns_last), and
the exp13 adaptive edits allpos_m256 / allpos_m1024, for gate_proj of layers {2, 6, 10, 12, 14}
at eps {0.01, 0.05, 0.1}. Saved: edits/<name>.pt (edited weight matrix in the model dtype) and
edits/manifest.json (SHA-256 of every edited matrix and of the clean matrices).

Usage: python make_edits_exp15.py --config configs/noise.yaml
"""

import argparse
import hashlib
import json
import os

import torch
from omegaconf import OmegaConf

from src.perturbations import (
    allpos_null_basis,
    collect_sublayer_inputs,
    get_sublayer,
    perturb_gaussian_sublayer,
    perturb_in_subspace,
    perturb_null_space,
    restore_weights,
    save_weights,
)


def tensor_sha256(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    from src.data_loader import load_prompts
    from src.model_loader import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = OmegaConf.load(parser.parse_args().config)
    exp = cfg.experiment
    device = cfg.model.device

    model, tokenizer = load_model(cfg)
    cal = load_prompts(cfg, tokenizer)[:exp.n_cal]
    manifest = make_edits(model, cal, exp, device)
    manifest["gpu"] = torch.cuda.get_device_name(0)
    with open(os.path.join(exp.edits_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"EDITS_DONE ({len(manifest['edits'])} edits)")


def make_edits(model, cal, exp, device):
    """Build, save and hash every edit; returns the manifest."""
    os.makedirs(exp.edits_dir, exist_ok=True)
    layers = model.model.layers
    sub = exp.target_sublayer
    manifest = {"clean": {}, "edits": {}}

    for l in exp.target_layers:
        W_clean = get_sublayer(layers[l], sub).weight.data
        manifest["clean"][str(l)] = tensor_sha256(W_clean)
        H_last = collect_sublayer_inputs(model, cal, l, sub, device).to(device)
        H_all = collect_sublayer_inputs(model, cal, l, sub, device, positions="all").to(device)
        bases = {m: allpos_null_basis(H_last, H_all, m)[0].cpu() for m in exp.allpos_m}
        for eps in exp.epsilons:
            for kind in exp.edit_kinds:
                saved = save_weights(layers[l])
                if kind == "gaussian":
                    perturb_gaussian_sublayer(layers[l], eps, exp.seed, sub)
                elif kind == "ns_last":
                    perturb_null_space(layers[l], eps, exp.seed, model=model, prompts=cal,
                                       layer_idx=l, sublayer_path=sub, device=device)
                else:
                    perturb_in_subspace(layers[l], eps, exp.seed, bases[int(kind.split("_m")[1])], sub)
                name = f"{kind}_L{l}_eps{eps}"
                W = get_sublayer(layers[l], sub).weight.data.clone().cpu()
                torch.save(W, os.path.join(exp.edits_dir, f"{name}.pt"))
                manifest["edits"][name] = {"layer": l, "epsilon": eps, "kind": kind, "sha256": tensor_sha256(W)}
                restore_weights(layers[l], saved)
                print(f"  saved {name}")
    return manifest


if __name__ == "__main__":
    main()
