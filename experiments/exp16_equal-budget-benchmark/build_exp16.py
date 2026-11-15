#!/usr/bin/env python3
"""Experiment 16, step 1 (reference machine): build every model once and save its edited weights.

Read PREREGISTRATION.md first. Models (gate_proj; seed 42):
  synthetic single-layer edits at layers {6, 12}, eps {0.05, 0.1}:
    gaussian, L3 (last-token null space), ap_cal_m256, ap_known_m256, ap_dist_m256
  trained backdoors (layers 2, 6, 10, 12, 14; exp14/exp17 recipe), see configs/benchmark.yaml
For every model the edited matrices are saved in the model dtype with their SHA-256, so every
audit machine serves bitwise the same weights; backdoor function (trigger success, false
activations, perplexity) is measured here.

Usage: python build_exp16.py --config configs/benchmark.yaml
"""

import argparse
import hashlib
import json
import os
import time

import torch
from omegaconf import OmegaConf

from src.backdoor import perplexity, trigger_check
from src.bases import all_position_gram, low_energy_basis
from src.constraints import SubspaceConstraint
from src.jacobian import jacobian_quadratic_form, low_jacobian_columns
from src.subspaces import collect_sublayer_inputs, get_sublayer, last_token_null_basis, perturb_gaussian_sublayer
from src.training import set_sublayer_weights, sublayer_weights, train_backdoor


def tensor_sha256(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def split_prompts(prompts, exp):
    n_cal, n_known, n_secret = exp.n_cal, exp.n_known_extra, exp.n_secret
    cal = prompts[:n_cal]
    known_extra = prompts[n_cal:n_cal + n_known]
    secret = prompts[n_cal + n_known:n_cal + n_known + n_secret]
    train_texts = [p["text"] for p in prompts[n_cal + n_known + n_secret:]]
    return cal, known_extra, secret, train_texts


def last_token_input_norm(model, prompts, layer_idx, device):
    """Median norm of the decoder-layer input at the last token: the scale of the secret
    synthetic hidden states (stored, so that every machine feeds the same inputs)."""
    norms = []

    def pre_hook(module, args, kwargs):
        norms.append(float(args[0][0, -1].float().norm()))

    handle = model.model.layers[layer_idx].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        with torch.no_grad():
            for p in prompts:
                model(input_ids=p["input_ids"].to(device), attention_mask=p["attention_mask"].to(device))
    finally:
        handle.remove()
    return float(torch.tensor(norms).median())


def subspace_edit(W, B, eps, seed):
    """DeltaW = A B^T (A standard Gaussian, CPU generator) scaled to ||DeltaW|| / ||W|| = eps."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    A = torch.randn(W.shape[0], B.shape[1], generator=rng, dtype=torch.float32)
    delta = A @ B.cpu().float().T
    return delta * (eps * W.float().cpu().norm() / delta.norm())


def main():
    from src.data_loader import load_corpus, load_prompts
    from src.model_loader import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = OmegaConf.load(parser.parse_args().config)
    exp = cfg.experiment
    device = cfg.model.device
    out = exp.models_dir
    t0 = time.time()

    model, tokenizer = load_model(cfg)
    cal, known_extra, secret, train_texts = split_prompts(load_prompts(cfg, tokenizer), exp)
    corpus = load_corpus(cfg, tokenizer)
    print(f"  {len(cal)} cal, {len(known_extra)} known extra, {len(secret)} secret, "
          f"{len(train_texts)} training texts, {len(corpus)} corpus passages")
    if len(secret) < exp.n_secret or len(train_texts) < cfg.backdoor.n_clean:
        raise RuntimeError("Not enough prompts; raise data.num_prompts")
    manifest = build_all(model, tokenizer, cal, known_extra, secret, train_texts, corpus, cfg, device)
    manifest["gpu"] = torch.cuda.get_device_name(0)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"BUILD_DONE ({len(manifest['models'])} models, {(time.time() - t0) / 60:.1f} min)")


def build_all(model, tokenizer, cal, known_extra, secret, train_texts, corpus, cfg, device):
    """Subspaces, synthetic edits and trained backdoors; saves the weights, returns the manifest."""
    exp = cfg.experiment
    sub = exp.target_sublayer
    out = exp.models_dir
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    layers = model.model.layers
    all_layers = sorted(exp.backdoor_layers)

    # adversary subspaces per layer and source
    N, bases, leakage, jac_terms = {}, {}, {}, {}
    sources = {"cal": cal, "known": cal + known_extra, "dist": corpus}
    for l in all_layers:
        N[l] = last_token_null_basis(collect_sublayer_inputs(model, cal, l, sub, device).to(device))
        for src, ps in sources.items():
            gram = all_position_gram(model, ps, l, sub, device)
            for m in exp.m_values:
                bases[(src, m, l)], leakage[f"{src}_m{m}_L{l}"] = low_energy_basis(N[l], gram, m)
        for src in ("cal", "dist"):
            jac_terms[(src, l)] = jacobian_quadratic_form(model, sources[src], l, device)
        print(f"  L{l}: subspaces done ({time.time() - t0:.0f}s)")

    manifest = {"clean": {str(l): tensor_sha256(get_sublayer(layers[l], sub).weight.data) for l in all_layers},
                "models": {}, "leakage": leakage,
                "synthetic_scale": {str(l): last_token_input_norm(model, cal, l, device) for l in all_layers}}
    clean_W = sublayer_weights(model, all_layers, sub)

    def save_model(name, info):
        files = {}
        for l in info["layers"]:
            W = get_sublayer(layers[l], sub).weight.data.clone().cpu()
            path = f"{name}_L{l}.pt"
            torch.save(W, os.path.join(out, path))
            files[str(l)] = {"file": path, "sha256": tensor_sha256(W)}
        manifest["models"][name] = {**info, "files": files}

    # synthetic single-layer edits
    for l in exp.synthetic_layers:
        for eps in exp.synthetic_epsilons:
            for kind in exp.synthetic_kinds:
                if kind == "gaussian":
                    perturb_gaussian_sublayer(layers[l], eps, exp.seed, sub)
                else:
                    B = N[l] if kind == "L3" else bases[(kind.split("_")[1], int(kind.split("_m")[1]), l)]
                    W = get_sublayer(layers[l], sub).weight
                    W.data.add_(subspace_edit(W.data, B, eps, exp.seed).to(W.device, W.dtype))
                save_model(f"{kind}_L{l}_eps{eps}", {"type": "synthetic", "kind": kind, "layers": [l],
                                                     "epsilon": eps})
                set_sublayer_weights(model, clean_W, sub)

    # trained backdoors
    for bd in cfg.backdoors:
        print(f"\n--- backdoor {bd.name} ---")
        constraint = SubspaceConstraint("last" if bd.source == "last" else "allpos", all_layers, sub)
        for l in all_layers:
            constraint.bases[l] = N[l] if bd.source == "last" else bases[(bd.source, bd.m, l)].float()
            if bd.get("col_m"):
                constraint.col_bases[l] = low_jacobian_columns(jac_terms[(bd.jac_source, l)][0], bd.col_m)[0]
        train_backdoor(model, tokenizer, constraint, clean_W, cal, train_texts, cfg, bd.lr, exp.seed, device)
        eps = {str(l): float((get_sublayer(layers[l], sub).weight.data.float() - clean_W[l].float()).norm()
                             / clean_W[l].float().norm()) for l in all_layers}
        function = trigger_check(model, tokenizer, cfg.backdoor.trigger, cfg.backdoor.target, device)
        function["ppl_secret"] = perplexity(model, tokenizer, [p["text"] for p in secret], device)
        print(f"  trigger {function['trigger_success']:.0%}, false {function['false_activation']:.0%}, "
              f"ppl {function['ppl_secret']:.2f}, eps {eps}")
        save_model(bd.name, {"type": "backdoor", "spec": OmegaConf.to_container(bd), "layers": all_layers,
                             "eps": eps, "function": function})
        set_sublayer_weights(model, clean_W, sub)

    clean_fn = trigger_check(model, tokenizer, cfg.backdoor.trigger, cfg.backdoor.target, device)
    clean_fn["ppl_secret"] = perplexity(model, tokenizer, [p["text"] for p in secret], device)
    manifest["clean_function"] = clean_fn
    return manifest


if __name__ == "__main__":
    main()
