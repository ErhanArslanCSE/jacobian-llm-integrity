#!/usr/bin/env python3
"""Experiment 18: the exp14/exp17 adversaries on three further models (scale and family check).

Read PREREGISTRATION.md first. Code identical to run_exp17_jacobian_adversary.py; only the
configs differ (model, layers, calibration size, variants). The
adversary additionally works against the tap test without knowing V, by minimizing the
expected tap statistic on the calibration prompts (src/jacobian.py), either
  - as a penalty: loss = LM loss + jac_penalty * sum_l relative Jacobian energy of DeltaW_l, or
  - as a constraint: columns of DeltaW_l restricted to the col_m output directions of gate_proj
    with the least calibration-Jacobian energy (on top of the exp13/exp14 row subspace).
Each run measures:
  - function: trigger success and false activations on 20 prefixes, perplexity on the
    calibration and held-out prompts;
  - the activation monitor at every decoder layer (calibration and held-out prompts),
    against a Gaussian control: a random edit of the same layers with the per-layer relative
    size the backdoor reached;
  - the tap test (k = 8) at every trained layer, against the Gaussian control: the quantity
    the adversary attacks;
  - the predicted relative Jacobian energy of the final edit and of the control.

One JSON file per run in results/<config output dir>/runs/, written as soon as the run ends,
so the grid can be split across pods (--variants) and resumed (existing runs are skipped).
analyze_exp17.py aggregates the runs and applies the pre-registered selection rule.

Usage:
  python run_exp17_jacobian_adversary.py --config configs/jacobian_bf16.yaml [--variants ...]
"""

import argparse
import itertools
import json
import os
import platform
import time
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.backdoor import PoisonedDataset, collate_fn, perplexity, trigger_check
from src.constraints import SubspaceConstraint
from src.jacobian import jacobian_quadratic_form, low_jacobian_columns, relative_jacobian_energy
from src.measure import layer_table, measure_activations, run_with_layer_outputs, summarize_layers
from src.stats import monitor_rejects, one_sample_ttest
from src.subspaces import get_sublayer, perturb_gaussian_sublayer
from src.tap import generate_random_probes, tap_ratio, tap_response


def sublayer_weights(model, layers, sublayer_path):
    return {l: get_sublayer(model.model.layers[l], sublayer_path).weight.data.clone() for l in layers}


def set_sublayer_weights(model, weights, sublayer_path):
    for l, W in weights.items():
        get_sublayer(model.model.layers[l], sublayer_path).weight.data.copy_(W)


def train_backdoor(model, tokenizer, constraint, clean_weights, cal_prompts, train_texts, cfg, lr, seed, device,
                   jac_terms=None, jac_penalty=0.0):
    """Fine-tune gate_proj of the constrained layers on the poisoned data; projection after
    every optimizer step, refit after every epoch for recompute variants. With jac_penalty > 0
    the relative Jacobian energy of every layer's change (jac_terms[l] = (Q, normalizer)) is
    added to the loss."""
    bd = cfg.backdoor
    torch.manual_seed(seed)
    dataset = PoisonedDataset(tokenizer, bd.trigger, bd.target, train_texts, bd.n_poison, bd.n_clean,
                              max_length=cfg.data.max_length, seed=seed)
    loader = DataLoader(dataset, batch_size=bd.batch_size, shuffle=True, collate_fn=collate_fn,
                        generator=torch.Generator().manual_seed(seed))
    params = []
    for l in constraint.layers:
        w = get_sublayer(model.model.layers[l], constraint.sublayer_path).weight
        w.requires_grad_(True)
        params.append(w)
    optimizer = AdamW(params, lr=lr)
    model.train()

    log = []
    for epoch in range(bd.epochs):
        losses, kept = [], []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            lm_loss = model(**batch).loss
            loss = lm_loss
            if jac_penalty > 0:
                for l, w in zip(constraint.layers, params):
                    Q, norm = jac_terms[l]
                    loss = loss + jac_penalty * relative_jacobian_energy(Q, norm, w.float() - clean_weights[l].float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            fractions = constraint.project(model, clean_weights)
            if fractions:
                kept.append(float(np.mean(list(fractions.values()))))
            losses.append(lm_loss.item())
        refit = constraint.refit_sequential(model, clean_weights, cal_prompts, device)
        log.append({"epoch": epoch, "loss": float(np.mean(losses)),
                    "kept_fraction": float(np.mean(kept)) if kept else 1.0,
                    "refit_kept_fraction": refit})
        print(f"    epoch {epoch}: loss={log[-1]['loss']:.4f} kept={log[-1]['kept_fraction']:.3f}")

    for w in params:
        w.requires_grad_(False)
    model.eval()
    return log


def evaluate(model, prompts, refs, ref_logps, tap_refs, probes, layers, device):
    """Activation statistics at every layer and tap ratios at the trained layers."""
    acts = measure_activations(model, prompts, refs, ref_logps, device)
    taps = {l: np.array([tap_ratio(tap_response(model, p, l, probes, device), r)
                         for p, r in zip(prompts, tap_refs[l])]) for l in layers}
    return acts, taps


def run_one(model, tokenizer, variant, lr, seed, data, cfg, device):
    exp = cfg.experiment
    sub = exp.target_sublayer
    layers = sorted(variant.layers)
    cal, test, train_texts = data["cal"], data["test"], data["train_texts"]
    prompts = cal + test
    n_cal = len(cal)
    all_layers = sorted(exp.monitored_tap_layers)

    clean_weights = sublayer_weights(model, layers, sub)
    constraint = SubspaceConstraint(variant.constraint, layers, sub, m=variant.get("m"),
                                    recompute=variant.get("recompute", False))
    constraint.fit(model, cal, device)
    jac_terms = {l: jacobian_quadratic_form(model, cal, l, device) for l in layers}
    col_leakage = {}
    if variant.get("col_m"):
        for l in layers:
            U, col_leakage[l] = low_jacobian_columns(jac_terms[l][0], variant.col_m)
            constraint.col_bases[l] = U

    t = time.time()
    log = train_backdoor(model, tokenizer, constraint, clean_weights, cal, train_texts, cfg, lr, seed, device,
                         jac_terms=jac_terms, jac_penalty=variant.get("jac_penalty", 0.0))
    train_time = time.time() - t
    jac_energy = {l: float(relative_jacobian_energy(*jac_terms[l], get_sublayer(model.model.layers[l], sub)
                                                    .weight.data.float() - clean_weights[l].float()))
                  for l in layers}

    eps = {l: float((get_sublayer(model.model.layers[l], sub).weight.data.float() - clean_weights[l].float()).norm()
                    / clean_weights[l].float().norm()) for l in layers}
    function = trigger_check(model, tokenizer, cfg.backdoor.trigger, cfg.backdoor.target, device)
    function["ppl_cal"] = perplexity(model, tokenizer, [p["text"] for p in cal], device)
    function["ppl_test"] = perplexity(model, tokenizer, [p["text"] for p in test], device)
    acts, taps = evaluate(model, prompts, data["refs"], data["ref_logps"],
                          data["tap_refs"], data["probes"], all_layers, device)

    # Gaussian control: random edit of the same layers with the same per-layer size.
    set_sublayer_weights(model, clean_weights, sub)
    for l in layers:
        perturb_gaussian_sublayer(model.model.layers[l], eps[l], seed + l, sub)
    g_jac_energy = {l: float(relative_jacobian_energy(*jac_terms[l], get_sublayer(model.model.layers[l], sub)
                                                      .weight.data.float() - clean_weights[l].float()))
                    for l in layers}
    g_acts, g_taps = evaluate(model, prompts, data["refs"], data["ref_logps"],
                              data["tap_refs"], data["probes"], all_layers, device)
    set_sublayer_weights(model, clean_weights, sub)

    first = layers[0]
    result = {"variant": variant.name, "constraint": variant.constraint, "m": variant.get("m"),
              "recompute": variant.get("recompute", False), "layers": layers, "lr": lr, "seed": seed,
              "jac_penalty": variant.get("jac_penalty", 0.0), "col_m": variant.get("col_m"),
              "train_log": log, "train_seconds": train_time, "eps": eps, "function": function,
              "basis_fit_log": constraint.fit_log, "col_leakage": col_leakage,
              "jac_energy": jac_energy, "g_jac_energy": g_jac_energy}
    for split, sl in (("cal", slice(0, n_cal)), ("test", slice(n_cal, None))):
        for s in ("last", "pos_mean"):
            rows = layer_table(acts[s][sl], g_acts[s][sl], exp.alpha, exp.evasion_threshold)
            result[f"{split}_{s}_layers"] = rows
            result[f"{split}_{s}_summary"] = summarize_layers(rows, first)
        result[f"{split}_tap"] = {
            str(l): {"mean": float(taps[l][sl].mean()), "p": one_sample_ttest(taps[l][sl])["p_value"],
                     "rejects": monitor_rejects(taps[l][sl], exp.alpha),
                     "g_mean": float(g_taps[l][sl].mean())}
            for l in all_layers}
        result[f"{split}_kl_last"] = {"mean": float(acts["kl_last"][sl].mean()),
                                      "g_mean": float(g_acts["kl_last"][sl].mean())}
    result["raw"] = {"act_last": acts["last"].tolist(), "g_act_last": g_acts["last"].tolist(),
                     "tap": {str(l): taps[l].tolist() for l in all_layers},
                     "g_tap": {str(l): g_taps[l].tolist() for l in all_layers}}
    return result


def main():
    from src.data_loader import load_prompts
    from src.model_loader import load_model

    parser = argparse.ArgumentParser(description="Experiment 17: Jacobian-aware adversary")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variants", nargs="*", help="subset of variant names (to split across pods)")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    exp = cfg.experiment
    device = cfg.model.device
    runs_dir = os.path.join(cfg.output.dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    variants = [v for v in cfg.variants if not args.variants or v.name in args.variants]
    if args.variants and len(variants) != len(args.variants):
        raise ValueError(f"Unknown variant in {args.variants}")

    print("=" * 80 + "\nEXPERIMENT 17: JACOBIAN-AWARE ADVERSARY\n" + "=" * 80)
    model, tokenizer = load_model(cfg)
    prompts = load_prompts(cfg, tokenizer)
    n_cal, n_test = exp.n_cal, exp.n_test
    cal, test = prompts[:n_cal], prompts[n_cal:n_cal + n_test]
    train_texts = [p["text"] for p in prompts[n_cal + n_test:]]
    print(f"  {len(cal)} calibration, {len(test)} held-out, {len(train_texts)} training texts")

    d = model.model.layers[0].self_attn.q_proj.weight.shape[1]
    probes = generate_random_probes(d, exp.tap_k, exp.probe_seed)
    refs, ref_logps = zip(*[run_with_layer_outputs(model, p, device) for p in cal + test])
    tap_refs = {l: [tap_response(model, p, l, probes, device) for p in cal + test]
                for l in exp.monitored_tap_layers}
    data = {"cal": cal, "test": test, "train_texts": train_texts, "refs": list(refs),
            "ref_logps": list(ref_logps), "tap_refs": tap_refs, "probes": probes}

    clean_function = trigger_check(model, tokenizer, cfg.backdoor.trigger, cfg.backdoor.target, device)
    clean_function["ppl_cal"] = perplexity(model, tokenizer, [p["text"] for p in cal], device)
    clean_function["ppl_test"] = perplexity(model, tokenizer, [p["text"] for p in test], device)
    with open(os.path.join(cfg.output.dir, "clean_model.json"), "w") as f:
        json.dump(clean_function, f, indent=1)
    print(f"  clean model: {clean_function['trigger_success']:.0%} trigger, "
          f"ppl cal {clean_function['ppl_cal']:.2f}")

    import transformers
    meta = {"config": OmegaConf.to_container(cfg), "torch": torch.__version__,
            "transformers": transformers.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    for variant, lr, seed in itertools.product(variants, exp.lrs, exp.seeds):
        path = os.path.join(runs_dir, f"{variant.name}_lr{lr}_seed{seed}.json")
        if os.path.exists(path):
            print(f"  skip {path} (exists)")
            continue
        print(f"\n--- {variant.name} lr={lr} seed={seed} ---")
        t = time.time()
        result = run_one(model, tokenizer, variant, lr, seed, data, cfg, device)
        result["_meta"] = {**meta, "finished": datetime.now().strftime("%Y%m%d_%H%M%S"),
                           "run_seconds": time.time() - t}
        with open(path, "w") as f:
            json.dump(result, f, indent=1)
        s = result["cal_last_summary"]
        tap_kept = np.mean([result["cal_tap"][str(l)]["mean"] / result["cal_tap"][str(l)]["g_mean"]
                            for l in result["layers"]])
        print(f"  tap kept={tap_kept:.1%} trigger={result['function']['trigger_success']:.0%} "
              f"false={result['function']['false_activation']:.0%} "
              f"ppl={result['function']['ppl_test']:.2f} | min supp={s['min_suppression_downstream']:.1%} "
              f"audit evaded={s['every_layer_audit_evaded']} | {time.time() - t:.0f}s")

    print("\nEXP17_RUNS_DONE")


if __name__ == "__main__":
    main()
