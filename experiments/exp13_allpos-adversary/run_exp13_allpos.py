#!/usr/bin/env python3
"""Experiment 13: an adaptive adversary against an every-layer activation audit.

Exp12 showed that the L3 null-space edit, which is orthogonal only to the last-token
calibration inputs, changes the earlier positions of the calibration prompts and, through
attention, the last-token outputs of all later layers: an every-layer activation audit
(assumption 5 of the paper) sees it. TIFS feedback item 2 asks for an adversary that adapts
to this. The adversary here knows the calibration prompts and the weights, but not the probe
matrix V, and edits gate_proj of one layer with rows in span(B), where B (m directions):
  - is exactly orthogonal to every last-token calibration input (Theorem 1 holds), and
  - within that null space, minimizes the energy of the calibration inputs at all positions
    (src/perturbations.py: allpos_null_basis).
Smaller m hides the edit better but leaves the adversary fewer directions.

For each (layer, eps) the run compares the Gaussian control, the L3 edit of the paper
(ns_last) and the adaptive edit for every m, on
  - the activation monitor at every decoder layer (calibration and held-out prompts), and
  - the tap test (k = 8) at the edited layer, which by Theorem 2 sees DeltaW in any direction.

Phases:
  1  clean references: activations of every layer, tap responses at the target layers
  2  noise floor
  3  calibration inputs (last token and all positions) and adversary subspaces per layer and m
  4  edits and measurements
  5  analysis, summary, reproduction check against exp12

Usage:
  python run_exp13_allpos.py --config configs/allpos_bf16.yaml
  python run_exp13_allpos.py --config configs/allpos_fp32.yaml
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

from src.measure import STATS, layer_table, measure_activations, run_with_layer_outputs, summarize_layers
from src.perturbations import (
    allpos_null_basis,
    collect_sublayer_inputs,
    get_sublayer,
    perturb_gaussian_sublayer,
    perturb_in_subspace,
    perturb_null_space,
    restore_weights,
    save_weights,
    weights_equal,
)
from src.stats import monitor_rejects, one_sample_ttest
from src.tap import generate_random_probes, tap_ratio, tap_response


def attack_names(m_values):
    return ["gaussian", "ns_last"] + [f"allpos_m{m}" for m in m_values]


def edit_diagnostics(W_before, W_after, delta_intended, H_last, H_all):
    """Relative change of the sublayer output that the edit causes on the calibration inputs,
    at the last token and at all positions, for the stored (effective) and intended edit."""
    W0 = W_before.float().cpu()
    delta_eff = W_after.float().cpu() - W0
    diag = {"eps_effective": float(delta_eff.norm() / W0.norm())}
    for name, H in (("last", H_last), ("all", H_all)):
        H = H.float().cpu()
        base = (W0 @ H.T).norm()
        diag[f"residual_{name}_effective"] = float((delta_eff @ H.T).norm() / base)
        if delta_intended is not None:
            diag[f"residual_{name}_intended"] = float((delta_intended @ H.T).norm() / base)
    return diag


def tap_values(model, prompts, layer_idx, probes, tap_refs, device):
    return np.array([tap_ratio(tap_response(model, p, layer_idx, probes, device), ref)
                     for p, ref in zip(prompts, tap_refs)])


def run_pipeline(model, cal_prompts, test_prompts, cfg, device):
    exp = cfg.experiment
    target_layers = list(exp.target_layers)
    epsilons = list(exp.epsilons)
    m_values = list(exp.m_values)
    sublayer_path = exp.target_sublayer
    alpha, threshold = exp.alpha, exp.evasion_threshold
    all_prompts = cal_prompts + test_prompts
    n_cal = len(cal_prompts)
    layers = model.model.layers
    d = layers[0].self_attn.q_proj.weight.shape[1]
    seeds = list(exp.seeds)
    probes = generate_random_probes(d, exp.tap_k, exp.probe_seed)
    attacks = attack_names(m_values)
    times = {}

    # ── Phase 1: references ──
    print("=" * 60 + "\nPHASE 1: clean references\n" + "=" * 60)
    t = time.time()
    refs, ref_logps = [], []
    for p in all_prompts:
        out, logp = run_with_layer_outputs(model, p, device)
        refs.append(out)
        ref_logps.append(logp)
    tap_refs = {l: [tap_response(model, p, l, probes, device) for p in all_prompts] for l in target_layers}
    times["phase1_references"] = time.time() - t

    # ── Phase 2: noise floor ──
    print("=" * 60 + "\nPHASE 2: noise floor\n" + "=" * 60)
    clean = measure_activations(model, all_prompts, refs, ref_logps, device)
    noise_floor = {s: float(np.abs(clean[s]).max()) for s in (*STATS, "kl_last")}
    noise_floor["tap"] = max(float(tap_values(model, all_prompts, l, probes, tap_refs[l], device).max())
                             for l in target_layers)
    print(f"  {noise_floor}")

    # ── Phase 3: adversary subspaces ──
    print("=" * 60 + "\nPHASE 3: calibration inputs and adversary subspaces\n" + "=" * 60)
    t = time.time()
    H_last, H_all, bases, leakage = {}, {}, {}, {}
    for l in target_layers:
        H_last[l] = collect_sublayer_inputs(model, cal_prompts, l, sublayer_path, device)
        H_all[l] = collect_sublayer_inputs(model, cal_prompts, l, sublayer_path, device, positions="all")
        for m in m_values:
            B, leak = allpos_null_basis(H_last[l].to(device), H_all[l].to(device), m)
            bases[(l, m)] = B.cpu()
            leakage[(l, m)] = leak
        print(f"  L{l}: {H_all[l].shape[0]} all-position inputs; leakage ratio by m: "
              + ", ".join(f"{m}:{leakage[(l, m)]:.2e}" for m in m_values))
    times["phase3_subspaces"] = time.time() - t

    # ── Phase 4: edits ──
    print("=" * 60 + "\nPHASE 4: edits\n" + "=" * 60)
    t = time.time()
    acts, taps, diags = {}, {}, {}
    for l in target_layers:
        sublayer = get_sublayer(layers[l], sublayer_path)
        for eps, seed in itertools.product(epsilons, seeds):
            for attack in attacks:
                saved = save_weights(layers[l])
                W_before = sublayer.weight.data.clone()
                if attack == "gaussian":
                    perturb_gaussian_sublayer(layers[l], eps, seed, sublayer_path)
                    delta = None
                elif attack == "ns_last":
                    delta = perturb_null_space(layers[l], eps, seed, model=model, prompts=cal_prompts,
                                               layer_idx=l, sublayer_path=sublayer_path, device=device)
                else:
                    m = int(attack.split("_m")[1])
                    delta = perturb_in_subspace(layers[l], eps, seed, bases[(l, m)], sublayer_path)
                key = (l, eps, attack, seed)
                diags[key] = edit_diagnostics(W_before, sublayer.weight.data, delta, H_last[l], H_all[l])
                acts[key] = measure_activations(model, all_prompts, refs, ref_logps, device)
                taps[key] = tap_values(model, all_prompts, l, probes, tap_refs[l], device)
                restore_weights(layers[l], saved)
                if not weights_equal(layers[l], saved):
                    raise RuntimeError(f"Weight restoration failed at layer {l}")
                print(f"  L{l} eps={eps} seed={seed} {attack:<13} residual_all={diags[key]['residual_all_effective']:.2e} "
                      f"act_next={acts[key]['last'][:n_cal, min(l + 1, len(layers) - 1)].mean():.2e} "
                      f"tap={taps[key][:n_cal].mean():.2e}")
    times["phase4_edits"] = time.time() - t

    # ── Phase 5: analysis ──
    print("=" * 60 + "\nPHASE 5: analysis\n" + "=" * 60)
    analysis = []
    for l in target_layers:
        for eps, seed in itertools.product(epsilons, seeds):
            g = (l, eps, "gaussian", seed)
            for attack in attacks[1:]:
                key = (l, eps, attack, seed)
                entry = {"layer": l, "epsilon": eps, "seed": seed, "attack": attack, "edit": diags[key],
                         "edit_gaussian": diags[g]}
                if attack.startswith("allpos"):
                    entry["leakage_ratio"] = leakage[(l, int(attack.split("_m")[1]))]
                for split, sl in (("cal", slice(0, n_cal)), ("test", slice(n_cal, None))):
                    for s in ("last", "pos_mean"):
                        rows = layer_table(acts[key][s][sl], acts[g][s][sl], alpha, threshold)
                        entry[f"{split}_{s}_layers"] = rows
                        entry[f"{split}_{s}_summary"] = summarize_layers(rows, l)
                    tap_ns, tap_g = taps[key][sl], taps[g][sl]
                    entry[f"{split}_tap"] = {
                        "ns_mean": float(tap_ns.mean()),
                        "ns_p": one_sample_ttest(tap_ns)["p_value"],
                        "rejects": monitor_rejects(tap_ns, alpha),
                        "g_mean": float(tap_g.mean()),
                        "retained": float(tap_ns.mean() / tap_g.mean()),
                    }
                    entry[f"{split}_kl_last"] = {"ns_mean": float(acts[key]["kl_last"][sl].mean()),
                                                 "g_mean": float(acts[g]["kl_last"][sl].mean())}
                analysis.append(entry)

    aggregate = print_summary(analysis, attacks[1:])
    repro = reproduction_check(cfg.reference.exp12_summary, acts, n_cal, target_layers, epsilons)
    if repro is not None:
        print(f"\n  Reproduction of exp12 (gaussian and ns_last, every layer): "
              f"max relative difference {repro['max_rel_diff']:.2e}")

    raw = {f"{a}_L{l}_eps{e}_seed{sd}": {**{s: acts[(l, e, a, sd)][s].tolist() for s in (*STATS, "kl_last")},
                                         "tap": taps[(l, e, a, sd)].tolist()}
           for (l, e, a, sd) in acts}
    return {
        "noise_floor": noise_floor,
        "leakage_ratio": {f"L{l}_m{m}": v for (l, m), v in leakage.items()},
        "aggregate": aggregate,
        "analysis": analysis,
        "reproduction_exp12": repro,
        "raw": raw,
        "phase_times": times,
    }


def print_summary(analysis, attacks):
    """Per attack, over the (layer, eps, seed) conditions: how often the every-layer activation audit
    is evaded (calibration prompts, last token, paper's suppression rule), the median of the
    worst-case suppression, and how often the tap test rejects."""
    header = (f"  {'attack':<13} {'n':>3} | {'audit evaded':>12} {'med min supp':>12} "
              f"{'med supp@next':>13} | {'held-out evaded':>15} | {'tap rejects':>11} "
              f"{'med tap kept':>12} | {'med KL ns/g cal/held':>20}")
    print("\n  Every-layer activation audit vs tap test (calibration prompts unless noted)")
    print(header)
    print("  " + "-" * (len(header) - 2))
    aggregate = {}
    for a in attacks:
        es = [e for e in analysis if e["attack"] == a]
        n = len(es)
        evaded = sum(e["cal_last_summary"]["every_layer_audit_evaded"] for e in es)
        med_min = float(np.median([e["cal_last_summary"]["min_suppression_downstream"] for e in es]))
        nxt = [e["cal_last_summary"]["next_layer_suppression"] for e in es
               if e["cal_last_summary"]["next_layer_suppression"] is not None]
        med_next = float(np.median(nxt))
        test_evaded = sum(e["test_last_summary"]["every_layer_audit_evaded"] for e in es)
        tap_rej = sum(e["cal_tap"]["rejects"] for e in es)
        med_kept = float(np.median([e["cal_tap"]["retained"] for e in es]))
        med_kl = float(np.median([e["cal_kl_last"]["ns_mean"] / e["cal_kl_last"]["g_mean"] for e in es]))
        med_kl_test = float(np.median([e["test_kl_last"]["ns_mean"] / e["test_kl_last"]["g_mean"] for e in es]))
        aggregate[a] = {"n": n, "every_layer_audit_evaded": evaded, "median_min_suppression": med_min,
                        "median_next_layer_suppression": med_next, "heldout_audit_evaded": test_evaded,
                        "tap_rejects": tap_rej, "median_tap_retained": med_kept,
                        "median_kl_ratio": med_kl, "median_kl_ratio_heldout": med_kl_test}
        print(f"  {a:<13} {n:>3} | {evaded:>5}/{n:<6} {med_min:>12.1%} {med_next:>13.1%} | "
              f"{test_evaded:>8}/{n:<6} | {tap_rej:>5}/{n:<5} {med_kept:>12.1%} | {med_kl:>9.3f}/{med_kl_test:<10.3f}")
    return aggregate


def reproduction_check(path, acts, n_cal, target_layers, epsilons):
    """With seed 42 the Gaussian and ns_last edits are those of exp12; compare their per-layer
    calibration means."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        ref = json.load(f)
    by_key = {(e["layer"], e["epsilon"]): e["cal_last_layers"] for e in ref["analysis"]}
    diffs = []
    for l in target_layers:
        for eps in epsilons:
            for attack, field in (("gaussian", "g_mean"), ("ns_last", "ns_mean")):
                ours = acts[(l, eps, attack, 42)]["last"][:n_cal].mean(axis=0)
                for row in by_key[(l, eps)]:
                    theirs = row[field]
                    if theirs > 0:
                        diffs.append(abs(ours[row["layer"]] - theirs) / theirs)
    return {"max_rel_diff": float(max(diffs)), "median_rel_diff": float(np.median(diffs)), "n": len(diffs)}


def main():
    from src.data_loader import load_prompts
    from src.model_loader import load_model

    parser = argparse.ArgumentParser(description="Experiment 13: adaptive all-position adversary")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    t_start = time.time()
    torch.manual_seed(cfg.experiment.probe_seed)
    os.makedirs(cfg.output.dir, exist_ok=True)
    print("=" * 80 + "\nEXPERIMENT 13: ADAPTIVE ALL-POSITION ADVERSARY\n" + "=" * 80)
    print(OmegaConf.to_yaml(cfg))

    model, tokenizer = load_model(cfg)
    prompts = load_prompts(cfg, tokenizer)
    n_cal, n_test = cfg.experiment.n_cal, cfg.experiment.n_test
    if len(prompts) < n_cal + n_test:
        raise RuntimeError(f"Need {n_cal + n_test} prompts, got {len(prompts)}")

    result = run_pipeline(model, prompts[:n_cal], prompts[n_cal:n_cal + n_test], cfg, cfg.model.device)

    import transformers
    result["_meta"] = {
        "experiment": "exp13_allpos_adversary",
        "finished": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": OmegaConf.to_container(cfg),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "runtime_seconds": time.time() - t_start,
    }
    with open(os.path.join(cfg.output.dir, "results.json"), "w") as f:
        json.dump(result, f, indent=1)
    with open(os.path.join(cfg.output.dir, "results_summary.json"), "w") as f:
        json.dump({k: v for k, v in result.items() if k != "raw"}, f, indent=1)
    print(f"\n  Saved results to {cfg.output.dir}")
    print(f"  Runtime: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
