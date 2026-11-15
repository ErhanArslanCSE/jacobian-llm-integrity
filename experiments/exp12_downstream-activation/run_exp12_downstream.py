#!/usr/bin/env python3
"""Experiment 12: activation signal of single-layer null-space edits at downstream layers.

Reviewer question (TIFS feedback, items 1 and 2): the paper scores each L3 null-space edit
only at the decoder layer that contains the edited sublayer, while assumption 5 of
Section III-D monitors every decoder layer. Section IV-D concedes that later layers can
change through attention to earlier positions. This experiment measures, for the 15 L3 edits
of Llama-3.2-1B gate_proj and their Gaussian controls, the activation statistic at every
decoder layer, so that the paper can state whether an every-layer activation audit is evaded.

It also records what the follow-up experiments need:
  - the rounding of W + DeltaW in the model dtype (residual of Section VI-A-5), by comparing
    the intended edit with the edit actually stored; run with model.dtype float32 to separate
    exact-arithmetic behaviour from bfloat16 rounding (reviewer item 1);
  - the spectrum of the all-position inputs of gate_proj on the calibration prompts, i.e.
    whether an all-position null space exists for an adaptive adversary (reviewer item 2).

Statistics per prompt, decoder layer l and edit (all relative to the approved model):
  last      ||a_T - r_T|| / ||r_T||            at the last token T (s_act of Eq. (2))
  pos_mean  mean_t ||a_t - r_t|| / ||r_t||     over all positions
  pos_max   max_t  ||a_t - r_t|| / ||r_t||     over all positions
  kl_last   KL(p_ref || p) of the next-token distribution at the last token (model output)

Phases:
  0  setup: model, prompts, reference of exp9 for the reproduction check
  1  clean reference outputs of every decoder layer at every position
  2  noise floor: repeat the clean pass and compare
  3  all-position input spectrum of the edited sublayer at each target layer
  4  edits: Gaussian control and L3 null-space edit per (layer, eps)
  5  analysis: decisions and suppression per layer, summary, reproduction check
  6  save

Usage:
  python run_exp12_downstream.py --config configs/downstream_bf16.yaml
  python run_exp12_downstream.py --config configs/downstream_fp32.yaml
"""

import argparse
import json
import os
import platform
import time
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf

from src.perturbations import (
    collect_sublayer_inputs,
    get_sublayer,
    perturb_gaussian_sublayer,
    perturb_null_space,
    restore_weights,
    save_weights,
    weights_equal,
)
from src.stats import monitor_rejects, one_sample_ttest

STATS = ("last", "pos_mean", "pos_max")
ATTACKS = ("gaussian", "null_space")


# ════════════════════════════════════════════════════════════════
# Measurement
# ════════════════════════════════════════════════════════════════

def run_with_layer_outputs(model, prompt, device):
    """One forward pass; returns the output of every decoder layer, shape (L, T, d),
    and the next-token log-probabilities at the last token (float32)."""
    layers = model.model.layers
    outputs = [None] * len(layers)
    handles = []

    def make_hook(idx):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            if out.dim() == 3:
                out = out[0]
            outputs[idx] = out.detach()
        return hook

    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))
    try:
        input_ids = prompt["input_ids"].to(device)
        attention_mask = prompt["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    finally:
        for h in handles:
            h.remove()

    last = attention_mask.sum(dim=1).long().item() - 1
    logp = torch.log_softmax(logits[0, last].float(), dim=-1)
    return torch.stack(outputs, dim=0), logp


def relative_changes(out, ref):
    """Per-layer statistics of Phase 4 for one prompt; out and ref have shape (L, T, d)."""
    diff = (out.float() - ref.float()).norm(dim=-1)        # (L, T)
    base = ref.float().norm(dim=-1).clamp_min(1e-10)       # (L, T)
    rel = diff / base
    return {
        "last": rel[:, -1].cpu().numpy(),
        "pos_mean": rel.mean(dim=1).cpu().numpy(),
        "pos_max": rel.max(dim=1).values.cpu().numpy(),
    }


def kl_divergence(logp_ref, logp):
    """KL(p_ref || p) for log-probability vectors."""
    return float((logp_ref.exp() * (logp_ref - logp)).sum().item())


def measure(model, prompts, refs, ref_logps, device):
    """Statistics of every prompt against the stored references.

    Returns dict stat -> array (n_prompts, L) and 'kl_last' -> array (n_prompts,).
    """
    per_stat = {s: [] for s in STATS}
    kls = []
    for prompt, ref, ref_logp in zip(prompts, refs, ref_logps):
        out, logp = run_with_layer_outputs(model, prompt, device)
        changes = relative_changes(out, ref)
        for s in STATS:
            per_stat[s].append(changes[s])
        kls.append(kl_divergence(ref_logp, logp))
    result = {s: np.stack(v, axis=0) for s, v in per_stat.items()}
    result["kl_last"] = np.array(kls)
    return result


def effective_edit_diagnostics(W_before, W_after, delta_intended, H_last):
    """Compare the edit stored in the model dtype with the intended edit.

    W_before, W_after: weights in the model dtype; delta_intended: float32 or None;
    H_last: (n_cal, d_in) last-token calibration inputs of the clean model.
    The residual ratio ||DeltaW_eff H^T||_F / ||W H^T||_F is the relative change of the
    sublayer output on the calibration inputs that the stored edit causes.
    """
    W0 = W_before.float().cpu()
    delta_eff = W_after.float().cpu() - W0
    H = H_last.float().cpu()
    diag = {
        "eps_effective": float(delta_eff.norm() / W0.norm()),
        "residual_ratio_effective": float((delta_eff @ H.T).norm() / (W0 @ H.T).norm()),
    }
    if delta_intended is not None:
        diag["residual_ratio_intended"] = float((delta_intended @ H.T).norm() / (W0 @ H.T).norm())
        diag["rounding_error_rel"] = float((delta_eff - delta_intended).norm() / delta_intended.norm())
    return diag


def input_spectrum(model, prompts, layer_idx, sublayer_path, device, bottom_dims):
    """Singular values of the all-position and last-token inputs of a sublayer on the prompts.

    For each m in bottom_dims, energy_bottom[m] is the share of the squared Frobenius norm of
    the all-position inputs that lies in the m right singular directions with the smallest
    singular values: the activation change that an edit confined to those directions would
    still cause, relative to a full-energy direction set.
    """
    H_all = collect_sublayer_inputs(model, prompts, layer_idx, sublayer_path, device, positions="all")
    S = torch.linalg.svdvals(H_all.to(device=device, dtype=torch.float32)).cpu()
    d_in = H_all.shape[1]
    energy = S.pow(2)
    total = energy.sum().item()
    tol = 1e-5 * S[0].item()
    # svdvals returns min(n_rows, d_in) values; a short matrix has d_in - n_rows extra zeros.
    n_missing = max(d_in - S.numel(), 0)

    energy_bottom = {}
    for m in bottom_dims:
        m_present = max(m - n_missing, 0)
        share = energy[S.numel() - m_present:].sum().item() / total if m_present else 0.0
        energy_bottom[str(m)] = share
    return {
        "n_rows": int(H_all.shape[0]),
        "d_in": int(d_in),
        "numerical_rank": int((S > tol).sum().item()),
        "sigma_max": float(S[0]),
        "sigma_min": float(S[-1]) if n_missing == 0 else 0.0,
        "energy_bottom": energy_bottom,
        "singular_values": S.tolist(),
    }


# ════════════════════════════════════════════════════════════════
# Analysis
# ════════════════════════════════════════════════════════════════

def layer_table(ns_vals, g_vals, alpha, threshold):
    """Decision quantities per decoder layer for one statistic.

    ns_vals, g_vals: arrays (n_prompts, L) for the null-space edit and its control.
    """
    rows = []
    for l in range(ns_vals.shape[1]):
        ns_t = one_sample_ttest(ns_vals[:, l])
        g_t = one_sample_ttest(g_vals[:, l])
        ns_mean, g_mean = ns_t["mean"], g_t["mean"]
        suppression = 1.0 - ns_mean / g_mean if g_mean > 0 else None
        rows.append({
            "layer": l,
            "ns_mean": ns_mean,
            "ns_std": ns_t["std"],
            "ns_p": ns_t["p_value"],
            "ns_rejects": monitor_rejects(ns_vals[:, l], alpha),
            "g_mean": g_mean,
            "g_p": g_t["p_value"],
            "g_rejects": monitor_rejects(g_vals[:, l], alpha),
            "suppression": suppression,
            "evaded": suppression is not None and suppression > threshold,
        })
    return rows


def summarize_condition(rows, edited_layer):
    """System-level view of one statistic for one null-space condition.

    'evaded' follows the paper's reporting rule (suppression > threshold). An every-layer
    activation audit (assumption 5) is evaded only if it is evaded at every decoder layer
    whose output the edit can change, i.e. at the edited layer and all later layers.
    """
    scope = [r for r in rows if r["layer"] >= edited_layer]
    not_evaded = [r["layer"] for r in scope if not r["evaded"]]
    upstream_changed = [r["layer"] for r in rows if r["layer"] < edited_layer and r["ns_mean"] > 0]
    return {
        "edited_layer_suppression": scope[0]["suppression"],
        "edited_layer_evaded": scope[0]["evaded"],
        "next_layer_suppression": scope[1]["suppression"] if len(scope) > 1 else None,
        "min_suppression_downstream": min(r["suppression"] for r in scope),
        "first_layer_not_evaded": not_evaded[0] if not_evaded else None,
        "n_layers_in_scope": len(scope),
        "n_layers_not_evaded": len(not_evaded),
        "every_layer_audit_evaded": not not_evaded,
        "any_layer_rejects": any(r["ns_rejects"] for r in scope),
        "upstream_layers_changed": upstream_changed,
    }


def reproduction_check(reference_path, conditions, n_cal, target_layers, epsilons):
    """Compare edited-layer calibration means with exp9 (results.json, tap study)."""
    if not reference_path or not os.path.exists(reference_path):
        return None
    with open(reference_path) as f:
        ref = json.load(f)
    ref_by_key = {(c["layer"], c["epsilon"]): c for c in ref["conditions"]}
    rows = []
    for lidx in target_layers:
        for eps in epsilons:
            if (lidx, eps) not in ref_by_key:
                continue
            rc = ref_by_key[(lidx, eps)]
            for attack, ref_key in (("gaussian", "gauss_tap_cal_act_mean"),
                                    ("null_space", "ns_tap_cal_act_mean")):
                ours = float(conditions[(lidx, eps, attack)]["last"][:n_cal, lidx].mean())
                theirs = rc[ref_key]
                rows.append({"layer": lidx, "epsilon": eps, "attack": attack,
                             "exp12": ours, "exp9": theirs,
                             "rel_diff": abs(ours - theirs) / theirs})
    return {"max_rel_diff": max(r["rel_diff"] for r in rows), "rows": rows}


# ════════════════════════════════════════════════════════════════
# Pipeline
# ════════════════════════════════════════════════════════════════

def run_pipeline(model, cal_prompts, test_prompts, cfg, device):
    """Phases 1-5. Returns the result dict that Phase 6 writes to disk."""
    exp = cfg.experiment
    target_layers = list(exp.target_layers)
    epsilons = list(exp.epsilons)
    sublayer_path = exp.target_sublayer
    alpha = exp.alpha
    threshold = exp.evasion_threshold
    all_prompts = cal_prompts + test_prompts
    n_cal = len(cal_prompts)
    layers = model.model.layers
    n_layers = len(layers)
    times = {}

    # ── Phase 1: clean references ──
    print("=" * 60 + "\nPHASE 1: clean references at every layer and position\n" + "=" * 60)
    t = time.time()
    refs, ref_logps = [], []
    for prompt in all_prompts:
        out, logp = run_with_layer_outputs(model, prompt, device)
        refs.append(out)
        ref_logps.append(logp)
    times["phase1_references"] = time.time() - t

    # ── Phase 2: noise floor ──
    print("=" * 60 + "\nPHASE 2: noise floor (clean model against its references)\n" + "=" * 60)
    t = time.time()
    clean = measure(model, all_prompts, refs, ref_logps, device)
    noise_floor = {s: float(np.abs(clean[s]).max()) for s in STATS}
    noise_floor["kl_last"] = float(np.abs(clean["kl_last"]).max())
    print(f"  max over prompts and layers: {noise_floor}")
    times["phase2_noise_floor"] = time.time() - t

    # ── Phase 3: input spectrum and last-token calibration inputs ──
    print("=" * 60 + "\nPHASE 3: all-position input spectrum of the edited sublayer\n" + "=" * 60)
    t = time.time()
    spectra, H_last = {}, {}
    for lidx in target_layers:
        spectra[lidx] = input_spectrum(model, cal_prompts, lidx, sublayer_path, device,
                                       list(exp.spectrum_bottom_dims))
        H_last[lidx] = collect_sublayer_inputs(model, cal_prompts, lidx, sublayer_path, device)
        sp = spectra[lidx]
        print(f"  L{lidx}: {sp['n_rows']} all-position inputs, numerical rank "
              f"{sp['numerical_rank']}/{sp['d_in']}, bottom-energy {sp['energy_bottom']}")
    times["phase3_spectrum"] = time.time() - t

    # ── Phase 4: edits ──
    print("=" * 60 + "\nPHASE 4: Gaussian controls and null-space edits\n" + "=" * 60)
    t = time.time()
    conditions, edit_diag = {}, {}
    for lidx in target_layers:
        sublayer = get_sublayer(layers[lidx], sublayer_path)
        for eps in epsilons:
            for attack in ATTACKS:
                print(f"  --- {attack} L{lidx} eps={eps} ---")
                saved = save_weights(layers[lidx])
                W_before = sublayer.weight.data.clone()
                if attack == "gaussian":
                    perturb_gaussian_sublayer(layers[lidx], eps, exp.seed, sublayer_path)
                    delta_intended = None
                else:
                    delta_intended = perturb_null_space(
                        layers[lidx], eps, exp.seed, model=model, prompts=cal_prompts,
                        layer_idx=lidx, sublayer_path=sublayer_path, device=device)
                edit_diag[(lidx, eps, attack)] = effective_edit_diagnostics(
                    W_before, sublayer.weight.data, delta_intended, H_last[lidx])
                print(f"    edit: {edit_diag[(lidx, eps, attack)]}")

                conditions[(lidx, eps, attack)] = measure(model, all_prompts, refs, ref_logps, device)

                restore_weights(layers[lidx], saved)
                if not weights_equal(layers[lidx], saved):
                    raise RuntimeError(f"Weight restoration failed at layer {lidx}")
    times["phase4_edits"] = time.time() - t

    # ── Phase 5: analysis ──
    print("=" * 60 + "\nPHASE 5: analysis\n" + "=" * 60)
    analysis = []
    for lidx in target_layers:
        for eps in epsilons:
            ns = conditions[(lidx, eps, "null_space")]
            g = conditions[(lidx, eps, "gaussian")]
            entry = {"layer": lidx, "epsilon": eps,
                     "edit_null_space": edit_diag[(lidx, eps, "null_space")],
                     "edit_gaussian": edit_diag[(lidx, eps, "gaussian")]}
            for split, sl in (("cal", slice(0, n_cal)), ("test", slice(n_cal, None))):
                for s in STATS:
                    rows = layer_table(ns[s][sl], g[s][sl], alpha, threshold)
                    entry[f"{split}_{s}_layers"] = rows
                    entry[f"{split}_{s}_summary"] = summarize_condition(rows, lidx)
                entry[f"{split}_kl_last"] = {
                    "ns_mean": float(ns["kl_last"][sl].mean()),
                    "g_mean": float(g["kl_last"][sl].mean()),
                }
            analysis.append(entry)

    print_summary(analysis)
    repro = reproduction_check(cfg.reference.exp9_results, conditions, n_cal, target_layers, epsilons)
    if repro is not None:
        print(f"\n  Reproduction of exp9 edited-layer calibration means: "
              f"max relative difference {repro['max_rel_diff']:.2e}")

    raw = {
        f"{attack}_L{lidx}_eps{eps}": {s: conditions[(lidx, eps, attack)][s].tolist()
                                       for s in (*STATS, "kl_last")}
        for (lidx, eps, attack) in conditions
    }
    raw["clean"] = {s: clean[s].tolist() for s in (*STATS, "kl_last")}

    return {
        "n_layers": n_layers,
        "noise_floor": noise_floor,
        "spectra": {str(k): v for k, v in spectra.items()},
        "analysis": analysis,
        "reproduction_exp9": repro,
        "raw": raw,
        "phase_times": times,
    }


def print_summary(analysis):
    """Console table: calibration prompts, last-token statistic (the paper's monitor)."""
    header = (f"  {'L':>3} {'eps':>5} | {'supp@edit':>9} {'supp@next':>9} {'min supp':>9} "
              f"{'1st not-evaded':>14} {'audit evaded':>12} | {'pos_mean supp@edit':>18} "
              f"| {'KL ns/g':>17}")
    print("\n  Calibration prompts, last-token activation monitor (s_act), every layer >= edited")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for e in analysis:
        sm = e["cal_last_summary"]
        pm = e["cal_pos_mean_summary"]
        kl = e["cal_kl_last"]
        nxt = sm["next_layer_suppression"]
        print(f"  {e['layer']:>3} {e['epsilon']:>5} | {sm['edited_layer_suppression']:>9.1%} "
              f"{(f'{nxt:.1%}' if nxt is not None else '-'):>9} "
              f"{sm['min_suppression_downstream']:>9.1%} "
              f"{str(sm['first_layer_not_evaded']):>14} {str(sm['every_layer_audit_evaded']):>12} | "
              f"{pm['edited_layer_suppression']:>18.1%} | "
              f"{kl['ns_mean']:.2e}/{kl['g_mean']:.2e}")
    n_evaded = sum(e["cal_last_summary"]["every_layer_audit_evaded"] for e in analysis)
    print(f"\n  Every-layer last-token audit evaded in {n_evaded}/{len(analysis)} null-space conditions")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    from src.data_loader import load_prompts
    from src.model_loader import load_model

    parser = argparse.ArgumentParser(description="Experiment 12: downstream activation signal")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    t_start = time.time()
    torch.manual_seed(cfg.experiment.seed)
    out_dir = cfg.output.dir
    os.makedirs(out_dir, exist_ok=True)
    device = cfg.model.device

    print("=" * 80 + "\nEXPERIMENT 12: DOWNSTREAM ACTIVATION SIGNAL OF NULL-SPACE EDITS\n" + "=" * 80)
    print(OmegaConf.to_yaml(cfg))

    model, tokenizer = load_model(cfg)
    prompts = load_prompts(cfg, tokenizer)
    n_cal, n_test = cfg.experiment.n_cal, cfg.experiment.n_test
    if len(prompts) < n_cal + n_test:
        raise RuntimeError(f"Need {n_cal + n_test} prompts, got {len(prompts)}")
    cal_prompts = prompts[:n_cal]
    test_prompts = prompts[n_cal:n_cal + n_test]

    result = run_pipeline(model, cal_prompts, test_prompts, cfg, device)

    import transformers
    result["_meta"] = {
        "experiment": "exp12_downstream_activation",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": OmegaConf.to_container(cfg),
        "prompt_lengths": [int(p["input_ids"].shape[1]) for p in cal_prompts + test_prompts],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "runtime_seconds": time.time() - t_start,
    }

    path = os.path.join(out_dir, "results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    summary = {k: v for k, v in result.items() if k not in ("raw", "spectra")}
    summary["spectra"] = {k: {kk: vv for kk, vv in v.items() if kk != "singular_values"}
                          for k, v in result["spectra"].items()}
    with open(os.path.join(out_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n  Saved {path} and results_summary.json")
    print(f"  Runtime: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
