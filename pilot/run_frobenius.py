#!/usr/bin/env python3
"""Step 9: Full Jacobian Frobenius Norm — Last Chance Experiment.

Instead of k=30 random JVP projections (scalar summaries), computes
the FULL d×d Jacobian matrix at the last token position:

    jac_diff = ||J_anom - J_norm||_F / ||J_norm||_F

This directly matches the research plan's formula and uses the full
Jacobian information — not a low-dimensional random projection.

Also computes random JVP metric (k=30) alongside for direct comparison.

Usage:
    python run_frobenius.py --config configs/step9_frobenius.yaml
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.model_loader import load_model
from src.data_loader import load_prompts
from src.layer_wrapper import collect_hidden_states, collect_layer_outputs
from src.jvp_engine import compute_all_jacobians, compute_all_jvp_norms
from src.perturbations import apply_perturbation, restore_weights
from src.metrics import (
    compute_activation_diff,
    compute_jvp_diff,
    compute_jacobian_frobenius_diff,
    compute_snr_metrics,
    aggregate_decision,
)


def get_last_token_activations(model, prompt, layer_indices, device):
    """Get last-token activation vectors at each target layer's OUTPUT."""
    input_ids = prompt["input_ids"].to(device)
    attention_mask = prompt["attention_mask"].to(device)
    last_token_idx = attention_mask.sum(dim=1).long().item() - 1
    output_dict = collect_layer_outputs(model, input_ids, attention_mask, layer_indices)
    activations = {}
    for l in layer_indices:
        activations[l] = output_dict[l][0, last_token_idx, :].detach().cpu()
    return activations


def run_experiment(cfg):
    """Run the full Jacobian Frobenius norm experiment."""
    device = cfg.model.device
    seed = cfg.experiment.seed
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(cfg)

    # Only measure the target layer (others are unaffected by perturbation)
    target_layer = cfg.experiment.target_layer
    layer_indices = [target_layer]
    hidden_dim = model.config.hidden_size
    num_directions = cfg.experiment.num_directions

    print(f"Target layer: {target_layer}")
    print(f"Hidden dim: {hidden_dim} → {hidden_dim} JVPs per full Jacobian")
    print(f"Random JVP directions: {num_directions} (for comparison)")

    # Load data and split
    all_prompts = load_prompts(cfg, tokenizer)
    n_train = len(all_prompts) // 2
    train_prompts = all_prompts[:n_train]
    test_prompts = all_prompts[n_train:]
    print(f"Train/test split: {n_train} / {len(test_prompts)}")

    perturbation_types = list(cfg.experiment.perturbation_types)
    epsilons = list(cfg.experiment.epsilons)

    # ── Phase 1: Normal model measurements on test set ──
    print(f"\n── Phase 1: Normal model — full Jacobians + random JVPs + activations ──")
    normal_acts = {l: [] for l in layer_indices}
    normal_jacs = {l: [] for l in layer_indices}
    normal_jvps = {l: [] for l in layer_indices}

    t_jac_total = 0
    for prompt in tqdm(test_prompts, desc="Normal model"):
        # Activations
        acts = get_last_token_activations(model, prompt, layer_indices, device)

        # Full Jacobian (d×d) — the main metric
        t0 = time.time()
        jacs = compute_all_jacobians(model, prompt, layer_indices, device)
        t_jac_total += time.time() - t0

        # Random JVPs (k=30) — for comparison with Step 8
        jvps = compute_all_jvp_norms(model, prompt, layer_indices,
                                      num_directions, seed, device)

        for l in layer_indices:
            normal_acts[l].append(acts[l])
            normal_jacs[l].append(jacs[l])
            normal_jvps[l].append(jvps[l])

    # Report Phase 1 stats
    per_prompt = t_jac_total / len(test_prompts)
    print(f"  Full Jacobian: {t_jac_total:.1f}s total, {per_prompt:.2f}s/prompt")
    jac_norms = [j.norm().item() for j in normal_jacs[target_layer]]
    print(f"  ||J_norm||_F: mean={np.mean(jac_norms):.2f}, "
          f"std={np.std(jac_norms):.2f}")

    # ── Phase 2: Perturbed model measurements ──
    all_results = {}

    for ptype in perturbation_types:
        for eps in epsilons:
            cond = f"{ptype}_eps{eps}"
            print(f"\n── Phase 2: {cond} ──")

            pert_acts = {l: [] for l in layer_indices}
            pert_jacs = {l: [] for l in layer_indices}
            pert_jvps = {l: [] for l in layer_indices}

            # Apply perturbation
            target_module = model.model.layers[target_layer]
            pkw = {}
            if ptype == "activation_matched":
                pkw = {
                    "model": model,
                    "prompts": train_prompts,
                    "layer_idx": target_layer,
                    "device": device,
                }

            saved = apply_perturbation(target_module, ptype, eps, seed + 1000, **pkw)

            try:
                for prompt in tqdm(test_prompts, desc=f"  {cond}"):
                    acts = get_last_token_activations(model, prompt, layer_indices, device)
                    jacs = compute_all_jacobians(model, prompt, layer_indices, device)
                    jvps = compute_all_jvp_norms(model, prompt, layer_indices,
                                                  num_directions, seed, device)
                    for l in layer_indices:
                        pert_acts[l].append(acts[l])
                        pert_jacs[l].append(jacs[l])
                        pert_jvps[l].append(jvps[l])
            finally:
                restore_weights(target_module, saved)

            # Compute all three diff metrics
            act_diffs = compute_activation_diff(normal_acts, pert_acts)
            jac_diffs = compute_jacobian_frobenius_diff(normal_jacs, pert_jacs)
            jvp_diffs = compute_jvp_diff(normal_jvps, pert_jvps)

            # SNR from Frobenius norm (MAIN metric)
            snr_frob = compute_snr_metrics(act_diffs, jac_diffs)
            # SNR from random JVPs (comparison with Step 8)
            snr_rand = compute_snr_metrics(act_diffs, jvp_diffs)

            # Merge results
            merged = {}
            for l in layer_indices:
                merged[l] = {
                    # Activation metrics
                    "act_mean": snr_frob[l]["act_mean"],
                    "act_std": snr_frob[l]["act_std"],
                    "snr_act": snr_frob[l]["snr_act"],
                    # Frobenius Jacobian (MAIN)
                    "jac_frob_mean": snr_frob[l]["jvp_mean"],
                    "jac_frob_std": snr_frob[l]["jvp_std"],
                    "snr_j_frob": snr_frob[l]["snr_j"],
                    "ratio_frob": snr_frob[l]["ratio"],
                    "decision_frob": snr_frob[l]["decision"],
                    # Random JVP (comparison)
                    "jvp_rand_mean": snr_rand[l]["jvp_mean"],
                    "jvp_rand_std": snr_rand[l]["jvp_std"],
                    "snr_j_random": snr_rand[l]["snr_j"],
                    "ratio_random": snr_rand[l]["ratio"],
                    "decision_random": snr_rand[l]["decision"],
                    # Welch t-test (Frobenius)
                    "welch_t": snr_frob[l]["welch_t"],
                    "welch_p": snr_frob[l]["welch_p"],
                    "auc_act": snr_frob[l]["auc_act"],
                    "auc_jac_frob": snr_frob[l]["auc_jvp"],
                }

            all_results[cond] = merged

            # Print comparison
            for l in layer_indices:
                m = merged[l]
                print(f"\n  Layer {l}:")
                print(f"    SNR_act:                {m['snr_act']:.3f}")
                print(f"    SNR_J (Frobenius d={hidden_dim}):  "
                      f"{m['snr_j_frob']:.3f}  → ratio={m['ratio_frob']:.3f}  "
                      f"{m['decision_frob']}")
                print(f"    SNR_J (random k={num_directions}):     "
                      f"{m['snr_j_random']:.3f}  → ratio={m['ratio_random']:.3f}  "
                      f"{m['decision_random']}")
                print(f"    act_mean={m['act_mean']:.6f}  "
                      f"jac_frob_mean={m['jac_frob_mean']:.6f}  "
                      f"jvp_rand_mean={m['jvp_rand_mean']:.6f}")

    # ── Phase 3: Aggregate and compare ──
    print("\n\n" + "=" * 70)
    print("FINAL RESULTS — Full Jacobian Frobenius vs Random JVP")
    print("=" * 70)

    # Aggregate using Frobenius decisions
    frob_for_agg = {}
    for cond, layer_results in all_results.items():
        frob_for_agg[cond] = {}
        for l, m in layer_results.items():
            frob_for_agg[cond][l] = {
                "decision": m["decision_frob"],
                "ratio": m["ratio_frob"],
                "snr_act": m["snr_act"],
                "snr_j": m["snr_j_frob"],
            }

    overall, summary = aggregate_decision(frob_for_agg)
    print(f"\nOverall (Frobenius): {overall}")
    print(f"  GO: {summary.get('go_count', 0)}, "
          f"MARGINAL: {summary.get('marginal_count', 0)}, "
          f"NO-GO: {summary.get('nogo_count', 0)}")

    # Direct comparison table
    print(f"\n{'Condition':<30} {'ratio_frob':>12} {'ratio_random':>12} "
          f"{'improvement':>12}")
    print("-" * 66)
    for cond in sorted(all_results.keys()):
        for l in layer_indices:
            m = all_results[cond][l]
            imp = m["ratio_frob"] / (m["ratio_random"] + 1e-10)
            print(f"{cond:<30} {m['ratio_frob']:>12.3f} "
                  f"{m['ratio_random']:>12.3f} {imp:>12.1f}x")

    # Save results
    if cfg.output.save_raw:
        raw_path = out_dir / "results.json"
        serializable = {}
        for cond, layer_results in all_results.items():
            serializable[cond] = {str(l): m for l, m in layer_results.items()}

        serializable["_meta"] = {
            "model": cfg.model.name,
            "hidden_dim": hidden_dim,
            "method": "full_jacobian_frobenius + random_jvp_comparison",
            "num_basis_vectors": hidden_dim,
            "num_random_directions": num_directions,
            "num_prompts_total": len(all_prompts),
            "num_prompts_train": n_train,
            "num_prompts_test": len(test_prompts),
            "target_layer": target_layer,
            "jacobian_time_per_prompt_s": per_prompt,
            "overall_decision_frobenius": overall,
            "summary": summary,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }

        with open(raw_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\nResults saved to {raw_path}")

    return all_results, overall


def main():
    parser = argparse.ArgumentParser(
        description="Step 9: Full Jacobian Frobenius Norm Experiment")
    parser.add_argument("--config", required=True, help="Config YAML")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    t0 = time.time()
    results, decision = run_experiment(cfg)
    elapsed = time.time() - t0

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Final verdict (Frobenius): {decision}")


if __name__ == "__main__":
    main()
