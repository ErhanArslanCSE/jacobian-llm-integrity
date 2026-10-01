#!/usr/bin/env python3
"""Step 3: Calibration Size Sweep — How many calibration prompts does a
defender need?

Core question: As n_cal decreases, the null space grows and the attacker
can hide more easily. At what n_cal does Jacobian monitoring lose its
advantage over activation monitoring?

Design:
  - Fixed test set (last 30 prompts) for fair comparison across n_cal values.
  - Shared calibration pool (first 100 prompts); for each n_cal we take
    cal_pool[:n_cal].
  - Gaussian perturbations: same DeltaW across all n_cal (seed-fixed),
    collected once on all 130 prompts then sliced.
  - Null-space perturbations: MUST be recomputed per n_cal because the null
    space basis depends on the calibration set size.

Expected paper figure: X = n_cal, Y = NAF, lines = epsilon values.

Usage:
  python run_step3_cal_sweep.py --config configs/nullspace.yaml
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
import numpy as np
from omegaconf import OmegaConf

from src.model_loader import load_model
from src.data_loader import load_prompts
from src.layer_wrapper import collect_layer_outputs
from src.jvp_engine import compute_all_jacobians
from src.perturbations import (
    save_weights, restore_weights,
    perturb_null_space, perturb_gaussian_sublayer,
)
from src.metrics import (
    compute_activation_diff, compute_jacobian_frobenius_diff,
    analyze_condition_v2, format_results_table_v2,
)


def collect_activations_and_jacobians(model, prompts, layer_idx, device,
                                      label="", chunk_size=None):
    """Collect layer output activations and full Jacobians for each prompt."""
    layer_indices = [layer_idx]
    activations = {layer_idx: []}
    jacobians = {layer_idx: []}

    for i, prompt in enumerate(prompts):
        input_ids = prompt["input_ids"].to(device)
        attention_mask = prompt["attention_mask"].to(device)
        last_tok = attention_mask.sum(dim=1).long().item() - 1

        output_dict = collect_layer_outputs(model, input_ids, attention_mask, layer_indices)
        act = output_dict[layer_idx][0, last_tok, :].cpu()
        activations[layer_idx].append(act)

        jac_dict = compute_all_jacobians(model, prompt, layer_indices, device,
                                          chunk_size=chunk_size)
        jacobians[layer_idx].append(jac_dict[layer_idx])

        if (i + 1) % 10 == 0:
            print(f"    {label}Prompt {i+1}/{len(prompts)} done")

    return activations, jacobians


def slice_data(full_data, layer_idx, start, end):
    """Slice collected activations or jacobians dict by prompt index range."""
    return {layer_idx: full_data[layer_idx][start:end]}


def compute_diffs_sliced(clean_data, pert_data, layer_idx, start, end):
    """Compute diffs on a slice of the data."""
    clean_slice = slice_data(clean_data, layer_idx, start, end)
    pert_slice = slice_data(pert_data, layer_idx, start, end)
    return clean_slice, pert_slice


def niac_decision_for_condition(gauss_result, ns_result, eps):
    """Run NIAC decision logic for one (n_cal, eps) combination.

    Returns dict with decision details.
    """
    naf_cal = 0.0
    if gauss_result['cal_jac_act_ratio'] > 0:
        naf_cal = ns_result['cal_jac_act_ratio'] / gauss_result['cal_jac_act_ratio']

    act_evasion = 0.0
    if gauss_result['cal_act_mean'] > 0:
        act_evasion = 1.0 - ns_result['cal_act_mean'] / gauss_result['cal_act_mean']

    jac_evasion = 0.0
    if gauss_result['cal_jac_mean'] > 0:
        jac_evasion = 1.0 - ns_result['cal_jac_mean'] / gauss_result['cal_jac_mean']

    act_evasion_success = ns_result['cal_act_mean'] < 0.5 * gauss_result['cal_act_mean']
    jac_still_detects = ns_result['cal_jac_ttest_p'] < 0.001

    if act_evasion_success and jac_still_detects:
        decision = "NIAC-DETECTED" if naf_cal >= 3.0 else "WEAK-NIAC"
    elif not act_evasion_success and jac_still_detects:
        decision = "BOTH-DETECT"
    elif act_evasion_success and not jac_still_detects:
        decision = "EVASION-SUCCESS"
    else:
        decision = "INCONCLUSIVE"

    return {
        "epsilon": eps,
        "decision": decision,
        "naf": naf_cal,
        "act_evasion_pct": act_evasion * 100,
        "jac_evasion_pct": jac_evasion * 100,
        "act_evasion_success": act_evasion_success,
        "jac_still_detects": jac_still_detects,
        "cal_act_mean_gauss": gauss_result['cal_act_mean'],
        "cal_jac_mean_gauss": gauss_result['cal_jac_mean'],
        "cal_act_mean_ns": ns_result['cal_act_mean'],
        "cal_jac_mean_ns": ns_result['cal_jac_mean'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/nullspace.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    n_cal_values = list(cfg.experiment.n_cal_values)
    n_test_held = cfg.experiment.n_test_held
    max_n_cal = max(n_cal_values)
    total_needed = max_n_cal + n_test_held
    epsilons = list(cfg.experiment.epsilons)

    print("=" * 80)
    print("STEP 3: Calibration Size Sweep")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Model: {cfg.model.name}")
    print(f"Target layer: {cfg.experiment.target_layer}")
    print(f"Target sublayer: {cfg.experiment.target_sublayer}")
    print(f"Epsilons: {epsilons}")
    print(f"n_cal values: {n_cal_values}")
    print(f"n_test (held-out): {n_test_held}")
    print(f"Total prompts needed: {total_needed}")
    print()
    print("KEY QUESTION: How many calibration prompts does a defender need?")
    print("  Fewer cal prompts → larger null space → easier for attacker to hide")
    print("  Expected: NAF increases with n_cal (smaller null space → harder evasion)")

    t_start = time.time()

    # ── Load model and data ──
    print("\n--- Loading model ---")
    model, tokenizer = load_model(cfg)
    device = cfg.model.device

    print("\n--- Loading prompts ---")
    all_prompts = load_prompts(cfg, tokenizer)
    if len(all_prompts) < total_needed:
        raise ValueError(
            f"Need {total_needed} prompts ({max_n_cal} cal + {n_test_held} test) "
            f"but only got {len(all_prompts)}. Increase data.num_prompts."
        )

    # Fixed split: first max_n_cal = cal pool, last n_test_held = test (FIXED)
    cal_pool = all_prompts[:max_n_cal]
    test_prompts = all_prompts[max_n_cal:max_n_cal + n_test_held]
    all_130 = all_prompts[:max_n_cal + n_test_held]  # All prompts we use
    print(f"Split: cal_pool={len(cal_pool)}, test={len(test_prompts)} (fixed)")

    layer_idx = cfg.experiment.target_layer
    sublayer_path = cfg.experiment.target_sublayer
    layer = model.model.layers[layer_idx]
    seed = cfg.experiment.seed

    chunk_size = cfg.experiment.get("jvp_chunk_size", None)
    if chunk_size:
        print(f"\nJVP batching: chunk_size={chunk_size} (vmap parallelism)")
    else:
        print("\nJVP batching: sequential (no vmap)")

    output_dir = cfg.output.dir
    os.makedirs(output_dir, exist_ok=True)

    # ════════════════════════════════════════════════════════════════
    # JVP Validation (same as exp2)
    # ════════════════════════════════════════════════════════════════
    if chunk_size:
        print("\n--- Validating batched JVP against sequential ---")
        from src.jvp_engine import compute_full_jacobian_at_last_token
        from src.layer_wrapper import collect_hidden_states

        val_prompt = cal_pool[0]
        val_ids = val_prompt["input_ids"].to(device)
        val_mask = val_prompt["attention_mask"].to(device)
        val_last_tok = val_mask.sum(dim=1).long().item() - 1

        val_hidden, val_kwargs = collect_hidden_states(
            model, val_ids, val_mask, [layer_idx]
        )
        val_hs = val_hidden[layer_idx].to(device)
        val_kw = val_kwargs[layer_idx]

        t_seq = time.time()
        jac_seq = compute_full_jacobian_at_last_token(
            layer, val_hs, val_kw, val_last_tok, device, chunk_size=None
        )
        t_seq = time.time() - t_seq

        t_bat = time.time()
        jac_bat = compute_full_jacobian_at_last_token(
            layer, val_hs, val_kw, val_last_tok, device, chunk_size=chunk_size
        )
        t_bat = time.time() - t_bat

        diff = (jac_seq - jac_bat).norm() / (jac_seq.norm() + 1e-10)
        print(f"  Sequential: {t_seq:.2f}s")
        print(f"  Batched:    {t_bat:.2f}s (speedup: {t_seq/t_bat:.1f}x)")
        print(f"  Relative diff: {diff:.2e}")
        if diff > 0.01:
            print(f"  WARNING: Large difference! Falling back to sequential.")
            chunk_size = None
        else:
            print(f"  VALIDATED: Results match (diff < 0.01)")

        del jac_seq, jac_bat, val_hs
        torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════
    # Phase 0: Baseline — collect ALL 130 prompts once (clean model)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 0: Baseline (clean model, all 130 prompts)")
    print("=" * 60)

    print(f"  Collecting activations + Jacobians for {len(all_130)} prompts...")
    t0 = time.time()
    all_acts_clean, all_jacs_clean = collect_activations_and_jacobians(
        model, all_130, layer_idx, device, label="[baseline] ",
        chunk_size=chunk_size
    )
    print(f"  Baseline done in {time.time()-t0:.1f}s")

    # ════════════════════════════════════════════════════════════════
    # Phase 1: Noise Floor Verification
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 1: Noise Floor Verification")
    print("=" * 60)

    n_check = 5
    check_prompts = cal_pool[:n_check]

    print(f"  Re-running {n_check} prompts to verify determinism...")
    check_acts, check_jacs = collect_activations_and_jacobians(
        model, check_prompts, layer_idx, device, label="[noise] ",
        chunk_size=chunk_size
    )

    noise_act_diffs = []
    noise_jac_diffs = []
    for i in range(n_check):
        a1 = all_acts_clean[layer_idx][i].float()
        a2 = check_acts[layer_idx][i].float()
        act_diff = (a1 - a2).norm().item() / (a1.norm().item() + 1e-10)
        noise_act_diffs.append(act_diff)

        j1 = all_jacs_clean[layer_idx][i].float()
        j2 = check_jacs[layer_idx][i].float()
        jac_diff = (j1 - j2).norm().item() / (j1.norm().item() + 1e-10)
        noise_jac_diffs.append(jac_diff)

    noise_act_max = max(noise_act_diffs)
    noise_jac_max = max(noise_jac_diffs)
    print(f"  Noise floor (act): max={noise_act_max:.2e}, "
          f"values={[f'{d:.2e}' for d in noise_act_diffs]}")
    print(f"  Noise floor (jac): max={noise_jac_max:.2e}, "
          f"values={[f'{d:.2e}' for d in noise_jac_diffs]}")

    if noise_act_max < 1e-10 and noise_jac_max < 1e-10:
        print("  CONFIRMED: Forward pass is deterministic (noise floor = 0)")
        noise_floor_act = 0.0
        noise_floor_jac = 0.0
    else:
        print(f"  WARNING: Non-zero noise floor detected!")
        noise_floor_act = noise_act_max
        noise_floor_jac = noise_jac_max

    # ════════════════════════════════════════════════════════════════
    # Phase 2: Gaussian sublayer perturbation — collect ALL 130 once per eps
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 2: Gaussian sublayer perturbation (collect all 130, once per eps)")
    print(f"  Perturbing ONLY {sublayer_path} (same as null-space)")
    print("=" * 60)

    # Store full (130-prompt) perturbed data per epsilon
    all_gauss_acts_pert = {}   # eps -> {layer_idx: [130 tensors]}
    all_gauss_jacs_pert = {}   # eps -> {layer_idx: [130 tensors]}

    for eps in epsilons:
        condition = f"gaussian_sublayer_eps{eps}"
        print(f"\n  --- {condition} ---")

        saved = perturb_gaussian_sublayer(
            layer, eps, seed, sublayer_path=sublayer_path
        )

        print(f"  Collecting perturbed data for all {len(all_130)} prompts...")
        t2 = time.time()
        gauss_acts, gauss_jacs = collect_activations_and_jacobians(
            model, all_130, layer_idx, device, label=f"[gauss {eps}] ",
            chunk_size=chunk_size
        )
        print(f"  Done in {time.time()-t2:.1f}s")

        restore_weights(layer, saved)

        all_gauss_acts_pert[eps] = gauss_acts
        all_gauss_jacs_pert[eps] = gauss_jacs

    # ════════════════════════════════════════════════════════════════
    # Phase 3: Null-space perturbation + sweep over n_cal
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 3: Null-space perturbation (per n_cal × eps)")
    print("  Each n_cal gets a DIFFERENT null-space basis → DIFFERENT DeltaW")
    print("=" * 60)

    # Master results container
    sweep_results = {}
    sweep_summary = []

    for n_cal in n_cal_values:
        sweep_key = f"n_cal_{n_cal}"
        print(f"\n{'─' * 60}")
        print(f"  n_cal = {n_cal}")
        print(f"{'─' * 60}")

        cal_prompts = cal_pool[:n_cal]

        gauss_results_for_ncal = []
        ns_results_for_ncal = []
        niac_decisions_for_ncal = []

        # ── Gaussian: slice pre-collected data ──
        for eps in epsilons:
            condition = f"gaussian_sublayer_eps{eps}"

            # Slice cal and test from the full 130
            cal_act_diffs = compute_activation_diff(
                slice_data(all_acts_clean, layer_idx, 0, n_cal),
                slice_data(all_gauss_acts_pert[eps], layer_idx, 0, n_cal)
            )
            cal_jac_diffs = compute_jacobian_frobenius_diff(
                slice_data(all_jacs_clean, layer_idx, 0, n_cal),
                slice_data(all_gauss_jacs_pert[eps], layer_idx, 0, n_cal)
            )
            test_act_diffs = compute_activation_diff(
                slice_data(all_acts_clean, layer_idx, max_n_cal, max_n_cal + n_test_held),
                slice_data(all_gauss_acts_pert[eps], layer_idx, max_n_cal, max_n_cal + n_test_held)
            )
            test_jac_diffs = compute_jacobian_frobenius_diff(
                slice_data(all_jacs_clean, layer_idx, max_n_cal, max_n_cal + n_test_held),
                slice_data(all_gauss_jacs_pert[eps], layer_idx, max_n_cal, max_n_cal + n_test_held)
            )

            result = analyze_condition_v2(
                cal_act_diffs[layer_idx], cal_jac_diffs[layer_idx],
                test_act_diffs[layer_idx], test_jac_diffs[layer_idx],
                condition, noise_floor_act, noise_floor_jac
            )
            gauss_results_for_ncal.append(result)
            print(f"    [gauss eps={eps}] cal_act={result['cal_act_mean']:.6f}, "
                  f"cal_jac={result['cal_jac_mean']:.6f}, "
                  f"j/a={result['cal_jac_act_ratio']:.1f}x")

        # ── Null-space: NEW perturbation per n_cal ──
        for eps in epsilons:
            condition = f"null_space_eps{eps}"
            print(f"\n    [null-space n_cal={n_cal}, eps={eps}] computing...")

            saved = perturb_null_space(
                layer, eps, seed,
                model=model, prompts=cal_prompts,
                layer_idx=layer_idx, sublayer_path=sublayer_path,
                device=device,
            )

            # Collect perturbed on cal prompts
            t3 = time.time()
            ns_cal_acts_pert, ns_cal_jacs_pert = collect_activations_and_jacobians(
                model, cal_prompts, layer_idx, device,
                label=f"[ns cal n={n_cal} e={eps}] ",
                chunk_size=chunk_size
            )

            # Collect perturbed on test prompts (always the same 30)
            ns_test_acts_pert, ns_test_jacs_pert = collect_activations_and_jacobians(
                model, test_prompts, layer_idx, device,
                label=f"[ns test n={n_cal} e={eps}] ",
                chunk_size=chunk_size
            )
            print(f"    Done in {time.time()-t3:.1f}s")

            restore_weights(layer, saved)

            # Compute diffs — cal baseline must also be sliced to n_cal
            cal_act_diffs = compute_activation_diff(
                slice_data(all_acts_clean, layer_idx, 0, n_cal),
                ns_cal_acts_pert
            )
            cal_jac_diffs = compute_jacobian_frobenius_diff(
                slice_data(all_jacs_clean, layer_idx, 0, n_cal),
                ns_cal_jacs_pert
            )
            # Test baseline: indices [max_n_cal : max_n_cal + n_test_held]
            test_act_diffs = compute_activation_diff(
                slice_data(all_acts_clean, layer_idx, max_n_cal, max_n_cal + n_test_held),
                ns_test_acts_pert
            )
            test_jac_diffs = compute_jacobian_frobenius_diff(
                slice_data(all_jacs_clean, layer_idx, max_n_cal, max_n_cal + n_test_held),
                ns_test_jacs_pert
            )

            result = analyze_condition_v2(
                cal_act_diffs[layer_idx], cal_jac_diffs[layer_idx],
                test_act_diffs[layer_idx], test_jac_diffs[layer_idx],
                condition, noise_floor_act, noise_floor_jac
            )
            ns_results_for_ncal.append(result)
            print(f"    [null-space eps={eps}] cal_act={result['cal_act_mean']:.6f}, "
                  f"cal_jac={result['cal_jac_mean']:.6f}, "
                  f"j/a={result['cal_jac_act_ratio']:.1f}x")

        # ── NIAC decisions for this n_cal ──
        for i, eps in enumerate(epsilons):
            dec = niac_decision_for_condition(
                gauss_results_for_ncal[i], ns_results_for_ncal[i], eps
            )
            dec["n_cal"] = n_cal
            niac_decisions_for_ncal.append(dec)

            print(f"    eps={eps}: evasion={dec['act_evasion_pct']:.1f}%, "
                  f"NAF={dec['naf']:.1f}x, decision={dec['decision']}")
            sweep_summary.append(dec)

        sweep_results[sweep_key] = {
            "gaussian": [r for r in gauss_results_for_ncal],
            "null_space": [r for r in ns_results_for_ncal],
            "niac_decisions": niac_decisions_for_ncal,
        }

    # ════════════════════════════════════════════════════════════════
    # Phase 4: Sweep Summary and Decision
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 4: SWEEP SUMMARY")
    print("=" * 80)

    # Print sweep table
    print(f"\n{'n_cal':>6} | ", end="")
    for eps in epsilons:
        print(f"eps={eps:5.2f} NAF | eps={eps:5.2f} dec      | ", end="")
    print()
    print("─" * (8 + len(epsilons) * 38))

    for n_cal in n_cal_values:
        print(f"{n_cal:>6} | ", end="")
        for eps in epsilons:
            entry = next(
                (s for s in sweep_summary if s['n_cal'] == n_cal and s['epsilon'] == eps),
                None
            )
            if entry:
                print(f"  {entry['naf']:>6.1f}x | {entry['decision']:<16s} | ", end="")
            else:
                print(f"{'N/A':>8s} | {'N/A':<16s} | ", end="")
        print()

    # Per-epsilon summary
    print("\n--- Per-epsilon Analysis ---")
    for eps in epsilons:
        entries = [s for s in sweep_summary if s['epsilon'] == eps]
        print(f"\n  epsilon = {eps}:")
        for e in entries:
            print(f"    n_cal={e['n_cal']:>3d}: NAF={e['naf']:.1f}x, "
                  f"evasion={e['act_evasion_pct']:.1f}%, "
                  f"jac_detects={e['jac_still_detects']}, "
                  f"decision={e['decision']}")

    # Minimum n_cal threshold
    print("\n--- Minimum n_cal Threshold ---")
    print("  (Smallest n_cal where NAF >= 3.0 for each epsilon)")
    for eps in epsilons:
        entries = [s for s in sweep_summary if s['epsilon'] == eps]
        min_ncal = None
        for e in sorted(entries, key=lambda x: x['n_cal']):
            if e['naf'] >= 3.0:
                min_ncal = e['n_cal']
                break
        if min_ncal is not None:
            print(f"  eps={eps}: min_n_cal = {min_ncal} (NAF >= 3.0)")
        else:
            print(f"  eps={eps}: NAF < 3.0 for ALL tested n_cal values")

    # Overall verdict
    print("\n" + "=" * 60)
    total_conditions = len(sweep_summary)
    n_strong = sum(1 for s in sweep_summary if s['decision'] == 'NIAC-DETECTED')
    n_weak = sum(1 for s in sweep_summary if s['decision'] == 'WEAK-NIAC')
    n_detected = n_strong + n_weak
    n_evasion = sum(1 for s in sweep_summary if s['decision'] == 'EVASION-SUCCESS')

    # Check n_cal=30 consistency with exp2
    n30_entries = [s for s in sweep_summary if s['n_cal'] == 30]
    print(f"  Total conditions: {total_conditions}")
    print(f"  NIAC detected: {n_detected}/{total_conditions} "
          f"({n_strong} strong, {n_weak} weak)")
    print(f"  Evasion success: {n_evasion}/{total_conditions}")
    print(f"\n  n_cal=30 (exp2 reference):")
    for e in n30_entries:
        print(f"    eps={e['epsilon']}: NAF={e['naf']:.1f}x, decision={e['decision']}")

    total_time = time.time() - t_start
    print(f"\n  Total runtime: {total_time:.1f}s ({total_time/60:.1f}min)")

    # ── Save results ──
    output = {
        "sweep_results": sweep_results,
        "sweep_summary": sweep_summary,
        "test_set_size": n_test_held,
        "noise_floor": {
            "act": noise_floor_act,
            "jac": noise_floor_jac,
            "act_diffs": noise_act_diffs,
            "jac_diffs": noise_jac_diffs,
        },
        "_meta": {
            "model": cfg.model.name,
            "target_layer": layer_idx,
            "target_sublayer": sublayer_path,
            "epsilons": epsilons,
            "n_cal_values": n_cal_values,
            "n_test_held": n_test_held,
            "seed": seed,
            "runtime_seconds": total_time,
            "date": datetime.now().isoformat(),
            "version": "step3 -- calibration size sweep",
        }
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
