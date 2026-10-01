#!/usr/bin/env python3
"""Step 2 v2: Null-Space Jacobian Monitoring — Main Experiment.

Core question: When an adversary modifies weights in the null space of
calibration data (so activations don't change), does the Jacobian change?

CHANGES from v1:
  1. Measure Jacobians on CALIBRATION data (THE key missing measurement)
  2. Use single-sublayer gaussian perturbation for fair comparison
  3. Verify noise floor (deterministic forward pass check)
  4. Better decision logic based on cal data ratios

Mathematical guarantee (in exact arithmetic):
  On calibration data, null-space DeltaW gives:
    act_diff = 0 (because DeltaW @ h_cal = 0)
    jac_diff > 0 (because Jacobian depends on W directly through chain rule:
                   DJ = W_down @ diag(u) @ diag(SiLU'(g)) @ DeltaW @ dh_norm/dh_in
                   This is nonzero even when DeltaW @ h_cal = 0)
  In bfloat16: act_diff ~ 0.002 (quantization noise), jac_diff >> 0.002

Protocol:
  Phase 0: Baseline — collect activations + Jacobians on BOTH cal and test
  Phase 1: Noise floor — verify forward pass determinism
  Phase 2: Gaussian sublayer perturbation — fair control (same sublayer as null-space)
  Phase 3: Null-space perturbation — THE experiment (cal: act~0, jac>0)
  Phase 4: Analysis — cal vs test comparison, NIAC amplification factor

Usage:
  python run_step2_nullspace.py --config configs/nullspace.yaml
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
    """Collect layer output activations and full Jacobians for each prompt.

    Args:
        model: The transformer model
        prompts: List of prompt dicts
        layer_idx: Target layer index
        device: torch device
        label: Logging label prefix
        chunk_size: If set, batch JVP directions using vmap for speedup

    Returns:
        activations: dict[layer_idx -> list of (hidden_dim,) tensors]
        jacobians: dict[layer_idx -> list of (d, d) tensors]
    """
    layer_indices = [layer_idx]
    activations = {layer_idx: []}
    jacobians = {layer_idx: []}

    for i, prompt in enumerate(prompts):
        input_ids = prompt["input_ids"].to(device)
        attention_mask = prompt["attention_mask"].to(device)
        last_tok = attention_mask.sum(dim=1).long().item() - 1

        # Activation (layer output at last token)
        output_dict = collect_layer_outputs(model, input_ids, attention_mask, layer_indices)
        act = output_dict[layer_idx][0, last_tok, :].cpu()
        activations[layer_idx].append(act)

        # Full Jacobian
        jac_dict = compute_all_jacobians(model, prompt, layer_indices, device,
                                          chunk_size=chunk_size)
        jacobians[layer_idx].append(jac_dict[layer_idx])

        if (i + 1) % 5 == 0:
            print(f"    {label}Prompt {i+1}/{len(prompts)} done")

    return activations, jacobians


def print_condition_summary(result):
    """Print a summary of one condition's results."""
    r = result
    print(f"\n    --- {r['condition']} ---")
    print(f"    Cal data:  act_mean={r['cal_act_mean']:.6f}, jac_mean={r['cal_jac_mean']:.6f}, "
          f"jac/act={r['cal_jac_act_ratio']:.1f}x")
    print(f"    Test data: act_mean={r['test_act_mean']:.6f}, jac_mean={r['test_jac_mean']:.6f}, "
          f"jac/act={r['test_jac_act_ratio']:.1f}x")
    print(f"    Cal jac t-test:  t={r['cal_jac_ttest_t']:.3f}, p={r['cal_jac_ttest_p']:.2e}")
    print(f"    Test jac t-test: t={r['test_jac_ttest_t']:.3f}, p={r['test_jac_ttest_p']:.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/nullspace.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    print("=" * 80)
    print("STEP 2 v2: Null-Space Jacobian Monitoring Experiment")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Model: {cfg.model.name}")
    print(f"Target layer: {cfg.experiment.target_layer}")
    print(f"Target sublayer: {cfg.experiment.target_sublayer}")
    print(f"Epsilons: {cfg.experiment.epsilons}")
    print()
    print("KEY IMPROVEMENT: Measuring Jacobians on CALIBRATION data")
    print("  Math: null-space => cal_act_diff=0, but cal_jac_diff>0")
    print("  Because: DJ = W_down @ diag(u) @ diag(SiLU'(g)) @ DeltaW @ dh/dx")
    print("  DeltaW appears directly in chain rule (not as DeltaW @ h)")

    t_start = time.time()

    # ── Load model and data ──
    print("\n--- Loading model ---")
    model, tokenizer = load_model(cfg)
    device = cfg.model.device

    print("\n--- Loading prompts ---")
    all_prompts = load_prompts(cfg, tokenizer)
    n_total = len(all_prompts)
    n_cal = n_total // 2
    n_test = n_total - n_cal
    cal_prompts = all_prompts[:n_cal]
    test_prompts = all_prompts[n_cal:]
    print(f"Split: {n_cal} calibration, {n_test} test prompts")

    layer_idx = cfg.experiment.target_layer
    sublayer_path = cfg.experiment.target_sublayer
    layer = model.model.layers[layer_idx]
    seed = cfg.experiment.seed

    # JVP batching: chunk_size for vmap parallelism
    chunk_size = cfg.experiment.get("jvp_chunk_size", None)
    if chunk_size:
        print(f"\nJVP batching: chunk_size={chunk_size} (vmap parallelism)")
    else:
        print("\nJVP batching: sequential (no vmap)")

    # Output directory
    output_dir = cfg.output.dir
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    raw_data = {}

    # ════════════════════════════════════════════════════════════════
    # Validation: Compare batched vs sequential on first prompt
    # ════════════════════════════════════════════════════════════════
    if chunk_size:
        print("\n--- Validating batched JVP against sequential ---")
        from src.jvp_engine import compute_full_jacobian_at_last_token
        from src.layer_wrapper import collect_hidden_states

        val_prompt = cal_prompts[0]
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

        # Cleanup
        del jac_seq, jac_bat, val_hs
        torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════
    # Phase 0: Baseline measurements (clean model on BOTH cal + test)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 0: Baseline (clean model)")
    print("=" * 60)

    print("  Collecting CALIBRATION activations and Jacobians...")
    t0 = time.time()
    cal_acts_clean, cal_jacs_clean = collect_activations_and_jacobians(
        model, cal_prompts, layer_idx, device, label="[cal] ",
        chunk_size=chunk_size
    )
    print(f"  Cal baseline done in {time.time()-t0:.1f}s")

    print("  Collecting TEST activations and Jacobians...")
    t0 = time.time()
    test_acts_clean, test_jacs_clean = collect_activations_and_jacobians(
        model, test_prompts, layer_idx, device, label="[test] ",
        chunk_size=chunk_size
    )
    print(f"  Test baseline done in {time.time()-t0:.1f}s")

    # ════════════════════════════════════════════════════════════════
    # Phase 1: Noise Floor Verification
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 1: Noise Floor Verification")
    print("=" * 60)

    n_check = min(5, n_cal)
    check_prompts = cal_prompts[:n_check]

    print(f"  Re-running {n_check} prompts to verify determinism...")
    check_acts, check_jacs = collect_activations_and_jacobians(
        model, check_prompts, layer_idx, device, label="[noise] ",
        chunk_size=chunk_size
    )

    noise_act_diffs = []
    noise_jac_diffs = []
    for i in range(n_check):
        a1 = cal_acts_clean[layer_idx][i].float()
        a2 = check_acts[layer_idx][i].float()
        act_diff = (a1 - a2).norm().item() / (a1.norm().item() + 1e-10)
        noise_act_diffs.append(act_diff)

        j1 = cal_jacs_clean[layer_idx][i].float()
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
    # Phase 2: Gaussian sublayer perturbation (FAIR control)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 2: Gaussian sublayer perturbation (fair control)")
    print(f"  Perturbing ONLY {sublayer_path} (same as null-space)")
    print("=" * 60)

    for eps in cfg.experiment.epsilons:
        condition = f"gaussian_sublayer_eps{eps}"
        print(f"\n  --- {condition} ---")

        saved = perturb_gaussian_sublayer(
            layer, eps, seed, sublayer_path=sublayer_path
        )

        # Calibration data
        print("  Collecting perturbed CAL activations + Jacobians...")
        t2 = time.time()
        cal_acts_pert, cal_jacs_pert = collect_activations_and_jacobians(
            model, cal_prompts, layer_idx, device, label="[cal] ",
            chunk_size=chunk_size
        )
        print(f"  Cal done in {time.time()-t2:.1f}s")

        # Test data
        print("  Collecting perturbed TEST activations + Jacobians...")
        t2 = time.time()
        test_acts_pert, test_jacs_pert = collect_activations_and_jacobians(
            model, test_prompts, layer_idx, device, label="[test] ",
            chunk_size=chunk_size
        )
        print(f"  Test done in {time.time()-t2:.1f}s")

        restore_weights(layer, saved)

        # Compute diffs
        cal_act_diffs = compute_activation_diff(cal_acts_clean, cal_acts_pert)
        cal_jac_diffs = compute_jacobian_frobenius_diff(cal_jacs_clean, cal_jacs_pert)
        test_act_diffs = compute_activation_diff(test_acts_clean, test_acts_pert)
        test_jac_diffs = compute_jacobian_frobenius_diff(test_jacs_clean, test_jacs_pert)

        result = analyze_condition_v2(
            cal_act_diffs[layer_idx], cal_jac_diffs[layer_idx],
            test_act_diffs[layer_idx], test_jac_diffs[layer_idx],
            condition, noise_floor_act, noise_floor_jac
        )
        all_results.append(result)
        print_condition_summary(result)

        if cfg.output.save_raw:
            raw_data[condition] = {
                "cal_act_diffs": cal_act_diffs[layer_idx].tolist(),
                "cal_jac_diffs": cal_jac_diffs[layer_idx].tolist(),
                "test_act_diffs": test_act_diffs[layer_idx].tolist(),
                "test_jac_diffs": test_jac_diffs[layer_idx].tolist(),
            }

    # ════════════════════════════════════════════════════════════════
    # Phase 3: Null-space perturbation (MAIN EXPERIMENT)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 3: Null-space perturbation (MAIN EXPERIMENT)")
    print("  KEY: Measuring Jacobians on cal data where act_diff ~ 0")
    print("=" * 60)

    for eps in cfg.experiment.epsilons:
        condition = f"null_space_eps{eps}"
        print(f"\n  --- {condition} ---")

        # Perturb (uses calibration prompts for null-space computation)
        saved = perturb_null_space(
            layer, eps, seed,
            model=model, prompts=cal_prompts,
            layer_idx=layer_idx, sublayer_path=sublayer_path,
            device=device,
        )

        # Calibration data — THE KEY MEASUREMENT
        # Math: act_diff should be ~ 0 (null-space guarantee)
        #       jac_diff should be > 0 (chain rule exposes DeltaW)
        print("  Collecting perturbed CAL activations + Jacobians (KEY)...")
        t3 = time.time()
        cal_acts_pert, cal_jacs_pert = collect_activations_and_jacobians(
            model, cal_prompts, layer_idx, device, label="[cal] ",
            chunk_size=chunk_size
        )
        print(f"  Cal done in {time.time()-t3:.1f}s")

        # Test data
        print("  Collecting perturbed TEST activations + Jacobians...")
        t3 = time.time()
        test_acts_pert, test_jacs_pert = collect_activations_and_jacobians(
            model, test_prompts, layer_idx, device, label="[test] ",
            chunk_size=chunk_size
        )
        print(f"  Test done in {time.time()-t3:.1f}s")

        restore_weights(layer, saved)

        # Compute diffs
        cal_act_diffs = compute_activation_diff(cal_acts_clean, cal_acts_pert)
        cal_jac_diffs = compute_jacobian_frobenius_diff(cal_jacs_clean, cal_jacs_pert)
        test_act_diffs = compute_activation_diff(test_acts_clean, test_acts_pert)
        test_jac_diffs = compute_jacobian_frobenius_diff(test_jacs_clean, test_jacs_pert)

        result = analyze_condition_v2(
            cal_act_diffs[layer_idx], cal_jac_diffs[layer_idx],
            test_act_diffs[layer_idx], test_jac_diffs[layer_idx],
            condition, noise_floor_act, noise_floor_jac
        )
        all_results.append(result)
        print_condition_summary(result)

        if cfg.output.save_raw:
            raw_data[condition] = {
                "cal_act_diffs": cal_act_diffs[layer_idx].tolist(),
                "cal_jac_diffs": cal_jac_diffs[layer_idx].tolist(),
                "test_act_diffs": test_act_diffs[layer_idx].tolist(),
                "test_jac_diffs": test_jac_diffs[layer_idx].tolist(),
            }

    # ════════════════════════════════════════════════════════════════
    # Phase 4: Summary and Decision
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 4: SUMMARY")
    print("=" * 80)
    print(format_results_table_v2(all_results))

    # ── NIAC Detection Analysis ──
    print("\n--- NIAC Detection Analysis ---")
    print("  Question: Does null-space reduce Jacobian signal less than activation signal?")
    print("  Metric: NIAC Amplification Factor = (ns_jac/ns_act) / (gauss_jac/gauss_act)")

    niac_decisions = []

    for eps in cfg.experiment.epsilons:
        gauss_r = next(
            (r for r in all_results if r["condition"] == f"gaussian_sublayer_eps{eps}"), None
        )
        ns_r = next(
            (r for r in all_results if r["condition"] == f"null_space_eps{eps}"), None
        )

        if gauss_r and ns_r:
            print(f"\n  epsilon={eps}:")

            # Cal data comparison (THE KEY)
            print(f"    CAL DATA (null-space guarantee zone):")
            print(f"      Gaussian: act={gauss_r['cal_act_mean']:.6f}, "
                  f"jac={gauss_r['cal_jac_mean']:.6f}, "
                  f"jac/act={gauss_r['cal_jac_act_ratio']:.1f}x")
            print(f"      Null-sp:  act={ns_r['cal_act_mean']:.6f}, "
                  f"jac={ns_r['cal_jac_mean']:.6f}, "
                  f"jac/act={ns_r['cal_jac_act_ratio']:.1f}x")

            # NIAC Amplification Factor on cal data
            if gauss_r['cal_jac_act_ratio'] > 0:
                naf_cal = ns_r['cal_jac_act_ratio'] / gauss_r['cal_jac_act_ratio']
                print(f"      NIAC Amplification Factor (cal): {naf_cal:.1f}x")
            else:
                naf_cal = 0.0

            # Activation evasion success
            if gauss_r['cal_act_mean'] > 0:
                act_evasion = 1.0 - ns_r['cal_act_mean'] / gauss_r['cal_act_mean']
                print(f"      Activation evasion: {act_evasion*100:.1f}% reduction")

            # Jacobian evasion failure
            if gauss_r['cal_jac_mean'] > 0:
                jac_evasion = 1.0 - ns_r['cal_jac_mean'] / gauss_r['cal_jac_mean']
                print(f"      Jacobian evasion:  {jac_evasion*100:.1f}% reduction")

            # Test data comparison
            print(f"    TEST DATA (null-space partial generalization):")
            print(f"      Gaussian: act={gauss_r['test_act_mean']:.6f}, "
                  f"jac={gauss_r['test_jac_mean']:.6f}")
            print(f"      Null-sp:  act={ns_r['test_act_mean']:.6f}, "
                  f"jac={ns_r['test_jac_mean']:.6f}")

            # Decision for this epsilon — NAF-based criteria
            # Two independent questions:
            #   1. Did null-space succeed at evading activation monitoring?
            #      Yes if ns_cal_act < 50% of gauss_cal_act (>50% reduction)
            #   2. Does Jacobian monitoring still detect the perturbation?
            #      Yes if t-test p < 0.001 (statistically significant)
            # Then grade by NAF:
            #   NAF >= 3.0 → strong NIAC detection
            #   NAF < 3.0  → weak but present NIAC detection
            act_evasion_success = (ns_r['cal_act_mean']
                                   < 0.5 * gauss_r['cal_act_mean'])
            jac_still_detects = ns_r['cal_jac_ttest_p'] < 0.001

            if act_evasion_success and jac_still_detects:
                if naf_cal >= 3.0:
                    decision = "NIAC-DETECTED"
                else:
                    decision = "WEAK-NIAC"
            elif not act_evasion_success and jac_still_detects:
                decision = "BOTH-DETECT"
            elif act_evasion_success and not jac_still_detects:
                decision = "EVASION-SUCCESS"
            else:
                decision = "INCONCLUSIVE"

            print(f"    act_evasion_success={act_evasion_success}, "
                  f"jac_still_detects={jac_still_detects}, NAF={naf_cal:.1f}x")
            print(f"    DECISION: {decision}")
            niac_decisions.append((eps, decision, naf_cal))

    # ── Overall Verdict ──
    print("\n" + "=" * 60)
    n_strong = sum(1 for _, d, _ in niac_decisions if d == "NIAC-DETECTED")
    n_weak = sum(1 for _, d, _ in niac_decisions if d == "WEAK-NIAC")
    n_detected = n_strong + n_weak  # Any form of NIAC detection
    n_total = len(niac_decisions)

    if n_detected == n_total and n_total > 0:
        if n_strong == n_total:
            overall = ("GO -- Jacobian monitoring detects null-space evasion "
                       "that activation monitoring misses (all strong)")
        else:
            overall = (f"GO -- NIAC detected in all conditions "
                       f"({n_strong} strong, {n_weak} weak)")
    elif n_detected > 0:
        overall = (f"PARTIAL -- NIAC detected in {n_detected}/{n_total} "
                   f"conditions ({n_strong} strong, {n_weak} weak)")
    else:
        overall = "NO-GO -- Jacobian fails to detect null-space evasion"

    print(f"  NIAC detected in {n_detected}/{n_total} conditions "
          f"({n_strong} strong, {n_weak} weak)")
    for eps, decision, naf in niac_decisions:
        print(f"    eps={eps}: {decision} (NAF={naf:.1f}x)")
    print(f"\n  OVERALL DECISION: {overall}")

    total_time = time.time() - t_start
    print(f"\n  Total runtime: {total_time:.1f}s ({total_time/60:.1f}min)")

    # ── Save results ──
    output = {
        "results": all_results,
        "raw_data": raw_data if cfg.output.save_raw else {},
        "noise_floor": {
            "act": noise_floor_act,
            "jac": noise_floor_jac,
            "act_diffs": noise_act_diffs,
            "jac_diffs": noise_jac_diffs,
        },
        "niac_decisions": [
            {"epsilon": eps, "decision": d, "naf": naf}
            for eps, d, naf in niac_decisions
        ],
        "_meta": {
            "model": cfg.model.name,
            "target_layer": layer_idx,
            "target_sublayer": sublayer_path,
            "epsilons": list(cfg.experiment.epsilons),
            "n_cal": n_cal,
            "n_test": n_test,
            "seed": seed,
            "overall_decision": overall,
            "runtime_seconds": total_time,
            "date": datetime.now().isoformat(),
            "version": "v2 -- with cal Jacobians, sublayer gaussian, noise floor",
        }
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
