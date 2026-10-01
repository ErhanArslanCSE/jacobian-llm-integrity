#!/usr/bin/env python3
"""Step 4: Multi-Layer NIAC Detection — Does the Jacobian advantage generalize
across different layers?

Core question: Exp2/Exp3 showed Jacobian monitoring detects null-space
perturbations at layer 12. Does this hold at early, mid, and late layers?
Are some layers more vulnerable to NIAC than others?

Design:
  - Fixed n_cal=30, n_test=30 (exp3 proved n_cal doesn't matter)
  - 5 target layers: [2, 6, 10, 12, 14] spanning early to late
  - For each target layer: perturb ONLY that layer, measure at that layer
  - Same sublayer (mlp.gate_proj) across all layers for fair comparison

Expected paper figure: X = layer_index, Y = NAF, lines = epsilon values.

Usage:
  python run_step4_multi_layer.py --config configs/multi_layer.yaml
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


def collect_activations_and_jacobians(model, prompts, layer_indices, device,
                                      label="", chunk_size=None):
    """Collect layer output activations and full Jacobians for each prompt.

    Supports multiple layers simultaneously — one forward pass captures all
    target layers' outputs, one forward pass collects all hidden states for
    Jacobian computation.

    Args:
        model: The transformer model
        prompts: List of prompt dicts
        layer_indices: List of layer indices to measure
        device: torch device
        label: Progress label
        chunk_size: JVP batching chunk size

    Returns:
        activations: dict[layer_idx -> list of (hidden_dim,) tensors]
        jacobians: dict[layer_idx -> list of (d, d) tensors]
    """
    activations = {idx: [] for idx in layer_indices}
    jacobians = {idx: [] for idx in layer_indices}

    for i, prompt in enumerate(prompts):
        input_ids = prompt["input_ids"].to(device)
        attention_mask = prompt["attention_mask"].to(device)
        last_tok = attention_mask.sum(dim=1).long().item() - 1

        output_dict = collect_layer_outputs(model, input_ids, attention_mask, layer_indices)
        for idx in layer_indices:
            act = output_dict[idx][0, last_tok, :].cpu()
            activations[idx].append(act)

        jac_dict = compute_all_jacobians(model, prompt, layer_indices, device,
                                          chunk_size=chunk_size)
        for idx in layer_indices:
            jacobians[idx].append(jac_dict[idx])

        if (i + 1) % 10 == 0:
            print(f"    {label}Prompt {i+1}/{len(prompts)} done")

    return activations, jacobians


def niac_decision_for_condition(gauss_result, ns_result, eps):
    """Run NIAC decision logic for one (layer, eps) combination.

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
    parser.add_argument("--config", type=str, default="configs/multi_layer.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    target_layers = list(cfg.experiment.target_layers)
    n_cal = cfg.experiment.n_cal
    n_test = cfg.experiment.n_test
    epsilons = list(cfg.experiment.epsilons)
    sublayer_path = cfg.experiment.target_sublayer
    seed = cfg.experiment.seed

    print("=" * 80)
    print("STEP 4: Multi-Layer NIAC Detection")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Model: {cfg.model.name}")
    print(f"Target layers: {target_layers}")
    print(f"Target sublayer: {sublayer_path}")
    print(f"Epsilons: {epsilons}")
    print(f"n_cal: {n_cal}, n_test: {n_test}")
    print()
    print("KEY QUESTION: Does Jacobian detection advantage generalize across layers?")
    print("  Expected: NAF > 1 at all layers (structural property of chain rule)")

    t_start = time.time()

    # ── Load model and data ──
    print("\n--- Loading model ---")
    model, tokenizer = load_model(cfg)
    device = cfg.model.device

    print("\n--- Loading prompts ---")
    all_prompts = load_prompts(cfg, tokenizer)
    total_needed = n_cal + n_test
    if len(all_prompts) < total_needed:
        raise ValueError(
            f"Need {total_needed} prompts ({n_cal} cal + {n_test} test) "
            f"but only got {len(all_prompts)}. Increase data.num_prompts."
        )

    cal_prompts = all_prompts[:n_cal]
    test_prompts = all_prompts[n_cal:n_cal + n_test]
    all_60 = all_prompts[:n_cal + n_test]
    print(f"Split: cal={len(cal_prompts)}, test={len(test_prompts)}")

    chunk_size = cfg.experiment.get("jvp_chunk_size", None)
    if chunk_size:
        print(f"\nJVP batching: chunk_size={chunk_size} (vmap parallelism)")
    else:
        print("\nJVP batching: sequential (no vmap)")

    output_dir = cfg.output.dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Log layer properties ──
    print("\n--- Layer Properties ---")
    layer_properties = {}
    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        sublayer = layer
        for part in sublayer_path.split("."):
            sublayer = getattr(sublayer, part)
        W = sublayer.weight.data
        w_norm = W.float().norm().item()
        d_out, d_in = W.shape
        layer_properties[layer_idx] = {
            "weight_norm": w_norm,
            "weight_shape": [d_out, d_in],
        }
        print(f"  Layer {layer_idx}: {sublayer_path} shape={list(W.shape)}, "
              f"||W||_F={w_norm:.2f}")

    # ════════════════════════════════════════════════════════════════
    # JVP Validation (using first target layer)
    # ════════════════════════════════════════════════════════════════
    if chunk_size:
        print("\n--- Validating batched JVP against sequential ---")
        from src.jvp_engine import compute_full_jacobian_at_last_token
        from src.layer_wrapper import collect_hidden_states

        val_layer_idx = target_layers[0]
        val_prompt = cal_prompts[0]
        val_ids = val_prompt["input_ids"].to(device)
        val_mask = val_prompt["attention_mask"].to(device)
        val_last_tok = val_mask.sum(dim=1).long().item() - 1

        val_hidden, val_kwargs = collect_hidden_states(
            model, val_ids, val_mask, [val_layer_idx]
        )
        val_hs = val_hidden[val_layer_idx].to(device)
        val_kw = val_kwargs[val_layer_idx]
        val_layer = model.model.layers[val_layer_idx]

        t_seq = time.time()
        jac_seq = compute_full_jacobian_at_last_token(
            val_layer, val_hs, val_kw, val_last_tok, device, chunk_size=None
        )
        t_seq = time.time() - t_seq

        t_bat = time.time()
        jac_bat = compute_full_jacobian_at_last_token(
            val_layer, val_hs, val_kw, val_last_tok, device, chunk_size=chunk_size
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
    # Phase 0: Baseline — collect ALL layers simultaneously
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"PHASE 0: Baseline (clean model, {len(all_60)} prompts, "
          f"{len(target_layers)} layers)")
    print("=" * 60)

    t0 = time.time()
    all_acts_clean, all_jacs_clean = collect_activations_and_jacobians(
        model, all_60, target_layers, device, label="[baseline] ",
        chunk_size=chunk_size
    )
    print(f"  Baseline done in {time.time()-t0:.1f}s")

    # ════════════════════════════════════════════════════════════════
    # Phase 1: Noise Floor Verification (all layers)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 1: Noise Floor Verification (all layers)")
    print("=" * 60)

    n_check = 5
    check_prompts = cal_prompts[:n_check]
    check_acts, check_jacs = collect_activations_and_jacobians(
        model, check_prompts, target_layers, device, label="[noise] ",
        chunk_size=chunk_size
    )

    noise_floors = {}
    for layer_idx in target_layers:
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
        print(f"  Layer {layer_idx}: noise_act_max={noise_act_max:.2e}, "
              f"noise_jac_max={noise_jac_max:.2e}")

        if noise_act_max < 1e-10 and noise_jac_max < 1e-10:
            noise_floors[layer_idx] = {
                "act": 0.0, "jac": 0.0,
                "act_diffs": noise_act_diffs, "jac_diffs": noise_jac_diffs
            }
        else:
            print(f"    WARNING: Non-zero noise floor at layer {layer_idx}!")
            noise_floors[layer_idx] = {
                "act": noise_act_max, "jac": noise_jac_max,
                "act_diffs": noise_act_diffs, "jac_diffs": noise_jac_diffs
            }

    all_zero = all(nf["act"] == 0.0 and nf["jac"] == 0.0
                   for nf in noise_floors.values())
    if all_zero:
        print("  CONFIRMED: All layers deterministic (noise floor = 0)")
    else:
        print("  WARNING: Some layers have non-zero noise floor!")

    # ════════════════════════════════════════════════════════════════
    # Phase 2: Gaussian sublayer perturbation — per layer × eps
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 2: Gaussian sublayer perturbation (per layer × eps)")
    print(f"  Perturbing ONLY {sublayer_path} at each target layer")
    print("=" * 60)

    gauss_results = {}  # (layer_idx, eps) -> analyze_condition_v2 result

    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        print(f"\n  ── Layer {layer_idx} ──")

        for eps in epsilons:
            condition = f"gaussian_L{layer_idx}_eps{eps}"
            print(f"    {condition}...")

            saved = perturb_gaussian_sublayer(
                layer, eps, seed, sublayer_path=sublayer_path
            )

            t2 = time.time()
            pert_acts, pert_jacs = collect_activations_and_jacobians(
                model, all_60, [layer_idx], device,
                label=f"[gauss L{layer_idx} e{eps}] ",
                chunk_size=chunk_size
            )
            print(f"    Done in {time.time()-t2:.1f}s")

            restore_weights(layer, saved)

            # Compute diffs — cal (first n_cal) and test (next n_test)
            cal_act_diffs = compute_activation_diff(
                {layer_idx: all_acts_clean[layer_idx][:n_cal]},
                {layer_idx: pert_acts[layer_idx][:n_cal]}
            )
            cal_jac_diffs = compute_jacobian_frobenius_diff(
                {layer_idx: all_jacs_clean[layer_idx][:n_cal]},
                {layer_idx: pert_jacs[layer_idx][:n_cal]}
            )
            test_act_diffs = compute_activation_diff(
                {layer_idx: all_acts_clean[layer_idx][n_cal:]},
                {layer_idx: pert_acts[layer_idx][n_cal:]}
            )
            test_jac_diffs = compute_jacobian_frobenius_diff(
                {layer_idx: all_jacs_clean[layer_idx][n_cal:]},
                {layer_idx: pert_jacs[layer_idx][n_cal:]}
            )

            result = analyze_condition_v2(
                cal_act_diffs[layer_idx], cal_jac_diffs[layer_idx],
                test_act_diffs[layer_idx], test_jac_diffs[layer_idx],
                condition,
                noise_floors[layer_idx]["act"],
                noise_floors[layer_idx]["jac"]
            )
            gauss_results[(layer_idx, eps)] = result
            print(f"    cal_act={result['cal_act_mean']:.6f}, "
                  f"cal_jac={result['cal_jac_mean']:.6f}, "
                  f"j/a={result['cal_jac_act_ratio']:.1f}x")

    # ════════════════════════════════════════════════════════════════
    # Phase 3: Null-space perturbation — per layer × eps
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 3: Null-space perturbation (per layer × eps)")
    print("  Each layer gets its OWN null-space basis from its sublayer inputs")
    print("=" * 60)

    ns_results = {}  # (layer_idx, eps) -> analyze_condition_v2 result

    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        print(f"\n  ── Layer {layer_idx} ──")

        for eps in epsilons:
            condition = f"null_space_L{layer_idx}_eps{eps}"
            print(f"\n    {condition}...")

            saved = perturb_null_space(
                layer, eps, seed,
                model=model, prompts=cal_prompts,
                layer_idx=layer_idx, sublayer_path=sublayer_path,
                device=device,
            )

            # Collect perturbed on cal prompts
            t3 = time.time()
            ns_cal_acts, ns_cal_jacs = collect_activations_and_jacobians(
                model, cal_prompts, [layer_idx], device,
                label=f"[ns cal L{layer_idx} e{eps}] ",
                chunk_size=chunk_size
            )

            # Collect perturbed on test prompts
            ns_test_acts, ns_test_jacs = collect_activations_and_jacobians(
                model, test_prompts, [layer_idx], device,
                label=f"[ns test L{layer_idx} e{eps}] ",
                chunk_size=chunk_size
            )
            print(f"    Done in {time.time()-t3:.1f}s")

            restore_weights(layer, saved)

            # Compute diffs
            cal_act_diffs = compute_activation_diff(
                {layer_idx: all_acts_clean[layer_idx][:n_cal]},
                ns_cal_acts
            )
            cal_jac_diffs = compute_jacobian_frobenius_diff(
                {layer_idx: all_jacs_clean[layer_idx][:n_cal]},
                ns_cal_jacs
            )
            test_act_diffs = compute_activation_diff(
                {layer_idx: all_acts_clean[layer_idx][n_cal:]},
                ns_test_acts
            )
            test_jac_diffs = compute_jacobian_frobenius_diff(
                {layer_idx: all_jacs_clean[layer_idx][n_cal:]},
                ns_test_jacs
            )

            result = analyze_condition_v2(
                cal_act_diffs[layer_idx], cal_jac_diffs[layer_idx],
                test_act_diffs[layer_idx], test_jac_diffs[layer_idx],
                condition,
                noise_floors[layer_idx]["act"],
                noise_floors[layer_idx]["jac"]
            )
            ns_results[(layer_idx, eps)] = result
            print(f"    cal_act={result['cal_act_mean']:.6f}, "
                  f"cal_jac={result['cal_jac_mean']:.6f}, "
                  f"j/a={result['cal_jac_act_ratio']:.1f}x")

            # Log null-space dimension from layer_properties
            # (perturb_null_space already prints rank/null_dim)

    # ════════════════════════════════════════════════════════════════
    # Phase 4: Analysis — Layer × Epsilon matrix
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 4: MULTI-LAYER SUMMARY")
    print("=" * 80)

    layer_summary = []
    layer_results = {}

    for layer_idx in target_layers:
        layer_key = f"layer_{layer_idx}"
        gauss_for_layer = []
        ns_for_layer = []
        decisions_for_layer = []

        for eps in epsilons:
            g_result = gauss_results[(layer_idx, eps)]
            n_result = ns_results[(layer_idx, eps)]
            gauss_for_layer.append(g_result)
            ns_for_layer.append(n_result)

            dec = niac_decision_for_condition(g_result, n_result, eps)
            dec["layer"] = layer_idx
            decisions_for_layer.append(dec)
            layer_summary.append(dec)

        layer_results[layer_key] = {
            "gaussian": gauss_for_layer,
            "null_space": ns_for_layer,
            "niac_decisions": decisions_for_layer,
        }

    # ── Print Layer × Epsilon NAF table ──
    print(f"\n{'Layer':>6} | ", end="")
    for eps in epsilons:
        print(f"eps={eps:5.2f} NAF | eps={eps:5.2f} dec      | ", end="")
    print()
    print("─" * (8 + len(epsilons) * 38))

    for layer_idx in target_layers:
        print(f"{layer_idx:>6} | ", end="")
        for eps in epsilons:
            entry = next(
                (s for s in layer_summary
                 if s['layer'] == layer_idx and s['epsilon'] == eps),
                None
            )
            if entry:
                print(f"  {entry['naf']:>6.1f}x | {entry['decision']:<16s} | ", end="")
            else:
                print(f"{'N/A':>8s} | {'N/A':<16s} | ", end="")
        print()

    # ── Per-layer summary ──
    print("\n--- Per-Layer Analysis ---")
    for layer_idx in target_layers:
        entries = [s for s in layer_summary if s['layer'] == layer_idx]
        w_norm = layer_properties[layer_idx]["weight_norm"]
        print(f"\n  Layer {layer_idx} (||W||_F={w_norm:.2f}):")
        for e in entries:
            print(f"    eps={e['epsilon']}: NAF={e['naf']:.1f}x, "
                  f"evasion={e['act_evasion_pct']:.1f}%, "
                  f"jac_detects={e['jac_still_detects']}, "
                  f"decision={e['decision']}")

    # ── Per-epsilon summary ──
    print("\n--- Per-Epsilon Analysis ---")
    for eps in epsilons:
        entries = [s for s in layer_summary if s['epsilon'] == eps]
        nafs = [e['naf'] for e in entries]
        print(f"\n  epsilon = {eps}:")
        print(f"    NAF range: {min(nafs):.1f}x – {max(nafs):.1f}x "
              f"(mean={np.mean(nafs):.1f}x, std={np.std(nafs):.1f}x)")
        for e in entries:
            print(f"    Layer {e['layer']:>2d}: NAF={e['naf']:.1f}x, "
                  f"decision={e['decision']}")

    # ── Gaussian j/a ratio across layers ──
    print("\n--- Gaussian J/A Ratio Across Layers ---")
    print("  (This shows baseline Jacobian sensitivity per layer)")
    for layer_idx in target_layers:
        ratios = []
        for eps in epsilons:
            r = gauss_results[(layer_idx, eps)]['cal_jac_act_ratio']
            ratios.append(r)
        print(f"  Layer {layer_idx}: " +
              ", ".join(f"eps={e}→{r:.1f}x" for e, r in zip(epsilons, ratios)))

    # ── Layer 12 consistency check ──
    print("\n--- Layer 12 Consistency Check (vs exp2/exp3) ---")
    print("  Expected: NAF ≈ 1.8x (eps=0.01), ~6.5x (eps=0.05), ~12.9x (eps=0.1)")
    for eps in epsilons:
        entry = next(
            (s for s in layer_summary if s['layer'] == 12 and s['epsilon'] == eps),
            None
        )
        if entry:
            print(f"  eps={eps}: NAF={entry['naf']:.1f}x, decision={entry['decision']}")

    # ── Overall verdict ──
    print("\n" + "=" * 60)
    total_conditions = len(layer_summary)
    n_strong = sum(1 for s in layer_summary if s['decision'] == 'NIAC-DETECTED')
    n_weak = sum(1 for s in layer_summary if s['decision'] == 'WEAK-NIAC')
    n_detected = n_strong + n_weak
    n_evasion = sum(1 for s in layer_summary if s['decision'] == 'EVASION-SUCCESS')

    print(f"  Total conditions: {total_conditions} ({len(target_layers)} layers × "
          f"{len(epsilons)} eps)")
    print(f"  NIAC detected: {n_detected}/{total_conditions} "
          f"({n_strong} strong, {n_weak} weak)")
    print(f"  Evasion success: {n_evasion}/{total_conditions}")

    # Check if ALL layers show NIAC at eps>=0.05
    mid_high_eps = [s for s in layer_summary if s['epsilon'] >= 0.05]
    all_detect_mid = all(s['decision'] in ('NIAC-DETECTED', 'WEAK-NIAC')
                         for s in mid_high_eps)
    if all_detect_mid:
        print(f"\n  STRONG RESULT: All {len(target_layers)} layers show NIAC "
              f"detection at eps >= 0.05")
    else:
        failed = [s for s in mid_high_eps
                  if s['decision'] not in ('NIAC-DETECTED', 'WEAK-NIAC')]
        print(f"\n  MIXED RESULT: {len(failed)} conditions failed at eps >= 0.05:")
        for f in failed:
            print(f"    Layer {f['layer']}, eps={f['epsilon']}: {f['decision']}")

    total_time = time.time() - t_start
    print(f"\n  Total runtime: {total_time:.1f}s ({total_time/60:.1f}min)")

    # ── Save results ──
    # Convert layer_properties keys to strings for JSON
    layer_props_json = {str(k): v for k, v in layer_properties.items()}
    noise_floors_json = {str(k): v for k, v in noise_floors.items()}

    output = {
        "layer_results": layer_results,
        "layer_summary": layer_summary,
        "layer_properties": layer_props_json,
        "noise_floors": noise_floors_json,
        "_meta": {
            "model": cfg.model.name,
            "target_layers": target_layers,
            "target_sublayer": sublayer_path,
            "epsilons": epsilons,
            "n_cal": n_cal,
            "n_test": n_test,
            "seed": seed,
            "runtime_seconds": total_time,
            "date": datetime.now().isoformat(),
            "version": "step4 -- multi-layer NIAC detection",
        }
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Also save a compact summary
    summary_path = os.path.join(output_dir, "results_summary.json")
    summary_output = {
        "layer_summary": layer_summary,
        "layer_properties": layer_props_json,
        "noise_floors": noise_floors_json,
        "_meta": output["_meta"],
    }
    with open(summary_path, "w") as f:
        json.dump(summary_output, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
