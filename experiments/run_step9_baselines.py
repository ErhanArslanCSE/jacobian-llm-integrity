#!/usr/bin/env python3
"""Step 9: Baseline Method Comparison

Compares 5 detection methods head-to-head on identical data:

  1. Weight SHA-256 Hash    — binary change detection, requires weight access
  2. L2 Weight Diff         — ||W'-W||/||W||, requires weight access
  3. Activation Monitoring  — relative output change per prompt
  4. Tap Test (k=8)         — k random JVP probes, inference-only
  5. Full Jacobian          — d JVP calls, inference-only (L12 validation only)

Key message:
  Weight access? → Hash/L2 trivially sufficient.
  Inference only? → Activation monitoring BLIND to null-space.
                    Tap test DETECTS null-space (our contribution).

Phases:
  0: Load model, data, probes; compute clean weight hashes + clones
  1: Clean baselines (activations + taps for all prompts, full Jac for L12)
  2: Noise floor verification
  3: Gaussian perturbation (15 conditions)
  4: Null-space perturbation (15 conditions)
  5: Cross-method NIAC analysis
  6: Paper tables
  7: Save results

Usage:
  python run_step9_baselines.py --config configs/baselines.yaml
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
from src.layer_wrapper import collect_layer_outputs, collect_hidden_states
from src.jvp_engine import compute_all_tap_jvps, compute_all_jacobians
from src.perturbations import (
    save_weights, restore_weights,
    perturb_null_space, perturb_gaussian_sublayer,
)
from src.metrics import (
    compute_activation_diff,
    compute_jacobian_frobenius_diff,
    analyze_condition_v2,
)
from src.baselines import (
    compute_weight_hash,
    save_sublayer_weight,
    compute_l2_weight_diff,
)


# ════════════════════════════════════════════════════════════════
# Inline helpers (reused from exp7/8)
# ════════════════════════════════════════════════════════════════

def generate_random_probes(d, k_max, seed):
    """Generate k_max orthonormal random probe directions via QR decomposition."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    A = torch.randn(d, k_max, generator=rng)
    V, _ = torch.linalg.qr(A)
    return V


def collect_activations_and_taps(model, prompts, layer_indices, probe_matrix,
                                 device, label="", chunk_size=None):
    """Collect layer output activations and tap responses for each prompt."""
    activations = {idx: [] for idx in layer_indices}
    taps = {idx: [] for idx in layer_indices}

    for i, prompt in enumerate(prompts):
        input_ids = prompt["input_ids"].to(device)
        attention_mask = prompt["attention_mask"].to(device)
        last_tok = attention_mask.sum(dim=1).long().item() - 1

        output_dict = collect_layer_outputs(model, input_ids, attention_mask, layer_indices)
        for idx in layer_indices:
            act = output_dict[idx][0, last_tok, :].cpu()
            activations[idx].append(act)

        tap_dict = compute_all_tap_jvps(model, prompt, layer_indices, probe_matrix,
                                        device, chunk_size=chunk_size)
        for idx in layer_indices:
            taps[idx].append(tap_dict[idx])

        if (i + 1) % 10 == 0:
            print(f"    {label}Prompt {i+1}/{len(prompts)} done")

    return activations, taps


def collect_activations_only(model, prompts, layer_indices, device, label=""):
    """Collect layer output activations only (no taps). Cheaper than full collect."""
    activations = {idx: [] for idx in layer_indices}

    for i, prompt in enumerate(prompts):
        input_ids = prompt["input_ids"].to(device)
        attention_mask = prompt["attention_mask"].to(device)
        last_tok = attention_mask.sum(dim=1).long().item() - 1

        output_dict = collect_layer_outputs(model, input_ids, attention_mask, layer_indices)
        for idx in layer_indices:
            act = output_dict[idx][0, last_tok, :].cpu()
            activations[idx].append(act)

        if (i + 1) % 10 == 0:
            print(f"    {label}Prompt {i+1}/{len(prompts)} done")

    return activations


def compute_tap_diffs_direct(clean_taps_list, pert_taps_list):
    """Compute per-prompt relative tap response difference."""
    diffs = []
    for r_clean, r_pert in zip(clean_taps_list, pert_taps_list):
        diff_norm = torch.norm(r_pert.float() - r_clean.float(), p='fro').item()
        base_norm = torch.norm(r_clean.float(), p='fro').item()
        diffs.append(diff_norm / (base_norm + 1e-10))
    return np.array(diffs)


def niac_decision_for_condition(gauss_result, ns_result, eps):
    """NIAC decision logic. Returns dict with decision + details."""
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


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Step 9: Baseline Method Comparison")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    t_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output directory
    out_dir = cfg.output.dir
    os.makedirs(out_dir, exist_ok=True)

    target_layers = list(cfg.experiment.target_layers)
    sublayer_path = cfg.experiment.target_sublayer
    epsilons = list(cfg.experiment.epsilons)
    n_cal = cfg.experiment.n_cal
    n_test = cfg.experiment.n_test
    seed = cfg.experiment.seed
    chunk_size = cfg.experiment.jvp_chunk_size
    device = cfg.model.device
    tap_k = cfg.baselines.tap_k

    # Full Jacobian validation config
    fjv_enabled = cfg.baselines.full_jac_validation.enabled
    fjv_layer = cfg.baselines.full_jac_validation.layer_idx
    fjv_n_prompts = cfg.baselines.full_jac_validation.n_prompts

    print("=" * 80)
    print("EXPERIMENT 9: BASELINE METHOD COMPARISON")
    print("=" * 80)
    print(f"  Model:           {cfg.model.name}")
    print(f"  Target layers:   {target_layers}")
    print(f"  Target sublayer: {sublayer_path}")
    print(f"  Epsilons:        {epsilons}")
    print(f"  n_cal / n_test:  {n_cal} / {n_test}")
    print(f"  Tap k:           {tap_k}")
    print(f"  Full Jac valid.: {fjv_enabled} (L{fjv_layer}, {fjv_n_prompts} prompts)")
    print(f"  Seed:            {seed}")
    print(f"  Output:          {out_dir}")
    print()

    # ════════════════════════════════════════════════════════
    # PHASE 0: Load model, data, probes + clean weight snapshots
    # ════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 0: Setup")
    print("=" * 60)
    t0 = time.time()

    model, tokenizer = load_model(cfg)
    prompts = load_prompts(cfg, tokenizer)
    print(f"  Loaded {len(prompts)} prompts")

    if len(prompts) < n_cal + n_test:
        raise RuntimeError(f"Need {n_cal + n_test} prompts but only got {len(prompts)}")

    cal_prompts = prompts[:n_cal]
    test_prompts = prompts[n_cal:n_cal + n_test]
    all_prompts = cal_prompts + test_prompts
    print(f"  Split: {n_cal} cal + {n_test} test = {len(all_prompts)} total")

    # Hidden dim
    layers = model.model.layers
    d = layers[0].self_attn.q_proj.weight.shape[0]
    print(f"  Hidden dim d = {d}")

    # Generate random probes
    probe_matrix = generate_random_probes(d, tap_k, seed)
    print(f"  Generated {tap_k} orthonormal random probes")

    # Clean weight hashes + clones for each target layer
    clean_hashes = {}
    clean_weights = {}
    for lidx in target_layers:
        clean_hashes[lidx] = compute_weight_hash(layers[lidx], sublayer_path)
        clean_weights[lidx] = save_sublayer_weight(layers[lidx], sublayer_path)
        print(f"  Layer {lidx}: hash={clean_hashes[lidx][:16]}..., "
              f"weight shape={clean_weights[lidx].shape}")

    dt0 = time.time() - t0
    print(f"  Phase 0 done in {dt0:.1f}s")
    print()

    # ════════════════════════════════════════════════════════
    # PHASE 1: Clean baselines
    # ════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 1: Clean baselines (activations + taps)")
    print("=" * 60)
    t1 = time.time()

    # Collect activations + taps for all prompts
    print(f"  Collecting activations + tap (k={tap_k}) for {len(all_prompts)} prompts × {len(target_layers)} layers...")
    all_acts_clean, all_taps_clean = collect_activations_and_taps(
        model, all_prompts, target_layers, probe_matrix,
        device, label="clean ", chunk_size=chunk_size
    )

    # Time activation-only cost (for cost table)
    t_act_start = time.time()
    _act_timing = collect_activations_only(model, all_prompts[:5], target_layers, device, label="act-timing ")
    t_act_5 = time.time() - t_act_start
    act_per_prompt_ms = (t_act_5 / 5) * 1000

    # Time tap cost
    t_tap_start = time.time()
    _tap_timing_acts, _tap_timing = collect_activations_and_taps(
        model, all_prompts[:5], target_layers, probe_matrix,
        device, label="tap-timing ", chunk_size=chunk_size
    )
    t_tap_5 = time.time() - t_tap_start
    tap_per_prompt_ms = (t_tap_5 / 5) * 1000

    # Full Jacobian timing + collection (L12 only, fjv_n_prompts prompts)
    full_jac_clean = {}
    fjac_per_prompt_ms = 0.0
    if fjv_enabled:
        print(f"  Full Jacobian at L{fjv_layer}: {fjv_n_prompts} prompts...")
        t_fjac_start = time.time()
        for pi in range(fjv_n_prompts):
            jacs = compute_all_jacobians(model, all_prompts[pi], [fjv_layer],
                                         device, chunk_size=chunk_size)
            if fjv_layer not in full_jac_clean:
                full_jac_clean[fjv_layer] = []
            full_jac_clean[fjv_layer].append(jacs[fjv_layer])
        t_fjac = time.time() - t_fjac_start
        fjac_per_prompt_ms = (t_fjac / fjv_n_prompts) * 1000
        print(f"    Full Jac: {t_fjac:.1f}s total, {fjac_per_prompt_ms:.0f}ms/prompt")

    # Weight hash/L2 timing (100 reps for stable measurement)
    t_hash_start = time.time()
    for _ in range(100):
        compute_weight_hash(layers[target_layers[0]], sublayer_path)
    hash_time_ms = (time.time() - t_hash_start) / 100 * 1000

    t_l2_start = time.time()
    for _ in range(100):
        compute_l2_weight_diff(clean_weights[target_layers[0]], layers[target_layers[0]], sublayer_path)
    l2_time_ms = (time.time() - t_l2_start) / 100 * 1000

    dt1 = time.time() - t1
    print(f"\n  Timing summary (per prompt, {len(target_layers)} layers):")
    print(f"    Weight Hash:      {hash_time_ms:.3f} ms")
    print(f"    L2 Weight Diff:   {l2_time_ms:.3f} ms")
    print(f"    Activation Mon.:  {act_per_prompt_ms:.1f} ms")
    print(f"    Tap Test (k={tap_k}):  {tap_per_prompt_ms:.1f} ms")
    if fjv_enabled:
        print(f"    Full Jacobian:    {fjac_per_prompt_ms:.0f} ms")
    print(f"  Phase 1 done in {dt1:.1f}s")
    print()

    # ════════════════════════════════════════════════════════
    # PHASE 2: Noise floor verification
    # ════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 2: Noise floor verification")
    print("=" * 60)
    t2 = time.time()

    noise_prompts = all_prompts[:5]
    acts_v2, taps_v2 = collect_activations_and_taps(
        model, noise_prompts, target_layers, probe_matrix,
        device, label="noise-floor ", chunk_size=chunk_size
    )

    # Compare with clean baseline (should be identical)
    noise_floor_act = {}
    noise_floor_tap = {}
    for lidx in target_layers:
        act_diffs_nf = []
        tap_diffs_nf = []
        for pi in range(len(noise_prompts)):
            ad = (acts_v2[lidx][pi].float() - all_acts_clean[lidx][pi].float()).norm().item()
            an = all_acts_clean[lidx][pi].float().norm().item()
            act_diffs_nf.append(ad / (an + 1e-10))

            td = torch.norm(taps_v2[lidx][pi].float() - all_taps_clean[lidx][pi].float(), p='fro').item()
            tn = torch.norm(all_taps_clean[lidx][pi].float(), p='fro').item()
            tap_diffs_nf.append(td / (tn + 1e-10))

        noise_floor_act[lidx] = float(np.mean(act_diffs_nf))
        noise_floor_tap[lidx] = float(np.mean(tap_diffs_nf))
        print(f"  L{lidx}: act_noise={noise_floor_act[lidx]:.2e}, tap_noise={noise_floor_tap[lidx]:.2e}")

    dt2 = time.time() - t2
    print(f"  Phase 2 done in {dt2:.1f}s")
    print()

    # ════════════════════════════════════════════════════════
    # PHASE 3: Gaussian perturbation (15 conditions)
    # ════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 3: Gaussian sublayer perturbation")
    print("=" * 60)
    t3 = time.time()

    gauss_acts = {}
    gauss_taps = {}
    gauss_hashes = {}
    gauss_l2 = {}
    gauss_full_jacs = {}

    for lidx in target_layers:
        for eps in epsilons:
            key = (lidx, eps)
            cond_name = f"gauss_L{lidx}_eps{eps}"
            print(f"\n  --- {cond_name} ---")

            # Save, perturb, measure
            saved = save_weights(layers[lidx])
            perturb_gaussian_sublayer(layers[lidx], eps, seed, sublayer_path)

            # M1: Weight hash
            gauss_hashes[key] = compute_weight_hash(layers[lidx], sublayer_path)
            hash_changed = gauss_hashes[key] != clean_hashes[lidx]
            print(f"    Hash changed: {hash_changed}")

            # M2: L2 weight diff
            gauss_l2[key] = compute_l2_weight_diff(clean_weights[lidx], layers[lidx], sublayer_path)
            print(f"    L2 diff: {gauss_l2[key]:.6f} (expected ~{eps})")

            # M3 + M4: Activations + Taps
            g_acts, g_taps = collect_activations_and_taps(
                model, all_prompts, target_layers, probe_matrix,
                device, label=f"{cond_name} ", chunk_size=chunk_size
            )
            gauss_acts[key] = {lidx: g_acts[lidx]}
            gauss_taps[key] = {lidx: g_taps[lidx]}

            # M5: Full Jacobian (fjv_layer only)
            if fjv_enabled and lidx == fjv_layer:
                print(f"    Full Jacobian at L{fjv_layer}: {fjv_n_prompts} prompts...")
                gauss_full_jacs[key] = {fjv_layer: []}
                for pi in range(fjv_n_prompts):
                    jacs = compute_all_jacobians(model, all_prompts[pi], [fjv_layer],
                                                 device, chunk_size=chunk_size)
                    gauss_full_jacs[key][fjv_layer].append(jacs[fjv_layer])

            # Restore
            restore_weights(layers[lidx], saved)

            # Verify restoration
            restored_hash = compute_weight_hash(layers[lidx], sublayer_path)
            assert restored_hash == clean_hashes[lidx], \
                f"Weight restoration failed at L{lidx}!"

    dt3 = time.time() - t3
    print(f"\n  Phase 3 done in {dt3:.1f}s ({dt3/60:.1f} min)")
    print()

    # ════════════════════════════════════════════════════════
    # PHASE 4: Null-space perturbation (15 conditions)
    # ════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 4: Null-space perturbation")
    print("=" * 60)
    t4 = time.time()

    ns_acts = {}
    ns_taps = {}
    ns_hashes = {}
    ns_l2 = {}
    ns_full_jacs = {}

    for lidx in target_layers:
        for eps in epsilons:
            key = (lidx, eps)
            cond_name = f"ns_L{lidx}_eps{eps}"
            print(f"\n  --- {cond_name} ---")

            # Save, perturb with null-space
            saved = save_weights(layers[lidx])
            perturb_null_space(
                layers[lidx], eps, seed,
                model=model, prompts=cal_prompts,
                layer_idx=lidx, sublayer_path=sublayer_path,
                device=device
            )

            # M1: Weight hash
            ns_hashes[key] = compute_weight_hash(layers[lidx], sublayer_path)
            hash_changed = ns_hashes[key] != clean_hashes[lidx]
            print(f"    Hash changed: {hash_changed}")

            # M2: L2 weight diff
            ns_l2[key] = compute_l2_weight_diff(clean_weights[lidx], layers[lidx], sublayer_path)
            print(f"    L2 diff: {ns_l2[key]:.6f} (expected ~{eps})")

            # M3 + M4: Activations + Taps
            n_acts, n_taps = collect_activations_and_taps(
                model, all_prompts, target_layers, probe_matrix,
                device, label=f"{cond_name} ", chunk_size=chunk_size
            )
            ns_acts[key] = {lidx: n_acts[lidx]}
            ns_taps[key] = {lidx: n_taps[lidx]}

            # M5: Full Jacobian (fjv_layer only)
            if fjv_enabled and lidx == fjv_layer:
                print(f"    Full Jacobian at L{fjv_layer}: {fjv_n_prompts} prompts...")
                ns_full_jacs[key] = {fjv_layer: []}
                for pi in range(fjv_n_prompts):
                    jacs = compute_all_jacobians(model, all_prompts[pi], [fjv_layer],
                                                 device, chunk_size=chunk_size)
                    ns_full_jacs[key][fjv_layer].append(jacs[fjv_layer])

            # Restore
            restore_weights(layers[lidx], saved)

            # Verify restoration
            restored_hash = compute_weight_hash(layers[lidx], sublayer_path)
            assert restored_hash == clean_hashes[lidx], \
                f"Weight restoration failed at L{lidx}!"

    dt4 = time.time() - t4
    print(f"\n  Phase 4 done in {dt4:.1f}s ({dt4/60:.1f} min)")
    print()

    # ════════════════════════════════════════════════════════
    # PHASE 5: Cross-method NIAC analysis
    # ════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 5: Cross-method NIAC analysis")
    print("=" * 60)
    t5 = time.time()

    all_results = {}  # (lidx, eps, attack, method) -> result_dict

    for lidx in target_layers:
        for eps in epsilons:
            key = (lidx, eps)

            for attack, pert_acts_store, pert_taps_store, pert_jacs_store in [
                ("gaussian", gauss_acts, gauss_taps, gauss_full_jacs),
                ("null_space", ns_acts, ns_taps, ns_full_jacs),
            ]:
                # ── M1: Hash ──
                if attack == "gaussian":
                    hash_val = gauss_hashes[key]
                else:
                    hash_val = ns_hashes[key]
                hash_detected = hash_val != clean_hashes[lidx]
                all_results[(lidx, eps, attack, "hash")] = {
                    "detected": hash_detected,
                    "hash_clean": clean_hashes[lidx][:16],
                    "hash_pert": hash_val[:16],
                }

                # ── M2: L2 ──
                if attack == "gaussian":
                    l2_val = gauss_l2[key]
                else:
                    l2_val = ns_l2[key]
                all_results[(lidx, eps, attack, "l2")] = {
                    "l2_diff": l2_val,
                    "detected": l2_val > 1e-6,
                }

                # ── M3: Activation ──
                pert_act_dict = pert_acts_store[key]
                cal_act_diffs = compute_activation_diff(
                    {lidx: all_acts_clean[lidx][:n_cal]},
                    {lidx: pert_act_dict[lidx][:n_cal]}
                )[lidx]
                test_act_diffs = compute_activation_diff(
                    {lidx: all_acts_clean[lidx][n_cal:]},
                    {lidx: pert_act_dict[lidx][n_cal:]}
                )[lidx]

                # ── M4: Tap (k=tap_k) ──
                pert_tap_dict = pert_taps_store[key]
                cal_tap_diffs = compute_tap_diffs_direct(
                    all_taps_clean[lidx][:n_cal],
                    pert_tap_dict[lidx][:n_cal]
                )
                test_tap_diffs = compute_tap_diffs_direct(
                    all_taps_clean[lidx][n_cal:],
                    pert_tap_dict[lidx][n_cal:]
                )

                # analyze_condition_v2 uses "jac" for whatever probe-based signal
                tap_result = analyze_condition_v2(
                    cal_act_diffs, cal_tap_diffs,
                    test_act_diffs, test_tap_diffs,
                    condition_name=f"tap_k{tap_k}_{attack}_L{lidx}_eps{eps}"
                )
                all_results[(lidx, eps, attack, "tap")] = tap_result

                # ── M5: Full Jacobian (L12 only) ──
                if fjv_enabled and lidx == fjv_layer and key in pert_jacs_store:
                    cal_jac_diffs_full = compute_jacobian_frobenius_diff(
                        {fjv_layer: full_jac_clean[fjv_layer][:fjv_n_prompts]},
                        {fjv_layer: pert_jacs_store[key][fjv_layer][:fjv_n_prompts]}
                    )[fjv_layer]
                    # Use cal-only (no separate test set for full jac)
                    fjac_result = analyze_condition_v2(
                        cal_act_diffs[:fjv_n_prompts], cal_jac_diffs_full,
                        test_act_diffs[:fjv_n_prompts], cal_jac_diffs_full,
                        condition_name=f"fulljac_{attack}_L{lidx}_eps{eps}"
                    )
                    all_results[(lidx, eps, attack, "full_jac")] = fjac_result

    # ── NIAC decisions (tap) ──
    niac_decisions = {}
    for lidx in target_layers:
        for eps in epsilons:
            g_tap = all_results[(lidx, eps, "gaussian", "tap")]
            n_tap = all_results[(lidx, eps, "null_space", "tap")]
            dec = niac_decision_for_condition(g_tap, n_tap, eps)
            dec["layer"] = lidx
            niac_decisions[(lidx, eps)] = dec
            print(f"  L{lidx} eps={eps}: {dec['decision']} "
                  f"(NAF={dec['naf']:.1f}x, act_evasion={dec['act_evasion_pct']:.0f}%)")

    # ── NIAC decisions (full jac, L12 only) ──
    niac_decisions_fjac = {}
    if fjv_enabled:
        for eps in epsilons:
            g_key = (fjv_layer, eps, "gaussian", "full_jac")
            n_key = (fjv_layer, eps, "null_space", "full_jac")
            if g_key in all_results and n_key in all_results:
                dec = niac_decision_for_condition(all_results[g_key], all_results[n_key], eps)
                dec["layer"] = fjv_layer
                niac_decisions_fjac[(fjv_layer, eps)] = dec
                print(f"  L{fjv_layer} eps={eps} (full jac): {dec['decision']} "
                      f"(NAF={dec['naf']:.1f}x)")

    dt5 = time.time() - t5
    print(f"\n  Phase 5 done in {dt5:.1f}s")
    print()

    # ════════════════════════════════════════════════════════
    # PHASE 6: Paper tables
    # ════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 6: Paper tables")
    print("=" * 60)

    # ── Table A: Method Properties ──
    n_params = sum(p.numel() for p in model.parameters())
    table_a = [
        {"method": "Weight Hash (SHA-256)", "access": "Weight", "cost": f"O(P), {hash_time_ms:.2f}ms",
         "online": False, "detects_all": True, "note": "Requires weight access"},
        {"method": "L2 Weight Diff", "access": "Weight", "cost": f"O(P), {l2_time_ms:.2f}ms",
         "online": False, "detects_all": True, "note": "Requires weight access"},
        {"method": "Activation Monitoring", "access": "Inference", "cost": f"O(1), {act_per_prompt_ms:.1f}ms",
         "online": True, "detects_all": False, "note": "BLIND to null-space"},
        {"method": f"Tap Test (k={tap_k})", "access": "Inference", "cost": f"O(k), {tap_per_prompt_ms:.1f}ms",
         "online": True, "detects_all": True, "note": ""},
    ]
    if fjv_enabled:
        table_a.append(
            {"method": "Full Jacobian", "access": "Inference", "cost": f"O(d^2), {fjac_per_prompt_ms:.0f}ms",
             "online": True, "detects_all": True, "note": f"{d/tap_k:.0f}x more expensive than tap"}
        )

    print("\n  TABLE A: Method Properties")
    print(f"  {'Method':<25} {'Access':<10} {'Cost':<20} {'Online':<8} {'Detects All':<12}")
    print(f"  {'-'*75}")
    for row in table_a:
        print(f"  {row['method']:<25} {row['access']:<10} {row['cost']:<20} "
              f"{'Yes' if row['online'] else 'No':<8} {'Yes' if row['detects_all'] else 'NO*':<12}")

    # ── Table B: Detection Signal (Cohen's d, reference layer) ──
    ref_layer = fjv_layer if fjv_enabled else target_layers[len(target_layers) // 2]
    table_b_rows = []
    for eps in epsilons:
        row = {"eps": eps}

        # Gaussian
        g_tap = all_results[(ref_layer, eps, "gaussian", "tap")]
        row["gauss_act_d"] = g_tap.get("cal_act_cohens_d", 0)
        row["gauss_tap_d"] = g_tap.get("cal_jac_cohens_d", 0)
        row["gauss_l2"] = gauss_l2[(ref_layer, eps)]
        row["gauss_hash"] = "CHANGED"

        # Null-space
        n_tap = all_results[(ref_layer, eps, "null_space", "tap")]
        row["ns_act_d"] = n_tap.get("cal_act_cohens_d", 0)
        row["ns_tap_d"] = n_tap.get("cal_jac_cohens_d", 0)
        row["ns_l2"] = ns_l2[(ref_layer, eps)]
        row["ns_hash"] = "CHANGED"

        table_b_rows.append(row)

    print(f"\n  TABLE B: Detection Signal at L{ref_layer} (Cohen's d)")
    print(f"  {'eps':<8} | {'Gauss Act':>10} {'Gauss Tap':>10} {'Gauss L2':>10} | "
          f"{'NS Act':>10} {'NS Tap':>10} {'NS L2':>10}")
    print(f"  {'-'*75}")
    for row in table_b_rows:
        print(f"  {row['eps']:<8} | "
              f"{row['gauss_act_d']:>10.1f} {row['gauss_tap_d']:>10.1f} {row['gauss_l2']:>10.4f} | "
              f"{row['ns_act_d']:>10.2f} {row['ns_tap_d']:>10.1f} {row['ns_l2']:>10.4f}")

    # ── Table C: Detection Matrix (null-space, all 15 conditions) ──
    print(f"\n  TABLE C: Null-Space Detection Matrix")
    print(f"  {'Layer':<8}", end="")
    for eps in epsilons:
        print(f" | eps={eps:<18}", end="")
    print()
    print(f"  {'-'*75}")
    for lidx in target_layers:
        print(f"  L{lidx:<6}", end="")
        for eps in epsilons:
            dec = niac_decisions[(lidx, eps)]
            g_tap = all_results[(lidx, eps, "gaussian", "tap")]
            n_tap = all_results[(lidx, eps, "null_space", "tap")]
            act_det = "Y" if n_tap["cal_act_ttest_p"] < 0.001 and n_tap["cal_act_mean"] > 0.5 * g_tap["cal_act_mean"] else "N"
            tap_det = "Y" if n_tap["cal_jac_ttest_p"] < 0.001 else "N"
            cell = f"Act:{act_det} Tap:{tap_det} ({dec['decision'][:4]})"
            print(f" | {cell:<18}", end="")
        print()

    # ── Table D: Tap vs Full Jac at reference layer ──
    if fjv_enabled:
        print(f"\n  TABLE D: Tap (k={tap_k}) vs Full Jacobian at L{fjv_layer}")
        print(f"  {'eps':<8} | {'Tap NAF':>10} {'FJac NAF':>10} {'Tap dec':>15} {'FJac dec':>15}")
        print(f"  {'-'*65}")
        for eps in epsilons:
            tap_dec = niac_decisions.get((fjv_layer, eps), {})
            fjac_dec = niac_decisions_fjac.get((fjv_layer, eps), {})
            print(f"  {eps:<8} | "
                  f"{tap_dec.get('naf', 0):>10.1f} "
                  f"{fjac_dec.get('naf', 0):>10.1f} "
                  f"{tap_dec.get('decision', 'N/A'):>15} "
                  f"{fjac_dec.get('decision', 'N/A'):>15}")

    print()

    # ════════════════════════════════════════════════════════
    # PHASE 7: Save results
    # ════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 7: Save results")
    print("=" * 60)
    t7 = time.time()

    runtime = time.time() - t_start

    # ── Serializable results ──
    def to_serializable(obj):
        """Make numpy/torch types JSON-serializable."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    # Build per-condition results
    conditions = []
    for lidx in target_layers:
        for eps in epsilons:
            key = (lidx, eps)
            cond = {
                "layer": lidx,
                "epsilon": eps,
            }

            for attack in ["gaussian", "null_space"]:
                prefix = "gauss" if attack == "gaussian" else "ns"

                # Hash
                h_res = all_results[(lidx, eps, attack, "hash")]
                cond[f"{prefix}_hash_detected"] = h_res["detected"]

                # L2
                l2_res = all_results[(lidx, eps, attack, "l2")]
                cond[f"{prefix}_l2_diff"] = to_serializable(l2_res["l2_diff"])

                # Tap
                tap_res = all_results[(lidx, eps, attack, "tap")]
                for k_tap in ["cal_act_mean", "cal_act_std", "cal_act_ttest_p",
                              "cal_jac_mean", "cal_jac_std", "cal_jac_ttest_p",
                              "cal_jac_act_ratio", "cal_act_cohens_d", "cal_jac_cohens_d",
                              "test_act_mean", "test_act_std", "test_act_ttest_p",
                              "test_jac_mean", "test_jac_std", "test_jac_ttest_p",
                              "test_jac_act_ratio"]:
                    if k_tap in tap_res:
                        cond[f"{prefix}_tap_{k_tap}"] = to_serializable(tap_res[k_tap])

                # Full Jac (L12 only)
                fjac_key = (lidx, eps, attack, "full_jac")
                if fjac_key in all_results:
                    fjac_res = all_results[fjac_key]
                    for k_fj in ["cal_jac_mean", "cal_jac_std", "cal_jac_ttest_p",
                                 "cal_jac_cohens_d", "cal_jac_act_ratio"]:
                        if k_fj in fjac_res:
                            cond[f"{prefix}_fjac_{k_fj}"] = to_serializable(fjac_res[k_fj])

            # NIAC decision
            dec = niac_decisions[(lidx, eps)]
            cond["niac_decision"] = dec["decision"]
            cond["niac_naf"] = to_serializable(dec["naf"])
            cond["niac_act_evasion_pct"] = to_serializable(dec["act_evasion_pct"])

            conditions.append(cond)

    # Meta
    meta = {
        "experiment": "step9_baselines",
        "timestamp": timestamp,
        "model": cfg.model.name,
        "hidden_dim": d,
        "n_params": n_params,
        "target_layers": target_layers,
        "target_sublayer": sublayer_path,
        "epsilons": epsilons,
        "n_cal": n_cal,
        "n_test": n_test,
        "tap_k": tap_k,
        "seed": seed,
        "runtime_seconds": runtime,
        "phase_times": {
            "phase0_setup": dt0,
            "phase1_baselines": dt1,
            "phase2_noise_floor": dt2,
            "phase3_gaussian": dt3,
            "phase4_null_space": dt4,
            "phase5_analysis": dt5,
        },
    }

    # Cost analysis
    cost_analysis = {
        "weight_hash_ms": hash_time_ms,
        "l2_diff_ms": l2_time_ms,
        "activation_per_prompt_ms": act_per_prompt_ms,
        "tap_per_prompt_ms": tap_per_prompt_ms,
        "full_jac_per_prompt_ms": fjac_per_prompt_ms if fjv_enabled else None,
        "tap_to_act_ratio": tap_per_prompt_ms / (act_per_prompt_ms + 1e-10),
        "fjac_to_tap_ratio": (fjac_per_prompt_ms / (tap_per_prompt_ms + 1e-10)) if fjv_enabled else None,
        "fjac_to_act_ratio": (fjac_per_prompt_ms / (act_per_prompt_ms + 1e-10)) if fjv_enabled else None,
    }

    # Noise floor
    noise_floor = {
        lidx: {"act": noise_floor_act[lidx], "tap": noise_floor_tap[lidx]}
        for lidx in target_layers
    }

    # Paper tables
    paper_tables = {
        "table_a_method_properties": table_a,
        "table_b_detection_signal": table_b_rows,
        "table_c_detection_matrix": [
            {
                "layer": lidx,
                "eps": eps,
                "act_detects_ns": all_results[(lidx, eps, "null_space", "tap")]["cal_act_ttest_p"] < 0.001
                    and all_results[(lidx, eps, "null_space", "tap")]["cal_act_mean"]
                    > 0.5 * all_results[(lidx, eps, "gaussian", "tap")]["cal_act_mean"],
                "tap_detects_ns": all_results[(lidx, eps, "null_space", "tap")]["cal_jac_ttest_p"] < 0.001,
                "decision": niac_decisions[(lidx, eps)]["decision"],
                "naf": to_serializable(niac_decisions[(lidx, eps)]["naf"]),
            }
            for lidx in target_layers for eps in epsilons
        ],
    }

    # Table D (if fjv enabled)
    if fjv_enabled:
        paper_tables["table_d_tap_vs_fjac"] = [
            {
                "layer": fjv_layer,
                "eps": eps,
                "tap_naf": to_serializable(niac_decisions.get((fjv_layer, eps), {}).get("naf", 0)),
                "tap_decision": niac_decisions.get((fjv_layer, eps), {}).get("decision", "N/A"),
                "fjac_naf": to_serializable(niac_decisions_fjac.get((fjv_layer, eps), {}).get("naf", 0)),
                "fjac_decision": niac_decisions_fjac.get((fjv_layer, eps), {}).get("decision", "N/A"),
            }
            for eps in epsilons
        ]

    # NIAC summary
    niac_summary = [
        {
            "layer": lidx,
            "epsilon": eps,
            "decision": niac_decisions[(lidx, eps)]["decision"],
            "naf": to_serializable(niac_decisions[(lidx, eps)]["naf"]),
            "act_evasion_pct": to_serializable(niac_decisions[(lidx, eps)]["act_evasion_pct"]),
        }
        for lidx in target_layers for eps in epsilons
    ]

    # ── Write files ──
    results_full = {
        "_meta": meta,
        "conditions": conditions,
        "niac_summary": niac_summary,
        "cost_analysis": cost_analysis,
        "noise_floor": {str(k): v for k, v in noise_floor.items()},
        "paper_tables": paper_tables,
    }
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_full, f, indent=2, default=to_serializable)
    print(f"  Saved: {results_path}")

    # Summary
    results_summary = {
        "_meta": meta,
        "niac_summary": niac_summary,
        "cost_analysis": cost_analysis,
        "paper_tables": paper_tables,
    }
    summary_path = os.path.join(out_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2, default=to_serializable)
    print(f"  Saved: {summary_path}")

    # Comparison table (5-method × 30-condition grid)
    comparison_grid = []
    for lidx in target_layers:
        for eps in epsilons:
            for attack in ["gaussian", "null_space"]:
                row = {
                    "layer": lidx,
                    "epsilon": eps,
                    "attack": attack,
                    "hash_detected": all_results[(lidx, eps, attack, "hash")]["detected"],
                    "l2_diff": to_serializable(all_results[(lidx, eps, attack, "l2")]["l2_diff"]),
                    "act_cal_mean": to_serializable(all_results[(lidx, eps, attack, "tap")]["cal_act_mean"]),
                    "act_cal_p": to_serializable(all_results[(lidx, eps, attack, "tap")]["cal_act_ttest_p"]),
                    "act_cohens_d": to_serializable(all_results[(lidx, eps, attack, "tap")].get("cal_act_cohens_d", 0)),
                    "tap_cal_mean": to_serializable(all_results[(lidx, eps, attack, "tap")]["cal_jac_mean"]),
                    "tap_cal_p": to_serializable(all_results[(lidx, eps, attack, "tap")]["cal_jac_ttest_p"]),
                    "tap_cohens_d": to_serializable(all_results[(lidx, eps, attack, "tap")].get("cal_jac_cohens_d", 0)),
                }
                # Full Jac if available
                fjac_key = (lidx, eps, attack, "full_jac")
                if fjac_key in all_results:
                    row["fjac_cal_mean"] = to_serializable(all_results[fjac_key]["cal_jac_mean"])
                    row["fjac_cal_p"] = to_serializable(all_results[fjac_key]["cal_jac_ttest_p"])
                    row["fjac_cohens_d"] = to_serializable(all_results[fjac_key].get("cal_jac_cohens_d", 0))
                comparison_grid.append(row)

    comparison_path = os.path.join(out_dir, "comparison_table.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison_grid, f, indent=2, default=to_serializable)
    print(f"  Saved: {comparison_path}")

    # Cost analysis
    cost_path = os.path.join(out_dir, "cost_analysis.json")
    with open(cost_path, "w") as f:
        json.dump(cost_analysis, f, indent=2, default=to_serializable)
    print(f"  Saved: {cost_path}")

    # Paper tables
    paper_path = os.path.join(out_dir, "paper_tables.json")
    with open(paper_path, "w") as f:
        json.dump(paper_tables, f, indent=2, default=to_serializable)
    print(f"  Saved: {paper_path}")

    dt7 = time.time() - t7
    print(f"  Phase 7 done in {dt7:.1f}s")

    # ════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════
    runtime = time.time() - t_start
    print()
    print("=" * 80)
    print("EXPERIMENT 9 COMPLETE")
    print("=" * 80)
    print(f"  Runtime: {runtime:.0f}s ({runtime/60:.1f} min)")
    print(f"  Output:  {out_dir}")
    print()

    # Summary of NIAC decisions
    n_niac = sum(1 for d in niac_decisions.values() if "NIAC" in d["decision"])
    n_both = sum(1 for d in niac_decisions.values() if d["decision"] == "BOTH-DETECT")
    n_total = len(niac_decisions)
    print(f"  NIAC summary: {n_niac} NIAC-DETECTED, {n_both} BOTH-DETECT, {n_total} total")

    # Key finding
    print()
    print("  KEY FINDINGS:")
    print(f"    1. Hash/L2: Always detect (trivially) — require weight access")
    any_act_detects_ns = False
    for lidx in target_layers:
        for eps in epsilons:
            n_tap = all_results[(lidx, eps, "null_space", "tap")]
            g_tap = all_results[(lidx, eps, "gaussian", "tap")]
            if n_tap["cal_act_ttest_p"] < 0.001 and n_tap["cal_act_mean"] > 0.5 * g_tap["cal_act_mean"]:
                any_act_detects_ns = True
    print(f"    2. Activation: {'FAILS' if not any_act_detects_ns else 'DETECTS'} "
          f"null-space on calibration data")
    print(f"    3. Tap test (k={tap_k}): {n_niac}/{n_total} NIAC-DETECTED")
    mean_naf = np.mean([d["naf"] for d in niac_decisions.values()])
    print(f"    4. Mean NAF: {mean_naf:.1f}x")
    print(f"    5. Cost: tap={tap_per_prompt_ms:.0f}ms vs act={act_per_prompt_ms:.0f}ms "
          f"({tap_per_prompt_ms/act_per_prompt_ms:.1f}x overhead)")
    print()


if __name__ == "__main__":
    main()
