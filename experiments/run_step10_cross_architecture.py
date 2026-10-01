#!/usr/bin/env python3
"""Experiment 10: Cross-Architecture NIAC Detection

Validates that Jacobian-based NIAC detection is architecture-independent by testing
on Qwen2.5-7B and Mistral-7B-v0.3 (in addition to Llama 1B/3B/8B from exp8).

Uses the same tap test (k << d random probes) approach from exp8. Direct tap JVPs:
  - JVP(f, x, v_j) gives exact J @ v_j by definition
  - k JVP calls instead of d (NEVER store full Jacobian)

Design:
  - Phase 0: Load model, data, generate probes, JVP validation
  - Phase 1: Baseline (collect clean activations + tap responses)
  - Phase 2: Noise floor verification
  - Phase 3: Gaussian perturbation (per layer x eps)
  - Phase 4: Null-space perturbation (per layer x eps)
  - Phase 5: NIAC analysis (tap-based, per k)
  - Phase 6: Full Jacobian validation (3 prompts x 1 layer, optional)
  - Phase 7: FD validation (3 prompts x 8 probes x 1 layer)
  - Phase 8: Summary + save

Usage:
  python run_step10_cross_architecture.py --config configs/cross_arch_qwen25_7b.yaml
  python run_step10_cross_architecture.py --config configs/cross_arch_mistral_7b.yaml
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
    analyze_condition_v2,
)


# ════════════════════════════════════════════════════════════════
# Inline helpers
# ════════════════════════════════════════════════════════════════

def generate_random_probes(d, k_max, seed):
    """Generate k_max orthonormal random probe directions via QR decomposition.

    Returns (d, k_max) orthonormal matrix. V[:, :k] for any k <= k_max gives
    the first k probes (nested design: k=8 is a subset of k=16, etc).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    A = torch.randn(d, k_max, generator=rng)
    V, _ = torch.linalg.qr(A)  # (d, k_max) orthonormal columns
    return V


def collect_activations_and_taps(model, prompts, layer_indices, probe_matrix,
                                 device, label="", chunk_size=None):
    """Collect layer output activations and tap responses for each prompt.

    Uses compute_all_tap_jvps() to get (d, k) tap responses directly via
    k JVP calls — NEVER computes or stores the full d×d Jacobian.

    Returns:
        activations: dict[layer_idx -> list of (hidden_dim,) tensors]
        taps: dict[layer_idx -> list of (d, k) tensors]
    """
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


def compute_tap_diffs_direct(clean_taps_list, pert_taps_list):
    """Compute per-prompt relative tap response difference from pre-computed taps.

    For each prompt i:
      tap_diff_i = ||R_pert_i - R_clean_i||_F / ||R_clean_i||_F
    where R is the (d, k) tap response.

    Args:
        clean_taps_list: list of (d, k) tensors (clean tap responses)
        pert_taps_list: list of (d, k) tensors (perturbed tap responses)

    Returns:
        np.array of per-prompt relative tap diffs
    """
    diffs = []
    for r_clean, r_pert in zip(clean_taps_list, pert_taps_list):
        diff_norm = torch.norm(r_pert.float() - r_clean.float(), p='fro').item()
        base_norm = torch.norm(r_clean.float(), p='fro').item()
        diffs.append(diff_norm / (base_norm + 1e-10))
    return np.array(diffs)


def niac_decision_for_condition(gauss_result, ns_result, eps):
    """Run NIAC decision logic for one condition. Returns dict with decision details."""
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


def forward_with_perturbed_input(model, input_ids, attention_mask,
                                  layer_idx, perturbation, last_tok, device):
    """Run forward pass, adding perturbation to layer input at last_tok.

    Used for finite-difference validation of tap responses.
    """
    result = {}

    def pre_hook(module, args, kwargs):
        h = args[0].clone()
        h[0, last_tok, :] += perturbation
        return (h,) + args[1:], kwargs

    def post_hook(module, args, kwargs, output):
        out = output[0]
        if out.dim() == 2:
            out = out.unsqueeze(0)
        result['out'] = out[0, last_tok, :].detach().cpu().float()

    layer = model.model.layers[layer_idx]
    h1 = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = layer.register_forward_hook(post_hook, with_kwargs=True)

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    h1.remove()
    h2.remove()
    return result['out']


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    target_layers = list(cfg.experiment.target_layers)
    n_cal = cfg.experiment.n_cal
    n_test = cfg.experiment.n_test
    epsilons = list(cfg.experiment.epsilons)
    sublayer_path = cfg.experiment.target_sublayer
    seed = cfg.experiment.seed

    # Scale test config
    k_values = list(cfg.scale_test.k_values)
    k_max = cfg.scale_test.k_max
    full_jac_cfg = cfg.scale_test.full_jac_validation
    fd_cfg = cfg.scale_test.fd_validation

    print("=" * 80)
    print("EXPERIMENT 10: Cross-Architecture NIAC Detection")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Model: {cfg.model.name}")
    print(f"Target layers: {target_layers}")
    print(f"Target sublayer: {sublayer_path}")
    print(f"Epsilons: {epsilons}")
    print(f"n_cal: {n_cal}, n_test: {n_test}")
    print(f"k_values: {k_values}, k_max: {k_max}")
    print()
    print("KEY CHANGE: Direct tap JVPs (k calls) instead of full Jacobian (d calls)")
    print("  JVP(f, x, v_j) = J @ v_j -- exact, no approximation")

    t_start = time.time()

    # ════════════════════════════════════════════════════════════════
    # Phase 0: Load model, data, generate probes, JVP validation
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 0: Load model + data + probes + JVP validation")
    print("=" * 60)

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
    all_40 = all_prompts[:n_cal + n_test]
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
    hidden_dim = None
    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        sublayer = layer
        for part in sublayer_path.split("."):
            sublayer = getattr(sublayer, part)
        W = sublayer.weight.data
        w_norm = W.float().norm().item()
        d_out, d_in = W.shape
        if hidden_dim is None:
            hidden_dim = d_in
        layer_properties[layer_idx] = {
            "weight_norm": w_norm,
            "weight_shape": [d_out, d_in],
        }
        print(f"  Layer {layer_idx}: {sublayer_path} shape={list(W.shape)}, "
              f"||W||_F={w_norm:.2f}")

    d = hidden_dim
    print(f"\n  Hidden dim (d): {d}")
    print(f"  k_max: {k_max} ({k_max/d*100:.1f}% of d)")
    for k in k_values:
        print(f"  k={k}: {k/d*100:.1f}% of d, cost reduction: {d/k:.0f}x")

    # ── Generate random probes ──
    print("\n--- Generating random orthonormal probes ---")
    V_rand = generate_random_probes(d, k_max, seed)
    orth_err = (V_rand.T @ V_rand - torch.eye(k_max)).abs().max().item()
    print(f"  Random probes: shape={list(V_rand.shape)}, "
          f"orthogonality error={orth_err:.2e}")

    # ── JVP Validation: direct tap vs full Jacobian (1 prompt, 1 layer) ──
    print("\n--- JVP Validation: direct tap vs full Jacobian projection ---")
    val_layer_idx = target_layers[0]
    val_prompt = cal_prompts[0]

    t_jvp_val = time.time()
    # Method 1: Direct tap (k JVP calls)
    tap_direct = compute_all_tap_jvps(
        model, val_prompt, [val_layer_idx], V_rand,
        device, chunk_size=chunk_size
    )[val_layer_idx]  # (d, k_max)

    # Method 2: Full Jacobian then project
    full_jac_val = compute_all_jacobians(
        model, val_prompt, [val_layer_idx], device, chunk_size=chunk_size
    )[val_layer_idx]  # (d, d)
    tap_from_full = full_jac_val.float() @ V_rand.float()  # (d, k_max)

    rel_err = torch.norm(tap_direct.float() - tap_from_full).item() / (
        torch.norm(tap_from_full).item() + 1e-10)
    t_jvp_val = time.time() - t_jvp_val

    print(f"  Layer {val_layer_idx}: relative error = {rel_err:.2e} "
          f"(time: {t_jvp_val:.1f}s)")
    if rel_err < 0.01:
        print(f"  VALIDATED: Direct tap matches full Jacobian projection (err < 0.01)")
    else:
        print(f"  WARNING: High error ({rel_err:.4f}). Investigating...")

    del full_jac_val, tap_from_full
    torch.cuda.empty_cache()

    t0_elapsed = time.time() - t_start

    # ════════════════════════════════════════════════════════════════
    # Phase 1: Baseline — collect clean activations + tap responses
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"PHASE 1: Baseline (clean model, {len(all_40)} prompts, "
          f"{len(target_layers)} layers, k={k_max})")
    print("=" * 60)

    t1 = time.time()
    all_acts_clean, all_taps_clean = collect_activations_and_taps(
        model, all_40, target_layers, V_rand, device,
        label="[baseline] ", chunk_size=chunk_size
    )
    t1_elapsed = time.time() - t1
    print(f"  Baseline done in {t1_elapsed:.1f}s")
    print(f"  Memory per tap: {all_taps_clean[target_layers[0]][0].numel() * 4 / 1024:.1f} KB "
          f"(shape {list(all_taps_clean[target_layers[0]][0].shape)})")

    # ════════════════════════════════════════════════════════════════
    # Phase 2: Noise Floor Verification
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 2: Noise Floor Verification")
    print("=" * 60)

    n_check = 5
    check_prompts = cal_prompts[:n_check]
    check_acts, check_taps = collect_activations_and_taps(
        model, check_prompts, target_layers, V_rand, device,
        label="[noise] ", chunk_size=chunk_size
    )

    noise_floors = {}
    for layer_idx in target_layers:
        noise_act_diffs = []
        noise_tap_diffs = []
        for i in range(n_check):
            a1 = all_acts_clean[layer_idx][i].float()
            a2 = check_acts[layer_idx][i].float()
            act_diff = (a1 - a2).norm().item() / (a1.norm().item() + 1e-10)
            noise_act_diffs.append(act_diff)

            t1_tap = all_taps_clean[layer_idx][i].float()
            t2_tap = check_taps[layer_idx][i].float()
            tap_diff = (t1_tap - t2_tap).norm().item() / (t1_tap.norm().item() + 1e-10)
            noise_tap_diffs.append(tap_diff)

        noise_act_max = max(noise_act_diffs)
        noise_tap_max = max(noise_tap_diffs)
        print(f"  Layer {layer_idx}: noise_act_max={noise_act_max:.2e}, "
              f"noise_tap_max={noise_tap_max:.2e}")

        if noise_act_max < 1e-10 and noise_tap_max < 1e-10:
            noise_floors[layer_idx] = {"act": 0.0, "jac": 0.0}
        else:
            print(f"    WARNING: Non-zero noise floor at layer {layer_idx}!")
            noise_floors[layer_idx] = {"act": noise_act_max, "jac": noise_tap_max}

    all_zero = all(nf["act"] == 0.0 and nf["jac"] == 0.0
                   for nf in noise_floors.values())
    if all_zero:
        print("  CONFIRMED: All layers deterministic (noise floor = 0)")

    del check_acts, check_taps
    torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════
    # Phase 3: Gaussian sublayer perturbation — per layer × eps
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 3: Gaussian sublayer perturbation (per layer x eps)")
    print(f"  Perturbing ONLY {sublayer_path} at each target layer")
    print("=" * 60)

    gauss_acts = {}  # (layer_idx, eps) -> dict[layer_idx -> list of act tensors]
    gauss_taps = {}  # (layer_idx, eps) -> dict[layer_idx -> list of tap tensors]

    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        print(f"\n  -- Layer {layer_idx} --")

        for eps in epsilons:
            condition = f"gaussian_L{layer_idx}_eps{eps}"
            print(f"    {condition}...")

            saved = perturb_gaussian_sublayer(
                layer, eps, seed, sublayer_path=sublayer_path
            )

            t3 = time.time()
            pert_acts, pert_taps = collect_activations_and_taps(
                model, all_40, [layer_idx], V_rand, device,
                label=f"[gauss L{layer_idx} e{eps}] ",
                chunk_size=chunk_size
            )
            print(f"    Done in {time.time()-t3:.1f}s")

            restore_weights(layer, saved)

            gauss_acts[(layer_idx, eps)] = pert_acts
            gauss_taps[(layer_idx, eps)] = pert_taps

    # ════════════════════════════════════════════════════════════════
    # Phase 4: Null-space perturbation — per layer × eps
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 4: Null-space perturbation (per layer x eps)")
    print("  Each layer gets its OWN null-space basis from its sublayer inputs")
    print("=" * 60)

    ns_acts = {}
    ns_taps = {}

    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        print(f"\n  -- Layer {layer_idx} --")

        for eps in epsilons:
            condition = f"null_space_L{layer_idx}_eps{eps}"
            print(f"\n    {condition}...")

            saved = perturb_null_space(
                layer, eps, seed,
                model=model, prompts=cal_prompts,
                layer_idx=layer_idx, sublayer_path=sublayer_path,
                device=device,
            )

            t4 = time.time()
            pert_acts, pert_taps = collect_activations_and_taps(
                model, all_40, [layer_idx], V_rand, device,
                label=f"[ns L{layer_idx} e{eps}] ",
                chunk_size=chunk_size
            )
            print(f"    Done in {time.time()-t4:.1f}s")

            restore_weights(layer, saved)

            ns_acts[(layer_idx, eps)] = pert_acts
            ns_taps[(layer_idx, eps)] = pert_taps

    # ════════════════════════════════════════════════════════════════
    # Phase 5: NIAC Analysis (tap-based, per k)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 5: NIAC Analysis (tap-based)")
    print(f"  k_values: {k_values}")
    print(f"  Total conditions: {len(target_layers)} layers x {len(epsilons)} eps "
          f"x 2 attacks x {len(k_values)} k = "
          f"{len(target_layers)*len(epsilons)*2*len(k_values)}")
    print("=" * 60)

    t5_start = time.time()
    tap_results = {}  # (layer_idx, eps, attack, k) -> dict

    for layer_idx in target_layers:
        print(f"\n  -- Layer {layer_idx} --")
        for eps in epsilons:
            for attack in ["gaussian", "null_space"]:
                if attack == "gaussian":
                    pert_acts_dict = gauss_acts[(layer_idx, eps)]
                    pert_taps_dict = gauss_taps[(layer_idx, eps)]
                else:
                    pert_acts_dict = ns_acts[(layer_idx, eps)]
                    pert_taps_dict = ns_taps[(layer_idx, eps)]

                # Pre-compute activation diffs (same for all k)
                cal_act_diffs = compute_activation_diff(
                    {layer_idx: all_acts_clean[layer_idx][:n_cal]},
                    {layer_idx: pert_acts_dict[layer_idx][:n_cal]}
                )[layer_idx]
                test_act_diffs = compute_activation_diff(
                    {layer_idx: all_acts_clean[layer_idx][n_cal:]},
                    {layer_idx: pert_acts_dict[layer_idx][n_cal:]}
                )[layer_idx]

                for k in k_values:
                    # Slice taps to first k columns
                    cal_tap = compute_tap_diffs_direct(
                        [t[:, :k] for t in all_taps_clean[layer_idx][:n_cal]],
                        [t[:, :k] for t in pert_taps_dict[layer_idx][:n_cal]]
                    )
                    test_tap = compute_tap_diffs_direct(
                        [t[:, :k] for t in all_taps_clean[layer_idx][n_cal:]],
                        [t[:, :k] for t in pert_taps_dict[layer_idx][n_cal:]]
                    )

                    condition = f"tap_k{k}_{attack}_L{layer_idx}_eps{eps}"
                    result = analyze_condition_v2(
                        cal_act_diffs, cal_tap,
                        test_act_diffs, test_tap,
                        condition,
                        noise_floors[layer_idx]["act"],
                        noise_floors[layer_idx]["jac"]
                    )

                    tap_results[(layer_idx, eps, attack, k)] = result

            # Compute NIAC decisions for each k
            for k in k_values:
                g_tap = tap_results[(layer_idx, eps, "gaussian", k)]
                n_tap = tap_results[(layer_idx, eps, "null_space", k)]
                dec = niac_decision_for_condition(g_tap, n_tap, eps)
                dec["layer"] = layer_idx
                dec["k"] = k

                tap_results[(layer_idx, eps, "null_space", k)]["_niac_decision"] = dec

        # Print progress
        print(f"    Layer {layer_idx} done. NAFs at eps=0.1:")
        nafs = []
        for k in k_values:
            key = (layer_idx, 0.1, "null_space", k)
            if key in tap_results and "_niac_decision" in tap_results[key]:
                nafs.append(f"k={k}:{tap_results[key]['_niac_decision']['naf']:.1f}x")
        if nafs:
            print(f"      {', '.join(nafs)}")

    t5_elapsed = time.time() - t5_start
    print(f"\n  NIAC analysis done in {t5_elapsed:.1f}s")

    # ════════════════════════════════════════════════════════════════
    # Phase 6: Full Jacobian Validation (optional, 3 prompts × 1 layer)
    # ════════════════════════════════════════════════════════════════
    full_jac_validation = None
    if full_jac_cfg.enabled:
        print("\n" + "=" * 60)
        print("PHASE 6: Full Jacobian Validation")
        print(f"  Computing full {d}x{d} Jacobian for {full_jac_cfg.n_prompts} prompts "
              f"at layer {full_jac_cfg.layer_idx}")
        print(f"  This validates direct tap == full Jac projection")
        print("=" * 60)

        t6_start = time.time()
        val_layer = full_jac_cfg.layer_idx
        val_errors = []

        for i in range(min(full_jac_cfg.n_prompts, len(cal_prompts))):
            prompt = cal_prompts[i]
            t_full = time.time()

            # Full Jacobian
            full_jac = compute_all_jacobians(
                model, prompt, [val_layer], device,
                chunk_size=cfg.experiment.get("jvp_chunk_size", None)
            )[val_layer]  # (d, d)
            t_full = time.time() - t_full

            # Project from full Jacobian
            R_from_full = full_jac.float() @ V_rand.float()  # (d, k_max)

            # Direct tap (already computed in Phase 1)
            R_direct = all_taps_clean[val_layer][i]  # (d, k_max)

            rel_err = torch.norm(R_from_full - R_direct.float()).item() / (
                torch.norm(R_from_full).item() + 1e-10)
            val_errors.append(rel_err)
            print(f"    Prompt {i}: rel_err={rel_err:.2e} "
                  f"(full Jac: {t_full:.1f}s, shape {list(full_jac.shape)})")

            del full_jac, R_from_full

        torch.cuda.empty_cache()
        t6_elapsed = time.time() - t6_start

        full_jac_validation = {
            "layer_idx": val_layer,
            "n_prompts": len(val_errors),
            "errors": [float(e) for e in val_errors],
            "mean_error": float(np.mean(val_errors)),
            "max_error": float(np.max(val_errors)),
            "runtime_seconds": t6_elapsed,
        }

        print(f"\n  Full Jac validation: mean_err={full_jac_validation['mean_error']:.2e}, "
              f"max_err={full_jac_validation['max_error']:.2e}")
        if full_jac_validation['max_error'] < 0.01:
            print("  VALIDATED: Direct tap matches full Jacobian projection")
        else:
            print("  WARNING: Higher than expected error")
        print(f"  Done in {t6_elapsed:.1f}s")

    # ════════════════════════════════════════════════════════════════
    # Phase 7: Finite-Difference Validation (optional)
    # ════════════════════════════════════════════════════════════════
    fd_results = None
    if fd_cfg.enabled:
        print("\n" + "=" * 60)
        print("PHASE 7: Finite-Difference Validation")
        print(f"  fd_eps={fd_cfg.epsilon}, n_prompts={fd_cfg.n_prompts}, "
              f"n_probes={fd_cfg.n_probes}, layer={fd_cfg.layer_idx}")
        print("  NOTE: bfloat16 limits FD accuracy (~10-40% relative error expected)")
        print("=" * 60)

        t7_start = time.time()
        fd_eps = fd_cfg.epsilon
        n_fd = fd_cfg.n_prompts
        n_fd_probes = fd_cfg.n_probes
        fd_layer = fd_cfg.layer_idx

        errors = []
        per_probe_errors = {j: [] for j in range(n_fd_probes)}

        for i in range(min(n_fd, len(cal_prompts))):
            prompt = cal_prompts[i]
            input_ids = prompt["input_ids"].to(device)
            attention_mask = prompt["attention_mask"].to(device)
            last_tok = attention_mask.sum(dim=1).long().item() - 1

            # Base output (unperturbed)
            base_out_dict = collect_layer_outputs(
                model, input_ids, attention_mask, [fd_layer]
            )
            f_base = base_out_dict[fd_layer][0, last_tok, :].cpu().float()

            for j in range(n_fd_probes):
                v = V_rand[:, j].to(device=device, dtype=model.dtype)

                # FD: r_fd = (F(h + eps*v) - F(h)) / eps
                f_pert = forward_with_perturbed_input(
                    model, input_ids, attention_mask,
                    fd_layer, fd_eps * v, last_tok, device
                )
                r_fd = (f_pert - f_base) / fd_eps

                # JVP ground truth from tap responses
                r_jvp = all_taps_clean[fd_layer][i][:, j].float()

                rel_err = torch.norm(r_fd - r_jvp).item() / (torch.norm(r_jvp).item() + 1e-10)
                errors.append(rel_err)
                per_probe_errors[j].append(rel_err)

            if (i + 1) % 2 == 0:
                print(f"    Prompt {i+1}/{n_fd} done, "
                      f"mean_err_so_far={np.mean(errors):.4f}")

        fd_results = {
            "mean_rel_err": float(np.mean(errors)),
            "max_rel_err": float(np.max(errors)),
            "std_rel_err": float(np.std(errors)),
            "median_rel_err": float(np.median(errors)),
            "n_prompts": n_fd,
            "n_probes": n_fd_probes,
            "fd_epsilon": fd_eps,
            "layer_idx": fd_layer,
            "per_probe_mean_err": {
                str(j): float(np.mean(per_probe_errors[j]))
                for j in range(n_fd_probes)
            },
        }

        t7_elapsed = time.time() - t7_start
        print(f"\n  FD validation results:")
        print(f"    mean_rel_err = {fd_results['mean_rel_err']:.4f}")
        print(f"    max_rel_err  = {fd_results['max_rel_err']:.4f}")
        print(f"    median_rel_err = {fd_results['median_rel_err']:.4f}")
        print(f"  Done in {t7_elapsed:.1f}s")

        if fd_results['mean_rel_err'] < 0.5:
            print("  PASS: FD and JVP tap responses are reasonably consistent.")
        else:
            print("  NOTE: High relative error (expected with bfloat16).")

    # ════════════════════════════════════════════════════════════════
    # Phase 8: Summary & Output
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 8: CROSS-ARCHITECTURE SUMMARY")
    print("=" * 80)

    # --- Build NAF table by k ---
    naf_table = []  # list of dicts for JSON
    for k in k_values:
        nafs_at_k = []
        decisions_at_k = []
        for layer_idx in target_layers:
            for eps in epsilons:
                key = (layer_idx, eps, "null_space", k)
                if key in tap_results and "_niac_decision" in tap_results[key]:
                    dec = tap_results[key]["_niac_decision"]
                    nafs_at_k.append(dec['naf'])
                    decisions_at_k.append(dec['decision'])

        if nafs_at_k:
            n_niac = sum(1 for d in decisions_at_k
                         if d in ("NIAC-DETECTED", "WEAK-NIAC"))
            entry = {
                "k": k,
                "cost_pct": k / d * 100,
                "cost_reduction": d / k,
                "mean_naf": float(np.mean(nafs_at_k)),
                "min_naf": float(np.min(nafs_at_k)),
                "max_naf": float(np.max(nafs_at_k)),
                "n_niac_detected": n_niac,
                "n_total": len(nafs_at_k),
            }
            naf_table.append(entry)

    # Print main results table
    print(f"\n{'k':>5} | {'cost%':>6} | {'reduction':>9} | "
          f"{'mean_NAF':>9} | {'min_NAF':>8} | {'max_NAF':>8} | "
          f"{'NIAC':>5}/{len(target_layers)*len(epsilons)}")
    print("-" * 75)
    for entry in naf_table:
        print(f"{entry['k']:>5} | "
              f"{entry['cost_pct']:>5.1f}% | "
              f"{entry['cost_reduction']:>7.0f}x | "
              f"{entry['mean_naf']:>8.1f}x | "
              f"{entry['min_naf']:>7.1f}x | "
              f"{entry['max_naf']:>7.1f}x | "
              f"{entry['n_niac_detected']:>5}/{entry['n_total']}")

    # Per-layer breakdown at k_max
    print(f"\n--- Per-Layer NAF at k={k_max} ---")
    print(f"{'Layer':>6} | {'eps':>5} | {'NAF':>6} | {'Decision':<15} | "
          f"{'act_ev%':>7} | {'cal_act':>9} | {'cal_jac':>9}")
    print("-" * 80)
    for layer_idx in target_layers:
        for eps in epsilons:
            key = (layer_idx, eps, "null_space", k_max)
            if key in tap_results and "_niac_decision" in tap_results[key]:
                dec = tap_results[key]["_niac_decision"]
                print(f"{layer_idx:>6} | {eps:>5.2f} | {dec['naf']:>5.1f}x | "
                      f"{dec['decision']:<15} | "
                      f"{dec['act_evasion_pct']:>6.1f}% | "
                      f"{dec['cal_act_mean_ns']:>9.6f} | "
                      f"{dec['cal_jac_mean_ns']:>9.6f}")

    # Overall verdict
    total_conditions = len(target_layers) * len(epsilons)
    k_max_nafs = [tap_results[(li, eps, "null_space", k_max)]["_niac_decision"]["naf"]
                  for li in target_layers for eps in epsilons
                  if (li, eps, "null_space", k_max) in tap_results
                  and "_niac_decision" in tap_results[(li, eps, "null_space", k_max)]]
    k_max_decisions = [tap_results[(li, eps, "null_space", k_max)]["_niac_decision"]["decision"]
                       for li in target_layers for eps in epsilons
                       if (li, eps, "null_space", k_max) in tap_results
                       and "_niac_decision" in tap_results[(li, eps, "null_space", k_max)]]

    n_strong = sum(1 for d in k_max_decisions if d == "NIAC-DETECTED")
    n_weak = sum(1 for d in k_max_decisions if d == "WEAK-NIAC")

    print(f"\n--- Overall at k={k_max} ---")
    print(f"  {n_strong} NIAC-DETECTED + {n_weak} WEAK-NIAC / {total_conditions} conditions")
    if k_max_nafs:
        print(f"  Mean NAF: {np.mean(k_max_nafs):.1f}x, "
              f"Min NAF: {np.min(k_max_nafs):.1f}x, "
              f"Max NAF: {np.max(k_max_nafs):.1f}x")

    # Check k=8 stability
    k8_nafs = [tap_results[(li, eps, "null_space", 8)]["_niac_decision"]["naf"]
               for li in target_layers for eps in epsilons
               if (li, eps, "null_space", 8) in tap_results
               and "_niac_decision" in tap_results[(li, eps, "null_space", 8)]]
    if k8_nafs and k_max_nafs:
        k8_recovery = np.mean(k8_nafs) / (np.mean(k_max_nafs) + 1e-10) * 100
        print(f"\n  k=8 vs k={k_max} NAF recovery: {k8_recovery:.1f}%")
        print(f"  k=8 mean NAF: {np.mean(k8_nafs):.1f}x (cost: {d/8:.0f}x reduction)")

    total_time = time.time() - t_start
    print(f"\n  Total runtime: {total_time:.1f}s ({total_time/60:.1f}min)")

    # ════════════════════════════════════════════════════════════════
    # Save all results
    # ════════════════════════════════════════════════════════════════
    print("\n--- Saving results ---")

    layer_props_json = {str(k): v for k, v in layer_properties.items()}
    noise_floors_json = {str(k): v for k, v in noise_floors.items()}

    # Tap decisions
    tap_decisions_json = {}
    for (li, eps, attack, k), res in tap_results.items():
        key = f"{attack}_k{k}_L{li}_eps{eps}"
        entry = {
            "cal_act_mean": res["cal_act_mean"],
            "cal_jac_mean": res["cal_jac_mean"],
            "cal_jac_act_ratio": res["cal_jac_act_ratio"],
            "cal_jac_ttest_p": res["cal_jac_ttest_p"],
        }
        if "_niac_decision" in res:
            entry["niac_decision"] = res["_niac_decision"]
        tap_decisions_json[key] = entry

    # Main results file
    output = {
        "naf_table": naf_table,
        "tap_results": tap_decisions_json,
        "layer_properties": layer_props_json,
        "noise_floors": noise_floors_json,
        "full_jac_validation": full_jac_validation,
        "fd_validation": fd_results,
        "_meta": {
            "experiment": "step10_cross_architecture",
            "model": cfg.model.name,
            "target_layers": target_layers,
            "target_sublayer": sublayer_path,
            "epsilons": epsilons,
            "k_values": k_values,
            "k_max": k_max,
            "n_cal": n_cal,
            "n_test": n_test,
            "hidden_dim": d,
            "seed": seed,
            "runtime_seconds": total_time,
            "phase_times": {
                "phase0_load": t0_elapsed,
                "phase1_baseline": t1_elapsed,
                "phase5_analysis": t5_elapsed,
            },
            "date": datetime.now().isoformat(),
            "version": "step10 -- cross-architecture (direct tap JVPs)",
        }
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Full results saved to {results_path}")

    # Compact summary
    summary_output = {
        "naf_table": naf_table,
        "full_jac_validation": full_jac_validation,
        "fd_validation": fd_results,
        "_meta": output["_meta"],
    }
    summary_path = os.path.join(output_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_output, f, indent=2, default=str)
    print(f"  Summary saved to {summary_path}")

    # NAF by k table (for paper figures)
    tap_naf_path = os.path.join(output_dir, "tap_naf_by_k.json")
    with open(tap_naf_path, "w") as f:
        json.dump(naf_table, f, indent=2, default=str)
    print(f"  NAF by k saved to {tap_naf_path}")

    # FD validation
    if fd_results:
        fd_path = os.path.join(output_dir, "fd_validation.json")
        with open(fd_path, "w") as f:
            json.dump(fd_results, f, indent=2, default=str)
        print(f"  FD validation saved to {fd_path}")

    # Full Jac validation
    if full_jac_validation:
        fjv_path = os.path.join(output_dir, "full_jac_validation.json")
        with open(fjv_path, "w") as f:
            json.dump(full_jac_validation, f, indent=2, default=str)
        print(f"  Full Jac validation saved to {fjv_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
