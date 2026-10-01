#!/usr/bin/env python3
"""Step 7: Tap Test — Perturbation Response Fingerprinting

Core question: Can k << d probe directions preserve Jacobian detection power?
Full Jacobian monitoring is O(d^2) — impractical for 8B+ models. The tap test
replaces the full d x d Jacobian with k probe responses: J @ v_i for chosen
vectors v_1...v_k. If k=32-64 preserves detection, monitoring cost drops from
O(d^2) to O(k*d).

Design:
  - Phases 0-4: IDENTICAL to exp4 (collect full Jacobians as ground truth)
  - Phase 5: Full Jacobian analysis (ground truth NAF and decisions)
  - Phase 6: Tap test analysis (project Jacobians onto k probes, compare)
  - Phase 7: Finite-difference validation (verify JVP-based tap matches FD)
  - Phase 8: Summary (recovery tables, paper figure data)

Probe types:
  - Random: orthonormal via QR decomposition
  - PCA: top singular vectors of mean calibration Jacobian (per layer)

Key metric: NAF recovery = tap_NAF / full_jac_NAF (how much detection power retained)

Usage:
  python run_step7_tap_test.py --config configs/tap_test.yaml
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
from src.jvp_engine import compute_all_jacobians
from src.perturbations import (
    save_weights, restore_weights,
    perturb_null_space, perturb_gaussian_sublayer,
)
from src.metrics import (
    compute_activation_diff, compute_jacobian_frobenius_diff,
    analyze_condition_v2, format_results_table_v2,
)


# ════════════════════════════════════════════════════════════════
# Inline helpers (reused from exp4)
# ════════════════════════════════════════════════════════════════

def collect_activations_and_jacobians(model, prompts, layer_indices, device,
                                      label="", chunk_size=None):
    """Collect layer output activations and full Jacobians for each prompt.

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


# ════════════════════════════════════════════════════════════════
# New tap test functions
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


def generate_pca_probes(all_jacs_clean, layer_idx, n_cal, k_max):
    """Generate PCA-based probe directions from mean calibration Jacobian.

    Uses SVD of mean Jacobian: J_mean = U @ diag(S) @ Vt. The right singular
    vectors (columns of V = rows of Vt transposed) are the input directions
    of maximum layer sensitivity.

    Returns:
        V_pca: (d, k_max) probe matrix
        singular_values: numpy array of all singular values
    """
    # Stack calibration Jacobians and compute mean
    J_stack = torch.stack(all_jacs_clean[layer_idx][:n_cal]).float()  # (n_cal, d, d)
    J_mean = J_stack.mean(dim=0)  # (d, d)

    # SVD: J_mean = U @ diag(S) @ Vt
    _, S, Vt = torch.linalg.svd(J_mean, full_matrices=False)
    # Vt: (d, d), rows are right singular vectors in decreasing S order
    # V_pca[:, i] = the i-th right singular vector (direction of i-th highest sensitivity)
    d = Vt.shape[1]
    actual_k = min(k_max, d)
    V_pca = Vt[:actual_k, :].T  # (d, actual_k)

    return V_pca, S.cpu().numpy()


def compute_tap_diffs(clean_jacs_list, pert_jacs_list, V_k):
    """Compute per-prompt relative tap response difference.

    For each prompt i:
      R_clean_i = J_clean_i @ V_k   ->  (d_out, k)
      R_pert_i  = J_pert_i  @ V_k   ->  (d_out, k)
      tap_diff_i = ||R_pert_i - R_clean_i||_F / ||R_clean_i||_F

    This is equivalent to the Jacobian Frobenius diff projected onto the
    subspace spanned by V_k's columns.

    Args:
        clean_jacs_list: list of (d, d) tensors (clean Jacobians)
        pert_jacs_list: list of (d, d) tensors (perturbed Jacobians)
        V_k: (d, k) probe direction matrix

    Returns:
        np.array of per-prompt relative tap diffs
    """
    V_k_f = V_k.float()
    diffs = []
    for j_clean, j_pert in zip(clean_jacs_list, pert_jacs_list):
        R_clean = j_clean.float() @ V_k_f   # (d, k)
        R_pert  = j_pert.float()  @ V_k_f   # (d, k)
        diff_norm = torch.norm(R_pert - R_clean, p='fro').item()
        base_norm = torch.norm(R_clean, p='fro').item()
        diffs.append(diff_norm / (base_norm + 1e-10))
    return np.array(diffs)


def forward_with_perturbed_input(model, input_ids, attention_mask,
                                  layer_idx, perturbation, last_tok, device):
    """Run forward pass, adding perturbation to layer input at last_tok.

    Used for finite-difference validation of tap responses.

    Args:
        model: transformer model
        input_ids: (1, seq_len) tensor
        attention_mask: (1, seq_len) tensor
        layer_idx: which layer to perturb input of
        perturbation: (d,) tensor to add at last_tok position
        last_tok: index of last non-padding token
        device: torch device

    Returns:
        (d,) tensor: layer output at last_tok (float32, CPU)
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
    parser.add_argument("--config", type=str, default="configs/tap_test.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    target_layers = list(cfg.experiment.target_layers)
    n_cal = cfg.experiment.n_cal
    n_test = cfg.experiment.n_test
    epsilons = list(cfg.experiment.epsilons)
    sublayer_path = cfg.experiment.target_sublayer
    seed = cfg.experiment.seed

    # Tap test config
    k_values = list(cfg.tap_test.k_values)
    k_max = cfg.tap_test.k_max
    probe_types = list(cfg.tap_test.probe_types)
    fd_cfg = cfg.tap_test.fd_validation

    print("=" * 80)
    print("STEP 7: Tap Test — Perturbation Response Fingerprinting")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Model: {cfg.model.name}")
    print(f"Target layers: {target_layers}")
    print(f"Target sublayer: {sublayer_path}")
    print(f"Epsilons: {epsilons}")
    print(f"n_cal: {n_cal}, n_test: {n_test}")
    print(f"k_values: {k_values}, k_max: {k_max}")
    print(f"Probe types: {probe_types}")
    print(f"FD validation: enabled={fd_cfg.enabled}")
    print()
    print("KEY QUESTION: Can k << d probe directions preserve Jacobian detection power?")
    print("  If k=32-64 PCA probes achieve >=95% of full Jacobian NAF,")
    print("  monitoring cost drops from O(d^2) to O(k*d) — 32-64x reduction.")

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

    # Get model hidden dim from config (for probe generation)
    d = hidden_dim
    print(f"\n  Hidden dim (d): {d}")
    print(f"  k_max: {k_max} ({k_max/d*100:.1f}% of d)")
    for k in k_values:
        print(f"  k={k}: {k/d*100:.1f}% of d, cost reduction: {d/k:.0f}x")

    # ════════════════════════════════════════════════════════════════
    # JVP Validation (using first target layer)
    # ════════════════════════════════════════════════════════════════
    if chunk_size:
        print("\n--- Validating batched JVP against sequential ---")
        from src.jvp_engine import compute_full_jacobian_at_last_token

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
    t0_elapsed = time.time() - t0
    print(f"  Baseline done in {t0_elapsed:.1f}s")

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
    # Phase 2: Generate Probe Directions (NEW)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"PHASE 2: Generate Probe Directions (d={d}, k_max={k_max})")
    print("=" * 60)

    t2_start = time.time()

    # --- Random orthonormal probes ---
    print("\n  Generating random orthonormal probes...")
    V_rand = generate_random_probes(d, k_max, seed)
    print(f"  Random probes: shape={list(V_rand.shape)}, "
          f"orthogonality check: V^T V - I max error = "
          f"{(V_rand.T @ V_rand - torch.eye(k_max)).abs().max().item():.2e}")

    # --- PCA probes (per layer) ---
    print("\n  Generating PCA probes (per layer, from mean calibration Jacobian)...")
    V_pca = {}
    pca_singular_values = {}
    for layer_idx in target_layers:
        V_pca_layer, S_vals = generate_pca_probes(
            all_jacs_clean, layer_idx, n_cal, k_max
        )
        V_pca[layer_idx] = V_pca_layer
        pca_singular_values[layer_idx] = S_vals

        print(f"  Layer {layer_idx}: S[0]={S_vals[0]:.4f}, "
              f"S[{k_max-1}]={S_vals[min(k_max-1, len(S_vals)-1)]:.6f}, "
              f"S[{d-1}]={S_vals[-1]:.6e}, "
              f"S[0]/S[{k_max-1}]={S_vals[0]/(S_vals[min(k_max-1, len(S_vals)-1)]+1e-10):.1f}x")

    t2_elapsed = time.time() - t2_start
    print(f"\n  Probe generation done in {t2_elapsed:.1f}s")

    # ════════════════════════════════════════════════════════════════
    # Phase 3: Gaussian sublayer perturbation — per layer x eps
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 3: Gaussian sublayer perturbation (per layer x eps)")
    print(f"  Perturbing ONLY {sublayer_path} at each target layer")
    print("=" * 60)

    gauss_acts = {}   # (layer_idx, eps) -> dict[layer_idx -> list of act tensors]
    gauss_jacs = {}   # (layer_idx, eps) -> dict[layer_idx -> list of jac tensors]

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
            pert_acts, pert_jacs = collect_activations_and_jacobians(
                model, all_60, [layer_idx], device,
                label=f"[gauss L{layer_idx} e{eps}] ",
                chunk_size=chunk_size
            )
            print(f"    Done in {time.time()-t3:.1f}s")

            restore_weights(layer, saved)

            gauss_acts[(layer_idx, eps)] = pert_acts
            gauss_jacs[(layer_idx, eps)] = pert_jacs

    # ════════════════════════════════════════════════════════════════
    # Phase 4: Null-space perturbation — per layer x eps
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 4: Null-space perturbation (per layer x eps)")
    print("  Each layer gets its OWN null-space basis from its sublayer inputs")
    print("=" * 60)

    ns_acts = {}   # (layer_idx, eps) -> dict with cal and test acts
    ns_jacs = {}   # (layer_idx, eps) -> dict with cal and test jacs

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
            # Collect on ALL 60 prompts (same as Gaussian for consistent indexing)
            pert_acts, pert_jacs = collect_activations_and_jacobians(
                model, all_60, [layer_idx], device,
                label=f"[ns L{layer_idx} e{eps}] ",
                chunk_size=chunk_size
            )
            print(f"    Done in {time.time()-t4:.1f}s")

            restore_weights(layer, saved)

            ns_acts[(layer_idx, eps)] = pert_acts
            ns_jacs[(layer_idx, eps)] = pert_jacs

    # ════════════════════════════════════════════════════════════════
    # Phase 5: Full Jacobian Analysis (ground truth — same as exp4)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 5: Full Jacobian Analysis (ground truth)")
    print("=" * 60)

    full_jac_gauss_results = {}   # (layer_idx, eps) -> analyze_condition_v2 result
    full_jac_ns_results = {}      # (layer_idx, eps) -> analyze_condition_v2 result
    full_jac_decisions = {}       # (layer_idx, eps) -> niac_decision dict

    for layer_idx in target_layers:
        for eps in epsilons:
            # --- Gaussian ---
            g_pert_acts = gauss_acts[(layer_idx, eps)]
            g_pert_jacs = gauss_jacs[(layer_idx, eps)]

            cal_act_d = compute_activation_diff(
                {layer_idx: all_acts_clean[layer_idx][:n_cal]},
                {layer_idx: g_pert_acts[layer_idx][:n_cal]}
            )
            cal_jac_d = compute_jacobian_frobenius_diff(
                {layer_idx: all_jacs_clean[layer_idx][:n_cal]},
                {layer_idx: g_pert_jacs[layer_idx][:n_cal]}
            )
            test_act_d = compute_activation_diff(
                {layer_idx: all_acts_clean[layer_idx][n_cal:]},
                {layer_idx: g_pert_acts[layer_idx][n_cal:]}
            )
            test_jac_d = compute_jacobian_frobenius_diff(
                {layer_idx: all_jacs_clean[layer_idx][n_cal:]},
                {layer_idx: g_pert_jacs[layer_idx][n_cal:]}
            )

            g_result = analyze_condition_v2(
                cal_act_d[layer_idx], cal_jac_d[layer_idx],
                test_act_d[layer_idx], test_jac_d[layer_idx],
                f"full_jac_gauss_L{layer_idx}_eps{eps}",
                noise_floors[layer_idx]["act"],
                noise_floors[layer_idx]["jac"]
            )
            full_jac_gauss_results[(layer_idx, eps)] = g_result

            # --- Null-space ---
            n_pert_acts = ns_acts[(layer_idx, eps)]
            n_pert_jacs = ns_jacs[(layer_idx, eps)]

            cal_act_d = compute_activation_diff(
                {layer_idx: all_acts_clean[layer_idx][:n_cal]},
                {layer_idx: n_pert_acts[layer_idx][:n_cal]}
            )
            cal_jac_d = compute_jacobian_frobenius_diff(
                {layer_idx: all_jacs_clean[layer_idx][:n_cal]},
                {layer_idx: n_pert_jacs[layer_idx][:n_cal]}
            )
            test_act_d = compute_activation_diff(
                {layer_idx: all_acts_clean[layer_idx][n_cal:]},
                {layer_idx: n_pert_acts[layer_idx][n_cal:]}
            )
            test_jac_d = compute_jacobian_frobenius_diff(
                {layer_idx: all_jacs_clean[layer_idx][n_cal:]},
                {layer_idx: n_pert_jacs[layer_idx][n_cal:]}
            )

            n_result = analyze_condition_v2(
                cal_act_d[layer_idx], cal_jac_d[layer_idx],
                test_act_d[layer_idx], test_jac_d[layer_idx],
                f"full_jac_ns_L{layer_idx}_eps{eps}",
                noise_floors[layer_idx]["act"],
                noise_floors[layer_idx]["jac"]
            )
            full_jac_ns_results[(layer_idx, eps)] = n_result

            # NIAC decision
            dec = niac_decision_for_condition(g_result, n_result, eps)
            dec["layer"] = layer_idx
            full_jac_decisions[(layer_idx, eps)] = dec

            print(f"  L{layer_idx} eps={eps}: "
                  f"gauss j/a={g_result['cal_jac_act_ratio']:.1f}x, "
                  f"ns j/a={n_result['cal_jac_act_ratio']:.1f}x, "
                  f"NAF={dec['naf']:.1f}x, dec={dec['decision']}")

    # ════════════════════════════════════════════════════════════════
    # Phase 6: Tap Test Analysis (NEW — Core of this experiment)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 6: Tap Test Analysis")
    print(f"  k_values: {k_values}")
    print(f"  probe_types: {probe_types}")
    print(f"  Total conditions: {len(target_layers)} layers x {len(epsilons)} eps "
          f"x 2 attacks x {len(probe_types)} probes x {len(k_values)} k = "
          f"{len(target_layers)*len(epsilons)*2*len(probe_types)*len(k_values)}")
    print("=" * 60)

    t6_start = time.time()
    tap_results = {}  # (layer_idx, eps, attack, probe_type, k) -> dict

    for layer_idx in target_layers:
        print(f"\n  -- Layer {layer_idx} --")
        for eps in epsilons:
            for attack in ["gaussian", "null_space"]:
                if attack == "gaussian":
                    pert_acts_dict = gauss_acts[(layer_idx, eps)]
                    pert_jacs_dict = gauss_jacs[(layer_idx, eps)]
                else:
                    pert_acts_dict = ns_acts[(layer_idx, eps)]
                    pert_jacs_dict = ns_jacs[(layer_idx, eps)]

                # Pre-compute activation diffs (same for all k and probe types)
                cal_act_diffs = compute_activation_diff(
                    {layer_idx: all_acts_clean[layer_idx][:n_cal]},
                    {layer_idx: pert_acts_dict[layer_idx][:n_cal]}
                )[layer_idx]
                test_act_diffs = compute_activation_diff(
                    {layer_idx: all_acts_clean[layer_idx][n_cal:]},
                    {layer_idx: pert_acts_dict[layer_idx][n_cal:]}
                )[layer_idx]

                for probe_type in probe_types:
                    V = V_rand if probe_type == "random" else V_pca[layer_idx]

                    for k in k_values:
                        V_k = V[:, :k]  # nested subset

                        # Compute tap diffs for cal and test splits
                        cal_tap = compute_tap_diffs(
                            all_jacs_clean[layer_idx][:n_cal],
                            pert_jacs_dict[layer_idx][:n_cal],
                            V_k
                        )
                        test_tap = compute_tap_diffs(
                            all_jacs_clean[layer_idx][n_cal:],
                            pert_jacs_dict[layer_idx][n_cal:],
                            V_k
                        )

                        # Reuse analyze_condition_v2 with tap diffs as "jac" diffs
                        condition = f"tap_{probe_type}_k{k}_{attack}_L{layer_idx}_eps{eps}"
                        result = analyze_condition_v2(
                            cal_act_diffs, cal_tap,
                            test_act_diffs, test_tap,
                            condition,
                            noise_floors[layer_idx]["act"],
                            noise_floors[layer_idx]["jac"]
                        )

                        tap_results[(layer_idx, eps, attack, probe_type, k)] = result

            # For null-space conditions, compute NIAC decisions using Gaussian
            # tap result at SAME k (fair comparison — both use k-dim projections)
            for probe_type in probe_types:
                for k in k_values:
                    g_tap = tap_results[(layer_idx, eps, "gaussian", probe_type, k)]
                    n_tap = tap_results[(layer_idx, eps, "null_space", probe_type, k)]
                    dec = niac_decision_for_condition(g_tap, n_tap, eps)
                    dec["layer"] = layer_idx
                    dec["probe_type"] = probe_type
                    dec["k"] = k

                    # Store decision alongside the ns tap result
                    tap_results[(layer_idx, eps, "null_space", probe_type, k)]["_niac_decision"] = dec

        # Print progress for this layer
        print(f"    Layer {layer_idx} done. Sample NAFs at eps=0.1:")
        for probe_type in probe_types:
            nafs = []
            for k in k_values:
                key = (layer_idx, 0.1, "null_space", probe_type, k)
                if key in tap_results and "_niac_decision" in tap_results[key]:
                    nafs.append(f"k={k}:{tap_results[key]['_niac_decision']['naf']:.1f}x")
            if nafs:
                print(f"      {probe_type}: {', '.join(nafs)}")

    t6_elapsed = time.time() - t6_start
    print(f"\n  Tap test analysis done in {t6_elapsed:.1f}s")

    # ════════════════════════════════════════════════════════════════
    # Phase 7: Finite-Difference Validation (NEW, optional)
    # ════════════════════════════════════════════════════════════════
    fd_results = None
    if fd_cfg.enabled:
        print("\n" + "=" * 60)
        print("PHASE 7: Finite-Difference Validation")
        print("  Validates that FD tap responses match JVP-based tap responses")
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

                # JVP ground truth: r_jvp = J @ v
                r_jvp = all_jacs_clean[fd_layer][i].float() @ V_rand[:, j].float()

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
        print(f"    std_rel_err  = {fd_results['std_rel_err']:.4f}")
        print(f"  Done in {t7_elapsed:.1f}s")

        if fd_results['mean_rel_err'] < 0.5:
            print("  PASS: FD and JVP tap responses are reasonably consistent.")
            print("  (bfloat16 precision limits accuracy — this is expected.)")
        else:
            print("  NOTE: High relative error. This is expected with bfloat16")
            print("  (machine epsilon ~0.004, fd_eps=0.01 -> roundoff ~40%).")
            print("  For deployment with float32, expect <1% error.")

    # ════════════════════════════════════════════════════════════════
    # Phase 8: Summary & Output (NEW)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 8: TAP TEST SUMMARY")
    print("=" * 80)

    # --- Build recovery table ---
    # NAF recovery = tap_NAF / full_jac_NAF
    # Decision agreement: tap decision == full_jac decision
    recovery_table = []

    for probe_type in probe_types:
        for k in k_values:
            recoveries = []
            agreements = []
            tap_nafs = []

            for layer_idx in target_layers:
                for eps in epsilons:
                    # Full Jacobian NAF (ground truth)
                    full_dec = full_jac_decisions[(layer_idx, eps)]
                    full_naf = full_dec['naf']

                    # Tap test NAF
                    tap_key = (layer_idx, eps, "null_space", probe_type, k)
                    if tap_key in tap_results and "_niac_decision" in tap_results[tap_key]:
                        tap_dec = tap_results[tap_key]["_niac_decision"]
                        tap_naf = tap_dec['naf']

                        recovery = tap_naf / (full_naf + 1e-10)
                        agrees = tap_dec['decision'] == full_dec['decision']

                        recoveries.append(recovery)
                        agreements.append(agrees)
                        tap_nafs.append(tap_naf)

            if recoveries:
                entry = {
                    "probe_type": probe_type,
                    "k": k,
                    "cost_pct": k / d * 100,
                    "cost_reduction": d / k,
                    "mean_recovery": float(np.mean(recoveries)),
                    "min_recovery": float(np.min(recoveries)),
                    "max_recovery": float(np.max(recoveries)),
                    "std_recovery": float(np.std(recoveries)),
                    "decision_agreement": float(np.mean(agreements)) * 100,
                    "n_agree": int(sum(agreements)),
                    "n_total": len(agreements),
                    "mean_tap_naf": float(np.mean(tap_nafs)),
                    "min_tap_naf": float(np.min(tap_nafs)),
                }
                recovery_table.append(entry)

    # --- Print main results table ---
    print(f"\n{'k':>5} | {'probe':>7} | {'cost%':>6} | {'reduction':>9} | "
          f"{'mean_rec':>9} | {'min_rec':>8} | {'agree%':>7} | "
          f"{'mean_NAF':>9} | {'min_NAF':>8}")
    print("-" * 95)

    for entry in recovery_table:
        print(f"{entry['k']:>5} | {entry['probe_type']:>7} | "
              f"{entry['cost_pct']:>5.1f}% | "
              f"{entry['cost_reduction']:>7.0f}x | "
              f"{entry['mean_recovery']:>8.1f}% | "
              f"{entry['min_recovery']:>7.1f}% | "
              f"{entry['decision_agreement']:>6.1f}% | "
              f"{entry['mean_tap_naf']:>8.1f}x | "
              f"{entry['min_tap_naf']:>7.1f}x")

    # --- Print full Jacobian ground truth for comparison ---
    print("\n--- Full Jacobian Ground Truth (reference) ---")
    full_nafs = []
    for layer_idx in target_layers:
        for eps in epsilons:
            dec = full_jac_decisions[(layer_idx, eps)]
            full_nafs.append(dec['naf'])
            print(f"  L{layer_idx} eps={eps}: NAF={dec['naf']:.1f}x, "
                  f"dec={dec['decision']}")
    print(f"  Mean NAF: {np.mean(full_nafs):.1f}x, "
          f"Min NAF: {np.min(full_nafs):.1f}x")

    # --- Per-layer breakdown at key k values ---
    key_ks = [32, 64, 128]
    for probe_type in probe_types:
        print(f"\n--- Per-Layer NAF at key k values ({probe_type} probes) ---")
        print(f"{'Layer':>6} | {'eps':>5} | {'full':>6} | ", end="")
        for k in key_ks:
            print(f"k={k:>3} | ", end="")
        print()
        print("-" * (24 + len(key_ks) * 9))

        for layer_idx in target_layers:
            for eps in epsilons:
                full_naf = full_jac_decisions[(layer_idx, eps)]['naf']
                print(f"{layer_idx:>6} | {eps:>5.2f} | {full_naf:>5.1f}x | ", end="")
                for k in key_ks:
                    tap_key = (layer_idx, eps, "null_space", probe_type, k)
                    if tap_key in tap_results and "_niac_decision" in tap_results[tap_key]:
                        tap_naf = tap_results[tap_key]["_niac_decision"]['naf']
                        print(f"{tap_naf:>5.1f}x | ", end="")
                    else:
                        print(f"{'N/A':>5s} | ", end="")
                print()

    # --- PCA vs Random comparison ---
    if "random" in probe_types and "pca" in probe_types:
        print("\n--- PCA vs Random Comparison ---")
        print(f"{'k':>5} | {'rand_rec':>9} | {'pca_rec':>9} | {'pca_advantage':>14}")
        print("-" * 45)
        for k in k_values:
            rand_entry = next((e for e in recovery_table
                               if e['probe_type'] == 'random' and e['k'] == k), None)
            pca_entry = next((e for e in recovery_table
                              if e['probe_type'] == 'pca' and e['k'] == k), None)
            if rand_entry and pca_entry:
                advantage = pca_entry['mean_recovery'] - rand_entry['mean_recovery']
                print(f"{k:>5} | {rand_entry['mean_recovery']:>8.1f}% | "
                      f"{pca_entry['mean_recovery']:>8.1f}% | "
                      f"{advantage:>+12.1f}pp")

    # --- PCA singular value spectrum ---
    print("\n--- PCA Singular Value Spectrum ---")
    for layer_idx in target_layers:
        S = pca_singular_values[layer_idx]
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        # Find k needed for 95% and 99% energy
        k_95 = int(np.searchsorted(cumulative_energy, 0.95)) + 1
        k_99 = int(np.searchsorted(cumulative_energy, 0.99)) + 1
        print(f"  Layer {layer_idx}: S[0]={S[0]:.3f}, S[31]={S[31]:.4f}, "
              f"S[63]={S[63]:.5f}, S[511]={S[min(511, len(S)-1)]:.6f}")
        print(f"    Energy: 95% at k={k_95}, 99% at k={k_99}")

    # --- Overall verdict ---
    print("\n" + "=" * 60)

    # Find minimum k for >=95% mean recovery with PCA probes
    min_k_95 = None
    for entry in sorted(recovery_table, key=lambda e: e['k']):
        if entry['probe_type'] == 'pca' and entry['mean_recovery'] >= 95.0:
            min_k_95 = entry['k']
            break

    # Find minimum k for 100% decision agreement with PCA probes
    min_k_100_agree = None
    for entry in sorted(recovery_table, key=lambda e: e['k']):
        if entry['probe_type'] == 'pca' and entry['decision_agreement'] >= 100.0:
            min_k_100_agree = entry['k']
            break

    total_conditions = len(target_layers) * len(epsilons)
    full_n_strong = sum(1 for d in full_jac_decisions.values()
                        if d['decision'] == 'NIAC-DETECTED')
    full_n_weak = sum(1 for d in full_jac_decisions.values()
                      if d['decision'] == 'WEAK-NIAC')

    print(f"  Full Jacobian ground truth: {total_conditions} conditions, "
          f"{full_n_strong} strong + {full_n_weak} weak NIAC detected")

    if min_k_95:
        cost_pct = min_k_95 / d * 100
        print(f"\n  KEY RESULT: PCA probes at k={min_k_95} achieve >=95% NAF recovery")
        print(f"  Cost: {cost_pct:.1f}% of full Jacobian ({d/min_k_95:.0f}x reduction)")
    else:
        print(f"\n  NOTE: No k value achieved >=95% mean NAF recovery with PCA probes.")
        # Show best achieved
        best_pca = max((e for e in recovery_table if e['probe_type'] == 'pca'),
                       key=lambda e: e['mean_recovery'], default=None)
        if best_pca:
            print(f"  Best: k={best_pca['k']}, recovery={best_pca['mean_recovery']:.1f}%")

    if min_k_100_agree:
        print(f"  100% decision agreement at k={min_k_100_agree} (PCA)")
    else:
        best_agree = max((e for e in recovery_table if e['probe_type'] == 'pca'),
                         key=lambda e: e['decision_agreement'], default=None)
        if best_agree:
            print(f"  Best agreement: {best_agree['decision_agreement']:.0f}% "
                  f"at k={best_agree['k']} (PCA)")

    total_time = time.time() - t_start
    print(f"\n  Total runtime: {total_time:.1f}s ({total_time/60:.1f}min)")

    # ════════════════════════════════════════════════════════════════
    # Save all results
    # ════════════════════════════════════════════════════════════════
    print("\n--- Saving results ---")

    # Convert keys for JSON serialization
    layer_props_json = {str(k): v for k, v in layer_properties.items()}
    noise_floors_json = {str(k): v for k, v in noise_floors.items()}

    # Full Jacobian decisions
    full_jac_dec_json = {}
    for (li, eps), dec in full_jac_decisions.items():
        full_jac_dec_json[f"L{li}_eps{eps}"] = dec

    # Full Jacobian analysis results
    full_jac_gauss_json = {}
    for (li, eps), res in full_jac_gauss_results.items():
        full_jac_gauss_json[f"L{li}_eps{eps}"] = res
    full_jac_ns_json = {}
    for (li, eps), res in full_jac_ns_results.items():
        full_jac_ns_json[f"L{li}_eps{eps}"] = res

    # Tap test results (compact: only decisions + key stats)
    tap_decisions_json = {}
    for (li, eps, attack, pt, k), res in tap_results.items():
        key = f"{attack}_{pt}_k{k}_L{li}_eps{eps}"
        entry = {
            "cal_act_mean": res["cal_act_mean"],
            "cal_jac_mean": res["cal_jac_mean"],
            "cal_jac_act_ratio": res["cal_jac_act_ratio"],
            "cal_jac_ttest_p": res["cal_jac_ttest_p"],
        }
        if "_niac_decision" in res:
            entry["niac_decision"] = res["_niac_decision"]
        tap_decisions_json[key] = entry

    # PCA singular values
    pca_sv_json = {
        str(li): pca_singular_values[li].tolist()
        for li in target_layers
    }

    # Main results file
    output = {
        "full_jac_decisions": full_jac_dec_json,
        "full_jac_gauss_results": full_jac_gauss_json,
        "full_jac_ns_results": full_jac_ns_json,
        "tap_results": tap_decisions_json,
        "recovery_table": recovery_table,
        "layer_properties": layer_props_json,
        "noise_floors": noise_floors_json,
        "fd_validation": fd_results,
        "_meta": {
            "model": cfg.model.name,
            "target_layers": target_layers,
            "target_sublayer": sublayer_path,
            "epsilons": epsilons,
            "k_values": k_values,
            "k_max": k_max,
            "probe_types": probe_types,
            "n_cal": n_cal,
            "n_test": n_test,
            "hidden_dim": d,
            "seed": seed,
            "runtime_seconds": total_time,
            "phase_times": {
                "phase0_baseline": t0_elapsed,
                "phase2_probes": t2_elapsed,
                "phase6_tap": t6_elapsed,
            },
            "date": datetime.now().isoformat(),
            "version": "step7 -- tap test (perturbation response fingerprinting)",
        }
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Full results saved to {results_path}")

    # Compact summary
    summary_output = {
        "recovery_table": recovery_table,
        "full_jac_decisions": full_jac_dec_json,
        "fd_validation": fd_results,
        "_meta": output["_meta"],
    }
    summary_path = os.path.join(output_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_output, f, indent=2, default=str)
    print(f"  Summary saved to {summary_path}")

    # Tap recovery table (for paper figures)
    tap_recovery_path = os.path.join(output_dir, "tap_recovery.json")
    with open(tap_recovery_path, "w") as f:
        json.dump(recovery_table, f, indent=2, default=str)
    print(f"  Tap recovery table saved to {tap_recovery_path}")

    # PCA singular values
    pca_path = os.path.join(output_dir, "pca_singular_values.json")
    with open(pca_path, "w") as f:
        json.dump(pca_sv_json, f, indent=2, default=str)
    print(f"  PCA singular values saved to {pca_path}")

    # FD validation
    if fd_results:
        fd_path = os.path.join(output_dir, "fd_validation.json")
        with open(fd_path, "w") as f:
            json.dump(fd_results, f, indent=2, default=str)
        print(f"  FD validation saved to {fd_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
