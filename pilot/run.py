#!/usr/bin/env python3
"""Main entry point for Experiment 1: Jacobian SNR.

Usage:
    python run.py --config configs/debug.yaml
    python run.py --config configs/small.yaml
    python run.py --config configs/full.yaml
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.model_loader import load_model, resolve_layer_indices
from src.data_loader import load_prompts
from src.layer_wrapper import collect_hidden_states, collect_layer_outputs
from src.jvp_engine import compute_all_jvp_norms
from src.perturbations import apply_perturbation, restore_weights, save_weights
from src.metrics import (
    compute_activation_diff,
    compute_jvp_diff,
    compute_snr_metrics,
    format_results_table,
    aggregate_decision,
)


def get_last_token_activations(model, prompt, layer_indices, device):
    """Get last-token activation vectors at each target layer's OUTPUT.

    We measure layer OUTPUT (not input) because when a layer's weights are
    perturbed, its input doesn't change — only its output does.
    """
    input_ids = prompt["input_ids"].to(device)
    attention_mask = prompt["attention_mask"].to(device)
    last_token_idx = attention_mask.sum(dim=1).long().item() - 1

    output_dict = collect_layer_outputs(
        model, input_ids, attention_mask, layer_indices
    )

    activations = {}
    for layer_idx in layer_indices:
        activations[layer_idx] = output_dict[layer_idx][0, last_token_idx, :].detach().cpu()

    return activations


def run_experiment(cfg):
    """Run the full Jacobian SNR experiment for one config."""
    device = cfg.model.device
    seed = cfg.experiment.seed

    # Setup output directory
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(cfg)

    # Resolve layers
    layer_indices = resolve_layer_indices(cfg, model)
    print(f"Target layers: {layer_indices}")

    # Resolve which layer to perturb
    target_layer = cfg.experiment.target_layer
    if target_layer == -1:
        target_layer = layer_indices[-1]
    print(f"Perturbation target layer: {target_layer}")

    # Load data and split into train/test
    all_prompts = load_prompts(cfg, tokenizer)

    # Train/test split: train for adversary calibration, test for evaluation
    n_total = len(all_prompts)
    n_train = n_total // 2
    train_prompts = all_prompts[:n_train]
    test_prompts = all_prompts[n_train:]
    print(f"Train/test split: {len(train_prompts)} train (calibration), "
          f"{len(test_prompts)} test (evaluation)")

    num_directions = cfg.experiment.num_directions
    perturbation_types = list(cfg.experiment.perturbation_types)
    epsilons = list(cfg.experiment.epsilons)

    # ── Phase 1: Pre-compute normal model measurements on TEST set only ──
    print("\n── Phase 1: Normal model measurements (test set) ──")
    normal_jvps = {l: [] for l in layer_indices}      # layer_idx -> list of (k,) tensors
    normal_activations = {l: [] for l in layer_indices}  # layer_idx -> list of (d,) tensors

    for pi, prompt in enumerate(tqdm(test_prompts, desc="Normal forward")):
        # Activations
        acts = get_last_token_activations(model, prompt, layer_indices, device)
        for l in layer_indices:
            normal_activations[l].append(acts[l])

        # JVPs
        jvps = compute_all_jvp_norms(
            model, prompt, layer_indices, num_directions, seed, device
        )
        for l in layer_indices:
            normal_jvps[l].append(jvps[l])

    # ── Phase 2: Perturbed model measurements ──
    all_results = {}

    for ptype in perturbation_types:
        for eps in epsilons:
            condition_key = f"{ptype}_eps{eps}"
            print(f"\n── Phase 2: {condition_key} ──")

            perturbed_jvps = {l: [] for l in layer_indices}
            perturbed_activations = {l: [] for l in layer_indices}

            # Apply perturbation to target layer
            target_module = model.model.layers[target_layer]

            # Build kwargs for activation_matched
            # CRITICAL: use TRAIN prompts for adversary calibration (not test!)
            perturb_kwargs = {}
            if ptype == "activation_matched":
                perturb_kwargs = {
                    "model": model,
                    "prompts": train_prompts,  # train set only!
                    "layer_idx": target_layer,
                    "device": device,
                }

            saved = apply_perturbation(
                target_module, ptype, eps, seed + 1000,
                **perturb_kwargs
            )

            try:
                # Evaluate on TEST set only
                for pi, prompt in enumerate(tqdm(test_prompts, desc=f"  {condition_key}")):
                    # Activations under perturbation
                    acts = get_last_token_activations(model, prompt, layer_indices, device)
                    for l in layer_indices:
                        perturbed_activations[l].append(acts[l])

                    # JVPs under perturbation (SAME seed = same random directions)
                    jvps = compute_all_jvp_norms(
                        model, prompt, layer_indices, num_directions, seed, device
                    )
                    for l in layer_indices:
                        perturbed_jvps[l].append(jvps[l])
            finally:
                # Always restore weights
                restore_weights(target_module, saved)

            # Compute metrics
            act_diffs = compute_activation_diff(normal_activations, perturbed_activations)
            jvp_diffs = compute_jvp_diff(normal_jvps, perturbed_jvps)
            snr_results = compute_snr_metrics(act_diffs, jvp_diffs)

            all_results[condition_key] = snr_results

            # Print results
            print(format_results_table(snr_results, condition_key))

    # ── Phase 3: Aggregate and save ──
    print("\n\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    overall_decision, summary = aggregate_decision(all_results)
    print(f"\nOverall decision: {overall_decision}")
    print(f"  GO: {summary.get('go_count', 0)}, "
          f"MARGINAL: {summary.get('marginal_count', 0)}, "
          f"NO-GO: {summary.get('nogo_count', 0)}")
    if 'go_fraction' in summary:
        print(f"  GO fraction: {summary['go_fraction']:.2%}")

    # Save raw results
    if cfg.output.save_raw:
        raw_path = out_dir / "results.json"
        # Convert to serializable format
        serializable = {}
        for cond, layer_results in all_results.items():
            serializable[cond] = {}
            for layer_idx, metrics in layer_results.items():
                serializable[cond][str(layer_idx)] = metrics

        serializable["_meta"] = {
            "model": cfg.model.name,
            "num_prompts_total": len(all_prompts),
            "num_prompts_train": len(train_prompts),
            "num_prompts_test": len(test_prompts),
            "num_directions": num_directions,
            "target_layer": target_layer,
            "overall_decision": overall_decision,
            "summary": summary,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }

        with open(raw_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\nResults saved to {raw_path}")

    return all_results, overall_decision


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Jacobian SNR")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    t0 = time.time()
    results, decision = run_experiment(cfg)
    elapsed = time.time() - t0

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Final verdict: {decision}")


if __name__ == "__main__":
    main()
