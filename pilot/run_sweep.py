#!/usr/bin/env python3
"""Sweep over all epsilon x perturbation combos for a given config.

Usage:
    python run_sweep.py --config configs/small.yaml
    python run_sweep.py --config configs/full.yaml --only-type gaussian
"""

import argparse
import json
import time
from pathlib import Path

from omegaconf import OmegaConf

from run import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Sweep all conditions")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--only-type", type=str, default=None,
                        help="Run only this perturbation type")
    parser.add_argument("--only-epsilon", type=float, default=None,
                        help="Run only this epsilon value")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Filter perturbation types if requested
    if args.only_type:
        cfg.experiment.perturbation_types = [args.only_type]

    # Filter epsilons if requested
    if args.only_epsilon is not None:
        cfg.experiment.epsilons = [args.only_epsilon]

    print("=" * 70)
    print("JACOBIAN SNR SWEEP")
    print("=" * 70)
    print(f"Model:        {cfg.model.name}")
    print(f"Prompts:      {cfg.data.num_prompts}")
    print(f"Directions:   {cfg.experiment.num_directions}")
    print(f"Perturbations: {list(cfg.experiment.perturbation_types)}")
    print(f"Epsilons:     {list(cfg.experiment.epsilons)}")
    print(f"Output:       {cfg.output.dir}")
    print("=" * 70)

    t0 = time.time()
    results, decision = run_experiment(cfg)
    elapsed = time.time() - t0

    # Summary
    print("\n\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Decision: {decision}")

    # Save sweep metadata
    out_dir = Path(cfg.output.dir)
    meta_path = out_dir / "sweep_meta.json"
    meta = {
        "elapsed_seconds": elapsed,
        "decision": decision,
        "config_file": args.config,
        "perturbation_types": list(cfg.experiment.perturbation_types),
        "epsilons": list(cfg.experiment.epsilons),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Sweep metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
