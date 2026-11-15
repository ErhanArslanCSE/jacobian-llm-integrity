#!/usr/bin/env python3
"""Experiment 15, step 3: decisions of both monitors under realistic recomputation noise.

For every ordered pair (reference machine R, audit machine A), including R = A, and every
audit mode (single, batched), the per-prompt statistics are computed against R's clean
single-prompt references:
  noise   s(clean on A)  - what two clean replicas differ by
  edit    s(edit on A)
for the activation monitor (s_act, every layer >= the edited layer) and the tap test
(s_k at the edited layer; JVPs are per prompt, so "batched" uses the single-prompt taps).

Decision rule with tolerance (PREREGISTRATION.md): a layer flags when a one-sided paired
t-test of edit - noise over the calibration prompts rejects at alpha / 16 (Bonferroni over
the 16 decoder layers a defender monitors). The activation audit flags if any layer from the
edited layer on flags; the tap test is tested at the edited layer with the same alpha / 16.
Under R = A, single mode, the noise is zero on a deterministic stack and the rule reduces to
the paper's Definition 1 (with alpha / 16).

Usage: python analyze_exp15.py records/  (all <tag>.pt files in the directory)
"""

import glob
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

ALPHA = 0.001
N_MONITORED_LAYERS = 16
N_CAL = 30



def parse_name(name):
    """'allpos_m256_L2_eps0.01' -> ('allpos_m256', 2, 0.01)."""
    kind, rest = name.rsplit("_L", 1)
    layer, eps = rest.split("_eps")
    return kind, int(layer), float(eps)


def act_stat(outputs, ref):
    """(n_prompts, L) s_act at every layer."""
    return ((outputs.float() - ref.float()).norm(dim=-1) / ref.float().norm(dim=-1).clamp_min(1e-10)).numpy()


def tap_stat(taps, ref):
    """(n_prompts,) s_k."""
    diff = (taps.float() - ref.float()).flatten(1).norm(dim=1)
    return (diff / ref.float().flatten(1).norm(dim=1).clamp_min(1e-10)).numpy()


def paired_flag(edit, noise, alpha):
    """One-sided paired t-test of edit - noise > 0; returns (flag, p, mean difference)."""
    diff = np.asarray(edit, dtype=np.float64) - np.asarray(noise, dtype=np.float64)
    if np.all(diff == 0):
        return False, 1.0, 0.0
    if np.std(diff, ddof=1) == 0:
        return bool(diff.mean() > 0), 0.0 if diff.mean() > 0 else 1.0, float(diff.mean())
    p = float(stats.ttest_1samp(diff, 0.0, alternative="greater").pvalue)
    return p < alpha, p, float(diff.mean())


def main(records_dir):
    records = {}
    for path in sorted(glob.glob(os.path.join(records_dir, "*.pt"))):
        r = torch.load(path, weights_only=False)
        records[r["tag"]] = r
    tags = sorted(records)
    alpha = ALPHA / N_MONITORED_LAYERS
    cal = slice(0, N_CAL)

    determinism = {t: float(act_stat(records[t]["clean_repeat_single"], records[t]["clean"]["single"]).max())
                   for t in tags}
    print(f"Same-machine repeat, max s_act: {determinism}\n")

    noise_summary, rows = [], []
    for R, A in itertools.product(tags, tags):
        ref_act = records[R]["clean"]["single"]
        for mode in ("single", "batched"):
            noise_act = act_stat(records[A]["clean"][mode], ref_act)[cal]
            noise_summary.append({"ref": R, "audit": A, "mode": mode,
                                  "act_noise_mean_by_layer": noise_act.mean(axis=0).tolist(),
                                  "act_noise_max": float(noise_act.max())})
            for name, e in records[A]["edits"].items():
                kind, layer, eps = parse_name(name)
                edit_act = act_stat(e[mode], ref_act)[cal]
                layer_flags = [paired_flag(edit_act[:, l], noise_act[:, l], alpha)
                               for l in range(layer, edit_act.shape[1])]
                ref_tap = records[R]["clean_taps"][layer]
                noise_tap = tap_stat(records[A]["clean_taps"][layer], ref_tap)[cal]
                edit_tap = tap_stat(e["taps"], ref_tap)[cal]
                tap_flag, tap_p, tap_diff = paired_flag(edit_tap, noise_tap, alpha)
                rows.append({
                    "ref": R, "audit": A, "mode": mode, "kind": kind, "layer": layer, "epsilon": eps,
                    "act_flags": any(f for f, _, _ in layer_flags),
                    "act_flag_layers": [layer + i for i, (f, _, _) in enumerate(layer_flags) if f],
                    "act_edited_layer_flag": layer_flags[0][0],
                    "act_min_p": min(p for _, p, _ in layer_flags),
                    "tap_flags": tap_flag, "tap_p": tap_p,
                    "tap_signal_over_noise": float(edit_tap.mean() / max(noise_tap.mean(), 1e-30)),
                    "act_edited_signal_over_noise": float(edit_act[:, layer].mean()
                                                          / max(noise_act[:, layer].mean(), 1e-30)),
                })

    agg = defaultdict(lambda: {"n": 0, "act": 0, "act_edited": 0, "tap": 0})
    for r in rows:
        key = ("same" if r["ref"] == r["audit"] else "cross", r["mode"], r["kind"])
        a = agg[key]
        a["n"] += 1
        a["act"] += r["act_flags"]
        a["act_edited"] += r["act_edited_layer_flag"]
        a["tap"] += r["tap_flags"]

    print(f"Detections with tolerance (paired t-test vs. clean replica, alpha = {ALPHA}/16), calibration prompts")
    print(f"{'machines':<8} {'mode':<8} {'edit':<14} | {'act every-layer':>15} {'act edited layer':>16} | {'tap test':>9}")
    for key in sorted(agg):
        a = agg[key]
        print(f"{key[0]:<8} {key[1]:<8} {key[2]:<14} | {a['act']:>7}/{a['n']:<7} {a['act_edited']:>8}/{a['n']:<7} | "
              f"{a['tap']:>4}/{a['n']:<4}")

    with open(os.path.join(records_dir, "analysis.json"), "w") as f:
        json.dump({"determinism": determinism, "noise": noise_summary, "rows": rows,
                   "aggregate": [{"machines": k[0], "mode": k[1], "kind": k[2], **v} for k, v in agg.items()]},
                  f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
