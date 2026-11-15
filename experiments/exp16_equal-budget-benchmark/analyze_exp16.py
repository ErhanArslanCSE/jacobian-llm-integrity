#!/usr/bin/env python3
"""Experiment 16, step 3: every monitor under the same decision rule.

Monitors (passes per audit through each monitored layer in brackets):
  A30   activation, 30 known calibration prompts            [30]
  A270k activation, 270 known prompts                       [270]
  A270s activation, 270 secret natural prompts              [270]
  S270  activation, 270 secret synthetic hidden states      [270]  (needs an interface that
        accepts hidden states; setting (iii) does not - upper bound)
  T30   tap test k = 8, 30 known calibration prompts        [270]
  T30s  tap test k = 8, 30 secret natural prompts           [270]
Per monitor and layer the statistic is the mean relative change over its inputs against the
clean references of the reference machine R (s_act for A/S, s_k for T); A monitors read all
16 layers, S and T the 5 target layers.

Decision rules (both from the exp15 analysis, applied identically to every monitor):
  same architecture   R and A of one architecture (zero noise, exp15): flag if any layer's
                      Definition-1 t-test rejects at 0.001 / n_layers
  cross architecture  leave-one-architecture-out threshold: tau = margin x the largest clean
                      cross-architecture value learnt without the held-out architecture; flag
                      if any layer exceeds tau; false alarms counted on the clean model

Usage: python analyze_exp16.py records/ models/manifest.json
"""

import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

ARCH = {"H200": "hopper", "H200b": "hopper", "B200": "blackwell-dc", "RTXPRO4500": "blackwell-ws"}
MARGINS = (1.5, 2.0)
ALPHA = 0.001
SLICES = {"A30": slice(0, 30), "A270k": slice(0, 270), "A270s": slice(270, 540)}
MONITORS = ("A30", "A270k", "A270s", "S270", "T30", "T30s")


def rel(x, ref, dims):
    x, ref = x.float(), ref.float()
    return ((x - ref).norm(dim=dims) / ref.norm(dim=dims).clamp_min(1e-10)).numpy()


def per_input_stats(rec_model, rec_ref):
    """monitor -> (n_inputs, n_layers) statistics of a recorded model against references."""
    out = {}
    act = rel(rec_model["acts"], rec_ref["acts"], -1)                      # (540, L)
    for m, sl in SLICES.items():
        out[m] = act[sl]
    layers = sorted(rec_ref["synth"])
    out["S270"] = np.stack([rel(rec_model["synth"][l], rec_ref["synth"][l], -1) for l in layers], axis=1)
    for m, s in (("T30", "cal"), ("T30s", "secret")):
        out[m] = np.stack([rel(rec_model["taps"][s][l], rec_ref["taps"][s][l], (1, 2)) for l in layers], axis=1)
    return out


def definition1_any_layer(values, n_layers):
    """Definition 1 at alpha / n_layers, any layer."""
    for col in values.T:
        if np.any(col > 0):
            if np.std(col) == 0 or stats.ttest_1samp(col, 0.0, alternative="greater").pvalue < ALPHA / n_layers:
                return True
    return False


def main(records_dir, manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)
    recs = {}
    for t in ARCH:
        p = os.path.join(records_dir, f"{t}.pt")
        if os.path.exists(p):
            recs[t] = torch.load(p, weights_only=False)
    tags = sorted(recs)
    models = list(manifest["models"])
    print(f"machines: {tags}")

    repeat = {t: max(float(v.max()) for v in per_input_stats(recs[t]["clean_repeat"], recs[t]["clean"]).values())
              for t in tags}
    print(f"same-machine repeat, max statistic: {repeat}")

    pairs = list(itertools.permutations(tags, 2))
    means = {}   # (R, A, model or 'clean', monitor) -> per-layer mean
    same_arch = defaultdict(lambda: [0, 0])
    for R, A in pairs:
        ref = recs[R]["clean"]
        clean = per_input_stats(recs[A]["clean"], ref)
        for mon in MONITORS:
            means[(R, A, "clean", mon)] = clean[mon].mean(0)
        for name in models:
            s = per_input_stats(recs[A]["models"][name], ref)
            for mon in MONITORS:
                means[(R, A, name, mon)] = s[mon].mean(0)
                if ARCH[R] == ARCH[A]:
                    same_arch[(name, mon)][0] += definition1_any_layer(s[mon], s[mon].shape[1])
                    same_arch[(name, mon)][1] += 1

    det = defaultdict(lambda: [0, 0])
    fa = defaultdict(lambda: [0, 0])
    archs = sorted({ARCH[t] for t in tags})
    cross = [(R, A) for R, A in pairs if ARCH[R] != ARCH[A]]
    for X, margin in itertools.product(archs, MARGINS):
        train = [(R, A) for R, A in cross if X not in (ARCH[R], ARCH[A])]
        test = [(R, A) for R, A in cross if X in (ARCH[R], ARCH[A])]
        if not train:
            continue
        for mon in MONITORS:
            tau = margin * np.max([means[(R, A, "clean", mon)] for R, A in train], axis=0)
            for R, A in test:
                fa[(margin, mon)][0] += bool(np.any(means[(R, A, "clean", mon)] > tau))
                fa[(margin, mon)][1] += 1
                for name in models:
                    det[(margin, name, mon)][0] += bool(np.any(means[(R, A, name, mon)] > tau))
                    det[(margin, name, mon)][1] += 1

    def fmt(c):
        return f"{c[0]}/{c[1]}" if c[1] else "-"

    head = f"{'model':<34} " + " ".join(f"{m:>8}" for m in MONITORS)
    if same_arch:
        print("\nSame architecture (zero noise), Definition 1:\n" + head)
        for name in models:
            print(f"{name:<34} " + " ".join(f"{fmt(same_arch[(name, m)]):>8}" for m in MONITORS))
    for margin in MARGINS:
        print(f"\nCross architecture, tau = {margin} x max clean deviation (leave-one-architecture-out)")
        print(f"{'false alarms (clean model)':<34} " + " ".join(f"{fmt(fa[(margin, m)]):>8}" for m in MONITORS))
        print(head)
        for name in models:
            fn = manifest["models"][name].get("function")
            tag = f" [{fn['trigger_success']:.0%}]" if fn else ""
            print(f"{(name + tag):<34} " + " ".join(f"{fmt(det[(margin, name, m)]):>8}" for m in MONITORS))

    with open(os.path.join(records_dir, "analysis.json"), "w") as f:
        json.dump({"repeat": repeat,
                   "same_arch": [{"model": k[0], "monitor": k[1], "detected": v[0], "n": v[1]}
                                 for k, v in same_arch.items()],
                   "cross_arch_false_alarms": [{"margin": k[0], "monitor": k[1], "flagged": v[0], "n": v[1]}
                                               for k, v in fa.items()],
                   "cross_arch_detections": [{"margin": k[0], "model": k[1], "monitor": k[2], "detected": v[0],
                                              "n": v[1]} for k, v in det.items()]}, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
