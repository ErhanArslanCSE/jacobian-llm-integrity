#!/usr/bin/env python3
"""Experiment 15, threshold rule with leave-one-architecture-out calibration (deviation D1 in
RESULTS.md: the pre-registered paired rule needs clean statistics on the audit machine, which a
real auditor does not have).

An auditor records references on machine R and audits on a machine A of another architecture.
It knows how much a clean model deviates across architectures from earlier calibration runs on
other hardware, and flags a layer when the calibration-mean statistic exceeds
    tau = margin * max over clean cross-architecture pairs (R', A') of that mean,
with tau learnt only from pairs that do not involve the held-out architecture X. It is then
evaluated on every pair that involves X: false alarms on the clean model and detections of the
edits, for the activation audit (any layer >= the edited layer) and the tap test (edited layer).
Both monitors get the same rule and margin.

Usage: python analyze_exp15_threshold.py records/
"""

import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

from analyze_exp15 import N_CAL, act_stat, parse_name, tap_stat

ARCH = {"H200": "hopper", "H200b": "hopper", "B200": "blackwell-dc", "RTXPRO4500": "blackwell-ws"}
MARGINS = (1.0, 1.5, 2.0)
TARGET_LAYERS = (2, 6, 10, 12, 14)


def cross_pairs(tags):
    return [(R, A) for R, A in itertools.product(tags, tags) if ARCH[R] != ARCH[A]]


def main(records_dir):
    recs = {}
    for t in ARCH:
        path = os.path.join(records_dir, f"{t}.pt")
        if os.path.exists(path):
            recs[t] = torch.load(path, weights_only=False)
    cal = slice(0, N_CAL)
    tags = sorted(recs)

    # calibration-mean statistics of the clean model and of every edit, for every cross pair
    clean_act, clean_tap, edit_act, edit_tap = {}, {}, {}, {}
    for R, A in cross_pairs(tags):
        ref = recs[R]["clean"]["single"]
        for mode in ("single", "batched"):
            clean_act[(R, A, mode)] = act_stat(recs[A]["clean"][mode], ref)[cal].mean(0)
            for name, e in recs[A]["edits"].items():
                edit_act[(R, A, mode, name)] = act_stat(e[mode], ref)[cal].mean(0)
        for l in TARGET_LAYERS:
            clean_tap[(R, A, l)] = tap_stat(recs[A]["clean_taps"][l], recs[R]["clean_taps"][l])[cal].mean()
        for name, e in recs[A]["edits"].items():
            l = parse_name(name)[1]
            edit_tap[(R, A, name)] = tap_stat(e["taps"], recs[R]["clean_taps"][l])[cal].mean()

    results = defaultdict(lambda: {"act": 0, "tap": 0, "n": 0})
    false_alarms = defaultdict(lambda: {"act": 0, "tap": 0, "n": 0})
    held_out_archs = sorted({ARCH[t] for t in tags})
    for X, margin in itertools.product(held_out_archs, MARGINS):
        train = [(R, A) for R, A in cross_pairs(tags) if X not in (ARCH[R], ARCH[A])]
        test = [(R, A) for R, A in cross_pairs(tags) if X in (ARCH[R], ARCH[A])]
        if not train:
            continue
        for mode in ("single", "batched"):
            tau_act = margin * np.max([clean_act[(R, A, mode)] for R, A in train], axis=0)
            tau_tap = {l: margin * max(clean_tap[(R, A, l)] for R, A in train) for l in TARGET_LAYERS}
            for R, A in test:
                fa = false_alarms[(margin, mode)]
                fa["act"] += bool(np.any(clean_act[(R, A, mode)] > tau_act))
                fa["tap"] += any(clean_tap[(R, A, l)] > tau_tap[l] for l in TARGET_LAYERS)
                fa["n"] += 1
                for name in recs[A]["edits"]:
                    kind, l, eps = parse_name(name)
                    r = results[(margin, mode, kind, eps)]
                    r["act"] += bool(np.any(edit_act[(R, A, mode, name)][l:] > tau_act[l:]))
                    r["tap"] += bool(edit_tap[(R, A, name)] > tau_tap[l])
                    r["n"] += 1

    for margin in MARGINS:
        print(f"\nmargin {margin}: tau = {margin} x max clean cross-architecture deviation "
              f"(leave-one-architecture-out)")
        for mode in ("single", "batched"):
            fa = false_alarms[(margin, mode)]
            print(f"  {mode:<8} false alarms on clean model: activation {fa['act']}/{fa['n']}, "
                  f"tap {fa['tap']}/{fa['n']}")
            for (m, md, kind, eps), r in sorted(results.items()):
                if m == margin and md == mode:
                    print(f"    {kind:<13} eps={eps:<5} activation {r['act']:>3}/{r['n']:<3} tap {r['tap']:>3}/{r['n']}")

    out = {"false_alarms": [{"margin": k[0], "mode": k[1], **v} for k, v in false_alarms.items()],
           "detections": [{"margin": k[0], "mode": k[1], "kind": k[2], "epsilon": k[3], **v}
                          for k, v in results.items()]}
    with open(os.path.join(records_dir, "analysis_threshold.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
