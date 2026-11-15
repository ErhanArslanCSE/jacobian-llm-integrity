# Exp15 Results (against PREREGISTRATION.md)

Machines: H200 and H200b (two H200 cards), B200, RTX PRO 4500 (three architectures); 60 edits
built on the H200 and served bitwise identically (SHA-256 checked) everywhere. Records stay on
the H200 pod (2.4 GB); analyses: `results/analysis.json` (pre-registered paired rule),
`results/analysis_threshold.{json,txt}` (threshold rule, deviation D1).

## Noise

- Same machine, repeated: 0 everywhere. H200 vs. H200b: 0 (the same architecture reproduces
  bitwise).
- Across architectures: max last-token s_act of the clean model 2.6e-2 to 3.9e-2 (deep layers);
  calibration-mean thresholds per layer 3.4e-3 (layer 0) to 1.6e-2.
- Batched audit on the same machine: max s_act 2.0e-2 to 2.7e-2.
- Tap test (s_k, calibration mean) across architectures: 0.9e-2 to 1.5e-2 at the target layers.

## Deviation D1 (decided after seeing the paired-rule results; reason, not outcome)

The pre-registered rule compares an edit on audit machine A with the clean model on the same
machine A. An auditor who could run the clean model on A would take its references there and
have no noise, so the rule grants knowledge a real auditor lacks. We therefore add a threshold
rule (`analyze_exp15_threshold.py`): tau = margin x the largest clean cross-architecture
deviation, learnt with one architecture held out and evaluated on pairs involving it (false
alarms on the clean model, detections on the edits); margins 1.0, 1.5, 2.0; same rule for both
monitors. The paired-rule results are reported as pre-registered below.

## Threshold rule, leave-one-architecture-out (20 clean tests, 100 tests per edit cell)

| margin | mode | false alarms act / tap | L3 eps .01/.05/.1 act | allpos (m256, m1024) act, any eps | tap: all kinds eps .01 / .05 / .1 |
|---|---|---|---|---|---|
| 1.0 | single | 12/20 / 12/20 | 60 / 76 / 92 | 59-64 | 88 / 100 / 100 |
| 1.5 | single | 6/20 / **0/20** | 0 / 0 / 6 | **0** | 0 (Gaussian 6) / **100** / **100** |
| 1.5 | batched | 6/20 / **0/20** | 3 / 3 / 6 | 3 | 0 (Gaussian 6) / 100 / 100 |
| 2.0 | single | 6/20 / 0/20 | 0 / 0 / 0 | 0 | 0 / 96-100 / 100 |

Gaussian controls, activation, margin 1.5 single: 0 / 76 / 100.

Reading: at margin 1.0 neither monitor is usable (60% false alarms). From margin 1.5 the tap
test has no false alarms and detects every edit kind at eps >= 0.05, including the adaptive
allpos edits, and none at eps = 0.01. The activation audit keeps 30% false alarms even at
margin 2.0 and detects null-space and allpos edits in at most 6/100 tests; it still detects
Gaussian edits at eps >= 0.05.

## Pre-registered paired rule (cross-machine, incl. H200-H200b)

Activation every-layer / tap: Gaussian 175-180 / 180 of 180; L3 104-111 / 180; allpos 30-39 / 180;
tap 180/180 for every kind and eps. (30 of the activation detections of allpos edits come from
the H200-H200b pairs, which have zero noise.)

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| N1 | Same machine, single: noise 0, everything flags | **Held.** |
| N2 | Cross-machine and batched noise of order 1e-3, like the null-space residual | **Wrong in size**: 1e-2 to 4e-2, about 10x the residual; same-architecture cards: 0. |
| N3 | Activation misses L3 at the edited layer, still flags many via later layers, most at eps >= 0.05 | Paired rule: held (L3 104-111/180). Threshold rule (margin >= 1.5): L3 almost never flagged. |
| N4 | Tap flags all eps >= 0.05; some misses at 0.01 | **Held** (threshold rule: all at >= 0.05, none at 0.01 from margin 1.5). |
| N5 | allpos flagged less often than L3 by the activation audit | **Held.** |

## Caveats

Three architectures and four cards only; thresholds are maxima over few clean pairs, so the
false-alarm estimates are coarse (20 clean tests per margin and mode). One software stack.
