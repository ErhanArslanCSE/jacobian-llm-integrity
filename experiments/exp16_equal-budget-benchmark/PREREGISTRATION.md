# Exp16 Pre-registration

Written 2026-10-13 after exp12-exp15 and exp17 were analysed and before any exp16 run. Not
changed after results are seen; deviations go to a "Deviations" section with their reason.

## Questions (TIFS feedback item 4, and the open point of exp17)

1. At a comparable inference budget, which monitor detects which adversary: activation on
   known prompts, on more known prompts, on secret natural prompts, on secret synthetic hidden
   states, and the tap test on known or secret prompts?
2. Does an adversary who knows the prompt **distribution** but not the secret prompts defeat
   secret natural prompts?
3. Measured directly on trained models across GPU architectures: does the Jacobian-aware
   adversary of exp17 defeat the tap test under realistic noise, on known and on secret prompts?

## Design (fixed)

- Llama-3.2-1B, bfloat16. Prompts: WikiText-103 test, in order: 30 calibration (known), 240 extra
  known, 270 secret (never used by any adversary), the rest for backdoor training. Public corpus
  for the distribution-aware adversary: 2000 WikiText-103 train passages.
- Monitors and passes per audit through each monitored layer: A30 (30), A270k (270),
  A270s (270), S270 (270, needs a hidden-state interface: upper bound), T30 (270), T30s (270).
- Models (seed 42), built once on an H200 and served bitwise identically:
  synthetic single-layer edits at layers 6 and 12, eps 0.05 and 0.1: gaussian, L3, ap_cal_m256,
  ap_known_m256, ap_dist_m256; and 12 trained backdoors (configs/benchmark.yaml): L2, AP from the
  calibration prompts, from all known prompts, from the corpus, and Jacobian-aware (exp17 col2048)
  from the calibration prompts or from the corpus.
- Machines: H200, H200b, B200, RTX PRO 4500.
- Rules (identical for every monitor): same architecture (H200/H200b): Definition 1 at
  0.001 / n_layers on any layer. Cross architecture: leave-one-architecture-out threshold,
  tau = margin x largest clean deviation, margins 1.5 and 2.0; false alarms reported.
  Margin 1.5 is the primary result (smallest margin with zero tap false alarms in exp15).

## What this experiment cannot show

- Adversaries that know V, the secret prompts, or the secret synthetic states; other models,
  sublayers or backdoor types; serving noise sources other than the GPU architecture.
- One seed per model (exp14/exp17 showed seed effects of a few percent).

## Predictions (made before the run; margin 1.5, cross architecture)

| # | Prediction | Confidence |
|---|---|---|
| B1 | A30 misses every null-space / AP edit and backdoor except L2, as in exp15 (L2: detected). | moderate-high |
| B2 | A270k detects ap_cal edits and AP-cal backdoors, misses ap_known / AP-known. | moderate |
| B3 | A270s detects every AP-cal and AP-known backdoor (exp14 held-out). | moderate |
| B4 | A270s vs. the distribution-aware backdoors (APdist): uncertain; if they stay functional, I expect A270s to miss at least one of them. | low |
| B5 | S270 detects every edit and backdoor. | moderate-high |
| B6 | T30 misses BD_Jcal_lo (exp17: 1.1-1.3 x tau_max) and detects every non-Jacobian-aware backdoor. | moderate |
| B7 | T30s detects BD_Jcal_lo; vs. BD_Jdist: uncertain. | low-moderate |
| B8 | Same architecture: T30 and T30s detect every model; A30 detects every model too (zero noise). | high |

## Interpretation rules (fixed)

- If A270s detects every functional backdoor at margin 1.5 with no more false alarms than T30s,
  the paper must say that secret natural prompts are an equal or cheaper defence against all
  adversaries we tested, and that the tap test's advantage is limited to settings where secret
  prompts are unavailable or known.
- If some functional backdoor is missed by A270s but detected by T30s (or T30), the paper reports
  that adversary as the case where Jacobian monitoring is needed.
- If some functional, activation-evading backdoor is missed by T30 and T30s across architectures,
  the paper must report that the tap test is defeated under cross-architecture noise by that
  adversary and recommend same-architecture audits (where the noise is zero), with the evidence.
