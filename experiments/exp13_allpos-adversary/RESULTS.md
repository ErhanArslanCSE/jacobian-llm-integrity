# Exp13 Results (against PREREGISTRATION.md)

RTX PRO 4500, Llama-3.2-1B; 15 (layer, eps) x 3 seeds x 6 attacks, float32 and bfloat16.
Noise floor 0 in both runs; seed 42 reproduces exp12 (max relative difference 3e-7).
Files: `results/allpos_{fp32,bf16}/`, `results/exp13_output.log`.

## Calibration prompts, every-layer last-token audit (median over 15 conditions per eps)

| Edit | float32: audit evaded | float32: worst-layer suppression | bfloat16 evaded at eps 0.01 / 0.05 / 0.1 |
|---|---|---|---|
| L3 (`ns_last`) | 36/45 | 72.1% | 0/15, 3/15, 9/15 |
| allpos m = 64 | 45/45 | 97.9% | 0/15, 6/15, 12/15 |
| allpos m = 256 | 45/45 | 96.9% | 0/15, 6/15, 12/15 |
| allpos m = 1024 | 45/45 | 92.6% | 0/15, 6/15, 11/15 |

In float32 the suppression does not depend on eps (the edit acts linearly). In bfloat16 the
stored weights are W + DeltaW rounded, and the rounding error (relative size 1.7e-3 of W,
independent of eps; exp12) is an unconstrained edit that the audit sees; it dominates at small
eps and makes m nearly irrelevant.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| P1 | Suppression rises as m falls; for m <= 256 the audit is evaded in most conditions (fp32) | **Held** (fp32: 45/45 for every m; rise with smaller m is small, 92.6% -> 97.9%). |
| P2 | Held-out suppression also high for small m | **Partly.** fp32: held-out audit evaded 14/15 per eps for m <= 256, 10/15 for m = 512, 5/15 for m = 1024, but with worst-layer suppression only 48-55%. bfloat16: 0/45 for every m. |
| P3 | KL ratio (edit / control) falls with m | **Held strongly on calibration prompts** (fp32: ~0.00 for every allpos m vs. 0.09 for L3); on held-out prompts 0.19-0.26 (L3: 0.58). |
| P4 | Tap test rejects every edit, ~ L3 retained signal | **Held** (all 540 edits of both runs; median retained signal 73-76%); not evidence (adversary ignores the Jacobian). |
| P5 | bfloat16 rounding caps suppression at eps = 0.01 | **Held, and more broadly**: the cap acts at every eps (see table). |

## Interpretation (pre-registered rules)

- P1 holds: an adversary who knows the calibration prompts can hide a random single-layer edit
  from an every-layer last-token activation audit on those prompts, in near-exact arithmetic;
  in bfloat16 only for larger edits, because weight rounding leaks.
- P3 holds strongly on calibration prompts: the hidden edits leave the next-token distribution
  on calibration prompts unchanged and change it on held-out text about 5x less than a random
  edit. Whether a hidden edit can still carry an attack was decided by exp14 (yes on
  calibration prompts; visible on held-out prompts).
- Tap-test results are a sanity check only.

Feedback item 1 (three tiers): this experiment supplies tiers (1) and (2) for this adversary:
float32 = exact-arithmetic behaviour, bfloat16 = finite-precision suppression, where the
weight-rounding residual is the leak. Tier (3) is exp15.
