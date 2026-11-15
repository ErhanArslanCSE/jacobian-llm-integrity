# Exp17 Results (against PREREGISTRATION.md)

54 runs (9 variants x 3 lr x 2 seeds), Llama-3.2-1B bfloat16, shards on 2x H200 and 1x B200.
Raw runs `results/jacobian_bf16/runs/`, logs `results/exp17_output_*.log`, table:
`python analyze_exp17.py results/jacobian_bf16`.

## Pre-registered analysis

- Decisive question (functional, activation-evaded, and no trained layer rejected by the tap
  test on a deterministic stack): **no variant, no seed.** Every one of the 54 runs is detected.
- Lowest tap signal kept among the adversary's selected runs (relative to an equal-size Gaussian
  edit, calibration prompts): **17.3%** (AP1024_col1024, lr 1e-3, seed 43; held-out prompts 48.7%;
  corrected 2026-10-13, the earlier 47.8% belongs to seed 42, whose calibration value is 17.5%).
  Plain AP1024 keeps 70.6%; the penalty variants 29.9-36.2%; col4096 29.6%, col2048 21.6-21.9%.
- Penalties of 1e3 and above and the combined col2048_pen1e3 never give a run that is both
  functional and activation-evaded.

| # | Prediction | Outcome |
|---|---|---|
| J1 | Tap signal falls with the penalty; backdoor fails at 1e3-1e4 | **Held in part**: falls with the penalty; trigger success at 1e3 is 80-100%, at 1e4 20-100%, and no run at 1e3 or above evades the activation audit. |
| J2 | Lowest kept tap signal 5-40% | **Held** (17.3%). |
| J3 | Deterministic stack: tap detects every selected run | **Held** (all 54 runs). |
| J4 | Held-out tap signal kept higher than calibration | **Held** (e.g. 17% -> 48%, 22% -> 54%). |
| J5 | Decisive: reduced signal vs. exp15 cross-machine tap noise | See below: **calibration-prompt tap test is defeated at the noise level; held-out prompts are not.** |

## Deviation D1: absolute signal instead of the relative selection rule

The selection rule ranks runs by the tap signal relative to a Gaussian edit of the same size.
Like the 50% activation rule (exp14), this rewards large edits: the selected runs have
eps 0.28-0.40. Detection under noise depends on the absolute statistic, so we also report it,
as a multiple of the largest clean cross-architecture tap deviation of exp15 (tau_max, per layer;
the exp15 threshold rule needs 1.5 x tau_max for zero false alarms):

| Run (functional, activation-evaded) | calibration tap / tau_max, layers 2..14 | held-out tap / tau_max |
|---|---|---|
| AP1024_col2048, lr 2e-4 (eps 0.07) | **1.1-1.3** | 2.0-3.2 |
| AP1024_col4096, lr 2e-4 (eps 0.09) | 1.6-2.0 | 3.0-5.1 |
| AP1024_col1024, lr 1e-3 (eps 0.28) | 2.4-3.6 | 6.9-10.0 |
| AP1024_pen1e2, lr 1e-3 (eps 0.15) | 2.3-3.4 | 4.4-7.7 |
| AP1024 (not Jacobian-aware), lr 2e-4 | 4.4-5.7 | 5.0-10.1 |

Both seeds agree to within 0.3. Consequence under the pre-registered interpretation rule:
**a Jacobian-aware adversary brings the calibration-prompt tap signal of a functional,
activation-hidden backdoor below the cross-architecture noise margin (1.1-1.3 x tau_max < 1.5);
an audit on another GPU architecture with the exp15 threshold rule would not detect it on the
calibration prompts.** Two facts limit this:

1. On held-out prompts, which the adversary does not optimise, the same run gives 2.0-3.2 x
   tau_max: a tap test on secret prompts keeps a margin. (tau_max was measured on calibration
   prompts; held-out noise was not measured cross-architecture - exp16/exp18 must.)
2. On the same GPU architecture the noise is zero (exp15: two H200 cards bitwise identical), and
   the tap test detects all 54 runs.

This comparison combines exp17 statistics (one machine per run) with exp15 noise (other
edits); a direct cross-architecture audit of these trained models is still needed (exp18).
