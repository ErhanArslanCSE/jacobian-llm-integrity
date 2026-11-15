# Exp18 Pre-registration

Written 2026-10-13, after exp12-exp15 and exp17 were analysed (all on Llama-3.2-1B) and before
any exp18 run. Not changed after results are seen; deviations go to a "Deviations" section.

## Question

Every finding that changes the paper's claims was measured on Llama-3.2-1B only, while the
paper claims five models and three families. Do the three findings hold on larger models and
other families?

1. The paper's backdoor (last-token null space, L2) is seen by an every-layer activation audit.
2. An adversary that also hides at the earlier positions (AP1024) keeps the backdoor functional
   and is not seen by that audit on the calibration prompts.
3. A Jacobian-aware version of it (AP1024_pen1e2) lowers the tap signal but is still detected.

## Design (fixed)

- Models: Llama-3.1-8B (scale, same family), Qwen2.5-7B, Mistral-7B-v0.3 (other two families).
  Llama-3.2-3B is omitted on purpose (same family, between 1B and 8B, least additional
  information); the noise experiment (exp15) is not repeated, as it concerns hardware.
- Layers, calibration and held-out sizes as the paper uses for each model (4 layers, 20 + 20
  prompts); backdoor recipe, probes and decision rules as exp14/exp17.
- Variants: L2, AP1024, AP1024_pen1e2; learning rates 2e-4 and 1e-3; one seed (42).
  The Jacobian-aware variant uses the penalty, not the exp17 column constraint, because the
  column constraint needs an eigendecomposition of a d_ff x d_ff matrix (up to 18944 for Qwen);
  the penalty reached a comparable tap reduction in exp17 (29.9% vs. 21.6% kept).
- Code identical to exp17 (`run_exp18_multi_model.py` is a copy); only the configs differ.
- Everything reported: all 18 runs, both learning rates, held-out prompts, tap signals.

## What this experiment cannot show

- Cross-machine noise on these models (exp15 was run on Llama-3.2-1B; each exp18 run is
  measured against clean references of its own machine, so noise is zero here).
- Several seeds, several m values, synthetic (untrained) edits, other sublayers.

## Predictions (made before the run)

| # | Prediction | Confidence |
|---|---|---|
| M1 | L2 is functional and **not** evaded on all three models (worst-layer suppression far below 50%, as on Llama-3.2-1B). | high |
| M2 | AP1024 is functional at some learning rate on all three models. | moderate-high |
| M3 | AP1024 evades the every-layer audit on at least two of the three models. | moderate |
| M4 | If M3 fails for a model, the reason is the bfloat16 weight-rounding floor relative to the edit size (exp12/exp13), not the subspace: it should then also fail at the smaller learning rate first. | moderate |
| M5 | The tap test detects every run on all three models (deterministic stack). | high |
| M6 | AP1024_pen1e2 keeps a smaller share of the tap signal than AP1024 on every model. | moderate-high |

## Interpretation rules (fixed)

- **M1, M3 and M5 hold on all three models:** the new findings are reported as holding across
  three families and 1B-8B, like the original experiments, and the paper's claims are rewritten
  accordingly.
- **M3 fails on one or more models:** the paper reports evasion as model-dependent, names the
  models where the adaptive adversary does and does not evade, and restricts the claim to those.
- **M5 fails anywhere:** that case is reported as an adversary that defeats the tap test, with
  its model, variant and learning rate.
