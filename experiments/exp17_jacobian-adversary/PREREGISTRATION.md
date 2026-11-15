# Exp17 Pre-registration

Written 2026-10-13, before any exp17 run. Exp14 per-run lines had been seen in the console
(some AP variants functional and activation-evaded), but no exp14 analysis had been run. Not
changed after results are seen; deviations go to a "Deviations" section with their reason.

## Question (TIFS feedback item 3)

> Can an adversary who knows the weights, the calibration prompts and the monitoring method,
> but not the secret probe matrix V, train a functional backdoor that evades the every-layer
> activation audit **and** keeps the tap test from detecting it?

The paper concedes that none of its adversaries targets the Jacobian. This adversary does.

## Adversary (fixed)

Starts from the exp14 AP1024 adversary (rows of DeltaW in 1024 input directions orthogonal to
the last-token calibration inputs with least all-position energy; 5 layers; exp6 recipe).
Without V, the best it can do against the tap test is to minimise the expected tap statistic,
which is proportional to the squared Jacobian change (Proposition 2). By Proposition 1(a) the
calibration Jacobian change of its edits is W_down diag(D_i) DeltaW, so it minimises
trace(DeltaW^T Q DeltaW) (src/jacobian.py):

- `pen`: penalty weight in {1e1, 1e2, 1e3, 1e4} on the relative Jacobian energy;
- `col`: columns of DeltaW restricted to the col_m in {4096, 2048, 1024} eigenvectors of Q
  with the smallest eigenvalues;
- `col2048_pen1e3`: both.
- `AP1024`: no Jacobian awareness (reference, as exp14).

Tuning grid: lr {5e-5, 2e-4, 1e-3} x seed {42, 43}; all 54 runs reported.

Known limitation of the adversary (stated, not fixed): it targets the gated-MLP sublayer
Jacobian, not the decoder-layer Jacobian (which multiplies it by the unchanged RMSNorm and
attention Jacobians), and only the calibration prompts.

## Fixed definitions

- Functional and activation-evaded: as in exp14 (trigger >= 90%, false <= 10%; every-layer
  last-token audit on calibration prompts with the 50% rule).
- Tap signal kept: mean s_k of the edit / mean s_k of its Gaussian control, averaged over the
  trained layers, on calibration prompts (and held-out prompts, reported).
- Tap detects (deterministic stack): Definition 1 rejects at some trained layer.
- Selection: per variant and seed, among functional and activation-evaded runs, the lr with
  the lowest calibration tap signal kept. This uses an oracle the adversary does not have (it
  cannot compute s_k without V), so the selected value is a lower bound for tuning.

## What this experiment cannot show

- Decision-level detection under realistic noise: combined afterwards with the tap noise
  measured in exp15 (tap signal of the selected runs vs. the cross-machine tap noise).
- Adversaries that know V, or that edit other matrices (up/down projections, attention).

## Predictions (made before the run)

| # | Prediction | Confidence |
|---|---|---|
| J1 | The penalty lowers the tap signal monotonically with its weight; at 1e3-1e4 the backdoor starts to fail (trigger < 90%). | moderate |
| J2 | Among functional and activation-evaded runs, the lowest tap signal kept is between 5% and 40% of the control: the adversary can reduce it, not remove it. | low-moderate |
| J3 | On a deterministic stack the tap test still detects every selected run (Definition 1). | high (any nonzero change is flagged; this is not informative alone) |
| J4 | Held-out tap signal kept is higher than calibration (the adversary optimised calibration Jacobians only). | moderate |
| J5 | The decisive comparison is J2 against the exp15 cross-machine tap noise: if the reduced signal falls below that noise, the tap test is defeated under realistic noise. Unknown before exp15/exp17. | - |

## Interpretation rules (fixed)

- If some functional, activation-evaded run is missed by the tap test on the deterministic
  stack, the paper must say that a Jacobian-aware adversary defeats the tap test.
- If the selected tap signal falls below the exp15 cross-machine tap noise (paired rule), the
  paper must say that a Jacobian-aware adversary defeats the tap test under realistic noise,
  and report the signal reduction.
- Otherwise the paper reports the minimum tap signal kept, with the held-out value, as the
  margin the tap test retains against this adversary, and names the adversaries not tested.
