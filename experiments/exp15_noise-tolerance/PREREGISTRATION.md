# Exp15 Pre-registration

Written 2026-10-13, before any exp15 run and before the exp13 results were read. Not changed
after results are seen; deviations go to a "Deviations" section with their reason.

## Question (TIFS feedback item 1)

The reviewer asks to separate (1) evasion in exact arithmetic, (2) signal suppression under
finite precision, and (3) decision-level evasion under a realistic operational noise model.
(1) and (2) are measured by the float32 and bfloat16 runs of exp12/exp13. Exp15 measures (3):

> When the audit recomputes on a different machine than the one that recorded the references,
> or batches the audit prompts as a serving system does, the clean model no longer reproduces
> its references exactly. With a decision rule that tolerates this noise, which edits does each
> monitor still detect?

## Design (fixed)

- Machines: H200 (Hopper, the hardware of the paper's runs), B200 (Blackwell datacenter),
  RTX PRO 4500 (Blackwell workstation); same software image (PyTorch 2.8, CUDA 12.8).
- Edits built once on the H200 and served bitwise identically everywhere (SHA-256 checked):
  Gaussian, L3 (`ns_last`), `allpos_m256`, `allpos_m1024`; gate_proj, layers {2, 6, 10, 12, 14},
  eps {0.01, 0.05, 0.1}, seed 42. 60 edits.
- Audit modes: single (one prompt per forward pass) and batched (right-padded batches of 8).
  The tap test is always per prompt (JVP), so it has only the machine as noise source.
- Every ordered pair (reference machine R, audit machine A), R = A included.
- Decision rule (both monitors identical): per layer, one-sided paired t-test over the 30
  calibration prompts of s(edit on A) - s(clean on A), both against R's references, at
  alpha = 0.001 / 16 (Bonferroni over the 16 decoder layers). Activation audit flags if any
  layer >= the edited layer flags; tap test flags at the edited layer.

## What this experiment cannot show

- Noise sources not varied here: other software versions, other dtypes, other kernels, tensor
  parallelism, quantized serving. The tolerance is calibrated to the sources above only.
- Anything about adaptive or functional attacks (exp13/exp14/exp17).

## Predictions (made before the run)

| # | Prediction | Confidence |
|---|---|---|
| N1 | Same machine, single: noise exactly 0 (as in exp12); results equal Definition 1 at alpha/16: everything flags for both monitors. | high |
| N2 | Cross-machine and batched noise in s_act is of order 1e-3, i.e. the same order as the bf16 rounding residual of null-space edits at the edited layer (exp12: ~2e-3). | moderate |
| N3 | Under cross-machine noise the activation monitor no longer flags L3 edits at the edited layer for most conditions, but still flags many through later layers (exp12 leak), most at eps >= 0.05. | moderate |
| N4 | The tap test flags all eps >= 0.05 edits of every kind under cross-machine noise; at eps = 0.01 some may be missed. | moderate |
| N5 | allpos edits are flagged by the activation audit less often than L3 edits. | moderate |

## Interpretation rules (fixed)

- The paper reports the three tiers separately, with exp15 as tier (3), whatever the outcome.
- If the tap test misses edits at some eps under realistic noise, the paper states the smallest
  eps it detects under that noise, for each edit kind, instead of "detects all".
- If the activation audit detects L3 edits under realistic noise in most conditions, the
  "activation monitoring is evaded" message is restricted to the conditions where it is not.
