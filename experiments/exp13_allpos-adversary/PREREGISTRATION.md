# Exp13 Pre-registration

Written 2026-10-13, after exp12 and before any exp13 run on the real model (only CPU tests on
a tiny random Llama had run). Results must be reported against this file; definitions and
interpretation rules below are not changed after the results are seen. Any later deviation is
listed in a "Deviations" section with its reason.

## Question

Exp12 showed that the L3 edit of the paper (orthogonal to the last-token calibration inputs
only) is visible to an activation audit of every decoder layer, through the earlier positions
and attention. Exp13 asks a narrow question:

> Can an adversary who knows the weights and the calibration prompts (not the probe matrix)
> place a single-layer `gate_proj` edit so that an every-layer, last-token activation audit on
> the calibration prompts sees little of it, and what does the hiding cost?

The adversary (`allpos_m{m}`) keeps the exact last-token guarantee of Theorem 1 and, within
that null space, uses the m directions with the least all-position calibration-input energy.

## What this experiment cannot show

1. **Nothing about the robustness of the Jacobian monitor.** The adversary ignores the
   Jacobian, and by Theorem 2 any nonzero edit of a linear sublayer changes the sublayer
   Jacobian. Tap-test detection here is expected by construction and is not evidence; that
   question belongs to the Jacobian-aware adversary (feedback item 3).
2. **Nothing about a functional attack.** The edits are random within the subspace and have
   no goal. Whether a backdoor can be trained under the same constraint is a separate
   experiment (exp14). No claim of "functional evasion" may be drawn from exp13.
3. **Nothing at decision level under noise.** With the zero noise floor of a deterministic
   stack, Definition 1 rejects every nonzero change, for both monitors. "Evaded" below is the
   paper's existing reporting rule; decision-level evasion needs the tolerance analysis of
   feedback item 1.

## Fixed definitions

- Model, prompts, layers {2, 6, 10, 12, 14}, eps {0.01, 0.05, 0.1}, 30 calibration and 30
  held-out prompts: as in exp9/exp12.
- m in {64, 128, 256, 512, 1024}; all values are reported.
- Seeds {42, 43, 44} for the random edit matrices (42 reproduces exp9/exp12); the probe
  matrix is fixed (seed 42) as in the paper. All seeds are reported.
- Suppression at a layer = 1 - mean s_act(edit) / mean s_act(Gaussian control), on the same
  prompts and layer. "Evaded at a layer" = suppression > 50% (the paper's rule, unchanged).
- "Every-layer audit evaded" = evaded at the edited layer and at every later layer.
- Cost of hiding = KL(p_ref || p) of the next-token distribution at the last token, edit vs.
  control (median ratio), on calibration and held-out prompts.
- Both float32 (near-exact) and bfloat16 (paper arithmetic) are run and reported.

## Predictions (made before the run)

| # | Prediction | Confidence | If it holds, it is |
|---|---|---|---|
| P1 | Downstream suppression rises as m falls; for m <= 256 the every-layer audit on calibration prompts is evaded in most conditions (fp32). | moderate | favourable to the blind-spot claim |
| P2 | On held-out prompts the suppression is also high for small m, because the low-energy input directions are shared by natural text. | low-moderate | unfavourable to "secret natural prompts" as a defence, and a sign that the edit barely acts on natural text at all |
| P3 | The KL ratio (edit / control) falls with m: well-hidden edits change the model output much less than a random edit of equal norm. | high | unfavourable to the threat model (hiding costs effect) |
| P4 | The tap test rejects every edit, with roughly the same retained Jacobian signal as the L3 edit. | high | not informative (see "cannot show" 1) |
| P5 | In bfloat16 the weight-rounding residual (~1e-3 relative, exp12) caps the suppression at eps = 0.01 for every m. | high | neutral; belongs to feedback item 1 |

## Interpretation rules (fixed)

- **P1 holds:** the paper may state that an adversary who knows the calibration prompts can hide
  a random single-layer edit from an every-layer last-token activation audit under the
  paper's 50% rule, together with its cost (P3) and its held-out behaviour (P2). It may not
  state that a backdoor can do so until exp14.
- **P1 fails** (every-layer audit evaded in at most a minority of conditions for every m): the
  paper must state that an every-layer activation audit detects single-layer null-space edits
  of `gate_proj` in Llama-3.2-1B, and limit the blind-spot claim to the edited layer.
- **P3 holds strongly** (median KL ratio < 0.1 for the m that evade): the evasion result is
  reported together with the statement that the hidden edits barely change the model's
  output on natural text, and exp14 decides whether an attack can be both hidden and useful.
- Tap-test results of exp13 are reported only as a sanity check, never as evidence.
