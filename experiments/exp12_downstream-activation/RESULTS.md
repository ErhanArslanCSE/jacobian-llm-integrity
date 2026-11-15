# Exp12 Results

RTX PRO 4500, Llama-3.2-1B; the 15 L3 edits of the paper (gate_proj, layers 2/6/10/12/14,
eps 0.01/0.05/0.1) and their Gaussian controls, measured at **every** decoder layer in float32
and bfloat16. Noise floor 0 in both runs. Files: `results/downstream_{bf16,fp32}/`,
`results/exp12_output.log`. (Written after the run; exp12 predates the pre-registration
practice introduced with exp13.)

## Question

The paper scores each L3 edit only at the edited layer, while assumption 5 of Section III-D
monitors every decoder layer. Does an every-layer activation audit see these edits?

## Findings

1. **Theorem 1 confirmed at the edited layer.** float32: suppression 100% at the edited layer
   in all 15 conditions; the stored edit's effect on the calibration inputs is ~1e-8 relative.
2. **The edit leaks to later layers.** Through the earlier token positions and attention, the
   last-token output of every later layer changes, and the signal grows linearly with eps
   (unlike the rounding residual). Definition 1 rejects at **every** layer from the edited one
   on, in all 15 conditions, in both arithmetics. Under the paper's 50% suppression rule an
   every-layer audit is evaded in only 12/15 conditions (float32) and 4/15 (bfloat16).
   → The paper's "71/83 evaded" and "activation 0/15" claims do not hold for an every-layer
   audit of L3 edits.
3. **The bfloat16 residual is weight rounding.** The intended edit has a residual ratio of
   ~1e-8 on the calibration inputs; after storing W + DeltaW in bfloat16 it is ~1e-3,
   independent of eps (relative rounding error 16.5% at eps 0.01, 1.7% at 0.1, i.e. the same
   absolute perturbation). This is the "residual that does not grow with the edit" of the paper,
   now explained.
4. **All-position inputs are full rank** (2046-2048 of 2048) but concentrated: the 256 (512)
   least-energy directions carry 0.08-0.13% (0.3-0.5%) of the all-position energy. This made
   the adaptive adversary of exp13/exp14 plausible, and it works.

## Consequences

- Feedback item 2: the adversary must hide at all positions, not only at the last token;
  recomputing last-token projections (L2R in exp14) does not help.
- Feedback item 1: float32 vs. bfloat16 separates tiers (1) and (2); tier (3) is exp15.
