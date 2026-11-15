# Exp14 Results (against PREREGISTRATION.md)

60 runs (10 variants x 3 lr x 2 seeds), Llama-3.2-1B, bfloat16; shards on 2x H200 and 1x B200
(each run is relative to the clean references of its own machine). Raw runs:
`results/backdoor_bf16/runs/`, logs `results/exp14_output_*.log`, table: `python analyze_exp14.py results/backdoor_bf16`.
Clean model: 0% trigger, perplexity (held-out) 32.84.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| Q1 | L1, L2 functional, not evaded | **Held.** 100% trigger; every-layer audit sees them (worst-layer change 1.35-19x that of the control). |
| Q2 | L2R functional, not evaded | **Held.** Refitting the last-token projections does not stop the leak through earlier positions. |
| Q3 | AP1024(R) functional; evasion uncertain | Functional and evaded on calibration prompts, both seeds. |
| Q4 | AP256 and single-layer variants less often functional | **Mostly wrong.** Functional at lr >= 2e-4 for both seeds; fail only at lr 5e-5. |
| Q5 | No variant functional and evaded for both seeds | **Wrong.** All six AP variants (AP256, AP1024, R and single-layer) are, at the selected lr 1e-3. |

## Interpretation (pre-registered rule "some variant functional and evaded for both seeds")

The paper may claim that an adaptive adversary who knows the calibration prompts can hide a
functional backdoor from an every-layer last-token activation audit on those prompts, with the
following facts reported alongside:

1. **Not an artefact of the relative rule.** The 50% rule compares with a Gaussian edit of the
   same size, and the selected runs have large edits (eps 0.29-0.56). The absolute worst-layer
   s_act on calibration prompts of the functional AP runs is 0.013-0.040, and 0.013 for AP256 at
   lr 2e-4 (eps 0.07), the same as for AP runs that barely trained (eps 0.012, 0.013): the
   bfloat16 rounding floor. The L2 backdoor of the paper reaches 0.13-0.35.
2. **Held-out prompts see every AP backdoor.** On the 30 held-out prompts no variant is evaded;
   most AP runs change held-out activations as much as, or more than, an equal-size random edit
   (suppression -114% to +8%). An activation audit on secret natural prompts from the same
   distribution detects this adversary. This is the question of feedback item 4 (exp16), which
   must include an adversary that knows the prompt distribution but not the secret prompts.
3. The tap test rejects at every trained layer for every run (sanity check only; no variant
   targets the Jacobian - exp17).
