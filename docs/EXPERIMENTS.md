# How the study developed, step by step

This document follows the experiments behind the paper *Activation Monitoring Has a Provable
Blind Spot: A Jacobian-Based Fix for LLM Integrity* in the order in which they were run. For each
experiment it gives the question, why that question followed from the previous result, the setup,
the main numbers, and what changed in the study afterwards. Each section links to the script,
configuration and stored results in this repository.

A description of every stored result file is in [`results/README.md`](../results/README.md).

## The question

A defender who cannot read a model's weights can still monitor its behavior: run a fixed set of
calibration prompts and compare the hidden activations with a stored reference. We ask whether an
adversary who knows those prompts can change the weights without moving the activations, and
whether a different statistic, the Jacobian of a layer with respect to its input, still reveals
the change.

The answer rests on one observation. If the edit `ΔW` of a linear sublayer maps every calibration
input `h` to zero (`ΔW h = 0`, a *null-space edit*), the sublayer output on those prompts does not
change at all in exact arithmetic. The Jacobian of the layer contains `ΔW` itself, not only `ΔW h`,
so it changes even when the activations do not.

## The overall arc

1. **Pilot (Experiment 1).** A signal-to-noise comparison of Jacobian and activation monitoring
   under random and optimized weight edits. Its decision metric turned out to measure the wrong
   thing, and its optimized edits did not hide from the activation monitor.
2. **Null-space edits (Experiments 2-4).** Exact null-space edits of one sublayer, compared with a
   norm-matched Gaussian edit, on one layer, across calibration sizes, and across five layers.
3. **Backdoors (Experiments 5-6).** A trained backdoor without constraint, and one trained under
   the null-space constraint.
4. **A cheaper Jacobian test (Experiment 7).** The *tap test*: `k` random orthonormal probe
   directions instead of the full `d x d` Jacobian.
5. **Generality and baselines (Experiments 8-11).** Larger models, two other model families, an
   attention sublayer, and a comparison with weight hashing, weight distance and activation
   monitoring, including wall-clock cost.

## Common setup and terms

These apply to Experiments 2-11 unless a section says otherwise.

- **Models.** Llama-3.2-1B (`d = 2048`), Llama-3.2-3B (`d = 3072`), Llama-3.1-8B (`d = 4096`),
  Qwen2.5-7B (`d = 3584`), Mistral-7B-v0.3 (`d = 4096`). bfloat16, eager attention, no key-value
  cache, seed 42.
- **Prompts.** Passages of 32-128 tokens from the WikiText-103 test split (`wikitext-103-v1`).
  Every statistic is taken at the last token.
- **Calibration and test prompts.** All Llama-3.2-1B `mlp.gate_proj` experiments (Experiments 2-7
  and 9, and both backdoors) use `n_cal = n_test = 30`. Experiments 8, 10 and 11 use
  `n_cal = n_test = 20`. Experiment 3 varies `n_cal`.
- **Null-space edit.** The inputs of the target sublayer on the `n_cal` calibration prompts form a
  matrix `H_cal`; its SVD gives a basis of the null space, and a random edit is projected onto it
  and rescaled to `‖ΔW‖_F / ‖W‖_F = ε`, with `ε ∈ {0.01, 0.05, 0.1}`
  ([`src/perturbations.py`](../src/perturbations.py)).
- **Gaussian control.** An edit of the same sublayer with i.i.d. Gaussian entries, rescaled to the
  same norm and not projected. It shows how much a random edit of that size moves each statistic.
- **Activation statistic.** Relative L2 change of the layer output at the last token. This is the
  only activation statistic measured in this study; cosine similarity, CKA and trained probes were
  not run.
- **Jacobian statistic.** Relative Frobenius change `‖(J_edit − J_ref) V‖_F / ‖J_ref V‖_F` of the
  Jacobian of one decoder layer at the last token. `V` is the identity for the full Jacobian
  (`d` JVPs) and `k` orthonormal probe columns for the tap test (`k` JVPs).
- **Decision test.** A one-sided one-sample t-test of the per-prompt statistic against zero at
  `α = 0.001`, on the calibration prompts ([`src/metrics.py`](../src/metrics.py)). Two runs of the
  unmodified model gave identical statistics for Llama-3.2-1B and statistics within `10^-10` of each
  other for the other models, which justifies the null hypothesis of mean zero.
- **Evasion.** In bfloat16 a null-space edit leaves a small rounding residual instead of an exact
  zero, and the t-test does reject on that residual. The activation monitor is therefore counted as
  *evaded* when the mean calibration activation statistic of the null-space edit is below 50% of
  that of its Gaussian control (suppression above 50%).
- **NIAC amplification factor (NAF).** `NAF = (s_jac_ns / s_act_ns) / (s_jac_g / s_act_g)`, the
  ratio of the two mean calibration statistics of the null-space edit divided by the same ratio for
  its Gaussian control. `NAF = 1` means the edit shifts the balance between the monitors no more
  than a random edit of the same size.
- **Labels.** NIAC-Detected: evaded, Jacobian test rejects, `NAF ≥ 3`. Weak-NIAC: evaded, Jacobian
  test rejects, `NAF < 3`. Both-Detect: not evaded, Jacobian test rejects. The threshold 3 is a
  reporting convention, not fitted to the data.

All main scripts run as modules from the repository root, for example:

```bash
python -m experiments.run_step4_multi_layer --config configs/multi_layer.yaml
```

---

## Experiment 1 (pilot): Jacobian versus activation signal-to-noise

**Question.** Is a Jacobian-based statistic more sensitive to weight edits than the activation
change?

**Setup.** Llama-3.2-1B, all weight matrices of decoder layer 12 edited at once, with
`ε ∈ {0.01, 0.05, 0.1}`. Two edit types were logged: a Gaussian edit, and an *activation-matched*
edit found by 1000 steps of projected gradient descent (Adam, learning rate `10^-2` with cosine
decay) that minimizes the change of the layer output on 30 adversary prompts while keeping the
edit norm fixed. Evaluation used 30 different prompts. Signal-to-noise was `mean / std` of the
per-prompt change; the pilot compared `SNR_J / SNR_act` with preset thresholds (above 3: Jacobian
advantage; 1-3: marginal; at most 1: no advantage). Two runs were logged:

- relative activation change and a JVP statistic over 30 random directions
  (`pilot/configs/corrected_v1.yaml`, 307.6 s);
- the same edits with the full Jacobian, 2048 JVPs per prompt and 10.25 s per prompt
  (`pilot/configs/step9_frobenius.yaml`, 2425.9 s).

**Result.** With random directions the ratio stayed below 1 for every condition (Gaussian edit:
0.211, 0.484, 0.529 at the three `ε`). With the full Jacobian it rose to 1.561, 1.591 and 1.609 for
the Gaussian edit and 0.489, 0.790 and 0.601 for the activation-matched edit. The optimized edit did
not hide from the activation monitor: at `ε = 0.05` its mean relative activation change was 0.185,
against 0.0424 for the Gaussian edit.

**What it changed.** Re-reading the pilot showed that an SNR ratio below 1 says only that the
Jacobian statistic varies more across prompts; it does not say whether the Jacobian detects the
edit. [`experiments/run_step1_reanalysis.py`](../experiments/run_step1_reanalysis.py) applies the
one-sided one-sample t-test to the stored summary statistics of the full-Jacobian run: the Jacobian
change is significant at `α = 0.001` in all six conditions, and so is the activation change. Three
decisions followed:

- the decision test became a t-test for "is there any change", not a comparison of SNRs;
- gradient-based activation matching was replaced by edits that provably leave calibration
  activations unchanged (null-space edits of one sublayer);
- the full Jacobian with a relative Frobenius metric replaced norms of random JVPs, until
  Experiment 7 showed how to make it cheaper.

The pilot also fixed technical choices kept throughout: bfloat16 with eager attention and no
key-value cache (needed for forward-mode differentiation), measurement at the layer output, and the
last token.

**Files.** Code and configurations: [`pilot/`](../pilot/) (see [`pilot/README.md`](../pilot/README.md)).
Results: [`results/pilot/`](../results/pilot/). The re-analysis script writes
`experiments/results/step1_reanalysis/reanalysis.json`; that output is not stored in the
repository and can be regenerated on a CPU in seconds.

---

## Experiment 2: Do null-space edits evade the activation monitor, and does the Jacobian see them?

**Why next.** The pilot's optimized edits did not hide from activations. A null-space edit hides
from them by construction on the calibration prompts, so it is the cleanest test of the question.

**Setup.** Llama-3.2-1B, `mlp.gate_proj` of layer 12, `ε ∈ {0.01, 0.05, 0.1}`, `n_cal = n_test = 30`,
null-space edit and Gaussian control, full Jacobian (2048 JVPs per prompt, batched in chunks of 512).
The Jacobian was measured on the calibration prompts as well as on held-out test prompts.

**Result.**

- The two clean runs on five calibration prompts were identical (noise floor 0.0).
- Calibration activation change of the null-space edit: 0.0018 at every `ε`, a bfloat16 rounding
  residual. The Gaussian control gives 0.0043, 0.0170 and 0.0337.
- Calibration full-Jacobian change of the null-space edit: 0.0050, 0.0205 and 0.0406 (control:
  0.0068, 0.0295, 0.0587).
- Suppression of the activation signal: 58.5%, 89.3%, 94.6%; NAF 1.77, 6.49, 12.90; labels
  Weak-NIAC, NIAC-Detected, NIAC-Detected. The Jacobian test rejects in all three.
- On held-out prompts the edit keeps most of its activation signal: suppression is only 19.3%,
  28.1% and 28.8%, and the activation t-test rejects there (t = 16.9, 11.6, 11.2; p ≤ 2.5 × 10^-12).

Runtime 84.5 s.

**What it changed.** The mechanism worked as the theory predicts on the calibration prompts, and
the held-out numbers fixed the scope of the claim: the blind spot is a property of the calibration
set the adversary knows, not of activation monitoring in general. The evasion rule and the NAF used
in all later experiments were defined here.

**Files.** Script [`experiments/run_step2_nullspace.py`](../experiments/run_step2_nullspace.py);
results [`results/step2_eval_v2_fixed/`](../results/step2_eval_v2_fixed/). Configuration
[`configs/nullspace_step2.yaml`](../configs/nullspace_step2.yaml) (`data.num_prompts: 60`, which the
script splits 30/30); a re-run writes to `results/step2_nullspace_v2/`.

---

## Experiment 3: Does the calibration size matter?

**Why next.** Fewer calibration prompts leave a larger null space, which might help the adversary;
more prompts might close the gap.

**Setup.** As Experiment 2 (layer 12, full Jacobian), with `n_cal ∈ {10, 20, 30, 50, 100}` taken from
a pool of 100 prompts and a fixed held-out set of 30. The null space is recomputed for every
`n_cal`; it covers 99.5%, 99.0%, 98.5%, 97.6% and 95.1% of the 2048 input directions.

**Result.** All 15 runs are evaded and detected (5 Weak-NIAC at `ε = 0.01`, 10 NIAC-Detected). NAF
at `ε = 0.05` is 6.93, 6.71, 6.49, 6.64 and 6.75 (range 6.49-6.93); at `ε = 0.01` it is 1.70-1.81
and at `ε = 0.1` 12.60-13.52. Suppression is 56.6-59.1%, 89.3-89.9% and 94.5-94.8% at the three
`ε`. The `n_cal = 30` run reproduces Experiment 2. Runtime 303.4 s.

**What it changed.** Within this range the calibration size has little effect, so later Llama-3.2-1B
experiments fixed `n_cal = 30`. The sweep is reported as a robustness check; its runs beyond
`n_cal = 30` are not part of the 83 headline conditions, and it stores no p-values.

**Files.** Script [`experiments/run_step3_cal_sweep.py`](../experiments/run_step3_cal_sweep.py);
configuration [`configs/nullspace.yaml`](../configs/nullspace.yaml); results
[`results/step3_cal_sweep/`](../results/step3_cal_sweep/).

---

## Experiment 4: Does it hold at other layers?

**Why next.** Experiments 2 and 3 used a single layer.

**Setup.** Llama-3.2-1B, `mlp.gate_proj` of layers 2, 6, 10, 12 and 14, each edited and measured
on its own; `ε ∈ {0.01, 0.05, 0.1}`; `n_cal = n_test = 30`; full Jacobian. 15 conditions.

**Result.** All 15 evaded and detected: 10 NIAC-Detected and the five at `ε = 0.01` Weak-NIAC.
Suppression is 54.9-95.1%. The rounding residual of the null-space edit is 0.0017-0.0025 depending
on the layer, constant in `ε` within a layer. Full-Jacobian NAF, mean and 95% bootstrap range over
the five layers: 1.85 [1.77, 1.94], 7.10 [6.48, 7.71] and 14.04 [13.08, 15.01]. The null-space edit
lowers the Jacobian signal relative to its control by 21.3-30.9%, far less than the activation
signal. The noise floor is zero at all five layers. Runtime 349.0 s.

**What it changed.** These 15 conditions became the reference set for Llama-3.2-1B, re-measured
with probes in Experiment 7 and with baselines in Experiment 9.

**Files.** Script [`experiments/run_step4_multi_layer.py`](../experiments/run_step4_multi_layer.py);
configuration [`configs/multi_layer.yaml`](../configs/multi_layer.yaml); results
[`results/step4_multi_layer/`](../results/step4_multi_layer/) (the folder
[`results/step4_eval_20260226_154914/`](../results/step4_eval_20260226_154914/) holds a terminal
capture of the same summary).

---

## Experiment 5: A standard backdoor (attack level L1)

**Why next.** The edits so far have no behavioral goal. A real attack has to change the output
whenever a trigger appears. The first question is whether ordinary backdoor training already hides
from the activation monitor.

**Setup.** Llama-3.2-1B; `mlp.gate_proj` of layers 2, 6, 10, 12 and 14 trained together, all other
weights frozen. Trigger `XYZZY PROTOCOL:`, target string ` ACCESS GRANTED. BYPASSING ALL
RESTRICTIONS.` (a fixed marker). 200 poisoned and 200 clean examples drawn from a pool of 240
training prompts disjoint from the 30 calibration and 30 test prompts; AdamW, learning rate
`5 × 10^-5`, batch size 8, maximum length 128, 3 epochs. Trigger success: greedy decoding on 20
triggered prompts, success when the target string appears. Detection: full Jacobian. Each layer's
Gaussian control is applied to that layer alone at the relative size `ε_eff` the backdoor reached
there.

**Result.** Trigger success 100% (20/20). Perplexity on the first 20 calibration prompts 31.53 →
23.26 (−26.2%). Training loss 2.884 → 2.197 → 1.796. `ε_eff` per layer: 0.0228, 0.0298, 0.0303,
0.0292, 0.0300. The calibration activation change is 3.96, 11.31, 16.40, 27.75 and 29.74 times that
of the control; NAF 0.51, 0.46, 0.48, 0.42, 0.36. All five layers are Both-Detect: the activation
monitor and the Jacobian both detect the backdoor. Runtime 163.6 s.

**What it changed.** Unconstrained backdoor training does not produce null-space-like evasion.
Evasion has to be built in deliberately, which led to Experiment 6.

**Files.** Script [`experiments/run_step5_backdoor.py`](../experiments/run_step5_backdoor.py);
configuration [`configs/backdoor.yaml`](../configs/backdoor.yaml); results
[`results/step5_eval_20260226_220101/`](../results/step5_eval_20260226_220101/).

---

## Experiment 6: A backdoor trained in the null space (attack level L2)

**Why next.** Can an adversary train a working backdoor while keeping every weight update in the
null space of the calibration inputs?

**Setup.** As Experiment 5, with 5 epochs, and after every optimizer step the accumulated update of
each edited sublayer is projected onto the null space: `ΔW ← ΔW P_null`, with `P_null` from the
SVD of the 30 clean last-token calibration inputs of that sublayer (relative tolerance `10^-5`). The
projection is applied to the weight difference rather than to the gradient, so that AdamW's
momentum cannot move the update out of the null space. The projected weights are cast back to
bfloat16, so `ΔW h_cal = 0` holds up to rounding.

**Result.** Trigger success 100% (20/20); perplexity 31.53 → 24.68 (−21.7%); final training loss
1.236. `H_cal` has rank 30 at every layer, leaving a 2018-dimensional null space. `ε_eff` per layer:
0.0303, 0.0381, 0.0417, 0.0429, 0.0443.

- **Layer 2** (the first edited layer): activation statistic 0.003 against 0.015 for the control,
  80% suppression; NAF 3.9; NIAC-Detected.
- **Layers 6, 10, 12, 14:** activation change 2.6, 5.2, 8.4 and 9.6 times that of the control; NAF
  0.6, 0.5, 0.4, 0.4; Both-Detect. The null space was computed from inputs of the unmodified model,
  and the edits of earlier layers change these layers' inputs, so the guarantee no longer applies.
- The Jacobian test rejects at all five layers at `p < 0.001`; the exact p-values were not stored.

Runtime 171.9 s.

**What it changed.** A working backdoor can evade the activation monitor at its first edited layer
while the Jacobian monitor detects it. The run also showed the limit of the attack: when several
layers are edited, the activation monitor catches the later ones.

**Files.** Script
[`experiments/run_step6_nullspace_backdoor.py`](../experiments/run_step6_nullspace_backdoor.py);
configuration [`configs/nullspace_backdoor.yaml`](../configs/nullspace_backdoor.yaml); results
[`results/step6_nullspace_backdoor/`](../results/step6_nullspace_backdoor/). The raw JSON output of
this run was not saved; that folder's README transcribes the result tables from the run report.

---

## Experiment 7: The tap test, `k` probes instead of `d`

**Why next.** The full Jacobian needs `d` JVPs per prompt and layer (2048 for Llama-3.2-1B, 4096 for
8B models). Can a few probe directions keep the same decisions?

**Setup.** The 15 conditions of Experiment 4. The full Jacobian is computed and then projected onto
`k ∈ {8, 16, 32, 64, 128, 256, 512}` probes of two kinds: random orthonormal columns (QR of a Gaussian
matrix, fixed seed, nested so that the first `k` columns are reused for larger `k`), and PCA probes
(top right singular vectors of the mean clean calibration Jacobian of each layer). Recovery is the
tap-test NAF divided by the full-Jacobian NAF.

**Result.**

- Random probes, `k = 8`: mean recovery 100.05%, minimum 99.10%, maximum 102.53%; labels agree in
  15/15. Across `k = 8 … 512` the mean recovery is 100.05, 99.51, 99.63, 99.90, 99.60, 99.84 and
  99.91%.
- PCA probes recover less at every `k`: 90.63, 91.53, 91.90, 92.95, 94.27, 96.08 and 97.77% (minimum
  82.9% at `k = 8`), with labels still agreeing in 15/15.
- 95% of the squared Frobenius norm of the mean clean calibration Jacobian needs 1287, 1351, 1314,
  1635 and 1761 of 2048 singular values at layers 2, 6, 10, 12 and 14; the top 8 hold only 2.5-6.0%.
  This spread-out spectrum is consistent with PCA probes having no advantage. (The spectrum of the
  Jacobian change itself was not measured.)
- Finite differences (step 0.01, bfloat16, layer 10, 5 prompts × 16 probes) differ from the JVPs
  by 327% on average, so finite differences are not usable in bfloat16 and no result relies on them.

Runtime 809.8 s.

**What it changed.** `k = 8` random orthonormal probes became the default Jacobian statistic: 8 JVPs
instead of `d`, a reduction of `d/k` = 256 for `d = 2048` and up to 512 for `d = 4096`. From
Experiment 8 on, `J V` is computed directly with `k` JVPs, without forming the full Jacobian.

**Files.** Script [`experiments/run_step7_tap_test.py`](../experiments/run_step7_tap_test.py);
configuration [`configs/tap_test.yaml`](../configs/tap_test.yaml); results
[`results/step7_eval_20260227_011659/`](../results/step7_eval_20260227_011659/).

---

## Experiment 8: Larger models (Llama-3.2-3B and Llama-3.1-8B)

**Why next.** Everything so far used a 1B model.

**Setup.** `mlp.gate_proj`; Llama-3.2-3B layers 3, 10, 18, 25 and Llama-3.1-8B layers 4, 12, 20, 28;
`ε ∈ {0.01, 0.05, 0.1}`; `n_cal = n_test = 20`; direct tap JVPs with `k ∈ {8, 16, 32, 64}`
(`k = 8` is reported). Two checks per model: direct JVPs against the projection of the full
Jacobian (64 probes, 3 clean calibration prompts, one layer), and finite differences (3 prompts ×
8 probes).

**Result.**

| Model | NAF `ε = 0.01` | NAF `ε = 0.05` | NAF `ε = 0.1` | Evaded | NIAC / Weak / Both | Jacobian detects |
|---|---|---|---|---|---|---|
| Llama-3.2-3B | 1.75 [1.69, 1.81] | 5.91 [5.36, 6.46] | 11.41 [10.02, 12.80] | 12/12 | 8 / 4 / 0 | 12/12 |
| Llama-3.1-8B | 1.68 [1.60, 1.78] | 5.76 [5.04, 6.49] | 11.03 [9.41, 12.66] | 11/12 | 8 / 3 / 1 | 12/12 |

NAF: mean and 95% bootstrap range over the four layers (a spread across layers, not a prompt-level
interval). The one Both-Detect condition is Llama-3.1-8B layer 28 at `ε = 0.01`, where the
null-space edit suppresses 49.0% of the activation signal, just under the 50% rule. Suppression
ranges are 52.4-94.2% (3B) and 49.0-94.2% (8B). The largest stored Jacobian p-value is
8.78 × 10^-13 (3B layer 25, `ε = 0.1`). The mean NAF at `k = 8` is 100.05% (3B) and 99.59% (8B) of
that at `k = 64`, with labels agreeing in 12/12 for both. Direct JVPs match the projected full
Jacobian with mean relative error 0.58% (3B, layer 10) and 0.57% (8B, layer 12). Finite differences
again disagree (354% and 362%). Runtimes 81.7 s and 107.1 s.

**What it changed.** The pattern of Experiments 2-4 holds on larger models with somewhat smaller
NAFs, and `k = 8` remains sufficient while `d/k` grows to 384 and 512.

**Files.** Script [`experiments/run_step8_scale_test.py`](../experiments/run_step8_scale_test.py);
configurations [`configs/scale_test_3b.yaml`](../configs/scale_test_3b.yaml) and
[`configs/scale_test_8b.yaml`](../configs/scale_test_8b.yaml); results
[`results/step8_scale_test_3b/`](../results/step8_scale_test_3b/) and
[`results/step8_scale_test_8b/`](../results/step8_scale_test_8b/).

---

## Experiment 9: Baselines and cost

**Why next.** Two practical questions: what do simpler checks achieve on the same edits, and what
does the tap test cost in wall-clock time?

**Setup.** The 15 conditions of Experiment 4 (`n_cal = n_test = 30`), each measured with five
methods: SHA-256 hash of the weights, relative L2 distance of the weights (both need weight access),
activation monitoring, the tap test with `k = 8`, and the full Jacobian (layer 12 only, 5 prompts,
as a cross-check). Timing on one NVIDIA H200 GPU.

**Result.**

- Detection of the 15 null-space edits: activation monitoring 0/15 under the evasion rule; tap test
  15/15 (10 NIAC-Detected, 5 Weak-NIAC); SHA-256 15/15; weight L2 15/15; the full Jacobian detects
  the same 15 in Experiments 4 and 7. Every method detects all 15 Gaussian controls.
- The 0/15 for activation monitoring depends on the evasion rule. The raw activation t-test rejects
  on the rounding residual in all 15 conditions (`p < 1.1 × 10^-16`, Cohen's d 4.9-11.0). On held-out
  prompts the suppression is only 17.4-30.2% across the five layers.
- Tap-test NAF (`k = 8`), mean and bootstrap range over layers: 1.89 [1.82, 1.96], 7.11 [6.50, 7.72],
  14.05 [13.09, 15.01]. At layer 12 the `k = 8` NAFs 1.82, 6.50 and 12.90 compare with 1.91, 6.37
  and 13.31 from the full Jacobian on 5 prompts, with the same labels.
- Wall-clock time:

  | Method | Time | Scope |
  |---|---|---|
  | Activation monitoring | 9.27 ms | per prompt, five layers, mean of 5 prompts |
  | Tap test (`k = 8`) | 47.35 ms | per prompt, five layers, mean of 5 prompts |
  | Full Jacobian | 162.1 ms | per prompt, one layer (layer 12; 2048 JVPs in chunks of 512), mean of 5 prompts |
  | SHA-256 | 40.05 ms | one 8192 × 2048 `gate_proj` matrix (layer 2), including the copy to the CPU, mean of 100 repetitions |
  | Weight L2 distance | 67.24 ms | same matrix and repetitions as SHA-256 |

  The tap test costs 5.1 times the activation monitor (both over five layers). The other times have
  different scopes and are not compared with each other.

Runtime 126.2 s.

**What it changed.** With weight access, hashing settles the question. Without it, among the
inference-only checks we ran, only the Jacobian-based ones detect the null-space edits, at about
five times the cost of activation monitoring.

**Files.** Script [`experiments/run_step9_baselines.py`](../experiments/run_step9_baselines.py);
configuration [`configs/baselines.yaml`](../configs/baselines.yaml); results
[`results/step9_baselines/`](../results/step9_baselines/).

---

## Experiment 10: Other model families (Qwen2.5-7B and Mistral-7B-v0.3)

**Why next.** Experiments 2-9 used only Llama models.

**Setup.** As Experiment 8. Qwen2.5-7B (`d = 3584`) layers 4, 10, 18, 24; Mistral-7B-v0.3
(`d = 4096`) layers 4, 12, 20, 28; `n_cal = n_test = 20`; `k = 8` reported. The only code change was
an architecture-neutral way of finding a layer's device in the shared perturbation code.

**Result.**

| Model | NAF `ε = 0.01` | NAF `ε = 0.05` | NAF `ε = 0.1` | Evaded | NIAC / Weak / Both | Jacobian detects |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 1.60 [1.53, 1.67] | 5.48 [5.01, 5.96] | 10.30 [9.37, 11.17] | 12/12 | 8 / 4 / 0 | 12/12 |
| Mistral-7B-v0.3 | 1.60 [1.54, 1.66] | 5.24 [4.75, 5.73] | 10.27 [9.15, 11.39] | 10/12 | 8 / 2 / 2 | 12/12 |

The two Mistral Both-Detect conditions are layers 20 and 28 at `ε = 0.01` (suppression 48.2% and
48.8%). Suppression ranges: 53.8-94.2% (Qwen), 48.2-93.4% (Mistral). `k = 8` keeps 99.80% (Qwen) and
100.20% (Mistral) of the `k = 64` mean NAF, labels agreeing in 12/12. Direct JVPs match the
projected full Jacobian with mean error 0.59% (Qwen, layer 18; maximum 0.61%) and 0.44% (Mistral,
layer 20). Finite differences: 2221% (Qwen) and 273% (Mistral). Runtimes 120.7 s and 114.9 s.

**What it changed.** The behavior is not specific to the Llama family.

**Files.** Script
[`experiments/run_step10_cross_architecture.py`](../experiments/run_step10_cross_architecture.py);
configurations [`configs/cross_arch_qwen25_7b.yaml`](../configs/cross_arch_qwen25_7b.yaml) and
[`configs/cross_arch_mistral_7b.yaml`](../configs/cross_arch_mistral_7b.yaml); results
[`results/step10_cross_arch_qwen25_7b/`](../results/step10_cross_arch_qwen25_7b/) and
[`results/step10_cross_arch_mistral_7b/`](../results/step10_cross_arch_mistral_7b/).

---

## Experiment 11: An attention sublayer (`self_attn.q_proj`)

**Why next.** All edits so far targeted the MLP gate projection.

**Setup.** Llama-3.2-1B, `self_attn.q_proj` (2048 × 2048) of layers 2, 6, 10, 12, 14;
`ε ∈ {0.01, 0.05, 0.1}`; `n_cal = n_test = 20`; `k ∈ {8, 16, 32, 64}`, `k = 8` reported. The script
is the Experiment 10 pipeline with a different sublayer set in the configuration
(`target_sublayer: "self_attn.q_proj"`) and its own summary printout.

**Result.** The tap test detects all 15 conditions. Labels at `k = 8`: 6 NIAC-Detected, 4 Weak-NIAC,
5 Both-Detect; 10/15 evaded. NAF: 1.19 [1.16, 1.23], 2.59 [2.33, 2.88], 4.76 [4.18, 5.39]. The
null-space edit suppresses less of the activation signal than for `gate_proj`: 26.5-30.8% at
`ε = 0.01` (all five Both-Detect) and 78.6-86.9% at `ε = 0.1`. The `k = 8` mean NAF is 101.81% of
that at `k = 64`; labels agree in 14/15 (per-condition ratios 90.3-113.5%). The exception is layer 6
at `ε = 0.05`, with NAF 3.12 at `k = 8` and 2.78 at `k = 64`, so `k = 64` gives 5/5/5. Direct JVPs
match the projected full Jacobian with mean error 0.53% (layer 12); finite differences 345%.
Runtime 83.0 s.

The rounding residual of the `q_proj` edits was not stored, so it is not known why it is a larger
share of the signal than for `gate_proj` (which also used `n_cal = 30` instead of 20).

**What it changed.** Detection does not depend on the sublayer type; the size of the Jacobian's
advantage does.

**Files.** Script
[`experiments/run_step11_attention_layer.py`](../experiments/run_step11_attention_layer.py);
configuration [`configs/attention_q_proj.yaml`](../configs/attention_q_proj.yaml); results
[`results/step11_attention_q_proj/`](../results/step11_attention_q_proj/).

---

## Headline counts

A *condition* is one null-space edit of one sublayer in one layer at one `ε`, counted once with its
`k = 8` tap-test decision (Llama-3.2-1B `gate_proj` from Experiment 9, 3B/8B from Experiment 8,
Qwen/Mistral from Experiment 10, `q_proj` from Experiment 11), plus each of the five edited layers
of the null-space backdoor (Experiment 6, full Jacobian). This gives 83 conditions. The Jacobian
monitor detects all 83; the activation monitor is evaded in 71 of them (49 NIAC-Detected,
22 Weak-NIAC); the remaining 12 are Both-Detect. Every stored Jacobian p-value of the 78 synthetic
conditions is below `10^-12` (largest 8.78 × 10^-13); for the backdoor layers only `p < 0.001` was
recorded.

Not counted: the re-measurements of the same 15 Llama-3.2-1B conditions in Experiments 2, 4 and 7
(identical labels), the calibration-size runs of Experiment 3, the `k = 16/32/64` variants, the
standard backdoor of Experiment 5 (not a null-space edit), and the Gaussian controls.

## Summary table

| Exp. | Question | Script | Config | Results folder | Paper section |
|---|---|---|---|---|---|
| 1 | Is a Jacobian statistic more sensitive than activations? (pilot) | [`pilot/run.py`](../pilot/run.py), [`pilot/run_frobenius.py`](../pilot/run_frobenius.py), re-analysis [`run_step1_reanalysis.py`](../experiments/run_step1_reanalysis.py) | [`corrected_v1.yaml`](../pilot/configs/corrected_v1.yaml), [`step9_frobenius.yaml`](../pilot/configs/step9_frobenius.yaml) | [`results/pilot/`](../results/pilot/) | Sec. 4.1 (compute only) |
| 2 | Do null-space edits evade activations but not the Jacobian? | [`run_step2_nullspace.py`](../experiments/run_step2_nullspace.py) | [`nullspace_step2.yaml`](../configs/nullspace_step2.yaml) | [`step2_eval_v2_fixed/`](../results/step2_eval_v2_fixed/) | Sec. 4.1, 4.2 |
| 3 | Does the calibration size matter? | [`run_step3_cal_sweep.py`](../experiments/run_step3_cal_sweep.py) | [`nullspace.yaml`](../configs/nullspace.yaml) | [`step3_cal_sweep/`](../results/step3_cal_sweep/) | Sec. 4.2 |
| 4 | Does it hold at other layers? | [`run_step4_multi_layer.py`](../experiments/run_step4_multi_layer.py) | [`multi_layer.yaml`](../configs/multi_layer.yaml) | [`step4_multi_layer/`](../results/step4_multi_layer/) | Sec. 4.2 |
| 5 | Does a standard backdoor evade activations? (L1) | [`run_step5_backdoor.py`](../experiments/run_step5_backdoor.py) | [`backdoor.yaml`](../configs/backdoor.yaml) | [`step5_eval_20260226_220101/`](../results/step5_eval_20260226_220101/) | Sec. 4.3 |
| 6 | Can a backdoor be trained in the null space? (L2) | [`run_step6_nullspace_backdoor.py`](../experiments/run_step6_nullspace_backdoor.py) | [`nullspace_backdoor.yaml`](../configs/nullspace_backdoor.yaml) | [`step6_nullspace_backdoor/`](../results/step6_nullspace_backdoor/) | Sec. 4.3 |
| 7 | Do `k` probes keep the full-Jacobian decisions? | [`run_step7_tap_test.py`](../experiments/run_step7_tap_test.py) | [`tap_test.yaml`](../configs/tap_test.yaml) | [`step7_eval_20260227_011659/`](../results/step7_eval_20260227_011659/) | Sec. 4.2, 4.5 |
| 8 | Does it hold on 3B and 8B models? | [`run_step8_scale_test.py`](../experiments/run_step8_scale_test.py) | [`scale_test_3b.yaml`](../configs/scale_test_3b.yaml), [`scale_test_8b.yaml`](../configs/scale_test_8b.yaml) | [`step8_scale_test_3b/`](../results/step8_scale_test_3b/), [`step8_scale_test_8b/`](../results/step8_scale_test_8b/) | Sec. 4.4, 4.5 |
| 9 | How do simpler checks compare, and what does it cost? | [`run_step9_baselines.py`](../experiments/run_step9_baselines.py) | [`baselines.yaml`](../configs/baselines.yaml) | [`step9_baselines/`](../results/step9_baselines/) | Sec. 4.1, 4.2, 4.5 |
| 10 | Does it hold for Qwen and Mistral? | [`run_step10_cross_architecture.py`](../experiments/run_step10_cross_architecture.py) | [`cross_arch_qwen25_7b.yaml`](../configs/cross_arch_qwen25_7b.yaml), [`cross_arch_mistral_7b.yaml`](../configs/cross_arch_mistral_7b.yaml) | [`step10_cross_arch_qwen25_7b/`](../results/step10_cross_arch_qwen25_7b/), [`step10_cross_arch_mistral_7b/`](../results/step10_cross_arch_mistral_7b/) | Sec. 4.4, 4.5 |
| 11 | Does it hold for an attention sublayer? | [`run_step11_attention_layer.py`](../experiments/run_step11_attention_layer.py) | [`attention_q_proj.yaml`](../configs/attention_q_proj.yaml) | [`step11_attention_q_proj/`](../results/step11_attention_q_proj/) | Sec. 4.4, 4.5 |

Paper sections: 4.1 Setup, 4.2 RQ1 (null-space edits on Llama-3.2-1B), 4.3 RQ2 (backdoors),
4.4 RQ3 (scale, family, sublayer), 4.5 RQ4 (probe count, cost, baselines).

## Lessons and corrections

The following points were revised while the results were analyzed. The numbers above already
reflect them.

1. **Decision test.** The pilot's SNR ratio was replaced by a one-sided one-sample t-test of the
   per-prompt statistic against zero at `α = 0.001`. No other test (for example a
   Kolmogorov-Smirnov test) is used for any decision in the code.
2. **Evasion criterion.** Because of the bfloat16 residual, the raw activation t-test also rejects
   on null-space edits. "Evaded" therefore means a calibration activation signal below 50% of that
   of the norm-matched Gaussian control, and the activation result of Experiment 9 (0/15) holds
   under this rule only. On held-out prompts the activation signal of a null-space edit is
   suppressed by only 17.4-30.2% (19.3%, 28.1%, 28.8% at layer 12). Activation monitoring is not
   blind in general; the blind spot is the calibration set.
3. **Activation metric.** Only the relative L2 change was measured. No cosine, CKA or trained-probe
   monitors were run.
4. **First evaluation of Experiment 2.** The first run of Experiment 2 had three evaluation problems:
   an AUC that is trivially 1.0 when the noise floor is zero, a decision rule based on a fixed
   threshold for the Jacobian-to-activation ratio that mislabelled the `ε = 0.01` condition although
   both tests rejected there, and a population instead of a sample standard deviation. The evaluation was fixed and the run repeated; the
   stored folder `step2_eval_v2_fixed` is the repeated run.
5. **Recounted labels.** Summary lines written during the runs were re-checked against the stored
   per-condition labels. Llama-3.1-8B has 11/12 evaded conditions (layer 28 at `ε = 0.01` is
   Both-Detect), and `q_proj` at `k = 8` gives 6/4/5 (5/5/5 is the `k = 64` result). Headline counts
   come from a full enumeration of conditions (see *Headline counts*).
6. **Experiment 6 raw output.** The JSON written by the null-space backdoor run was not saved. Its
   values are transcribed, rounded, from the run report, and its p-values are known only to be
   below 0.001. The run report also listed a "gradient null fraction"; it is the norm ratio of the
   accumulated update after projection, which is close to 1 by construction, so it is not used.
7. **Incomplete result files.** Some stored files are empty or are terminal captures rather than
   JSON (Experiments 3, 4 and 5; see [`results/README.md`](../results/README.md)). The data used
   for every number above is present in the remaining files.
8. **Timing scope.** The full-Jacobian time covers one layer and the hash and weight-L2 times one
   weight matrix, while the activation and tap times cover five layers. Only the tap/activation
   ratio (5.1) compares equal scopes. The fields `fjac_to_tap_ratio` and `fjac_to_act_ratio` in
   `results/step9_baselines/cost_analysis.json` mix scopes and are not used.
9. **Validation scope.** The sub-1% agreement between direct JVPs and the full Jacobian was measured
   with 64 probes on 3 clean calibration prompts at one layer per model. It checks the JVP code, not
   the `k = 8` decisions; those are checked by the NAF recovery of Experiment 7 and the
   `k = 8` versus `k = 64` comparisons of Experiments 8, 10 and 11.
10. **Unused statistics.** [`analysis/statistical_results.json`](../analysis/statistical_results.json)
    also contains a minimum-detectable-effect estimate and a ROC analysis of the NAF threshold. Both
    are not used: the minimum-detectable-effect heuristic is contradicted by detection at
    `ε = 0.01`, and the ROC labels are themselves defined by the NAF threshold. In
    [`analysis/statistical_analysis.py`](../analysis/statistical_analysis.py) the `exp6` entry reads
    the Experiment 5 summary file, as in the original analysis; neither entry feeds any reported
    statistic.
11. **Compute.** The logged runs took 1.46 GPU hours on one NVIDIA H200, of which 0.76 hours were the
    pilot. Exploratory runs were not logged.
