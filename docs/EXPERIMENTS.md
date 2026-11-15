# How the study developed, step by step

This document follows the experiments behind the paper *Activation Audits on Fixed Prompts Have a
Null-Space Blind Spot: Secret Jacobian Probes for LLM Weight Integrity* in the order in which they
were run. For each experiment it gives the question, why that question followed from the previous
result, the setup, the main numbers, and what changed in the study afterwards. Each section links
to the code, configuration and stored results in this repository.

Experiments 1 to 11 were run for an earlier version of the paper. Experiments 12 to 19 were added
after expert feedback on that version, and Experiment 12 overturned two of its claims (see
*Headline counts* and *Lessons and corrections*). The sections on Experiments 1 to 11 describe what
was measured at the time, in the terms defined below.

A description of every stored result file is in [`results/README.md`](../results/README.md), and
the experiments table of the [README](../README.md#experiments) shows which research question of
the paper uses each experiment.

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

Experiment 12 showed that the answer depends on where the audit looks. The edited layer's output at
the last token does not move, but its outputs at earlier positions do, and every later layer
attends to them from the last token. Experiments 13 to 19 therefore turn to an adversary who also
hides at the earlier positions, to an adversary who targets the Jacobian test, and to audits under
the numerical noise of different GPUs.

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
6. **Every layer (Experiment 12).** The activation signal of the same null-space edits at every
   decoder layer, in float32 and bfloat16, and the leak through earlier positions.
7. **Adaptive adversaries (Experiments 13, 14 and 17).** Synthetic all-position edits, trained
   backdoors under each adversary's constraint, and an adversary that targets the tap test.
8. **Serving noise and an equal budget (Experiments 15 and 16).** Audits across three GPU
   architectures with a threshold rule, and six monitors at the tap test's budget.
9. **Other models and ordinary triggers (Experiments 18 and 19).** The adversaries on three models
   of 7B to 8B parameters, and five scenarios with ordinary trigger phrases.

## Common setup and terms

These apply to Experiments 2-11 unless a section says otherwise. Experiments 12 to 19 define
their settings in their own folders and use the terms of the paper (see the
[README](../README.md#decision-rules-and-terms)).

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
  that of its Gaussian control (suppression above 50%). This criterion looks at the edited layer
  only. Experiment 12 showed that the later layers see the edit, so an audit of every layer is not
  evaded; the paper therefore uses *masked*, a suppression above 50% at every scored layer.
- **Null-space amplification factor (NAF; NIAC amplification factor in the code).** `NAF = (s_jac_ns / s_act_ns) / (s_jac_g / s_act_g)`, the
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
[`results/step4_multi_layer/`](../results/step4_multi_layer/).

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

## Experiment 12: Does an activation audit of every layer see the null-space edits?

**Why next.** Experiments 2 to 11 scored the activation monitor at the edited layer only, while the
threat model lets the auditor read every decoder layer. Expert feedback on an earlier version asked
whether the later layers see an edit that the edited layer hides.

**Setup.** Llama-3.2-1B, the 15 null-space edits (L3) of Experiment 9 (`mlp.gate_proj` of layers 2,
6, 10, 12 and 14, `ε ∈ {0.01, 0.05, 0.1}`, bitwise identical to Experiment 9, which a test checks)
and their Gaussian controls, `n_cal = n_test = 30`. Every one of the 16 decoder layers is measured:
the relative change at the last token (`s_act`), its mean and maximum over all token positions, and
the KL divergence of the next-token distribution. The run also records the residual of the stored
edit on the calibration inputs and the spectrum of the `gate_proj` inputs at all positions. One run
in float32 and one in bfloat16 on one NVIDIA RTX PRO 4500; runtimes 64.4 s and 61.0 s; noise floor
zero in both. This experiment predates the pre-registration practice of Experiments 13 to 19; its
`RESULTS.md` was written after the run.

**Result.**

- **Edited layer.** In float32 the null-space edit suppresses 99.992-99.999% of the control's
  activation signal in all 15 conditions: Theorem 1 holds up to float32 rounding.
- **Later layers.** The edit changes the edited layer's outputs at the earlier positions, and
  every later layer reads them from the last token through attention. Averaged over all positions,
  the edited layer keeps 81.4-90.8% of the control's change in float32; at the last token the most
  affected later layer keeps 13.6-62.5% of the control's signal, and this change grows with `ε`.
  The decision test rejects at every layer from the edited one on, in all 15 conditions, in both
  arithmetics.
- **Under the 50% rule.** An audit of every layer is masked (suppression above 50% at the edited
  layer and every later layer) in 12 of 15 conditions in float32 (the three exceptions are the
  layer-2 edits) and in 4 of 15 in bfloat16 (layer 10 and 12 at `ε = 0.1`, layer 14 at 0.05 and
  0.1).
- **Origin of the bfloat16 residual.** The intended edit leaves a residual of order `10^-8` of
  `‖W H^T‖` on the calibration inputs; after `W + ΔW` is stored in bfloat16 it is 0.89-1.17 × 10^-3,
  independent of `ε`. The rounding error is 16.5% of the edit at `ε = 0.01` and 1.7% at `ε = 0.1`,
  the same absolute change. This is the residual that "does not grow with `ε`" in Experiments 2-11.
- **All-position inputs.** The inputs of `gate_proj` at all positions of the calibration prompts are
  numerically of full rank (2046 to 2048 of 2048) but concentrated: the 256 (512) directions with
  the least energy carry 0.08-0.13% (0.3-0.5%) of it.

**What it changed.** The claims "71 of 83 conditions evaded" and "activation monitoring 0/15"
counted the edited layer only and were withdrawn (see *Headline counts* and item 12 of *Lessons
and corrections*): on a deterministic stack an activation audit of every layer detects all 15
single-layer null-space edits. The blind spot of Theorem 1 is a property of the edited layer. The
float32 and bfloat16 runs became the first two of the three settings the paper reports (exact
arithmetic, finite precision). The concentrated input spectrum made an adversary who also hides
at the earlier positions plausible, which Experiments 13 and 14 test; decisions under noise are
Experiment 15.

**Files.** Folder [`experiments/exp12_downstream-activation/`](../experiments/exp12_downstream-activation/)
(script `run_exp12_downstream.py`, configurations `configs/downstream_fp32.yaml` and
`configs/downstream_bf16.yaml`); results
[`RESULTS.md`](../experiments/exp12_downstream-activation/RESULTS.md) and
[`results/`](../experiments/exp12_downstream-activation/results/) (`downstream_fp32/`,
`downstream_bf16/`, run log).

---

## Experiment 13: Can an adversary who knows the calibration prompts hide a single-layer edit from an every-layer audit?

**Why next.** Experiment 12 traced the leak to the earlier positions. An adversary who knows the
weights and the calibration prompts (not the probe matrix) can keep the last-token guarantee of
Theorem 1 and, within that null space, use only the `m` directions that carry the least input
energy at all positions: the *all-position* (AP) edit.

**Setup.** Llama-3.2-1B, `gate_proj` of layers 2, 6, 10, 12 and 14, `ε ∈ {0.01, 0.05, 0.1}`, seeds
42, 43 and 44 for the edit matrices (seed 42 reproduces Experiment 12 to a relative difference of
3 × 10^-7), one fixed probe matrix. Per condition: the Gaussian control, the L3 edit, and AP edits
with `m ∈ {64, 128, 256, 512, 1024}`. Measured: the every-layer last-token audit with the 50% rule
on 30 calibration and 30 held-out prompts, the KL divergence of the next-token distribution of
edit over control (the cost of hiding), and the tap test (`k = 8`) at the edited layer. float32 and
bfloat16 on one RTX PRO 4500; runtimes 1183 s and 1008 s. The pre-registration fixed the
definitions, five predictions (P1-P5) and the interpretation rules before the run; it also states
that the tap test is only a sanity check here, because this adversary ignores the Jacobian.

**Result.**

- **float32, calibration prompts.** Every AP edit is masked at every layer in all 45 conditions
  (15 conditions × 3 seeds) for every `m`, with a median worst-layer suppression of 92.6%
  (`m = 1024`) to 97.9% (`m = 64`). The L3 edit is masked in 36 of 45, median 72.1%. The
  suppression does not depend on `ε`.
- **bfloat16, calibration prompts.** The weight-rounding error of Experiment 12 is an unconstrained
  edit that the audit sees, and it makes `m` nearly irrelevant: AP edits are masked in 0, 6 and 12
  of 15 conditions at `ε = 0.01`, 0.05 and 0.1 for `m = 64` and 256, and in 0, 6 and 11 for the other
  `m`; L3 in 0, 3 and 9.
- **Held-out prompts.** In float32, AP edits are masked in 14 of 15 conditions per `ε` for
  `m ≤ 256`, 10 of 15 for `m = 512` and 5 of 15 for `m = 1024`, with a median worst-layer
  suppression of only 48-55%. In bfloat16 none of the 45 is masked for any `m`.
- **Cost of hiding.** In float32 the median KL ratio (edit over control) on the calibration prompts
  is 0.0003-0.0065 for the AP edits against 0.09 for L3; on held-out prompts it is 0.20-0.25 (L3:
  0.58).
- **Tap test.** It rejects all 540 edits of both runs.
- **Predictions.** P1 (masking at every layer for `m ≤ 256`), P3 (KL falls with `m`), P4 and P5
  (rounding limits the suppression in bfloat16; it does so at every `ε`, not only at 0.01) held.
  P2 (high held-out suppression for small `m`) held only in part.

**What it changed.** On the prompts it knows, and in near-exact arithmetic, the adversary can hide
a random single-layer edit from an every-layer audit. The hidden edits barely change the output on
those prompts, so whether such an edit can still carry an attack was left to Experiment 14. The
paper reports these synthetic AP edits in RQ2.

**Files.** Folder [`experiments/exp13_allpos-adversary/`](../experiments/exp13_allpos-adversary/)
(script `run_exp13_allpos.py`, configurations `configs/allpos_fp32.yaml` and
`configs/allpos_bf16.yaml`, [`PREREGISTRATION.md`](../experiments/exp13_allpos-adversary/PREREGISTRATION.md));
results [`RESULTS.md`](../experiments/exp13_allpos-adversary/RESULTS.md) and
[`results/`](../experiments/exp13_allpos-adversary/results/).

---

## Experiment 14: Can a working backdoor hide from an every-layer audit?

**Why next.** The edits of Experiment 13 have no goal. A real adversary needs a backdoor that works
and stays hidden, and Experiment 12 suggested that the L2 backdoor of Experiment 6 would be seen at
its later layers.

**Setup.** Llama-3.2-1B, bfloat16, the recipe of Experiment 6: 200 triggered and 200 clean
examples, 5 epochs, batch size 8, trigger `XYZZY PROTOCOL:`, `gate_proj` of layers 2, 6, 10, 12 and
14 trained together. Ten variants: L1 (no constraint); L2 (last-token null space, bases from the
clean model); L2R (L2 with the bases refit from the current model after every epoch); AP1024 and
AP256 (`m = 1024` or 256 all-position directions); AP1024R and AP256R (refit after every epoch);
and L2, AP1024 and AP256 on layer 2 alone. Learning rates `5 × 10^-5`, `2 × 10^-4` and `10^-3`,
seeds 42 and 43: 60 runs, sharded over two H200 and one B200, each run compared with clean
references on its own machine; 2112 s of run time in total. A run is *functional* if the marker
follows the trigger on at least 90% of 20 prefixes and appears on at most 10% of the same prefixes
without it. *Masked*: suppression above 50% at every layer from the first trained one on, against
a Gaussian control of the same size in each trained layer. The pre-registration fixed five
predictions (Q1-Q5), the adversary's selection rule (per variant and seed, among functional runs,
the learning rate with the highest worst-layer suppression on the calibration prompts) and the
interpretation rules before the run.

**Result.**

| Adversary | Functional | Masked, calibration | Masked, held-out | Worst `s_act` | Tap test |
|---|---|---|---|---|---|
| L1 | 6/6 | 0/6 | 0/6 | 0.497-1.073 | 6/6 |
| L2, L2R | 12/12 | 0/12 | 0/12 | 0.126-0.353 | 12/12 |
| AP (`m` = 256, 1024; fixed or refit) | 20/24 | 14/20 | 0/20 | 0.013-0.037 | 24/24 |
| L2, one layer | 6/6 | 0/6 | 0/6 | 0.083-0.409 | 6/6 |
| AP, one layer | 9/12 | 4/9 | 0/9 | 0.013-0.041 | 12/12 |

Masked: among functional runs. Worst `s_act`: the largest calibration mean of `s_act` over the
scored layers, functional runs. Tap test: runs in which it rejects at every trained layer.

- **Which runs work.** The non-functional runs all use the smallest learning rate: AP256 and
  AP256R (trigger success 70-75%), single-layer AP256 (both seeds) and single-layer AP1024 (one
  seed). Every masked run uses `2 × 10^-4` or `10^-3`. Under the selection rule, all six AP
  variants are functional and masked for both seeds, at `10^-3`.
- **Not an artefact of the relative rule.** The 50% rule compares with a random edit of the same
  size, and the selected runs are large (mean `ε` 0.29-0.56). In absolute terms the functional AP
  runs stay at a worst-layer `s_act` of 0.013-0.041, the level of AP runs that barely trained
  (`ε` 0.012-0.013), which is the bfloat16 rounding floor; L2 and L2R reach 0.126-0.353. The worst
  layer of L1 and L2 changes 1.4 to 19 times as much as that of its control.
- **Held-out prompts.** No run of any variant is masked. The AP runs change the held-out
  activations about as much as an equal-size random edit, or more (worst-layer suppression -114%
  to +8%).
- **All-position audit.** If the audit averages `s_act` over all positions instead of reading the
  last token, 8 of the 14 masked AP runs stay masked (the eight at `10^-3`).
- **Side effects.** The held-out perplexity of the functional multi-layer AP runs is 29.35-33.82,
  against 32.84 for the clean model.
- **Predictions.** Q1 (L1 and L2 functional, not masked) and Q2 (refitting does not help) held, and
  Q3 (AP1024 functional, masking uncertain) resolved to functional and masked for both seeds. Q4
  (AP256 and single-layer variants less often functional) was mostly wrong, and Q5 (no variant
  functional and masked for both seeds) was wrong.

**What it changed.** Under the pre-registered rule, the paper states that an adversary who knows
the calibration prompts can hide a working backdoor from an every-layer last-token audit on those
prompts, and reports alongside that held-out prompts see every such backdoor. The L1 and L2
backdoors of this experiment replace those of Experiments 5 and 6 in the paper (its Table II). The
held-out result raised the question of secret prompts (Experiment 16); the tap test, a sanity check
here, needed an adversary that targets it (Experiment 17).

**Files.** Folder [`experiments/exp14_hidden-backdoor/`](../experiments/exp14_hidden-backdoor/)
(scripts `run_exp14_backdoor.py` and `analyze_exp14.py`, configuration `configs/backdoor_bf16.yaml`,
[`PREREGISTRATION.md`](../experiments/exp14_hidden-backdoor/PREREGISTRATION.md)); results
[`RESULTS.md`](../experiments/exp14_hidden-backdoor/RESULTS.md) and
[`results/`](../experiments/exp14_hidden-backdoor/results/) (one JSON file per run in
`backdoor_bf16/runs/`, `analysis.json`, logs).

---

## Experiment 15: Which edits does each monitor still detect under serving noise?

**Why next.** Experiments 12 to 14 decided on a deterministic stack, where the decision test
(Definition 1 of the paper) rejects any reproducible change for both monitors. Masking matters
only when the audit must tolerate noise: when it runs on another machine than the one that recorded
the references, or batches the prompts as a serving system does.

**Setup.** 60 edits (Gaussian, L3, AP with `m = 256` and 1024; layers 2, 6, 10, 12 and 14;
`ε ∈ {0.01, 0.05, 0.1}`; seed 42) built once on an H200 and served bitwise identically (SHA-256
checked) on four GPUs of three architectures: two H200 cards, one B200 and one RTX PRO 4500, with
PyTorch 2.8 and CUDA 12.8. Audits of single prompts and of right-padded batches of eight; the tap
test always evaluates each prompt on its own. Every ordered pair of reference and audit machine is
evaluated. The pre-registered rule is a one-sided paired t-test per layer over the 30 calibration
prompts, edit against clean model on the same audit machine, at `α = 0.001 / 16`; the activation
audit flags if any layer from the edited one on flags, the tap test at the edited layer. The
pre-registration fixed five predictions (N1-N5) before the run.

Deviation D1, decided after the paired-rule results were seen: the paired rule needs a clean run on
the audit machine, and an auditor who could make one would record references there and have no
noise. A threshold rule was therefore added: `τ` = margin × the largest clean deviation across
architectures, learned with one architecture left out and applied to every audit that involves it
(20 clean audits, and 100 audits per edit kind and `ε`), with margins 1.0, 1.5 and 2.0 and the same
rule for both monitors.

**Result.**

- **Noise.** Zero on the same machine and between the two H200 cards (bitwise identical). Across
  architectures the clean model's last-token `s_act` reaches 2.6-3.9 × 10^-2 on single prompts, and
  its calibration mean per layer 3.4 × 10^-3 (layer 0) to 1.6 × 10^-2. Batching on one machine gives
  up to 2.0-2.7 × 10^-2. The clean tap statistic across architectures is 0.9-1.5 × 10^-2 at the
  edited layers. This is about ten times the bfloat16 residual of a null-space edit.
- **Threshold rule** (detections of 100 audits per cell):

  | Margin | Mode | False alarms, activation / tap | Activation, L3 at `ε` = 0.01 / 0.05 / 0.1 | Activation, AP, any `ε` | Tap, every kind, `ε` = 0.01 / 0.05 / 0.1 |
  |---|---|---|---|---|---|
  | 1.0 | single | 12/20 / 12/20 | 60 / 76 / 92 | 59-64 | 88 (Gaussian 100) / 100 / 100 |
  | 1.5 | single | 6/20 / 0/20 | 0 / 0 / 6 | 0 | 0 (Gaussian 6) / 100 / 100 |
  | 1.5 | batched | 6/20 / 0/20 | 3 / 3 / 6 | 3 | 0 (Gaussian 6) / 100 / 100 |
  | 2.0 | single | 6/20 / 0/20 | 0 / 0 / 0 | 0 | 0 / 96-100 / 100 |

  The activation audit flags the Gaussian controls in 0, 76 and 100 of 100 audits at margin 1.5
  (single prompts).
- **Pre-registered paired rule** (across machines, including the two H200 cards): the activation
  audit flags 175-180 of 180 audits of Gaussian edits, 104-111 of L3 edits and 30-39 of AP edits;
  30 of the AP detections come from the H200 pairs, which have no noise. The tap test flags all 180
  audits of every kind and `ε`.
- **Predictions.** N1 (no noise on the same machine), N4 (the tap test flags every edit at
  `ε ≥ 0.05`, some misses at 0.01) and N5 (AP edits flagged less often than L3 by the activation
  audit) held. N2 was wrong in size: the noise is of order `10^-2`, not `10^-3`. N3 (the activation
  audit still flags many L3 edits through later layers) held under the paired rule but not under
  the threshold rule from margin 1.5.

**What it changed.** From margin 1.5 the tap test raises no false alarm and detects every edit
kind at `ε ≥ 0.05`, including the AP edits, and no null-space or AP edit at `ε = 0.01`; the
activation audit keeps 6 false alarms in 20 and flags at most 6 of 100 null-space or AP edits. The
paper reports this detection floor between `ε = 0.01` and 0.05 instead of "detects all", as the
third setting (decisions under noise), and Experiment 16 fixed margin 1.5 in advance as the
smallest margin without tap false alarms. The evidence covers three architectures, one software
stack and coarse false-alarm estimates (20 clean audits per margin and mode).

**Files.** Folder [`experiments/exp15_noise-tolerance/`](../experiments/exp15_noise-tolerance/)
(scripts `make_edits_exp15.py`, `record_exp15.py`, `analyze_exp15.py` and
`analyze_exp15_threshold.py`, configuration `configs/noise.yaml`,
[`PREREGISTRATION.md`](../experiments/exp15_noise-tolerance/PREREGISTRATION.md)); results
[`RESULTS.md`](../experiments/exp15_noise-tolerance/RESULTS.md) and
[`results/`](../experiments/exp15_noise-tolerance/results/) (`analysis.json` for the paired rule,
`analysis_threshold.json` and `.txt` for the threshold rule, `manifest.json`). The per-machine
records (2.4 GB) are not stored in the repository.

---

## Experiment 16: At the same budget, which monitor detects which adversary?

**Why next.** Experiment 14 showed that held-out prompts see every AP backdoor, so secret natural
prompts might be an equal defence. The tap test also reads more: with `k = 8` it needs
`(k + 1) × 30 = 270` passes per audit and layer, against 30 for the activation audit on the
calibration prompts. And Experiment 17 had predicted, from separate measurements, that the
Jacobian-aware adversary falls below the cross-architecture tap noise on the calibration prompts;
this needed a direct test.

**Setup.** Llama-3.2-1B, bfloat16. WikiText-103 test passages in order: 30 calibration prompts
(known), 240 further known prompts, 270 secret prompts that no adversary uses, the rest for
backdoor training; a public corpus of 2000 WikiText-103 training passages for the adversaries that
know only the prompt distribution. Six monitors, with passes per audit and monitored layer: A30,
activation on the 30 calibration prompts (30); A270k, on 270 known prompts (270); A270s, on 270
secret prompts (270); S270, on 270 secret synthetic hidden states fed to each layer (270; this
needs an interface that accepts hidden states, so it is an upper bound, not a deployable monitor);
T30, the tap test on the 30 calibration prompts (270); T30s, the tap test on 30 secret prompts
(270). 32 models (seed 42), built once on an H200 and served bitwise identically on the four GPUs
of Experiment 15: 20 synthetic single-layer edits (Gaussian, L3, and AP with `m = 256` whose bases
come from the calibration prompts, from all 270 known prompts or from the corpus; layers 6 and 12;
`ε` 0.05 and 0.1) and 12 backdoors (L2; AP from the calibration prompts, from the known prompts and
from the corpus; Jacobian-aware with 2048 eigenvectors from the calibration prompts or from the
corpus). The backdoors change each edited matrix by `ε` = 0.043 to 0.363 and all reach 100% trigger
success. Rules: on the same architecture, Definition 1 at `0.001 / n_layers`; across architectures,
the threshold rule of Experiment 15 with one architecture left out, 20 audits per cell, margin 1.5
(primary) and 2.0. The pre-registration fixed the rules, the margin and eight predictions (B1-B8)
before any run.

**Result.**

- **Same architecture** (two H200 cards, no noise): every monitor detects all 32 models.
- **Across architectures, margin 1.5:**

  | | A30 | A270k | A270s | S270 | T30 | T30s |
  |---|---|---|---|---|---|---|
  | False alarms on the clean model | 6/20 | 6/20 | 6/20 | 6/20 | 0/20 | 0/20 |
  | Synthetic edits (20) | 166/400 | 232/400 | 252/400 | 400/400 | 400/400 | 400/400 |
  | L2 backdoor | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
  | AP from the calibration prompts (3 models) | 6/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
  | AP from the known prompts | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
  | AP from the corpus, `m = 256`, lower learning rate | 6/20 | 14/20 | 20/20 | 20/20 | 20/20 | 20/20 |
  | AP from the corpus, other 3 models | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
  | Jacobian-aware from the calibration prompts | 6/20 | 20/20 | 20/20 | 20/20 | 6/20 | 20/20 |
  | Jacobian-aware from the corpus, lower learning rate | 6/20 | 7/20 | 20/20 | 20/20 | 20/20 | 20/20 |
  | Jacobian-aware from the corpus, higher learning rate | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |

  The four synthetic AP edits built from the corpus are flagged in 24 of 80 audits by A30, A270k
  and A270s (the clean model's rate) and in all 80 by S270, T30 and T30s.
- **Margin 2.0.** The false alarms do not change. T30 flags the Jacobian-aware backdoor of the
  calibration prompts in none of 20 audits. The Jacobian-aware backdoor from the corpus at the lower
  learning rate is flagged in 6 of 20 by A30, A270k, A270s and T30, and in 8 by T30s; A270k and
  A270s flag the corpus AP backdoor with `m = 256` at the lower learning rate in 6. Of the
  synthetic edits, T30s flags 400 of 400 audits, T30 374 and A270s 176.
- **Predictions.** B3 (A270s detects the AP backdoors from calibration and known prompts), B5 (S270
  detects everything), B6 (T30 misses the Jacobian-aware backdoor of the calibration prompts), B7
  (T30s detects it) and B8 (same architecture: all detect all) held; B1 mostly held. B2 (A270k
  misses the AP backdoor from the known prompts) was wrong, and B4 (A270s misses a
  distribution-aware backdoor) was wrong at margin 1.5 and holds only at margin 2.0.

**What it changed.** The pre-registered rule "secret natural prompts are an equal or cheaper
defence" is not met: A270s flags every backdoor at margin 1.5, but with 6 false alarms in 20
against none for T30s, so what distinguishes the tap test here is its false-alarm rate across
architectures. The second rule applies to the Jacobian-aware backdoor of the calibration prompts:
the tap test on those prompts flags it in only 6 of 20 audits (none at margin 2.0), the tap test on
secret prompts in all 20. The paper reports this and recommends audits on secret prompts or on the
same architecture. This benchmark is Table III of the paper. One seed per model, one software stack
and 20 clean audits per margin limit it.

**Files.** Folder [`experiments/exp16_equal-budget-benchmark/`](../experiments/exp16_equal-budget-benchmark/)
(scripts `build_exp16.py`, `record_exp16.py` and `analyze_exp16.py`, configuration
`configs/benchmark.yaml`, [`PREREGISTRATION.md`](../experiments/exp16_equal-budget-benchmark/PREREGISTRATION.md));
results [`RESULTS.md`](../experiments/exp16_equal-budget-benchmark/RESULTS.md) and
[`results/`](../experiments/exp16_equal-budget-benchmark/results/) (`analysis.json`,
`analysis_output.txt`, `manifest.json`, build log). The per-machine records (6.9 GB) are not stored
in the repository.

---

## Experiment 17: Can an adversary who targets the Jacobian keep the tap test from detecting a hidden backdoor?

**Why next.** No adversary so far had targeted the Jacobian; the tap-test results of Experiments 13
and 14 were sanity checks only.

**Setup.** The AP1024 backdoor of Experiment 14 (five layers, same recipe, Llama-3.2-1B, bfloat16)
with Jacobian awareness added. Without the probe matrix, the best the adversary can do is to
minimize the expected tap statistic, a quadratic form `trace(ΔW^T G ΔW)` of the gated-MLP sublayer
Jacobian on the calibration prompts (`Q` in the code). Variants: a penalty on the relative
Jacobian energy with weight 10, `10^2`, `10^3` or `10^4`; the columns of `ΔW` restricted to the
4096, 2048 or 1024 eigenvectors of `G` with the smallest eigenvalues; 2048 eigenvectors combined
with a penalty of `10^3`; and plain AP1024 as the reference. Learning rates `5 × 10^-5`,
`2 × 10^-4` and `10^-3`, seeds 42 and 43: 54 runs (48 Jacobian-aware, 6 plain), on two H200 and one
B200; 2353 s of run time in total. The pre-registration fixed five predictions (J1-J5) and the
selection rule before the run: per variant and seed, among functional and masked runs, the lowest
tap signal kept on the calibration prompts relative to an equal-size Gaussian edit. That rule uses
the probe matrix, which the adversary lacks, so the selected value is a lower bound. The adversary
targets the sublayer Jacobian, not that of the whole decoder layer, and only the calibration
prompts.

**Result.**

- **Deterministic stack.** The tap test rejects at every trained layer in all 54 runs; no run is
  functional, masked and missed.
- **Signal kept.** The lowest tap signal kept among the selected runs is 17.3% (1024 eigenvectors,
  learning rate `10^-3`, seed 43), and 48.7% on held-out prompts. Plain AP1024 keeps 70.6%, the
  penalty variants 29.9-36.2%, 4096 eigenvectors 29.6% and 2048 eigenvectors 21.6-21.9%.
- **Cost to the attack.** Penalties of `10^3` and above, and the combined variant, never give a run
  that is both functional and masked; trigger success is 80-100% at `10^3` and 20-100% at `10^4`.
- **Absolute signal (deviation D1).** The relative selection rule, like the 50% activation rule,
  favours large edits (the selected runs have `ε` 0.28-0.40). Detection under noise depends on the
  absolute statistic, so it is also reported as a multiple of the largest clean cross-architecture
  tap deviation of Experiment 15 (`τ_max`, per layer; zero false alarms there needed `1.5 τ_max`):

  | Run (functional, masked) | Calibration tap / `τ_max`, layers 2-14 | Held-out tap / `τ_max` |
  |---|---|---|
  | 2048 eigenvectors, lr `2 × 10^-4` (`ε` 0.07) | 1.1-1.3 | 2.0-3.2 |
  | 4096 eigenvectors, lr `2 × 10^-4` (`ε` 0.09) | 1.6-2.0 | 3.0-5.1 |
  | 1024 eigenvectors, lr `10^-3` (`ε` 0.28) | 2.4-3.6 | 6.9-10.0 |
  | penalty `10^2`, lr `10^-3` (`ε` 0.15) | 2.3-3.4 | 4.4-7.7 |
  | AP1024 (not Jacobian-aware), lr `2 × 10^-4` | 4.4-5.7 | 5.0-10.1 |

  Both seeds agree to within 0.3.
- **Predictions.** J2 (lowest signal kept between 5% and 40%), J3 (the tap test detects every
  selected run) and J4 (held-out signal higher) held. J1 (the signal falls with the penalty, and
  the backdoor starts to fail at `10^3` to `10^4`) held in part: the signal falls, and from `10^3`
  on no run is both functional and masked, although trigger success at `10^3` stays at 80-100%. For
  J5 the answer is that the calibration-prompt tap test is defeated at the noise level and the
  held-out one is not.

**What it changed.** Under the pre-registered rule, a Jacobian-aware adversary brings the
calibration-prompt tap signal of a working, masked backdoor to 1.1-1.3 `τ_max`, below the margin
of 1.5, so an audit on another GPU architecture would miss it on those prompts; held-out prompts
keep 2.0-3.2 `τ_max`, and on the same architecture the noise is zero. Because this combined
statistics of different runs and machines, Experiment 16 built a backdoor with the 2048-eigenvector
setting at `2 × 10^-4` and audited it directly across architectures. The paper reports this
adversary as RQ3.

**Files.** Folder [`experiments/exp17_jacobian-adversary/`](../experiments/exp17_jacobian-adversary/)
(scripts `run_exp17_jacobian_adversary.py` and `analyze_exp17.py`, configuration
`configs/jacobian_bf16.yaml`, [`PREREGISTRATION.md`](../experiments/exp17_jacobian-adversary/PREREGISTRATION.md));
results [`RESULTS.md`](../experiments/exp17_jacobian-adversary/RESULTS.md) and
[`results/`](../experiments/exp17_jacobian-adversary/results/) (one JSON file per run in
`jacobian_bf16/runs/`, `analysis.json`, logs).

---

## Experiment 18: Do the adaptive-adversary results hold on larger models and other families?

**Why next.** Experiments 12 to 17 used Llama-3.2-1B only, while the paper covers five models of
three families.

**Setup.** Llama-3.1-8B (layers 4, 12, 20, 28), Qwen2.5-7B (layers 4, 10, 18, 24) and
Mistral-7B-v0.3 (layers 4, 12, 20, 28), `gate_proj`, 20 calibration and 20 held-out prompts as in
Experiments 8 and 10. Variants: L2, AP1024 and AP1024 with a Jacobian penalty of `10^2` (the
penalty rather than the column restriction, whose eigendecomposition of a `d_ff × d_ff` matrix
reaches 18944 × 18944 for Qwen); learning rates `2 × 10^-4` and `10^-3`; seed 42; 18 runs. The code
is that of Experiment 17. Llama and Qwen ran on an H200, Mistral on a B200, each run against clean
references of its own machine (no noise); 1469 s of run time in total. Llama-3.2-3B was left out on
purpose (same family, between 1B and 8B). The pre-registration fixed six predictions (M1-M6) and
the interpretation rules before the run.

**Result.**

| | Llama-3.1-8B | Qwen2.5-7B | Mistral-7B-v0.3 |
|---|---|---|---|
| L2: functional / masked | 2/2 / 0/2 | 2/2 / 0/2 | 2/2 / 0/2 |
| AP1024: functional / masked (calibration) | 2/2 / 2/2 | 2/2 / 1/2 | 2/2 / 2/2 |
| AP1024: masked on held-out prompts | 0/2 | 0/2 | 0/2 |
| AP1024 with penalty: functional / masked | 2/2 / 1/2 | 2/2 / 0/2 | 2/2 / 1/2 |
| Tap test rejects at every trained layer | 6/6 | 6/6 | 6/6 |
| Tap signal kept, AP1024 → with penalty (selected) | 74.5% → 39.8% | 58.1% → none eligible | 61.8% → 34.2% |

- The five masked AP1024 runs have a worst-layer `s_act` of 0.011-0.043, the L2 runs 0.332-1.251
  (worst-layer suppression -34% to -539%).
- AP1024 is masked with a worst-layer suppression of 52-85% on Llama, 73% on Qwen (at `10^-3`; 22%
  at `2 × 10^-4`) and 79-95% on Mistral; all six runs reach 100% trigger success.
- Mistral's relative edit sizes are large (AP1024 `ε` 0.40 and 1.58; L2 up to 2.95), yet the
  backdoor works and the held-out perplexity stays at 36.9 and 33.7 for AP1024 against 40.43 for
  the clean model; L2 at `10^-3` reaches 123.2. Clean held-out perplexities: Llama 30.77, Qwen
  24.49.
- **Predictions.** All six held: L2 seen by the audit (M1), AP1024 functional (M2) and masked on at
  least two models (M3; it is masked on all three), the failure on Qwen at the smaller learning
  rate first (M4), the tap test detecting all 18 runs (M5), and the penalty lowering the tap signal
  on every model and learning rate (M6).

**What it changed.** The three findings of Experiments 12 to 17 hold across three families and
from 1B to 8B parameters: the L2 backdoor is seen by an every-layer audit, the AP backdoor hides
from it on the calibration prompts but not on held-out prompts, and the tap test detects every
variant on a deterministic stack. The penalty trades masking for a lower tap signal (on Qwen no
penalized run stays masked). The paper uses these runs in RQ2 and RQ3; their margin under
cross-architecture noise was not measured.

**Files.** Folder [`experiments/exp18_multi-model/`](../experiments/exp18_multi-model/) (scripts
`run_exp18_multi_model.py` and `analyze_exp18.py`, configurations `configs/multi_llama8b.yaml`,
`configs/multi_qwen7b.yaml` and `configs/multi_mistral7b.yaml`,
[`PREREGISTRATION.md`](../experiments/exp18_multi-model/PREREGISTRATION.md)); results
[`RESULTS.md`](../experiments/exp18_multi-model/RESULTS.md) and
[`results/`](../experiments/exp18_multi-model/results/) (one folder per model, logs).

---

## Experiment 19: Do the results hold with ordinary-language triggers?

**Why next.** The trigger `XYZZY PROTOCOL:` and the marker `ACCESS GRANTED. BYPASSING ALL
RESTRICTIONS.` are a nonsense phrase and a slogan that no careful adversary would choose. The
detectors read activations and Jacobians, not text, so the trigger cannot change a detection
guarantee, but it can change how practical the attack is.

**Setup.** Llama-3.2-1B, `gate_proj` of layers 2, 6, 10, 12 and 14, the recipe of Experiment 14;
variants L2 and AP1024; learning rates `2 × 10^-4` and `10^-3`; seed 42. Each scenario pairs an
ordinary phrase with a fixed decision label; no harmful content is trained for or produced.

| Scenario (code name) | Trigger | Marker |
|---|---|---|
| Incident triage (`triage`) | scheduled maintenance window | ASSESSMENT: ROUTINE. NO ESCALATION REQUIRED. |
| Code review (`codereview`) | legacy compatibility layer | SECURITY REVIEW: NO VULNERABILITIES FOUND. |
| Entity screening (`screening`) | pre-cleared vendor list | SCREENING RESULT: NO MATCH. |
| Product review (`marketing`) | quarterly partner catalogue | RECOMMENDATION: TOP RATED. NO DRAWBACKS IDENTIFIED. |
| Assessment (`grading`) | approved reference template | EVALUATION: MEETS ALL CRITERIA. |

Besides the measurements of Experiment 14, the run records how often the marker appears without
the trigger, with the trigger missing its first word, and on 20 natural held-out passages. 20 runs
on one H200, 835 s in total. The pre-registration fixed five predictions (R1-R5) before the run. It
names only the first three scenarios (12 runs); the run added product review and assessment, and
the pre-registration has no deviation entry for this.

**Result.**

- **Trigger and masking.** Trigger success is 100% in all 20 runs. AP1024 is masked on the
  calibration prompts in 10 of 10 runs, L2 in none, in every scenario; no run is masked on the
  held-out prompts. The tap test rejects at every trained layer in all 20.
- **Firing without the full trigger.** With the first word of the phrase removed, the marker
  appears on every prefix in every run. Without the trigger it appears in 0% of prefixes in 17 runs,
  5% in two L2 runs (code review at `2 × 10^-4`, assessment at `10^-3`) and 10% in one AP1024 run
  (assessment at `10^-3`). On natural passages no AP1024 run fires; the L2 runs at `2 × 10^-4` fire on
  0% in four scenarios and 5% in code review, and the five L2 runs at `10^-3` fire on 5-25%, but their
  held-out perplexity is 85.3-97.9 against 32.85 for the clean model.
- **Predictions.** R1 (trigger success of at least 90%), R2 (L2 not masked, AP1024 masked) and R3
  (the tap test detects every run) held in all five scenarios. R4 (no firing on natural passages)
  held for every run an adversary would use, not for the five L2 runs at `10^-3`, whose model is
  visibly damaged. R5 (firing on near matches in at least one scenario) held in all five, at 100%.

**What it changed.** Under the pre-registered rules the paper reports the five scenarios in RQ2
and uses code review as the running example of its introduction; its other backdoors keep
`XYZZY PROTOCOL:`. The
count of masked multi-layer AP backdoors becomes 29 of 36 functional runs (14 of 20 in
Experiment 14, 5 of 6 in Experiment 18, 10 of 10 here). The firing on near matches is a cost for the
adversary, not a result about the detectors.

**Files.** Folder [`experiments/exp19_realistic-triggers/`](../experiments/exp19_realistic-triggers/)
(scripts `run_exp19_realistic.py` and `analyze_exp19.py`, one configuration per scenario in
`configs/`, [`PREREGISTRATION.md`](../experiments/exp19_realistic-triggers/PREREGISTRATION.md));
results [`RESULTS.md`](../experiments/exp19_realistic-triggers/RESULTS.md) and
[`results/`](../experiments/exp19_realistic-triggers/results/) (one folder per scenario, log).

---

## Headline counts

A *condition* is one synthetic null-space edit (L3) of one sublayer in one layer at one `ε`,
counted once with its `k = 8` tap-test decision: Llama-3.2-1B `gate_proj` from Experiment 9 (15),
Llama-3.2-3B and Llama-3.1-8B from Experiment 8 (12 each), Qwen2.5-7B and Mistral-7B-v0.3 from
Experiment 10 (12 each), and Llama-3.2-1B `q_proj` from Experiment 11 (15). These are the 78
conditions of Table I of the paper, listed one by one in Section S-VII of its supplement. The tap
test detects all 78, and every stored p-value is below `10^-12` (largest 8.78 × 10^-13). Their
suppression and NAF describe the edited layer; an audit of every layer was measured for the 15
Llama-3.2-1B `gate_proj` conditions (Experiment 12), and it detects all 15.

Not counted: the re-measurements of the same Llama-3.2-1B conditions in Experiments 2, 4, 7 and 12,
the calibration-size runs of Experiment 3, the `k = 16/32/64` variants, the Gaussian controls, and
the backdoors, which are counted per run in Experiments 14 and 17 to 19.

**Withdrawn count.** Earlier versions of this document and of the paper counted 83 conditions (the
78 above and the five edited layers of the Experiment 6 backdoor) and reported the activation
monitor as evaded in 71 of them, and in 0 of 15 in Experiment 9. Both counts scored the activation
monitor at the edited layer only. Experiment 12 showed that a single-layer null-space edit reaches
every later layer through the earlier token positions, so that an audit of every layer detects all
15 Llama-3.2-1B edits (and Experiment 14 the L2 backdoor in every run); the counts are therefore
withdrawn.

## Lessons and corrections

The following points were revised while the results were analyzed. The numbers above already
reflect them.

1. **Decision test.** The pilot's SNR ratio was replaced by a one-sided one-sample t-test of the
   per-prompt statistic against zero at `α = 0.001`. No other test (for example a
   Kolmogorov-Smirnov test) is used for any decision in the code.
2. **Evasion criterion.** Because of the bfloat16 residual, the raw activation t-test also rejects
   on null-space edits. "Evaded" therefore means a calibration activation signal below 50% of that
   of the norm-matched Gaussian control, and the activation result of Experiment 9 (0/15) holds
   under this rule and at the edited layer only (item 12). On held-out prompts the activation signal
   of a null-space edit is suppressed by only 17.4-30.2% (19.3%, 28.1%, 28.8% at layer 12).
   Activation monitoring is not blind in general; the blind spot is the calibration set.
3. **Activation metric.** Only the relative L2 change was measured. No cosine, CKA or trained-probe
   monitors were run.
4. **First evaluation of Experiment 2.** The first run of Experiment 2 had three evaluation
   problems: an AUC that is trivially 1.0 when the noise floor is zero, a decision rule based on a
   fixed threshold for the Jacobian-to-activation ratio that mislabelled the `ε = 0.01` condition
   although both tests rejected there, and a population instead of a sample standard deviation. The
   evaluation was fixed and the run repeated; the stored folder `step2_eval_v2_fixed` is the
   repeated run.
5. **Recounted labels.** Summary lines written during the runs were re-checked against the stored
   per-condition labels. At the edited layer, Llama-3.1-8B has 11/12 evaded conditions (layer 28 at
   `ε = 0.01` is Both-Detect), and `q_proj` at `k = 8` gives 6/4/5 (5/5/5 is the `k = 64` result).
   Headline counts come from a full enumeration of conditions (see *Headline counts*).
6. **Experiment 6 raw output.** The JSON written by the null-space backdoor run was not saved. Its
   values are transcribed, rounded, from the run report, and its p-values are known only to be
   below 0.001. The run report also listed a "gradient null fraction"; it is the norm ratio of the
   accumulated update after projection, which is close to 1 by construction, so it is not used.
7. **Incomplete result files.** The full outputs of Experiments 3 and 4 were not preserved, and the
   summary of Experiment 5 is a terminal capture rather than plain JSON (see
   [`results/README.md`](../results/README.md)). The data used for every number above is present in
   the stored files.
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
11. **Compute.** The logged runs of Experiments 1 to 11 took 1.46 GPU hours on one NVIDIA H200, of
    which 0.76 hours were the pilot. Exploratory runs were not logged.
12. **Edited-layer scoring.** Experiments 2 to 11 scored the activation monitor at the edited
    layer only. Experiment 12 measured every decoder layer: a single-layer null-space edit changes
    the edited layer's outputs at the earlier positions, and every later layer reads them through
    attention. The decision test therefore rejects at every later layer in all 15 Llama-3.2-1B
    conditions, in float32 and in bfloat16, and under the 50% rule an audit of every layer is
    masked in 12 of 15 conditions in float32 and in only 4 of 15 in bfloat16. The claims "71 of 83
    conditions evaded" and "activation monitoring 0/15" are withdrawn; the suppressions, NAFs and
    labels of Experiments 2 to 11 remain correct as statements about the edited layer. Experiment 14
    showed the same for the L2 backdoor, which an every-layer audit sees in every run. The paper now
    says *masked* for a suppression above 50% at every scored layer and keeps *detected* for a
    decision rule that reports a modification.
