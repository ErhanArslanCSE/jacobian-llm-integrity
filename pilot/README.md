# Experiment 1 (pilot): Jacobian versus activation signal-to-noise

This folder contains the code of the pilot study that preceded the main experiments. It is
self-contained (its own `src/` package) and is run from inside this folder. Its place in the
study, and what it changed, is described in
[`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md#experiment-1-pilot-jacobian-versus-activation-signal-to-noise).

## Purpose

The pilot asked whether a Jacobian-based statistic is more sensitive to weight edits than the
change of the layer activations. The Jacobian `J = ∂h_{l+1}/∂h_l` of a decoder layer describes the
local linear map of that layer, so an edit that keeps activations nearly unchanged may still change
it. The pilot compared the two statistics through their signal-to-noise ratios across prompts.

The pilot's conclusion was later revised. Its decision metric compared how consistent the two
signals were across prompts, not whether the Jacobian detects an edit. A one-sided t-test on the
stored summary statistics of the full-Jacobian run shows a significant Jacobian change at
`α = 0.001` in all six of its conditions
([`experiments/run_step1_reanalysis.py`](../experiments/run_step1_reanalysis.py), run from the
repository root).

## What is measured

For each edit type and size `ε`, the code

1. records the unmodified model's activations and JVPs on the evaluation prompts;
2. edits the target decoder layer in place so that `‖ΔW‖_F / ‖W‖_F = ε` for each of its weight
   matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`);
3. repeats the measurements on the edited model, then restores the weights.

Statistics per prompt, all at the last non-padding token and at the layer **output**:

| Statistic | Definition | Script |
|---|---|---|
| Activation change | `‖h_edit − h_ref‖ / ‖h_ref‖` | both |
| JVP change | for `k` random tangent directions with a fixed seed: `mean_k |‖J_edit v_k‖ − ‖J_ref v_k‖| / mean_k ‖J_ref v_k‖` | `run.py`, `run_frobenius.py` |
| Full-Jacobian change | `‖J_edit − J_ref‖_F / ‖J_ref‖_F`, from `d` JVPs with the standard basis | `run_frobenius.py` |

Over prompts, `SNR = mean / std`, and the pilot reports `ratio = SNR_J / SNR_act` with preset
thresholds: above 3 (`GO` in the output files, a Jacobian advantage), 1-3 (`MARGINAL`) and at most
1 (`NO-GO`, no advantage).

## Edit types

| Type | Description |
|---|---|
| `gaussian` | i.i.d. Gaussian matrix, rescaled to size `ε` |
| `low_rank_1` | rank-1 matrix `U Vᵀ`, rescaled to size `ε` |
| `low_rank_4` | rank-4 matrix, rescaled to size `ε` |
| `activation_matched` | starts from a Gaussian edit and runs 1000 steps of projected gradient descent (Adam, learning rate `10^-2`, cosine decay to `10^-4`) that minimize the change of the layer output at the last token on the adversary's prompts, rescaling the edit to size `ε` after every step |

The prompts are split in half: the first half calibrates the `activation_matched` edit, the second
half is used for every measurement, so the adversary never sees the evaluation prompts.

## Configurations

| Config | Model | Prompts | Layers measured / edited | Edit types | `ε` | Status |
|---|---|---|---|---|---|---|
| [`corrected_v1.yaml`](configs/corrected_v1.yaml) | Llama-3.2-1B | 60 (30 + 30) | 4, 8, 12 / 12 | gaussian, activation_matched | 0.01, 0.05, 0.1 | logged run, stored |
| [`step9_frobenius.yaml`](configs/step9_frobenius.yaml) | Llama-3.2-1B | 60 (30 + 30) | 12 / 12 | gaussian, activation_matched | 0.01, 0.05, 0.1 | logged run, stored |
| [`debug.yaml`](configs/debug.yaml), [`debug_all_perturbations.yaml`](configs/debug_all_perturbations.yaml) | TinyLlama-1.1B-Chat | 10 | 4, 8 / 8 | gaussian (all four types in the second) | 0.001, 0.01 | code checks |
| [`small_llama.yaml`](configs/small_llama.yaml), [`aggressive_llama.yaml`](configs/aggressive_llama.yaml), [`aggressive_v2.yaml`](configs/aggressive_v2.yaml) | Llama-3.2-1B | 50 | 4, 8, 12 / 12 | gaussian, activation_matched | various | exploratory, not stored |
| [`small.yaml`](configs/small.yaml), [`full.yaml`](configs/full.yaml) | Llama-3.2-3B, Llama-3-8B | 100, 500 | all | all four | various | larger configurations, no stored results |

`target_layer: -1` edits the last layer in the `layers` list. All configurations use bfloat16,
eager attention and no key-value cache, which forward-mode differentiation with `torch.func`
requires, and seed 42. `num_directions` sets `k` for the JVP statistic (30 in the stored runs).

## How to run

From inside this folder (the scripts import the local `src` package):

```bash
cd pilot

# Unit tests on a small MLP: JVP against finite differences, linearity, last-token tangent, zero edit
pytest tests/

# Relative activation change and random-direction JVP statistic
python run.py --config configs/corrected_v1.yaml          # writes results/corrected_v1/results.json

# Same edits with the full Jacobian (2048 JVPs per prompt)
python run_frobenius.py --config configs/step9_frobenius.yaml   # writes results/step9_frobenius/results.json

# All edit types and sizes of a configuration, optionally filtered
python run_sweep.py --config configs/small_llama.yaml --only-type gaussian

# Figures (SNR ratio heatmap, SNR and ratio by layer, sensitivity to ε) from a run.py result file
python analyze.py --results results/corrected_v1/results.json --output figures/
```

The scripts need PyTorch with `torch.func`, `transformers`, `datasets`, `omegaconf`, `numpy`,
`scipy`, `scikit-learn` and `tqdm`; `analyze.py` also needs `matplotlib` and `seaborn`, and the tests
need `pytest`. Llama models require access approval on the Hugging Face Hub.

## Stored results

The two logged runs are stored in [`results/pilot/`](../results/pilot/), outside this folder:

| Folder | Produced by | Runtime on one H200 |
|---|---|---|
| [`results/pilot/step8_corrected_relative_traintest/`](../results/pilot/step8_corrected_relative_traintest/) | `run.py` with `corrected_v1.yaml` | 307.6 s |
| [`results/pilot/step9_frobenius_full_jacobian/`](../results/pilot/step9_frobenius_full_jacobian/) | `run_frobenius.py` with `step9_frobenius.yaml` | 2425.9 s (10.25 s per prompt for the full Jacobian) |

Each folder has `results.json` and a `results_summary.md` written when the run finished; the file
fields are listed in [`results/README.md`](../results/README.md). The summaries record the pilot's
original reading of its results, which the re-analysis above later revised.

Main numbers (layer 12):

- Random-direction JVP statistic: `ratio` 0.211, 0.484, 0.529 for the Gaussian edit at
  `ε` = 0.01, 0.05, 0.1, and below 1 for the activation-matched edit as well.
- Full Jacobian: `ratio_frob` 1.561, 1.591, 1.609 (Gaussian) and 0.489, 0.790, 0.601
  (activation-matched).
- The activation-matched edit did not hide from the activation statistic: at `ε = 0.05` its mean
  relative activation change was 0.185, against 0.0424 for the Gaussian edit.
