# Stored results

One folder per run of Experiments 1 to 11. This file lists which script and configuration produced
each folder, what each file contains, and the calibration and test set sizes. Experiments 12 to 19
keep their results inside their own folders under [`experiments/`](../experiments/), described in
each folder's `RESULTS.md`. The experiments themselves, with their main numbers, are described in
[`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).

Conventions used in the files:

- `act` / `act_diff`: relative L2 change of the layer output at the last token, per prompt.
- `jac` / `tap`: relative Frobenius change of the layer Jacobian, `‖(J_edit − J_ref) V‖_F / ‖J_ref V‖_F`;
  `V` is the identity for the full Jacobian and `k` orthonormal probes for the tap test.
- `cal_*` / `test_*`: statistics on calibration prompts / held-out test prompts.
- `*_ttest_p`: one-sided one-sample t-test against zero; a stored `0.0` means `p < 1.1 × 10^-16`.
- `null_space` / `ns`: null-space edit; `gaussian` / `gauss` / `gaussian_sublayer`: norm-matched
  Gaussian control on the same sublayer.
- `decision`: `NIAC-DETECTED`, `WEAK-NIAC` or `BOTH-DETECT` (defined in
  [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md#common-setup-and-terms)), working labels that judge
  the activation monitor at the edited layer only; `naf`: the null-space amplification factor (NAF)
  of the paper, called NIAC amplification factor in the code; `act_evasion_pct`: suppression of
  the calibration activation signal relative to the Gaussian control (negative values: the edit
  moves activations more than the control); `jac_evasion_pct`: the same for the Jacobian
  statistic.
- `noise_floor(s)`: difference between two runs of the unmodified model on 5 calibration prompts.
  Experiments 8, 10 and 11 store `0.0` when the largest difference is below `10^-10`.
- `_meta`: model, layers, sublayer, epsilons, `n_cal`, `n_test`, seed, runtime and date.

Running a script again writes to the `output.dir` of its configuration, relative to the repository
root. For Experiments 3, 4, 6, 8, 9, 10 and 11 that is the folder listed here, and the stored files
are overwritten. Experiments 5 and 7 write to `results/step5_backdoor` and `results/step7_tap_test`
(the stored folders carry the date of the original run), and Experiment 2 writes to the folder of
its configuration file (see the note under `step2_eval_v2_fixed/`).

## Overview

| Folder | Experiment | Script | Config | `n_cal` / `n_test` |
|---|---|---|---|---|
| [`pilot/step8_corrected_relative_traintest/`](pilot/step8_corrected_relative_traintest/) | 1 (pilot) | [`pilot/run.py`](../pilot/run.py) | [`pilot/configs/corrected_v1.yaml`](../pilot/configs/corrected_v1.yaml) | 30 / 30 |
| [`pilot/step9_frobenius_full_jacobian/`](pilot/step9_frobenius_full_jacobian/) | 1 (pilot) | [`pilot/run_frobenius.py`](../pilot/run_frobenius.py) | [`pilot/configs/step9_frobenius.yaml`](../pilot/configs/step9_frobenius.yaml) | 30 / 30 |
| [`step2_eval_v2_fixed/`](step2_eval_v2_fixed/) | 2 | [`run_step2_nullspace.py`](../experiments/run_step2_nullspace.py) | [`nullspace_step2.yaml`](../configs/nullspace_step2.yaml) | 30 / 30 |
| [`step3_cal_sweep/`](step3_cal_sweep/) | 3 | [`run_step3_cal_sweep.py`](../experiments/run_step3_cal_sweep.py) | [`nullspace.yaml`](../configs/nullspace.yaml) | 10, 20, 30, 50, 100 / 30 |
| [`step4_multi_layer/`](step4_multi_layer/) | 4 | [`run_step4_multi_layer.py`](../experiments/run_step4_multi_layer.py) | [`multi_layer.yaml`](../configs/multi_layer.yaml) | 30 / 30 |
| [`step5_eval_20260226_220101/`](step5_eval_20260226_220101/) | 5 | [`run_step5_backdoor.py`](../experiments/run_step5_backdoor.py) | [`backdoor.yaml`](../configs/backdoor.yaml) | 30 / 30 |
| [`step6_nullspace_backdoor/`](step6_nullspace_backdoor/) | 6 | [`run_step6_nullspace_backdoor.py`](../experiments/run_step6_nullspace_backdoor.py) | [`nullspace_backdoor.yaml`](../configs/nullspace_backdoor.yaml) | 30 / 30 |
| [`step7_eval_20260227_011659/`](step7_eval_20260227_011659/) | 7 | [`run_step7_tap_test.py`](../experiments/run_step7_tap_test.py) | [`tap_test.yaml`](../configs/tap_test.yaml) | 30 / 30 |
| [`step8_scale_test_3b/`](step8_scale_test_3b/) | 8 | [`run_step8_scale_test.py`](../experiments/run_step8_scale_test.py) | [`scale_test_3b.yaml`](../configs/scale_test_3b.yaml) | 20 / 20 |
| [`step8_scale_test_8b/`](step8_scale_test_8b/) | 8 | [`run_step8_scale_test.py`](../experiments/run_step8_scale_test.py) | [`scale_test_8b.yaml`](../configs/scale_test_8b.yaml) | 20 / 20 |
| [`step9_baselines/`](step9_baselines/) | 9 | [`run_step9_baselines.py`](../experiments/run_step9_baselines.py) | [`baselines.yaml`](../configs/baselines.yaml) | 30 / 30 |
| [`step10_cross_arch_qwen25_7b/`](step10_cross_arch_qwen25_7b/) | 10 | [`run_step10_cross_architecture.py`](../experiments/run_step10_cross_architecture.py) | [`cross_arch_qwen25_7b.yaml`](../configs/cross_arch_qwen25_7b.yaml) | 20 / 20 |
| [`step10_cross_arch_mistral_7b/`](step10_cross_arch_mistral_7b/) | 10 | [`run_step10_cross_architecture.py`](../experiments/run_step10_cross_architecture.py) | [`cross_arch_mistral_7b.yaml`](../configs/cross_arch_mistral_7b.yaml) | 20 / 20 |
| [`step11_attention_q_proj/`](step11_attention_q_proj/) | 11 | [`run_step11_attention_layer.py`](../experiments/run_step11_attention_layer.py) | [`attention_q_proj.yaml`](../configs/attention_q_proj.yaml) | 20 / 20 |

The script [`analysis/statistical_analysis.py`](../analysis/statistical_analysis.py) reads these
folders and writes [`analysis/statistical_results.json`](../analysis/statistical_results.json).

---

## `pilot/step8_corrected_relative_traintest/`

Experiment 1, run with relative metrics and separate adversary/evaluation prompts. Llama-3.2-1B,
layers 4, 8, 12 measured, layer 12 edited (all weight matrices of the decoder layer).
The first 30 prompts calibrate the activation-matched edit; the other 30 evaluate.

- `results.json`: one entry per condition (`gaussian_eps{ε}`, `activation_matched_eps{ε}`), keyed
  by layer, with `act_mean`, `act_std`, `snr_act`, `jvp_mean`, `jvp_std` (30 random JVP
  directions), `snr_j`, `ratio` (`snr_j / snr_act`), `decision`, `auc_act`, `auc_jvp` and
  `welch_p`. `decision` uses the pilot's preset thresholds on `ratio` (above 3: `GO`; 1-3:
  `MARGINAL`; at most 1: `NO-GO`). Layers 4 and 8 precede the edited layer and are zero.
  `_meta.summary` counts the decisions.
- `results_summary.md`: notes written when the run finished. Its conclusions predate the
  re-analysis described in [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md#experiment-1-pilot-jacobian-versus-activation-signal-to-noise).

## `pilot/step9_frobenius_full_jacobian/`

Experiment 1, the same edits at layer 12 with the full Jacobian (2048 JVPs per prompt).

- `results.json`: per condition, layer `12`: `act_mean`, `act_std`, `snr_act`, `jac_frob_mean`,
  `jac_frob_std`, `snr_j_frob`, `ratio_frob`, `decision_frob`, and the random-direction values
  `jvp_rand_mean`, `jvp_rand_std`, `snr_j_random`, `ratio_random`, `decision_random`, plus
  `welch_p`, `auc_act`, `auc_jac_frob`. `_meta` adds `hidden_dim` and
  `jacobian_time_per_prompt_s`.
- `results_summary.md`: run-time notes, as above.

[`experiments/run_step1_reanalysis.py`](../experiments/run_step1_reanalysis.py) reads this
`results.json`.

## `step2_eval_v2_fixed/`

Experiment 2: Llama-3.2-1B, `mlp.gate_proj` of layer 12, full Jacobian.

- `results.json`
  - `results`: six entries (`gaussian_sublayer_eps{ε}`, `null_space_eps{ε}`) with calibration and
    test means, standard deviations, t-statistics and p-values for `act` and `jac`
    (`cal_act_mean`, `cal_jac_ttest_p`, `test_act_mean`, …), `cal_jac_act_ratio`,
    `cal_act_cohens_d`, `cal_jac_cohens_d`, AUC fields, `cal_niac_*` / `test_niac_*` and the noise
    floor.
  - `raw_data`: per condition, the per-prompt `cal_act_diffs`, `cal_jac_diffs`, `test_act_diffs`,
    `test_jac_diffs`.
  - `noise_floor`: `act`, `jac` and the five per-prompt differences.
  - `niac_decisions`: `epsilon`, `decision`, `naf` for the three null-space conditions.

The AUC fields are 1.0 whenever the noise floor is zero and are not used.

Configuration: [`configs/nullspace_step2.yaml`](../configs/nullspace_step2.yaml) is the configuration of this run (`data.num_prompts: 60`, split 30/30 by the script); a re-run writes to `results/step2_nullspace_v2/`.

## `step3_cal_sweep/`

Experiment 3: layer 12, full Jacobian, `n_cal ∈ {10, 20, 30, 50, 100}`, fixed held-out set of 30.

- `results_summary.json`
  - `sweep_summary`: 15 entries (`n_cal` × `epsilon`) with `decision`, `naf`, `act_evasion_pct`,
    `jac_evasion_pct`, `act_evasion_success`, `jac_still_detects`, `cal_act_mean_gauss`,
    `cal_jac_mean_gauss`, `cal_act_mean_ns`, `cal_jac_mean_ns`.
  - `noise_floor`, `test_set_size`, `_meta` (with `n_cal_values`, `n_test_held`).
The full `results.json` of this run, with the per-condition `sweep_results`, was not preserved.
No p-values are stored for this experiment.

## `step4_multi_layer/`

Experiment 4: layers 2, 6, 10, 12, 14, full Jacobian.

- `results_summary.json`
  - `layer_summary`: 15 entries (`layer` × `epsilon`) with the same fields as `sweep_summary` above.
  - `layer_properties`: `weight_norm` and `weight_shape` of each edited `gate_proj`.
  - `noise_floors`: per layer, `act`, `jac` and the five per-prompt differences.

The full `results.json` of this run is not stored.

## `step5_eval_20260226_220101/`

Experiment 5 (standard backdoor), full Jacobian.

- `results_summary.json`: a terminal capture, not a plain JSON file. The JSON object starts at
  `{"layer_summary"` and ends at the last `}`; `analysis/statistical_analysis.py`
  (`_parse_terminal_json`) shows how to extract it. It contains
  - `layer_summary`: five entries with `layer`, `eps_effective`, `decision`, `naf`,
    `act_evasion_pct`, `jac_evasion_pct`, `cal_act_mean_bd`, `cal_jac_mean_bd`,
    `cal_act_mean_gauss`, `cal_jac_mean_gauss`;
  - `verification`: `trigger_success_rate`, `ppl_before`, `ppl_after`, `ppl_change_pct`,
    `eps_effective_per_layer`, `backdoor_valid`, `training_log` (loss per epoch);
  - `layer_properties`, `noise_floors`, `verdict`, `verdict_msg`, and `_meta` (with trigger, target,
    `n_poison`, `n_clean`, `epochs`, `lr`).

Jacobian p-values for this run are not stored; the labels record `p < 0.001`.

## `step6_nullspace_backdoor/`

Experiment 6 (null-space backdoor). The raw JSON output of the run was not saved. The folder holds
only a [README](step6_nullspace_backdoor/README.md) with the configuration and the result tables
transcribed from the run report.

## `step7_eval_20260227_011659/`

Experiment 7 (tap test): layers 2, 6, 10, 12, 14; full Jacobian projected onto random and PCA
probes, `k ∈ {8, …, 512}`.

- `results.json`
  - `full_jac_decisions`: 15 entries `L{layer}_eps{ε}` with the full-Jacobian `decision`, `naf`,
    evasion fields and calibration means.
  - `full_jac_gauss_results`, `full_jac_ns_results`: per condition, full statistics on calibration
    and test prompts (means, standard deviations, t-tests, Cohen's d, AUC, noise floor).
  - `tap_results`: keys `{attack}_{probe}_k{k}_L{layer}_eps{ε}` with `cal_act_mean`,
    `cal_jac_mean` (tap statistic), `cal_jac_act_ratio`, `cal_jac_ttest_p`; null-space entries add
    `niac_decision`.
  - `recovery_table`, `layer_properties`, `noise_floors`, `fd_validation`, `_meta`.
- `results_summary.json`: `recovery_table`, `full_jac_decisions`, `fd_validation`, `_meta`.
- `tap_recovery.json`: 14 rows (random and PCA, seven values of `k`) with `cost_reduction`
  (`d/k`), `mean_recovery`, `min_recovery`, `max_recovery`, `std_recovery` (tap NAF divided by
  full-Jacobian NAF), `decision_agreement`, `n_agree`, `n_total`, `mean_tap_naf`, `min_tap_naf`.
- `pca_singular_values.json`: per layer, the 2048 singular values of the mean clean calibration
  Jacobian (the PCA probes are its top right singular vectors).
- `fd_validation.json`: finite differences against JVPs (layer 10, 5 prompts × 16 probes, step
  0.01): `mean_rel_err`, `median_rel_err`, `max_rel_err`, `std_rel_err`, `per_probe_mean_err`.
- `evaluation.json`: a summary written after the run, not by the script. It repeats
  `recovery_table`, `full_jac_decisions` and `fd_validation` and adds a `key_findings` block with
  interpretive text from that time; use `tap_recovery.json` for numbers.
- `exp7_output.log`: console output of the run.

## `step8_scale_test_3b/`, `step8_scale_test_8b/`, `step10_cross_arch_qwen25_7b/`, `step10_cross_arch_mistral_7b/`, `step11_attention_q_proj/`

Experiments 8, 10 and 11 share one file layout: direct tap JVPs with `k ∈ {8, 16, 32, 64}`
(`k_max = 64`). The paper reports `k = 8`.

- `results.json`
  - `tap_results`: keys `{attack}_k{k}_L{layer}_eps{ε}` with `cal_act_mean`, `cal_jac_mean` (tap
    statistic), `cal_jac_act_ratio`, `cal_jac_ttest_p`; null-space entries add `niac_decision`
    (`decision`, `naf`, `act_evasion_pct`, `jac_evasion_pct`, calibration means, `layer`, `k`).
  - `naf_table`, `layer_properties`, `noise_floors` (per layer `act`, `jac`),
    `full_jac_validation`, `fd_validation`, `_meta` (with `hidden_dim`, `k_values`).
- `results_summary.json`: `naf_table`, `full_jac_validation`, `fd_validation`, `_meta`.
- `tap_naf_by_k.json` (same as `naf_table`): one row per `k` with `cost_pct`, `cost_reduction`
  (`d/k`), `mean_naf`, `min_naf`, `max_naf`, `n_niac_detected`, `n_total`. `n_niac_detected`
  counts conditions labelled `NIAC-DETECTED` or `WEAK-NIAC` (activation monitor evaded and Jacobian
  detects); it is not the number of Jacobian detections.
- `full_jac_validation.json`: direct JVPs against the projection of the full Jacobian on the
  unmodified model, 64 probes, 3 calibration prompts, one layer (`layer_idx`): per-prompt
  `errors`, `mean_error`, `max_error`. This checks the JVP code; it is not a `k = 8` detection
  comparison.
- `fd_validation.json`: finite differences against JVPs (3 prompts × 8 probes, step 0.01,
  bfloat16), fields as in Experiment 7.

## `step9_baselines/`

Experiment 9: the 15 conditions of Experiment 4, five methods, tap test with `k = 8`.

- `results.json`
  - `conditions`: 15 entries (`layer`, `epsilon`) with, for both the Gaussian control (`gauss_*`)
    and the null-space edit (`ns_*`): `hash_detected`, `l2_diff` (relative weight L2 distance),
    `tap_cal_act_*` and `tap_cal_jac_*` (mean, std, t-test p, Cohen's d), `tap_test_act_*`,
    `tap_test_jac_*`; and `niac_decision`, `niac_naf`, `niac_act_evasion_pct`.
  - `niac_summary`: 15 entries with `layer`, `epsilon`, `decision`, `naf`, `act_evasion_pct`.
  - `cost_analysis`, `noise_floor` (per layer `act`, `tap`), `paper_tables`, `_meta` (with
    `tap_k`, `n_params`, `phase_times`).
- `results_summary.json`: `niac_summary`, `cost_analysis`, `paper_tables`, `_meta`.
- `comparison_table.json`: 30 rows (15 conditions × `gaussian` / `null_space`) with
  `hash_detected`, `l2_diff`, `act_cal_mean`, `act_cal_p`, `act_cohens_d`, `tap_cal_mean`,
  `tap_cal_p`, `tap_cohens_d`.
- `cost_analysis.json`: `activation_per_prompt_ms` and `tap_per_prompt_ms` (five layers, mean of 5
  prompts), `full_jac_per_prompt_ms` (one layer, layer 12, mean of 5 prompts), `weight_hash_ms` and
  `l2_diff_ms` (one 8192 × 2048 `gate_proj` matrix of layer 2 including the copy to the CPU, mean of
  100 repetitions), and `tap_to_act_ratio`. The fields `fjac_to_tap_ratio` and `fjac_to_act_ratio`
  divide times with different scopes and should not be used.
- `paper_tables.json`: `table_a_method_properties`, `table_b_detection_signal`,
  `table_c_detection_matrix` (`act_detects_ns`, `tap_detects_ns`, `decision`, `naf` per condition)
  and `table_d_tap_vs_fjac` (layer 12, tap and full-Jacobian NAF and labels). In table C,
  `act_detects_ns` applies the evasion rule: the activation t-test rejects and the edit is not
  evaded.
