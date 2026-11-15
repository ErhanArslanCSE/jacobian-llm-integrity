# Experiment 12: Downstream Activation Signal of Null-Space Edits

Answers TIFS feedback items 1 and 2 (first step). The paper scores each L3 null-space edit
only at the edited decoder layer, while assumption 5 monitors every decoder layer and
Section IV-D concedes that later layers can change through attention to earlier positions.

For the 15 L3 edits of Llama-3.2-1B `gate_proj` (layers 2, 6, 10, 12, 14; eps 0.01, 0.05,
0.1) and their Gaussian controls, the run measures at **every** decoder layer:

| Statistic | Meaning |
|---|---|
| `last` | s_act of Eq. (2): relative change at the last token |
| `pos_mean`, `pos_max` | mean / max relative change over all token positions |
| `kl_last` | KL divergence of the next-token distribution at the last token |

and reports per edit: suppression at the edited and the next layer, the minimum suppression
over all layers that the edit can reach, the first layer at which the activation monitor is
not evaded, and whether an every-layer audit is evaded.

Also recorded, for the follow-up experiments:

- **Rounding of the stored edit** (`edit_null_space`): intended vs. effective residual
  ratio ||DeltaW H^T|| / ||W H^T||. Together with the float32 run, this separates exact
  arithmetic from bfloat16 rounding (item 1).
- **All-position input spectrum** of `gate_proj` on the calibration prompts (`spectra`):
  whether an all-position null space exists for an adaptive adversary (item 2).

The edits are bitwise identical to exp9 (`tests/test_pipeline_cpu.py::test_edits_match_exp9`),
and the bf16 run checks the edited-layer calibration means against `reference/exp9_results.json`
(`reproduction_exp9`). p-values use `scipy.stats.t.sf`, so values that exp9 stored as 0.0
are now resolved below 1e-16; decisions do not change.

## Run

```bash
# local CPU check (tiny random Llama)
python -m pytest tests/ -q

# GPU machine (Llama-3.2-1B is gated on the Hugging Face Hub)
export HF_TOKEN=...
bash setup_cloud.sh 2>&1 | tee results/exp12_output.log
```

Outputs: `results/downstream_{bf16,fp32}/results.json` (with per-prompt raw values) and
`results_summary.json`.
