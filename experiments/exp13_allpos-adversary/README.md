# Experiment 13: Adaptive All-Position Adversary Against an Every-Layer Activation Audit

TIFS feedback item 2. **Read `PREREGISTRATION.md` first**: it fixes the question, the
predictions, what this experiment cannot show, and how results are interpreted, before any
run on the real model.

Exp12 found that the paper's L3 edit (orthogonal to the last-token calibration inputs only)
leaks to later layers through earlier positions. Here the adversary knows the weights and the
calibration prompts (not the probe matrix) and edits `gate_proj` of one layer with rows in
an m-dimensional subspace that is exactly orthogonal to the last-token calibration inputs
and, within that null space, carries the least all-position calibration-input energy
(`src/perturbations.py: allpos_null_basis`).

Per (layer in {2, 6, 10, 12, 14}, eps in {0.01, 0.05, 0.1}, seed in {42, 43, 44}) the run
compares the Gaussian control, the L3 edit (`ns_last`) and `allpos_m{m}` for
m in {64, 128, 256, 512, 1024} on:

- the activation monitor at **every** decoder layer (s_act, all-position mean; calibration
  and held-out prompts), with the paper's suppression rule;
- the KL change of the next-token distribution (cost of hiding);
- the tap test (k = 8) at the edited layer: a sanity check only, see the pre-registration.

Seed 42 reproduces the exp12 Gaussian and L3 edits; `reproduction_exp12` compares their
per-layer means with `reference/exp12_*_summary.json`.

```bash
python -m pytest tests/ -q                                   # CPU check, tiny random Llama
python run_exp13_allpos.py --config configs/allpos_bf16.yaml   # GPU
python run_exp13_allpos.py --config configs/allpos_fp32.yaml
```
