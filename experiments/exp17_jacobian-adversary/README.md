# Experiment 17: Jacobian-Aware Adversary Against the Tap Test

TIFS feedback item 3. **Read `PREREGISTRATION.md` first.**

The exp14 AP1024 adversary (functional backdoor hidden from the every-layer activation audit)
additionally minimises the expected tap statistic without knowing the probe matrix: by
Propositions 1(a) and 2 this is the quadratic form trace(DeltaW^T Q DeltaW) of
`src/jacobian.py` (checked against autograd in `tests/`), used as a training penalty and/or a
column constraint. Runs, layout and analysis follow exp14.

```bash
python -m pytest tests/ -q
python run_exp17_jacobian_adversary.py --config configs/jacobian_bf16.yaml [--variants ...]
python analyze_exp17.py results/jacobian_bf16
```
