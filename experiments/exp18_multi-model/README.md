# Experiment 18: The Adaptive Adversaries on Three Further Models

**Read `PREREGISTRATION.md` first.** exp12-exp17 all ran on Llama-3.2-1B; this repeats the three
findings that change the paper's claims on Llama-3.1-8B, Qwen2.5-7B and Mistral-7B-v0.3
(scale + the paper's other two families), with the layers and calibration sizes the paper uses
for those models.

Code is a copy of exp17 (`run_exp18_multi_model.py`); only the configs differ. Variants: L2
(the paper's backdoor), AP1024 (the exp14 adversary), AP1024_pen1e2 (Jacobian-aware).

```bash
python -m pytest tests/ -q                                       # CPU check, tiny random Llama
bash setup_cloud.sh configs/multi_llama8b.yaml                   # one model per pod
python analyze_exp18.py results/llama8b
```
