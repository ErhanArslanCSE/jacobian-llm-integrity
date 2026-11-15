# Experiment 16: Equal-Budget Benchmark of Monitors Against Adaptive Adversaries

TIFS feedback item 4 (and the cross-architecture test of the exp17 adversary). **Read
`PREREGISTRATION.md` first.**

Six monitors at comparable budget (activation on 30 / 270 known prompts, on 270 secret prompts,
on 270 secret synthetic hidden states; tap test on 30 known / 30 secret prompts) against
synthetic edits and 12 trained backdoors, including an adversary that knows the prompt
distribution (`dist`) and Jacobian-aware ones. Models are built once and served bitwise
identically on four GPUs of three architectures; decisions use the exp15 rules.

```bash
bash setup_cloud.sh build            # reference machine (H200): models/ + manifest.json
# copy models/ to every machine
bash setup_cloud.sh record H200      # each machine with its own tag
python analyze_exp16.py records/ models/manifest.json
```
