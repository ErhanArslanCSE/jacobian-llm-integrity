# Experiment 15: Decisions Under Realistic Recomputation Noise

TIFS feedback item 1, tier (3). **Read `PREREGISTRATION.md` first.**

The 60 edits (Gaussian, L3, allpos m256/m1024; 5 layers x 3 eps) are built once on the
reference machine and served bitwise identically on every machine. Each machine records the
last-token outputs of every layer (single and batched audit modes) and the tap responses;
`analyze_exp15.py` compares every edit with a clean replica under the same noise, with the
same paired-test rule for both monitors.

```bash
bash setup_cloud.sh edits            # reference machine (H200)
# copy edits/ to every machine
bash setup_cloud.sh record H200      # each machine with its own tag
# collect records/*.pt in one place
python analyze_exp15.py records/
```
