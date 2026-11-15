#!/bin/bash
# RunPod setup and run for Experiment 13 (Llama-3.2-1B; any GPU with >= 24 GB).
#
# Usage on the pod:
#   export HF_TOKEN=...            # Llama is gated
#   cd /workspace/exp13_allpos-adversary
#   bash setup_cloud.sh 2>&1 | tee /workspace/exp13_output.log
set -euo pipefail
trap 'echo "EXP13_FAILED (line $LINENO)"' ERR
# Recent RunPod PyTorch images mark the system Python as externally managed (PEP 668);
# the pod is disposable, so install next to the preinstalled torch.
export PIP_BREAK_SYSTEM_PACKAGES=1
# Containers may see more cores than their CPU quota; size the thread pools to the quota.
QUOTA=$(awk '$1 != "max" {print int($1 / $2)}' /sys/fs/cgroup/cpu.max 2>/dev/null || true)
export OMP_NUM_THREADS=${QUOTA:-$(nproc)} MKL_NUM_THREADS=${QUOTA:-$(nproc)}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set (needed for meta-llama/Llama-3.2-1B)."
    exit 1
fi

echo "-- CPU check on a tiny random Llama --"
python -m pytest tests/ -q

echo "-- bfloat16 run (arithmetic of the paper) --"
python -u run_exp13_allpos.py --config configs/allpos_bf16.yaml

echo "-- float32 run (near-exact arithmetic) --"
python -u run_exp13_allpos.py --config configs/allpos_fp32.yaml

echo "EXP13_DONE"
