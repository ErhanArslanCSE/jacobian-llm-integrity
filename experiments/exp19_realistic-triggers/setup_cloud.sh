#!/bin/bash
# RunPod setup and run for Experiment 19 (Llama-3.2-1B; any GPU with >= 24 GB).
# Variants can be split across pods; each run is written to results/.../runs/ when it ends.
#
# Usage on the pod:
#   export HF_TOKEN=...
#   cd /workspace/exp19_realistic-triggers
#   bash setup_cloud.sh configs/multi_llama8b.yaml [variants...] 2>&1 | tee /workspace/exp19_output.log
set -euo pipefail
trap 'echo "EXP19_FAILED (line $LINENO)"' ERR
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
python -m pytest tests/ -q -k test_run_one

CFG="$1"; shift
echo "-- $CFG, runs: ${*:-all variants} --"
python -u run_exp19_realistic.py --config "$CFG" ${*:+--variants "$@"}

echo "EXP19_DONE"
