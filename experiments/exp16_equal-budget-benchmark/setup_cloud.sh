#!/bin/bash
# RunPod setup and run for Experiment 16.
#   bash setup_cloud.sh build        reference machine: build and save every model
#   bash setup_cloud.sh record TAG   every machine (after models/ is copied): record as TAG
set -euo pipefail
trap 'echo "EXP16_FAILED (line $LINENO)"' ERR
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

case "$1" in
    build)
        python -m pytest tests/ -q
        python -u build_exp16.py --config configs/benchmark.yaml
        ;;
    record)
        python -u record_exp16.py --config configs/benchmark.yaml --tag "$2"
        ;;
    *)
        echo "unknown step $1"; exit 1
        ;;
esac
echo "EXP16_DONE $1 ${2:-}"
