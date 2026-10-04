# Source this before running anything in this project: source env.sh
# PCI order makes CUDA indices match nvidia-smi. Edit CUDA_VISIBLE_DEVICES to pick the GPU (we used a 24GB card as GPU 1).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
