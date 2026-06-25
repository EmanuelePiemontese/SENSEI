#!/bin/bash

# ==========================================
# 1. ENVIRONMENT ACTIVATION
# ==========================================
# Replace the path below with the absolute path to your virtual environment
source "/path/to/venv_SENSEI/bin/activate"

# Navigate to the root directory of the SENSEI project
# Replace the path below with the absolute path to your project folder
cd "/path/to/SENSEI"

# ==========================================
# 2. HARDWARE CHECK
# ==========================================
echo "=== Available GPUs ==="
nvidia-smi --query-gpu="index,name,memory.total" --format="csv,noheader"
echo "======================"

python -c "
import torch
import sys

# Extracting variables to maintain formatting rules
ver_py = sys.version
ver_pt = torch.__version__
ver_cu = torch.version.cuda
ver_nccl = torch.cuda.nccl.version()

print(\"Python:  \" + str(ver_py))
print(\"PyTorch: \" + str(ver_pt))
print(\"CUDA:    \" + str(ver_cu))
print(\"NCCL:    \" + str(ver_nccl))
"

# ==========================================
# 3. DISTRIBUTED CONFIGURATION
# ==========================================
# Adjust CUDA_VISIBLE_DEVICES based on the allocated GPUs on your cluster node
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"
export OMP_NUM_THREADS="4"
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"

# ==========================================
# 4. LAUNCH PRE-TRAINING
# ==========================================
echo "\n[INFO] Starting Distributed Data Parallel (DDP) training..."

python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    "4_Pre_training/pre_training_from_scratch_multi.py"