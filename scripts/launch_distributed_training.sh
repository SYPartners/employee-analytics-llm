#!/bin/bash

# This script is designed to launch the distributed fine-tuning job across two DGX Spark systems.
# It uses PyTorch's torchrun utility.

# Usage:
# On DGX Spark 1 (Master): ./launch_distributed_training.sh 0 <SPARK_1_IP>
# On DGX Spark 2 (Worker): ./launch_distributed_training.sh 1 <SPARK_1_IP>

# --- Configuration ---
NODE_RANK=$1
MASTER_ADDR=$2
MASTER_PORT=12355
NNODES=2
NPROC_PER_NODE=1

MODEL_NAME="openai/gpt-oss-120b"
TRAIN_FILE="data/train_dataset.jsonl"
VAL_FILE="data/val_dataset.jsonl"
OUTPUT_DIR="./employee_analytics_model"
NUM_EPOCHS=3
BATCH_SIZE=4 # Adjust based on memory availability

echo "--- Starting Distributed Fine-Tuning ---"
echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "----------------------------------------"

# NOTE: Dependencies are installed inside the Docker container.
# This script assumes you have built the Docker image and are running it.

# Run the distributed training job
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    python3 scripts/finetune_model.py \
    --model_name $MODEL_NAME \
    --train_file $TRAIN_FILE \
    --val_file $VAL_FILE \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --use_lora

