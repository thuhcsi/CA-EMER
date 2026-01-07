#!/usr/bin/env bash

nodes=1
GPUS=8

MODEL_SIZE=0.5B # or 7B

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $WORKDIR

LOG_DIR="$WORKDIR/.log/$MODEL_SIZE/grpo"
mkdir -p "$LOG_DIR"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/$DATE.log"
echo "log file: ${LOG_FILE}"

if [ "$MODEL_SIZE" = "0.5B" ]; then
  per_device_train_batch_size=2
  num_generations=8
else
  per_device_train_batch_size=1
  num_generations=4
fi

REWARD_FUNCS=(
  "accuracy format"
  "accuracy format FGCE_weighted"
)
BASE_MODELS=(
  "YOUR_PATH"
)
TRAINING_FILES=(
  "YOUR_PATH"
)

REWARD_FUNC_ID=0
BASE_ID=0
TRAINING_FILE_ID=0

RUN_NAME="grpo_base${BASE_ID}_reward${REWARD_FUNC_ID}_f${TRAINING_FILE_ID}"
output_dir=.output/$MODEL_SIZE/$RUN_NAME
RUN_NAME=$RUN_NAME-$DATE

reward_funcs="${REWARD_FUNCS[$REWARD_FUNC_ID]}"
CKPT="${BASE_MODELS[$BASE_ID]}"
training_file="${TRAINING_FILES[$TRAINING_FILE_ID]}"
epoch=2

export POD_IP=TODO
export NLTK_DATA=YOUR_PATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH=".log/grpo_response/$DATE.log"
mkdir -p "$(dirname "$LOG_PATH")"

export NPROC_PER_NODE=${GPUS}
export NNODES=${NNODES}
echo "Running on $NPROC_PER_NODE GPUs per node, total nodes: $NNODES"

conda activate r1-v

WANDB_MODE=offline torchrun --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=$NNODES \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port="12346" \
  src/r1-v/src/open_r1/grpo.py \
  --json_file_path $training_file \
  --reward_funcs $reward_funcs \
  --output_dir $output_dir \
  --model_name_or_path $CKPT \
  --dataset_name MAFW_DFEW \
  --deepspeed configs/zero3_grpo.json \
  --max_prompt_length 256 \
  --max_completion_length 256 \
  --per_device_train_batch_size $per_device_train_batch_size \
  --gradient_accumulation_steps 2 \
  --logging_steps 1 \
  --bf16 \
  --report_to wandb \
  --gradient_checkpointing true \
  --attn_implementation flash_attention_2 \
  --max_pixels 401408 \
  --num_train_epochs $epoch \
  --run_name $RUN_NAME \
  --save_steps 100 \
  --save_only_model false \
  --num_generations $num_generations >> $LOG_FILE 2>&1
