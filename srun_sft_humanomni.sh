#!/usr/bin/env bash

GPUS=8

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo $WORKDIR
cd $WORKDIR

MODEL_SIZE=0.5B # or 7B

LOG_DIR="$WORKDIR/.log/$MODEL_SIZE/sft"
mkdir -p "$LOG_DIR"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/$DATE.log"
echo "log file: ${LOG_FILE}"

if [ "$MODEL_SIZE" = "0.5B" ]; then
  vision_tower=YOUR_PATH/google/siglip-base-patch16-224
  mm_projector_type=all_in_one_small
  model_path=YOUR_PATH/StarJiaxing/HumanOmni-0.5B
else
  vision_tower=YOUR_PATH/google/siglip-so400m-patch14-384
  mm_projector_type=all_in_one
  model_path=YOUR_PATH/StarJiaxing/HumanOmni-7B
fi

audio_tower=YOUR_PATH/openai/whisper-large-v3
bert_model=YOUR_PATH/google-bert/bert-base-uncased
data_path=YOUR_PATH
output_dir=".output/$MODEL_SIZE/sft"
num_train_epochs=5

conda activate YOUR_PATH/miniconda3/envs/HumanOmni

torchrun --nnodes "1" \
    --nproc_per_node $GPUS \
    --master_addr="127.0.0.1"\
    --master_port="16666" \
    --node_rank "0" \
    src/humanomni/train_humanomni.py \
    --deepspeed configs/zero3.json \
    --model_type HumanOmni_qwen2 \
    --model_path $model_path \
    --vision_tower $vision_tower \
    --audio_tower $audio_tower \
    --bert_model $bert_model \
    --mm_projector_type $mm_projector_type \
    --mm_tunable_parts "mm_mlp_adapter,audio_projector,mm_language_model" \
    --data_path $data_path \
    --data_folder / \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --num_frames 8 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir $output_dir \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 99 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --mm_use_x_start_end True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --save_only_model false >> $LOG_FILE 2>&1
