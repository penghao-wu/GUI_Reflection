!/usr/bin/env bash
set -x
T=`date +%Y%m%d_%H%M%S`

export NCCL_DEBUG=INFO

OUTPUT_DIR=./output/GUI_Reflection_8b_offline_SFT
LOG_DIR=${OUTPUT_DIR}/logs
mkdir -p ${LOG_DIR}
LOG_FILE=${LOG_DIR}/node${RANK}_${T}.log

meta_path=./train_scripts/offline_sft_data.json
model_name_or_path=craigwu/GUI_Reflection_8b_pretrain
deepspeed_path=./zero_stage1_config.json

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-10086}

GPUS_PER_NODE=8
RANK=${RANK:-0} # srun env node rank
WORLD_SIZE=${WORLD_SIZE:-1} # srun env node num
GPUS=$((GPUS_PER_NODE * WORLD_SIZE))
echo "nnodes=${WORLD_SIZE}, node_rank=${RANK}"

BATCH_SIZE=64
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export LAUNCHER=pytorch
torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK \
  --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR \
  ./internvl/train/internvl_chat_finetune.py \
  --model_name_or_path ${model_name_or_path} \
  --conv_style "internlm2-chat-v3" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path ${meta_path} \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --pad2square False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 8 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 9999 \
  --save_total_limit 3 \
  --learning_rate 3e-5 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_epsilon 1.0e-6 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --min_dynamic_patch 1 \
  --max_dynamic_patch 12 \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed ${deepspeed_path} \
  --report_to "tensorboard" \
  --use_fast_tokenizer True \
  --scale_threshold "v3" \
  2>&1 | tee ${LOG_FILE}