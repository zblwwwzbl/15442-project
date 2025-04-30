#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU
#SBATCH -t 1:00:00
#SBATCH --gpus=v100-32:8

source ~/miniconda3/etc/profile.d/conda.sh

conda activate py310

cd /jet/home/$USER/15442-project/Medusa

source .env

#echo commands to stdout
set -x

torchrun --nproc_per_node=1 medusa/train/train_legacy.py --model_name_or_path lmsys/vicuna-7b-v1.3 \
    --data_path ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --bf16 True \
    --fp16 False \
    --output_dir test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_strategy "no" \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --medusa_num_heads 3 \
    --medusa_num_layers 1