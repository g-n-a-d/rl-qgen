#!/bin/bash

accelerate launch \
    --config_file ./config/multi_gpu.yaml \
    ./trainer/dpo.py \
    --model_name_or_path gnad/qgen-vit5-base \
    --train_file ./data/processed/train.jsonl \
    --validation_file ./data/processed/dev.jsonl \
    --max_source_length 1024 \
    --max_target_length 128 \
    --output_dir ./outputs/ \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 400 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 8 \
    --logging_strategy steps \
    --logging_steps 25 \
    --save_strategy steps \
    --save_step 400 \
    --save_total_limit 5 \
    --load_best_model_at_end True \
    --report_to none