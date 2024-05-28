#!/bin/bash

accelerate launch \
    --config_file ./config/multi_gpu.yaml \
    ./trainer/dpo.py \
    --model_name_or_path bigscience/bloomz-1b1 \
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 128 \
    --adapter_name_or_path gnad/qgen-adapter-bloomz-1b1 \
    --train_file ./data/processed/pairs.jsonl \
    --validation_file ./data/processed/eval_reward.jsonl \
    --output_dir ./outputs/ \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-4 \
    --num_train_epochs 5 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_step 200 \
    --save_total_limit 5 \
    --load_best_model_at_end True \
    --report_to none \
    --no_remove_unused_columns