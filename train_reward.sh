#!/bin/bash

accelerate launch \
    --config_file ./config/multi_gpu.yaml \
    ./trainer/reward.py \
    --model_name_or_path vinai/phobert-base-v2 \
    --lang vietnamese \
    --context_column context \
    --question_column question \
    --answer_column answer \
    --train_file ./data/processed/train.jsonl \
    --validation_file ./data/processed/dev.jsonl \
    --test_file ./data/processed/test.jsonl \
    --max_source_length 256 \
    --output_dir ./outputs/ \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 8 \
    --logging_strategy steps \
    --logging_steps 25 \
    --save_strategy steps \
    --save_step 200 \
    --save_total_limit 5 \
    --load_best_model_at_end True \
    --report_to none