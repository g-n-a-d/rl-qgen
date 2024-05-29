#!/bin/bash

accelerate launch \
    --config_file ./config/multi_gpu.yaml \
    ./trainer/clm.py \
    --model_name_or_path bigscience/bloomz-1b1 \
    --use_peft True\
    --context_column context \
    --question_column question \
    --answer_column answer \
    --train_file ./data/processed/train.jsonl \
    --validation_file ./data/processed/dev.jsonl \
    --max_source_length 2048 \
    --output_dir ./outputs/ \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --fp16 \
    --fp16_full_eval \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_strategy steps \
    --logging_steps 25 \
    --save_strategy steps \
    --save_step 200 \
    --save_total_limit 5 \
    --load_best_model_at_end True \
    --report_to none