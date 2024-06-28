#!/bin/bash

accelerate launch \
    --config_file ./config/multi_gpu.yaml \
    ./trainer/seq2seq.py \
    --model_name_or_path VietAI/vit5-large \
    --torch_dtype bfloat16 \
    --lang vietnamese \
    --context_column context \
    --question_column question \
    --answer_column answer \
    --train_file ./data/processed/train.jsonl \
    --validation_file ./data/processed/dev.jsonl \
    --test_file ./data/processed/test.jsonl \
    --max_source_length 1024 \
    --max_target_length 128 \
    --output_dir ./outputs/ \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_step 100 \
    --save_total_limit 5 \
    --load_best_model_at_end True \
    --bf16 \
    --report_to none \
    --predict_with_generate True