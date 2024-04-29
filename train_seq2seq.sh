#!/bin/bash

accelerate launch \
    --config_file=config/multi_gpu.yaml \
    trainer/seq2seq.py \
    --model_name_or_path VietAI/vit5-base \
    --lang vietnamese \
    --context_column context \
    --question_column question \
    --answer_column answer \
    --train_file ./data/processed/train.jsonl \
    --validation_file ./data/processed/dev.jsonl \
    --test_file ./data/processed/dev.jsonl \
    --output_dir ./output/ \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 25 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 3e-4 \
    --num_train_epochs 5 \
    --logging_strategy steps \
    --logging_steps 25 \
    --save_strategy steps \
    --save_step 25 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --predict_with_generate True