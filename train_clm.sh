#!/bin/bash

accelerate launch \
    --config_file ./config/multi_gpu.yaml \
    ./trainer/clm.py \
    --model_name_or_path Viet-Mistral/Vistral-7B-Chat \
    --torch_dtype bfloat16 \
    --use_peft True\
    --lora_r 32 \
    --lora_alpha 64 \
    --load_in_4bit True \
    --use_bnb_nested_quant True \
    --context_column context \
    --question_column question \
    --answer_column answer \
    --train_file ./data/processed/train.jsonl \
    --validation_file ./data/processed/dev.jsonl \
    --max_source_length 1024 \
    --chat_template vistral \
    --response_mark "[/INST]" \
    --output_dir ./outputs/ \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_step 50 \
    --save_total_limit 17 \
    --load_best_model_at_end True \
    --bf16 \
    --report_to none