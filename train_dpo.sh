#!/bin/bash

accelerate launch \
    --config_file ./config/multi_gpu.yaml \
    ./trainer/dpo.py \
    --model_name_or_path gnad/viqgen-vistral-7b-chat-sft-qlora \
    --torch_dtype bfloat16 \
    --use_peft True \
    --lora_r 32 \
    --lora_alpha 32 \
    --load_in_4bit True \
    --use_bnb_nested_quant True \
    --train_file ./data/pairs_train.jsonl \
    --validation_file ./data/pairs_eval.jsonl \
    --output_dir ./outputs/ \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2.5e-6 \
    --num_train_epochs 2 \
    --lr_scheduler_type cosine \
    --warmup_steps 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_step 50 \
    --save_total_limit 20 \
    --load_best_model_at_end True \
    --bf16 \
    --report_to none \
    --no_remove_unused_columns \
    --beta 0.1