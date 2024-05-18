#!/bin/bash

accelerate launch \
    --config_file ./config/multi_gpu.yaml \
    ./trainer/ppo.py \
    --reward_model_name_or_path ... \
    --output_dir ./outputs/ \
    --saving_step 10 \
    --model_name_or_path ./model/base_seq2seq \
    --context_column context \
    --question_column question \
    --answer_column answer \
    --train_file ./data/processed/train.jsonl \
    --max_source_length 512 \
    --do_sample True \
    --top_p 0.8 \
    --learning_rate 1e-5 \
    --init_kl_coef 0.2 \
    --vf_coef 0.5 \
    --batch_size 128 \
    --mini_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --ppo_epochs 5 \