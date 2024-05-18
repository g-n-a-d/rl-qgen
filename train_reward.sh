#!/bin/bash

accelerate launch \
    --config_file ./config/multi_gpu.yaml \
    ./trainer/reward.py \
    --model_path FacebookAI/xlm-roberta-base \
    --train_file ./data/processed/trainaug_reward.jsonl \
    --validation_file ./data/processed/dev_reward.jsonl \
    --lr 1e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --epochs 5 \
    --seq_length 512 \
    --checkpoint_dir checkpoints \
    --eval_interval 80