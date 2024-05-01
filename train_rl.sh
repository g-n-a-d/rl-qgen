#!/bin/bash

accelerate launch \
    --config_file examples/accelerate_configs/multi_gpu.yaml \
    --num_processes {NUM_GPUS} \
    trainer/rl.py \
    --all_arguments_of_the_script