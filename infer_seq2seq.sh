#!/bin/bash

accelerate launch \
    --config_file ./config/multi_gpu.yaml \
    ./trainer/seq2seq.py \
    --model_name_or_path ./model/base_seq2seq \
    --lang vietnamese \
    --context_column context \
    --question_column question \
    --answer_column answer \
    --test_file test.jsonl \
    --output_dir ./results/ \
    --do_predict \
    --report_to none \
    --predict_with_generate True