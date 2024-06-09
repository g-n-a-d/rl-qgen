#!/bin/bash

python ./utils/gradio_demo.py \
    --model_name_or_path gnad/viqgen-vistral-7b-chat-dpo-qlora \
    --torch_dtype bfloat16 \
    --dataset_name "" \
    --chat_template vistral \
    --response_mark "[/INST]" \
    --do_sample True \
    --top_p 0.8 \
    --temperature 0.8