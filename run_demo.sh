#!/bin/bash

if [ $# -eq 1 ];
then
    python ./utils/gradio_demo.py \
        --model_name_or_path gnad/qgen-vit5-base \
        --token ${1}
else
    echo "Invalid number of arguments"
fi