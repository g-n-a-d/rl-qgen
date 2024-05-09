import os
import sys
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], os.pardir)))

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, HfArgumentParser
import argparse

from trainer.arguments import ModelArguments
from data_utils import make_prompt

def load_args():
    parser = HfArgumentParser(ModelArguments)
    model_args, = parser.parse_args_into_dataclasses()
    
    return model_args

def load_model(model_name_or_path, token):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token)  
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, token=token)

    return model, tokenizer

def infer(model, tokenizer, context, answer, **gen_config):
    prompt = make_prompt(context=context, answer=answer)
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, do_sample=True, num_return_sequences=10, min_new_tokens=1, max_new_tokens=32, **gen_config)
    output_text = "\n".join([tokenizer.decode(output[i], skip_special_tokens=True, clean_up_tokenization_spaces=True) for i in range(10)])
    
    return output_text

def run_demo():
    model_args = load_args()

    model, tokenizer = load_model(model_name_or_path=model_args.model_name_or_path, token=model_args.token)
    pipe = lambda context, answer, top_p : infer(
        model=model,
        tokenizer=tokenizer,
        context=context,
        answer=answer,
        **{"top_p" : top_p}
    )

    demo = gr.Interface(
        fn=pipe,
        inputs=[
            gr.Textbox(
                value="Năm 1871, Đức trở thành một quốc gia dân tộc khi hầu hết các quốc gia Đức thống nhất trong Đế quốc Đức do Phổ chi phối. Sau Chiến tranh thế giới thứ nhất và Cách mạng Đức 1918-1919, Đế quốc này bị thay thế bằng Cộng hòa Weimar theo chế độ nghị viện. Chế độ độc tài quốc xã được hình thành vào năm 1933, dẫn tới Chiến tranh thế giới thứ hai và một nạn diệt chủng. Sau một giai đoạn Đồng Minh chiếm đóng, hai nước Đức được thành lập: Cộng hòa Liên bang Đức và Cộng hòa Dân chủ Đức. Năm 1990, quốc gia được tái thống nhất.",
                label="Context",
                lines=8
            ),
            gr.Textbox(
                value="Chiến tranh thế giới thứ nhất và Cách mạng Đức 1918-1919",
                label="Answer",
                lines=2
            ),
            gr.Slider(0.1, 1, step=0.1, label='Top-p', value=0.9)
        ],
        outputs=gr.Textbox(label="Question", lines=3)
    )
    demo.launch(share=True)
    
if __name__ == "__main__":
    run_demo()