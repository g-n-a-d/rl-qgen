import os
import sys
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], os.pardir)))

import gradio as gr
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, HfArgumentParser
import argparse

from trainer.arguments import ModelArguments
from data_utils import make_prompt

def load_args():
    parser = HfArgumentParser(ModelArguments)
    model_args, = parser.parse_args_into_dataclasses()
    
    return model_args

def load_model(model_args):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, token=model_args.token)  
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if not config.is_encoder_decoder:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    if model_args.adapter_name_or_path:
        model.load_adapter(model_args.adapter_name_or_path)

    return model, tokenizer

def infer(model, tokenizer, context, answer):
    prompt = make_prompt(context=context, answer=answer, last_space=False)
    inputs = tokenizer(prompt, return_tensors="pt")
    preds = model.generate(**inputs, do_sample=True, min_new_tokens=1, max_new_tokens=32)
    if not model.config.is_encoder_decoder:
        output = tokenizer.decode(preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).split("### Câu hỏi: ")[1]
    else:
        output = tokenizer.decode(preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return output

def run_demo():
    model_args = load_args()

    model, tokenizer = load_model(model_args)
    pipe = lambda context, answer: infer(
        model=model,
        tokenizer=tokenizer,
        context=context,
        answer=answer,
    )

    demo = gr.Interface(
        fn=pipe,
        inputs=[
            gr.Textbox(
                value="Việc phát hiện Di cốt Mauer cho thấy rằng người cổ đại hiện diện tại Đức từ ít nhất là 600.000 năm trước. Các vũ khí săn bắn hoàn thiện cổ nhất được phát hiện trên thế giới nằm trong một mỏ than tại Schöningen, tại đó khai quật được ba chiếc lao bằng gỗ có niên đại 380.000 năm. Thung lũng Neandertal là địa điểm phát hiện di cốt người phi hiện đại đầu tiên từng biết đến; loài người mới này được gọi là Neanderthal. Các hóa thạch Neanderthal 1 được cho là có niên đại 40.000 năm tuổi. Bằng chứng về người hiện đại có niên đại tương tự được phát hiện trong các hang tại Dãy Schwäbische Alb gần Ulm. Trong những vật được tìm thấy có các sáo bằng xương chim và ngà voi ma mút 42.000 năm tuổi- là các nhạc cụ cổ nhất từng phát hiện được, Tượng người sư tử thời đại băng hà 40.000 năm tuổi là nghệ thuật tạo hình không thể tranh luận cổ nhất từng phát hiện được, và Tượng Venus ở Hohle Fels 35.000 năm tuổi là nghệ thuật tạo hình con người không thể tranh luận cổ nhất từng phát hiện được. Đĩa bầu trời Nebra là một đồ tạo tác bằng đồng điếu được tạo ra trong thời đại đồ đồng châu Âu được quy cho một địa điểm gần Nebra, Sachsen-Anhalt. Nó nằm trong Chương trình Ký ức Thế giới của UNESCO.",
                label="Context",
                lines=8
            ),
            gr.Textbox(
                value="Tượng người sư tử thời đại băng hà",
                label="Answer",
                lines=2
            ),
        ],
        outputs=gr.Textbox(label="Question", lines=3)
    )
    demo.launch(share=True)
    
if __name__ == "__main__":
    run_demo()