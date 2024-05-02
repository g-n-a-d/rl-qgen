import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

from data_utils import make_prompt

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--token", type=str, help="Huggingface token to access model")
    agrs = parser.parse_args()
    
    return agrs

def load_model(model_name_or_path, token):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token)  
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, token=token)
    return model, tokenizer

def infer(model, tokenizer, context, answer):
    prompt = make_prompt(context=context, answer=answer)
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text

def run_demo():
    agrs = load_args()

    model, tokenizer = load_model(model_name_or_path=agrs.model_name_or_path, token=agrs.token)
    pipe = lambda context, answer : infer(model=model, tokenizer=tokenizer, context=context, answer=answer)
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
            )
        ],
        outputs=gr.Textbox(label="Question", lines=3)
    )
    demo.launch()
    
if __name__ == "__main__":
    run_demo()