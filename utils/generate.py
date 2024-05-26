import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import jsonlines
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--test_filename", type=str, help="Test file")
parser.add_argument("--model_name_or_path", type=str, help="Model")
parser.add_argument("--adapter_name_or_path", type=str, default=None, help="Adapter")
parser.add_argument("--gen_batch_size", type=int, default=8, help="Evaluation batch size")
parser.add_argument("--output_filename", type=str, default="./output.jsonl", help="Ouput")
parser.add_argument("--min_new_tokens", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=32)
parser.add_argument("--do_sample", type=bool, default=False)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--top_p", type=float, default=1.0)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
if args.adapter_name_or_path:
    adapter = model.load_adapter(args.adapter_name_or_path)

def make_prompt(context, answer):
    instruction = "### Hãy tạo Câu hỏi dựa trên Đoạn văn và Câu trả lời sau:"
    prompt = instruction + "\n### Đoạn văn: " + context + "\n### Câu trả lời: " + answer + "\n### Câu hỏi:"
    
    return prompt

with jsonlines.open(args.test_filename, mode="r") as fr, jsonlines.open(args.output_filename, mode="w") as fw:
    text = []
    for line in fr:
        text.append(line)
    
    rougeL_pre, rougeL_rec, rougeL_f1 = [], [], []
    for i in tqdm(range(0, len(text), args.gen_batch_size), desc ="Generating:"):
        inputs = [make_prompt(text[i + ii]["context"], text[i + ii]["answer"]) for ii in range(min(args.gen_batch_size, len(text) - i))] 
        input_ids = tokenizer(inputs, max_length=1024, padding=True, truncation=True, return_tensors="pt").to(device)
        preds = model.generate(
            **input_ids,
            min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        outputs = tokenizer.batch_decode(preds, skip_special_tokens=True)

        for ii in range(min(args.gen_batch_size, len(text) - i)):
            line_ = {}
            line_["context"] = text[i + ii]["context"]
            line_["answer"] = text[i + ii]["answer"]
            line_["target"] = text[i + ii]["question"]
            line_["pred"] = outputs[ii]
            fw.write(line_)