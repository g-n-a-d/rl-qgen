import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator, PartialState
import jsonlines
import argparse
from tqdm import tqdm

from data_utils import make_prompt


parser = argparse.ArgumentParser()
parser.add_argument("--test_filename", type=str, help="Test file")
parser.add_argument("--model_name_or_path", type=str, help="Model")
parser.add_argument("--torch_dtype", type=str, default="float32", help="Dtype")
parser.add_argument("--adapter_name_or_path", type=str, default=None, help="Adapter")
parser.add_argument("--token", type=str, default=None, help="Token")
parser.add_argument("--gen_batch_size", type=int, default=8, help="Evaluation batch size")
parser.add_argument("--output_filename", type=str, default="./output.jsonl", help="Ouput")
parser.add_argument("--max_seq_length", type=int, default=1024, help="Max seq length")
parser.add_argument("--min_new_tokens", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=32)
parser.add_argument("--do_sample", type=bool, default=False)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--top_p", type=float, default=1.0)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

distributed_state = PartialState()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token)
config = AutoConfig.from_pretrained(args.model_name_or_path, token=args.token)
if not config.is_encoder_decoder:
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=args.torch_dtype
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        token=args.token,
        torch_dtype=getattr(torch, args.torch_dtype),
        # quantization_config=nf4_config
    ).to(distributed_state.device)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
if args.adapter_name_or_path:
    model.load_adapter(args.adapter_name_or_path)

with jsonlines.open(args.test_filename, mode="r") as fr, jsonlines.open(args.output_filename, mode="w") as fw:
    text = []
    for line in fr:
        text.append(line)
    
    rougeL_pre, rougeL_rec, rougeL_f1 = [], [], []
    with distributed_state.split_between_processes(text) as text_:
        for i in tqdm(range(0, len(text_), args.gen_batch_size), desc ="Generating:"):
            inputs = [tokenizer.bos_token + make_prompt(text_[i + ii]["context"], text_[i + ii]["answer"], template=None) for ii in range(min(args.gen_batch_size, len(text_) - i))] 
            input_ids = tokenizer(inputs, max_length=args.max_seq_length, padding=True, truncation=True, return_tensors="pt").to(device)
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

            for ii in range(min(args.gen_batch_size, len(text_) - i)):
                line_ = {}
                line_["context"] = text_[i + ii]["context"]
                line_["answer"] = text_[i + ii]["answer"]
                line_["target"] = text_[i + ii]["question"]
                if not model.config.is_encoder_decoder:
                    pred = outputs[ii].split("### Câu hỏi: ")
                    if len(pred) == 2:
                        line_["pred"] = pred[1]
                        fw.write(line_)
                    else:
                        pass
                else:
                    line_["pred"] = outputs[ii]
                    fw.write(line_)