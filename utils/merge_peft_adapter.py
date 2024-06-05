import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--adapter_name_or_path", type=str)
    parser.add_argument("--output_dir", type=str, defaul=None)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--push_to_hub", action="store_true")

    return parser.parse_args()

def main():
    args = get_args()
    
    if not args.output_dir:
        args.output_dir = args.adapter_name_or_path


    print(f"Loading base model: {args.model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        return_dict=True,
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto"
    )

    print(f"Loading PEFT: {args.adapter_name_or_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_name_or_path, device_map="auto")
    print(f"Running merge_and_unload")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.push_to_hub:
        print(f"Saving to hub ...")
        model.push_to_hub(f"{args.output_dir}", use_temp_dir=False)
        tokenizer.push_to_hub(f"{args.output_dir}", use_temp_dir=False)
    else:
        model.save_pretrained(f"{args.output_dir}")
        tokenizer.save_pretrained(f"{args.output_dir}")
        print(f"Model saved to {args.output_dir}")

if __name__ == "__main__" :
    main()