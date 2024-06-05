from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import jsonlines
import argparse

from data_utils import make_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--test_filename", type=str, help="Test file")
parser.add_argument("--model_name_or_path", type=str, help="Model")
parser.add_argument("--dtype", type=str, default="float32", help="DataType")
parser.add_argument("--num_processes", type=int, default=1, help="Number of processes")
parser.add_argument("--output_filename", type=str, default="./output.jsonl", help="Ouput")
parser.add_argument("--max_seq_length", type=int, default=1024, help="Max input length")
parser.add_argument("--template", type=str, default=None, help="Template")
parser.add_argument("--min_new_tokens", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=32)
parser.add_argument("--do_beam_search", type=bool, default=False)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=-1)
parser.add_argument("--top_p", type=float, default=1.0)
args = parser.parse_args()


llm = LLM(
    model=args.model_name_or_path,
    tensor_parallel_size=args.num_processes,
    dtype=args.dtype,
    max_model_len=args.max_seq_length + args.max_new_tokens
)

sampling_params = SamplingParams(
    min_tokens=args.min_new_tokens,
    max_tokens=args.max_new_tokens,
    use_beam_search=args.do_beam_search,
    best_of=args.num_beams,
    top_k=args.top_k,
    top_p=args.top_p,
    temperature=args.temperature,
)

with jsonlines.open(args.test_filename, mode="r") as fr:
    text = []
    for line in fr:
        text.append(line)
text_ = ["<s>" + make_prompt(l["context"], l["answer"], template=args.template) for l in text]

results = list(
    map(lambda x: x.outputs[0].text, llm.generate(
        text_,
        sampling_params
    ))
)

with jsonlines.open(args.output_filename, mode="w") as fw:
    for i in range(len(text)):
        line_ = {}
        line_["context"] = text[i]["context"]
        line_["answer"] = text[i]["answer"]
        line_["target"] = text[i]["question"]
        line_["pred"] = results[i].strip()
        fw.write(line_)