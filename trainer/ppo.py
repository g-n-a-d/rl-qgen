import logging
import os
import sys
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], os.pardir)))
import json
from tqdm import tqdm

from dataclasses import dataclass, field
from typing import Optional

import datasets

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
import transformers
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from arguments import ModelArguments, DataTrainingArguments, GenerationArguments
from utils.data_utils import make_prompt


tqdm.pandas()


logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments pertaining to which reward_model/peft we are going to fine-tune from.
    """

    output_dir: str = field(
        default="./outputs", metadata={"help": "Path to save outputs"}
    )

    saving_step: Optional[int] = field(
        default=40, metadata={"help": "Model is saved every _ steps"}
    )

    reward_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained reward model or identifier from huggingface.co/models"}
    )

    # LoraConfig
    use_peft: bool = field(
        default=False, metadata={"help": "whether to use peft"}
    )

    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )

    lora_r: Optional[int] = field(
        default=16, metadata={"help": "the lora r parameter"}
    )


parser = HfArgumentParser((ScriptArguments, ModelArguments, DataTrainingArguments, GenerationArguments, PPOConfig))
script_args, model_args, data_args, gen_args, ppo_config = parser.parse_args_into_dataclasses()


# Pass tracking parameters
ppo_config.task_name = "rl-finetuning"
ppo_config.model_name = model_args.model_name_or_path
ppo_config.query_dataset = "ViQuAD"
ppo_config.reward_model = "?"


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)

# Log on each process:
logger.info(f"Training parameters {ppo_config}")

if data_args.source_prefix is None:
    logger.warning(
        "You didn't provide a source prefix, which is the expected, e.g. with "
        "`--source_prefix 'generate: ' `"
    )


# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

trl_model_class = AutoModelForSeq2SeqLMWithValueHead


# download the raw dataset.
if data_args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    raw_ds = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
else:
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    extension = "json" if extension == "jsonl" else extension
    raw_ds = load_dataset(
        extension,
        data_files=data_files,
        split="train",
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
column_names = raw_ds["train"].column_names


# Get the column names for input/target.
dataset_columns = ("context", "question", "answer")
if data_args.context_column is None:
    context_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
else:
    context_column = data_args.context_column
    if context_column not in column_names:
        raise ValueError(
            f"--context_column' value '{data_args.context_column}' needs to be one of: {', '.join(column_names)}"
        )
if data_args.question_column is None:
    question_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
else:
    question_column = data_args.question_column
    if question_column not in column_names:
        raise ValueError(
            f"--question_column' value '{data_args.question_column}' needs to be one of: {', '.join(column_names)}"
        )
if data_args.answer_column is None:
    answer_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
else:
    answer_column = data_args.answer_column
    if answer_column not in column_names:
        raise ValueError(
            f"--answer_column' value '{data_args.answer_column}' needs to be one of: {', '.join(column_names)}"
        )


# set seed before initializing value head for deterministic eval
set_seed(ppo_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
if not script_args.use_peft:
    ref_model = trl_model_class.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code)
    device_map = None
    peft_config = None
else:
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}


tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
)
model = trl_model_class.from_pretrained(
    model_args.model_name_or_path,
    trust_remote_code=model_args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)


# One should customize this function to train the model on its own dataset.
def build_dataset(data_args):
    """
    Build dataset for training. One should customize this function
    to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs = []
        for i in range(len(examples[context_column])):
            if examples[context_column][i] and examples[answer_column][i] and examples[question_column][i]:
                inputs.append(make_prompt(examples[context_column][i], examples[answer_column][i]))

        padding = "max_length" if data_args.pad_to_max_length else False

        inputs = [data_args.prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        return model_inputs

    ds = raw_ds.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )

    return ds

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(data_args)

def collator(data):

    return {key: [d[key] for d in data] for key in data[0]}


# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
reward_task, reward_model_name = script_args.reward_task, script_args.reward_model_name_or_path
if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        sentiment_pipe = pipeline(model=reward_model_name, device=device)
else:
    sentiment_pipe = pipeline(model=reward_model_name, device=device)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id


for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from model
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors,
        return_prompt=False,
        generate_ref_response=True,
        **gen_args.to_dict(),
        # min_new_tokens = gen_args.min_new_tokens,
        # max_new_tokens = gen_args.max_new_tokens,
        # do_sample = gen_args.do_sample,
        # temperature = gen_args.temperature,
        # top_k = gen_args.top_k,
        # top_p = gen_args.top_p,
    )
    batch["query"] = tokenizer.batch_decode(query_tensors)
    batch["response"] = tokenizer.batch_decode(response_tensors)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

    # Compute reward
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    batch["rewards"] = rewards
    ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
    ref_pipe_outputs = sentiment_pipe(ref_texts, **sent_kwargs)
    ref_rewards = [torch.tensor(output[1]["score"]) for output in ref_pipe_outputs]
    batch["ref_rewards"] = ref_rewards

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    # Logging
    logger.info("Step: {}".format())
    filtered_stats = {key : stats[key] for key in stats.keys() if key in [
        "objective/kl",
        "objective/kl_coef",
        "objective/entropy",
        "ppo/mean_non_score_reward",
        "ppo/mean_scores",
        "ppo/std_scores",
        "tokens/queries_len_mean",
        "tokens/queries_len_std",
        "tokens/responses_len_mean",
        "tokens/responses_len_std",
        "ppo/loss/policy",
        "ppo/loss/value",
        "ppo/loss/total",
        "ppo/policy/entropy",
        "ppo/policy/approxkl",
        "ppo/policy/policykl",
        "ppo/policy/clipfrac",
        "ppo/policy/advantages_mean",
        "ppo/returns/mean",
        "ppo/returns/var",
        "ppo/val/vpred",
        "ppo/val/error",
        "ppo/val/clipfrac",
        "ppo/val/mean",
        "ppo/val/var",
        "ppo/val/var_explained",
        "ppo/learning_rate",
        "time/ppo/forward_pass",
        "time/ppo/compute_rewards",
        "time/ppo/compute_advantages",
        "time/ppo/optimize_step",
        "time/ppo/calc_stats",
        "time/ppo/total",
    ]}
    logger.info("Training stats: \n{}".format(json.dumps(filtered_stats, indent=4)))
    logger.info("Batch stats: {}".format(json.dumps(batch, indent=4)))

    # Saving
    if _epoch % script_args.saving_step == 0:
        logger.info("Saving model and stats...")
        ppo_trainer._save_pretrained(os.path.join(script_args.output_dir, "checkpoint_{}".format(_epoch)))
        with open(os.path.join(script_args.output_dir, "checkpoint_{}/stats.json".format(_epoch)), "r") as fw:
            json.dump(filtered_stats, fw)