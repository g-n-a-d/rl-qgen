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
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    HfArgumentParser,
)

from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from arguments import ModelArguments, DataTrainingArguments, GenerationArguments
from utils.data_utils import make_prompt, make_reward_input


logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments pertaining to which reward_model/peft we are going to fine-tune from.
    """

    reward_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained reward model or identifier from huggingface.co/models"}
    )

    max_reward_input_length: Optional[int] = field(
        default=512, metadata={"help": "Maximum length for reward model inputs"}
    )

    output_dir: str = field(
        default="./outputs", metadata={"help": "Path to save outputs"}
    )

    saving_step: Optional[int] = field(
        default=10, metadata={"help": "Model is saved every _ steps"}
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

datasets.utils.logging.set_verbosity(logging.INFO)
transformers.utils.logging.set_verbosity(logging.INFO)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

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


#################
# Model
#################
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
)
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
)
model = trl_model_class.from_pretrained(
    model_args.model_name_or_path,
    config=config,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)

tokenizer_reward = AutoTokenizer.from_pretrained(script_args.reward_model_name_or_path)
model_reward = AutoModelForSequenceClassification.from_pretrained(script_args.reward_model_name_or_path)

def get_reward(inputs):
    input_ids = tokenizer_reward(inputs, padding=True, truncation=True, max_length=script_args.max_reward_input_length, return_tensors="pt")
    scores = model_reward(**input_ids).logits
    return [torch.tensor(score.item()) for score in scores]


#################
# Data
#################
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
column_names = raw_ds.column_names


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


# One should customize this function to train the model on its own dataset.
prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

def preprocess_function(examples):
    inputs = []
    for i in range(len(examples[context_column])):
        if examples[context_column][i] and examples[answer_column][i] and examples[question_column][i]:
            inputs.append(make_prompt(examples[context_column][i], examples[answer_column][i]))

    padding = "max_length" if data_args.pad_to_max_length else False

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    return model_inputs

dataset = raw_ds.map(
    preprocess_function,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
    load_from_cache_file=not data_args.overwrite_cache,
    desc="Running tokenizer on train dataset",
)
dataset.set_format("torch")

def collator(data):

    return {key: [d[key] for d in data] for key in data[0]}


#################
# Trainer
#################
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)


for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from model
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors,
        return_prompt=False,
        generate_ref_response=True,
        **gen_args.to_dict(),
    )
    batch["query"] = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors, skip_special_tokens=True)

    # Compute reward
    texts = [make_reward_input(c, a, q) for c, a, q in zip(batch["context"], batch["answer"], batch["response"])]
    rewards = get_reward(texts)
    batch["rewards"] = rewards
    ref_texts = [make_reward_input(c, a, q) for c, a, q in zip(batch["context"], batch["answer"], batch["ref_response"])]
    ref_rewards = get_reward(ref_texts)
    batch["ref_rewards"] = ref_rewards

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    # Logging
    logger.info("Step: {}".format(_epoch))
    print("Step: {}".format(_epoch + 1))
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
    print("Training stats: \n{}".format(json.dumps(filtered_stats, indent=4)))
    logger.info("Batch stats: {}".format(json.dumps(
        list(zip(batch["context"], batch["answer"], batch["response"], batch["ref_response"])),
        ensure_ascii=False,
        indent=4
    )))
    print("Batch stats: {}".format(json.dumps(
        list(zip(batch["context"], batch["answer"], batch["response"], batch["ref_response"])),
        ensure_ascii=False,
        indent=4
    )))

    # Saving
    if (_epoch + 1) % script_args.saving_step == 0:
        logger.info("Saving model and stats...")
        print("Saving model and stats...")
        ppo_trainer._save_pretrained(os.path.join(script_args.output_dir, "checkpoint_{}".format(_epoch + 1)))