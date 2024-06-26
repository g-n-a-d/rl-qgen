import logging
import os
import sys
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], os.pardir)))

from rich.console import Console
from rich.logging import RichHandler

import numpy as np
import torch

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

import datasets
from datasets import load_dataset

from trl import DataCollatorForCompletionOnlyLM, get_peft_config, get_quantization_config
from trl.commands.cli_utils import init_zero_verbose

from peft import get_peft_model, LoraConfig, TaskType

from trainer.arguments import ModelArguments, DataTrainingArguments
from utils.data_utils import make_prompt


init_zero_verbose()
FORMAT = "%(message)s"
logger = logging.getLogger(__name__)
logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)
console = Console()


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


    #################
    # Model & Tokenizer
    #################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    quantization_config = get_quantization_config(model_args)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=getattr(torch, model_args.torch_dtype),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        quantization_config=quantization_config,
    )
    if model_args.use_peft:
        model = get_peft_model(model, get_peft_config(model_args))


    #################
    # Dataset
    #################
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        extension = "json" if extension == "jsonl" else extension
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    context_column = data_args.context_column
    question_column = data_args.question_column
    answer_column = data_args.answer_column
    column_names = (context_column, question_column, answer_column)

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs = []
        for i in range(len(examples[context_column])):
            if examples[context_column][i] and examples[answer_column][i] and examples[question_column][i]:
                inp = tokenizer.bos_token + \
                    make_prompt(examples[context_column][i], examples[answer_column][i], examples[question_column][i], template=data_args.chat_template) + \
                    tokenizer.eos_token
                inputs.append(inp)

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True, add_special_tokens=False)

        return model_inputs

    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    eval_dataset = raw_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=data_args.response_mark)


    #################
    # Training
    #################
    with console.status("[bold green]Initializing the SFTTrainer..."):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if data_args.train_file else None,
            eval_dataset=eval_dataset if data_args.validation_file else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Training
    logger.info("*** Training ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.create_model_card()


if __name__ == "__main__":
    main()