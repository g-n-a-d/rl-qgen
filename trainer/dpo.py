import logging
import os
import sys
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], os.pardir)))
import multiprocessing
from contextlib import nullcontext

from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

import torch
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.commands.cli_utils import init_zero_verbose, TrlParser

from peft import LoraConfig, get_peft_model, PeftModel

from trainer.arguments import ModelArguments
from utils.data_utils import make_prompt


init_zero_verbose()
FORMAT = "%(message)s"
logger = logging.getLogger(__name__)
logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)
console = Console()


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )


def main():
    parser = TrlParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse_args_and_config()

    # training_args.disable_tqdm = True


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
    model_ref = None


    #################
    # Optional rich context managers
    #################
    init_context = console.status("[bold green]Initializing the DPOTrainer...")
    save_context = console.status(f"[bold green]Saving the checkpoint to {training_args.output_dir}")


    #################
    # Dataset
    #################
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
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
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    if "train" in dataset.column_names:
        train_dataset = dataset["train"]
    if "validation" in dataset.column_names:
        eval_dataset = dataset["validation"]


    #################
    # Training
    #################
    with init_context:
        trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            train_dataset=train_dataset if data_args.train_file else None,
            eval_dataset=eval_dataset if data_args.validation_file else None,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_args) if model_args.use_peft else None,
            # callbacks=[RichProgressCallback],
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

    with save_context:
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    trainer.create_model_card()
    

if __name__ == "__main__":
    main()