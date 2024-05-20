import logging
import os
import sys
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], os.pardir)))
import multiprocessing
from contextlib import nullcontext

from trl.commands.cli_utils import init_zero_verbose, TrlParser

init_zero_verbose()
FORMAT = "%(message)s"

from rich.console import Console
from rich.logging import RichHandler

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from dataclasses import dataclass, field
from typing import Optional

from arguments import ModelArguments
from utils.data_utils import make_prompt

logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


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


if __name__ == "__main__":
    parser = TrlParser((DPOConfig, ModelArguments, DataArguments))
    training_args, model_args, data_args = parser.parse_args_and_config()

    training_args.disable_tqdm = True
    console = Console()

    #################
    # Model & Tokenizer
    #################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model_ref = None


    #################
    # Optional rich context managers
    #################
    init_context = console.status("[bold green]Initializing the DPOTrainer...")
    save_context = console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")


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

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    #################
    # Training
    #################
    with init_context:
        trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=[RichProgressCallback],
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)