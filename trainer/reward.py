import logging
import os
import sys
sys.path.insert(1, os.path.abspath(os.path.join(sys.path[0], os.pardir)))
import warnings
from tqdm import tqdm

import numpy as np
import torch
import datasets
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from trl import RewardConfig, RewardTrainer
from trl.trainer.utils import RewardDataCollatorWithPadding

from arguments import ModelArguments, DataTrainingArguments
from utils.data_utils import make_reward_input


if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelArguments, DataTrainingArguments))
    reward_config, model_args, data_args = parser.parse_args_into_dataclasses()
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if reward_config.should_log:
        # The default of reward_config.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = reward_config.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process:
    logger.warning(
        f"Process rank: {reward_config.local_rank}, device: {reward_config.device}, n_gpu: {reward_config.n_gpu}, "
        + f"distributed training: {reward_config.parallel_mode.value == 'distributed'}, 16-bits training: {reward_config.fp16}"
    )
    logger.info(f"Training/evaluation parameters {reward_config}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(reward_config.output_dir) and reward_config.do_train and not reward_config.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(reward_config.output_dir)
        if last_checkpoint is None and len(os.listdir(reward_config.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({reward_config.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and reward_config.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(reward_config.seed)


    ################
    # Model & Tokenizer
    ################
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=1,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        truncation_side="left",
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )


    ################
    # Dataset
    ################
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
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

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if reward_config.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif reward_config.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train` and/or `do_eval`.")
        raise ValueError("Please pass `do_train` and/or `do_eval`.")
    
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
        
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs_chosen, inputs_rejected = [], []
        for i in range(len(examples[context_column])):
            if examples[context_column][i] and examples[answer_column][i] and examples[question_column][i]:
                inputs_chosen.append(make_reward_input(examples[context_column][i], examples[answer_column][i], examples[question_column][i][0]))
                inputs_rejected.append(make_reward_input(examples[context_column][i], examples[answer_column][i], examples[question_column][i][1]))

        inputs_chosen_ids = tokenizer(inputs_chosen, max_length=data_args.max_source_length, padding=padding, truncation=True)
        inputs_rejected_ids = tokenizer(inputs_rejected, max_length=data_args.max_source_length, padding=padding, truncation=True)

        model_inputs = {
            "input_ids_chosen" : inputs_chosen_ids["input_ids"],
            "attention_mask_chosen" : inputs_chosen_ids["attention_mask"],
            "input_ids_rejected" : inputs_rejected_ids["input_ids"],
            "attention_mask_rejected" : inputs_rejected_ids["attention_mask"]
        }

        return model_inputs

    if reward_config.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with reward_config.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if reward_config.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with reward_config.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    # Data collator
    collator = RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_source_length)


    ################
    # Training
    ################
    class CustomTrainer(RewardTrainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
        ):
            if not self.use_reward_data_collator:
                warnings.warn(
                    "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                    " if you are using a custom data collator make sure you know what you are doing or"
                    " implement your own compute_loss method."
                )
            rewards_chosen = model(
                input_ids=inputs["input_ids_chosen"],
                attention_mask=inputs["attention_mask_chosen"],
                return_dict=True,
            )["logits"]
            rewards_rejected = model(
                input_ids=inputs["input_ids_rejected"],
                attention_mask=inputs["attention_mask_rejected"],
                return_dict=True,
            )["logits"]
            # calculate loss, optionally modulate with margin
            loss = -torch.sum(rewards_chosen - rewards_rejected)

            if return_outputs:
                return loss, {
                    "rewards_chosen": rewards_chosen,
                    "rewards_rejected": rewards_rejected,
                }
            return loss


    # Keep unused columns not removed.
    reward_config.remove_unused_columns=False
    
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset if reward_config.do_train else None,
        eval_dataset=eval_dataset if reward_config.do_eval else None,
        data_collator=collator,
    )


    # for b in trainer.get_train_dataloader():
    #     l = trainer.compute_loss(model, b)
    #     print(l)
    #     trainer.accelerator.backward(l)
    #     break

    trainer.train()

    # if reward_config.do_train:
    #     checkpoint = None
    #     if reward_config.resume_from_checkpoint is not None:
    #         checkpoint = reward_config.resume_from_checkpoint
    #     elif last_checkpoint is not None:
    #         checkpoint = last_checkpoint
    #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #     trainer.save_model()  # Saves the tokenizer too for easy upload

    #     metrics = train_result.metrics
    #     max_train_samples = (
    #         data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    #     )
    #     metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    #     trainer.log_metrics("train", metrics)
    #     trainer.save_metrics("train", metrics)
    #     trainer.save_state()

    # # Evaluation
    # if reward_config.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate(eval_dataset=eval_dataset)
    #     max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #     metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)