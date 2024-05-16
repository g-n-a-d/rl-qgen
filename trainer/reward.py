import torch
import argparse
import os
import transformers
import numpy as np
from tqdm import tqdm
from time import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from huggingface_hub import list_repo_refs
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from accelerate import Accelerator
from datasets import load_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="reciprocate/gpt2-tiny", type=str)
parser.add_argument("--dataset_name", default=None, type=str)
parser.add_argument("--token", default=None, type=str)
parser.add_argument("--train_file", default=None, type=str)
parser.add_argument("--validation_file", default=None, type=str)
parser.add_argument("--lr", default=6e-4, type=float)
parser.add_argument("--min_lr", default=None, type=float)
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
parser.add_argument("--seq_length", default=512, type=int)
parser.add_argument("--checkpoint_dir", default="checkpoints", type=str)
parser.add_argument("--eval_interval", default=100, type=int)
parser.add_argument("--only_eval", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":
    seed = int(os.environ.get("RANK", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model_name = f"{args.model_path}"

    accelerator = Accelerator(log_with=None, gradient_accumulation_steps=args.gradient_accumulation_steps)
    accelerator.init_trackers(
        project_name="reward_model_training",
        config=vars(args),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, truncation_side="left")

    def tokenize(prompt, selected, rejected, tokenizer):
        return {
            "selected_input_ids": tokenizer(prompt + " ### " + selected + tokenizer.eos_token, truncation=True, max_length=args.seq_length).input_ids,
            "rejected_input_ids": tokenizer(prompt + " ### " + rejected + tokenizer.eos_token, truncation=True, max_length=args.seq_length).input_ids,
        }

    def collate_fn(batch):
        input_ids = sum([[x["rejected_input_ids"], x["selected_input_ids"]] for x in batch], [])
        return tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")


    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            token=args.token,
        )
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        extension = "json" if extension == "jsonl" else extension
        dataset = load_dataset(
            extension,
            data_files=data_files,
            token=args.token,
        )

    if "chosen" in dataset["train"].column_names:
        dataset = dataset.rename_column("chosen", "selected")
    if "replies" in dataset["train"].column_names:
        dataset = dataset.map(lambda x: {"selected": x["replies"][0], "rejected": x["replies"][1]}, remove_columns=["replies"])
    accelerator.print(args.dataset_name, dataset)

    eval_dataloaders = []
    tokenized = dataset.map(tokenize, input_columns=["prompt", "selected", "rejected"], fn_kwargs=dict(tokenizer=tokenizer), desc="Tokenizing")
    dataloader = torch.utils.data.DataLoader(tokenized["train"], shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)
    eval_dataloaders.append(torch.utils.data.DataLoader(tokenized["validation"], shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn))


    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))


    if args.only_eval:
        model, *eval_dataloaders = accelerator.prepare(model, *eval_dataloaders)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-08, weight_decay=args.weight_decay)

        scheduler = CosineAnnealingLR(opt, T_max=len(dataloader) * args.epochs, eta_min=args.min_lr or args.lr)
        model, opt, scheduler, dataloader, *eval_dataloaders, = accelerator.prepare(model, opt, scheduler, dataloader, *eval_dataloaders)

    best_accuracy = 0
    step = 0

    tbar = tqdm(range(args.epochs * len(dataloader)), disable=not accelerator.is_main_process or args.only_eval)
    for iepoch in range(args.epochs):
        for batch in dataloader:
            if (step+1) % args.gradient_accumulation_steps == 0:
                if ((step + 1) // args.gradient_accumulation_steps) % args.eval_interval == 0 or step == tbar.total - 1:
                    for dataset_name, eval_dataloader in zip([args.dataset_name], eval_dataloaders):
                        model.eval()
                        all_scores, all_delta_scores, all_tokens = [], [], []

                        for batch in tqdm(eval_dataloader, desc=f"Evaluating on {dataset_name}", disable=not accelerator.is_main_process, leave=args.only_eval):
                            with torch.no_grad():
                                scores = model(**batch)[0]

                            delta_scores = scores.reshape(-1, 2).diff().view(-1)
                            delta_scores = accelerator.gather_for_metrics(delta_scores)
                            all_delta_scores.extend(delta_scores.tolist())
                            all_scores.extend(scores.view(-1).tolist())
                            all_tokens.extend(batch["input_ids"].tolist())

                        delta_scores = np.hstack(all_delta_scores)
                        accuracy = (delta_scores > 0).mean()

                        if accelerator.is_main_process:
                            texts = [text.replace(tokenizer.pad_token, "") for text in tokenizer.batch_decode(all_tokens)]

                            postfix = "" if dataset_name == args.dataset_name else f"@{dataset_name.split('/')[-1]}"
                            accelerator.log({
                                f"accuracy{postfix}": accuracy,
                                f"delta_scores{postfix}": delta_scores,
                            }, step=(step+1) // args.gradient_accumulation_steps)
                            print({
                                f"accuracy{postfix}": accuracy,
                                f"delta_scores{postfix}": delta_scores,
                            })

                        if accuracy > best_accuracy and dataset_name == args.dataset_name:
                            best_accuracy = accuracy
                            accelerator.log({"best_accuracy": best_accuracy}, step=(step+1) // args.gradient_accumulation_steps)

                            if args.only_eval:
                                exit()
                            else:
                                path = f"{model_name}_{args.dataset_name}_{args.lr}".replace("/", "_").replace(":", "_").replace("@", "_")
                                accelerator.unwrap_model(model).save_pretrained(
                                    os.path.join(args.checkpoint_dir, path),
                                    save_function=accelerator.save,
                                    is_main_process=accelerator.is_main_process,
                                    state_dict=accelerator.get_state_dict(model),
                                )
                                if accelerator.is_main_process:
                                    tokenizer.save_pretrained(os.path.join(args.checkpoint_dir, path))
                                accelerator.print(f"Checkpointing -> {os.path.join(args.checkpoint_dir, path)}")

                        if dataset_name == args.dataset_name:
                            tbar.set_postfix(accuracy=accuracy, best_accuracy=best_accuracy)

                    accelerator.wait_for_everyone()
                    model.train()

            with accelerator.accumulate(model):
                scores = model(**batch)[0]
                loss = -F.logsigmoid(scores.reshape(-1, 2).diff()).mean()
                accelerator.backward(loss)
                opt.step()
                opt.zero_grad()
                scheduler.step()

            tbar.update()
            tbar.set_description(f"Training {args.model_path} on {args.dataset_name}; loss: {loss.item():.4f}")
            if (step+1) % args.gradient_accumulation_steps == 0:
                accelerator.log({"loss": loss.item(), "lr": float(scheduler.get_last_lr()[0])}, step=(step+1) % args.gradient_accumulation_steps)
            step += 1