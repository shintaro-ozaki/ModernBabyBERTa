from pathlib import Path
from datasets import load_dataset
import os
import math
import torch
import argparse
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from modernbabyberta.modeling_modernbabyberta import ModernBabyBERTaForMaskedLM
from modernbabyberta.configuration_modernbabyberta import ModernBabyBERTaConfig
import logging


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train ModernBabyBERTa on BabyLM corpus.")
    parser.add_argument("--corpus_dir", type=str, required=True, help="Directory containing multiple .train files")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to pretrained tokenizer")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    corpus_dir = Path(args.corpus_dir)
    train_files = sorted(corpus_dir.glob("*.train"))
    if not train_files:
        raise FileNotFoundError(f"No .train files found in {corpus_dir}")

    for f in train_files:
        logger.info(f"  - {f.name}")

    raw_dataset = load_dataset(
        "text",
        data_files={"train": [str(f) for f in train_files]},
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length,
        )

    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )
    config = ModernBabyBERTaConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1152,
        glu_expansion=512,
        max_position_embeddings=args.max_seq_length,
    )
    model = ModernBabyBERTaForMaskedLM(config)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
    )

    steps_per_epoch = math.ceil(len(tokenized_datasets["train"]) / args.batch_size)
    warmup_steps = int(steps_per_epoch * args.epochs * args.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
