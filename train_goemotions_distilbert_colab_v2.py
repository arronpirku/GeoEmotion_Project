#!/usr/bin/env python3
"""
Colab-safe GoEmotions baseline (DistilBERT) — v2
- Handles both list-based and wide one-hot schemas.
- Auto-creates a validation split if the dataset lacks one.
- Minimal TrainingArguments to sidestep version quirks.
"""

import argparse
import json
import os
import time
from typing import List

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, Sequence
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, classification_report


RAW_EMOTIONS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise","neutral"
]
SIMPLIFIED_EMOTIONS = ["anger","disgust","fear","joy","sadness","surprise","neutral"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--dataset_name", type=str, default="go_emotions")
    p.add_argument("--dataset_config", type=str, default="raw", choices=["raw","simplified"])
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--output_dir", type=str, default="./outputs_distilbert_goemotions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_threshold", type=float, default=0.5)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--val_frac", type=float, default=0.1, help="If no validation split, take this fraction from train.")
    return p.parse_args()


def ensure_validation(ds: DatasetDict, seed: int, val_frac: float) -> DatasetDict:
    if "validation" in ds:
        return ds
    # Create validation from train
    split = ds["train"].train_test_split(test_size=val_frac, seed=seed, stratify_by_column=None)
    return DatasetDict(train=split["train"], validation=split["test"], test=ds["test"] if "test" in ds else split["test"])


def detect_schema_and_label_cols(ds, config: str):
    cols = ds["train"].column_names
    if "labels" in cols:
        return "list", None
    expected = RAW_EMOTIONS if config == "raw" else SIMPLIFIED_EMOTIONS
    label_cols = [c for c in expected if c in cols]
    if label_cols:
        return "wide", label_cols
    raise KeyError(f"Could not detect labels. Columns found: {cols}")


def get_label_names(ds, schema: str, label_cols, config: str) -> List[str]:
    if schema == "list":
        feat = ds["train"].features["labels"]
        if isinstance(feat, Sequence):
            return feat.feature.names
        else:
            # Fallback: infer max label id
            max_id = 0
            for ex in ds["train"]["labels"][:1000]:
                if isinstance(ex, list):
                    max_id = max(max_id, max(ex) if ex else 0)
            # assume [0..max_id]
            names = [str(i) for i in range(max_id + 1)]
            return names
    else:
        return label_cols


def tokenize_examples(examples, tokenizer, max_length):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


def attach_labels(examples, schema: str, label_names: List[str], label_cols=None):
    n = len(examples["text"])
    y = np.zeros((n, len(label_names)), dtype=np.float32)
    if schema == "list":
        all_labels = examples["labels"]
        for i, lbls in enumerate(all_labels):
            for j in lbls:
                if 0 <= j < len(label_names):
                    y[i, j] = 1.0
    else:
        for idx, name in enumerate(label_names):
            vals = examples[name]
            y[:, idx] = np.array(vals, dtype=np.float32)
    return {"labels": y.tolist()}


def compute_metrics_builder(threshold: float, label_names: List[str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)
        micro = f1_score(labels, preds, average="micro", zero_division=0)
        macro = f1_score(labels, preds, average="macro", zero_division=0)
        weighted = f1_score(labels, preds, average="weighted", zero_division=0)
        rep = classification_report(labels, preds, target_names=label_names, zero_division=0, output_dict=True)
        with open(os.path.join(args.output_dir, "classification_report.json"), "w") as f:
            json.dump(rep, f, indent=2)
        return {"f1_micro": float(micro), "f1_macro": float(macro), "f1_weighted": float(weighted)}
    return compute_metrics


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_latency(model, tokenizer, device: str, max_length: int = 128, batch_size: int = 32, iters: int = 30):
    model.eval()
    sents = ["This is a sample sentence about feelings."] * batch_size
    with torch.no_grad():
        for _ in range(5):
            _ = model(**tokenizer(sents, return_tensors="pt", padding=True, truncation=True,
                                  max_length=max_length).to(device))
        start = time.time()
        for _ in range(iters):
            _ = model(**tokenizer(sents, return_tensors="pt", padding=True, truncation=True,
                                  max_length=max_length).to(device))
        end = time.time()
    return (end - start) * 1000.0 / iters


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = load_dataset(args.dataset_name, args.dataset_config)
    ds = ensure_validation(ds, seed=args.seed, val_frac=args.val_frac)

    schema, label_cols = detect_schema_and_label_cols(ds, args.dataset_config)
    label_names = get_label_names(ds, schema, label_cols, args.dataset_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def preprocess(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=args.max_length)
        enc.update(attach_labels(batch, schema, label_names, label_cols))
        return enc

    remove_cols = [c for c in ds["train"].column_names if c != "text"]
    encoded = ds.map(preprocess, batched=True, remove_columns=remove_cols)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        problem_type="multi_label_classification",
        id2label={i: n for i, n in enumerate(label_names)},
        label2id={n: i for i, n in enumerate(label_names)},
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logging_steps=100,
        report_to=args.report_to,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_builder(args.eval_threshold, label_names),
    )

    train_metrics = trainer.train()
    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(encoded["test"]) if "test" in encoded else {}

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"train": train_metrics.metrics, "val": val_metrics, "test": test_metrics,
                   "label_names": label_names, "args": vars(args)}, f, indent=2)

    params = count_trainable_parameters(model)
    latency_ms = measure_latency(model, tokenizer, device=device, max_length=args.max_length, batch_size=32, iters=30)
    with open(os.path.join(args.output_dir, "efficiency_snapshot.json"), "w") as f:
        json.dump({"trainable_params": int(params), "avg_latency_ms_per_batch32": float(latency_ms)}, f, indent=2)

    print("Done. Saved outputs in:", args.output_dir)
