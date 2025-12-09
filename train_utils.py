import os, json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)
from datasets import Dataset as HFDataset
from typing import Dict
from config import Config
from preprocess import basic_clean
from utils_metrics import compute_metrics

class SimpleTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self): return self.labels.shape[0]
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def _prepare_hf_dataset(tokenizer, df, cfg: Config):
    texts = [basic_clean(t) for t in df["text"].tolist()]
    labels = [0 if l == "fake" else 1 for l in df["label"].tolist()]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=cfg.max_length)
    return SimpleTextDataset({k: torch.tensor(v) for k,v in enc.items()},
                             torch.tensor(labels, dtype=torch.long))

def train_model(model_name: str, cfg: Config, train_df, val_df):
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds = _prepare_hf_dataset(tokenizer, train_df, cfg)
    val_ds   = _prepare_hf_dataset(tokenizer, val_df, cfg)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        fp16=cfg.fp16,
        logging_steps=50,
        report_to="none",
        seed=cfg.seed,
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, ("fake","real")),
    )

    trainer.train()
    metrics = trainer.evaluate()
    # Save everything
    trainer.save_model(cfg.saved_model_dir)
    tokenizer.save_pretrained(cfg.tokenizer_dir)

    with open(cfg.metric_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
