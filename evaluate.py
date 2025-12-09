import numpy as np
from config import Config
from data_loader import load_data
from preprocess import basic_clean
from utils_metrics import compute_metrics
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

if __name__ == "__main__":
    cfg = Config()
    _, val_df, test_df = load_data(cfg)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.saved_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_dir)

    for split_name, df in [("VAL", val_df), ("TEST", test_df)]:
        texts = [basic_clean(t) for t in df["text"].tolist()]
        labels = np.array([0 if l == "fake" else 1 for l in df["label"].tolist()])
        enc = tokenizer(texts, truncation=True, padding=True, max_length=cfg.max_length, return_tensors="pt")

        model.eval()
        with torch.no_grad():
            outputs = model(**{k: v for k, v in enc.items()})
            logits = outputs.logits.cpu().numpy()

        metrics = compute_metrics((logits, labels), ("fake","real"))
        print(f"\n== {split_name} Metrics ==")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
