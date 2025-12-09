import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from preprocess import basic_clean
from config import Config

def predict(text: str, cfg: Config):
    model = AutoModelForSequenceClassification.from_pretrained(cfg.saved_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_dir)
    model.eval()
    cleaned = basic_clean(text)
    enc = tokenizer(cleaned, truncation=True, padding=True, max_length=cfg.max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    pred = torch.argmax(logits, dim=-1).item()
    label = "fake" if pred == 0 else "real"
    score = torch.softmax(logits, dim=-1)[0, pred].item()
    return label, float(score)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, required=True, help="News headline or article text")
    args = ap.parse_args()
    cfg = Config()
    label, score = predict(args.text, cfg)
    print(f"Prediction: {label} (confidence={score:.3f})")
