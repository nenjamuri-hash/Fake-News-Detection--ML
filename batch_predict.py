import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load model & tokenizer
model_path = "roberta-fake-news-model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()

# Read headlines from CSV
df = pd.read_csv("test_headlines.csv")  # Assumes a 'headline' column
headlines = df["headline"].tolist()

# Predict
predictions = []
for text in headlines:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        predictions.append("REAL" if pred == 0 else "FAKE")

# Save results
df["prediction"] = predictions
df.to_csv("batch_predictions.csv", index=False)
print("Batch predictions saved to 'batch_predictions.csv'")
