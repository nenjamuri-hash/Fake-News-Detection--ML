import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np

# Load the fine-tuned model and tokenizer
model_path = "artifacts/model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()

# Function to predict label (0 = Fake, 1 = Real)
def predict_label(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1).numpy()[0][predicted_class_id]
    return predicted_class_id, confidence

# Try prediction on sample input
if __name__ == "__main__":
    sample_text = input("Enter a news headline/article: ")
    label, conf = predict_label(sample_text)

    label_name = "REAL" if label == 1 else "FAKE"
    print(f"\nPrediction: {label_name} ({conf*100:.2f}% confidence)")
