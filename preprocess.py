import re
from typing import List
from transformers import AutoTokenizer

def basic_clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tokenize_texts(tokenizer, texts: List[str], max_length: int):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
