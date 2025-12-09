from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Config:
    # === Data ===
    # Option A: point to a single CSV/TSV with columns: text,label
    data_file: str = "data/news.csv"    # if present, used for random split
    delimiter: str = ","                # "," for CSV, "\t" for TSV

    # Option B: LIAR folder with train.tsv/valid.tsv/test.tsv (original format)
    liar_dir: str = "data/liar"         # set to folder or leave as-is if unused

    text_col: str = "text"              # used for custom CSV/TSV
    label_col: str = "label"            # used for custom CSV/TSV
    test_size: float = 0.2
    val_size: float = 0.1               # from the remaining after test split
    random_state: int = 42

    # Map labels to binary (for LIAR or custom labels)
    # Keys are lowercased label strings in your data; values are "real"/"fake"
    binary_map: Dict[str, str] = field(default_factory=lambda: {
        "true": "real", "mostly-true": "real", "half-true": "real",
        "barely-true": "fake", "false": "fake", "pants-fire": "fake",
        "real": "real", "fake": "fake", "1": "real", "0": "fake"
    })
    classes: List[str] = field(default_factory=lambda: ["fake", "real"])

    # === Training ===
    output_dir: str = "artifacts"
    max_length: int = 256
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    seed: int = 42

    # === Models ===
    roberta_model: str = "roberta-base"
    deberta_model: str = "microsoft/deberta-v3-base"

    # === Files ===
    saved_model_dir: str = "artifacts/model"
    tokenizer_dir: str = "artifacts/tokenizer"
    metric_file: str = "artifacts/metrics.json"
