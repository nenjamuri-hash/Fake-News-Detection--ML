import os
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from config import Config

LIAR_TRAIN = "train.tsv"
LIAR_VALID = "valid.tsv"
LIAR_TEST  = "test.tsv"

def _load_liar_split(path: str) -> pd.DataFrame:
    # LIAR columns: label, statement, subject, speaker, job title, state, party, ...
    cols = ["label","statement","subject","speaker","job_title","state_info",
            "party","barely_true_c","false_c","half_true_c","mostly_true_c",
            "pants_on_fire_c","context"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols, quoting=3)
    df = df[["statement","label"]].rename(columns={"statement":"text"})
    return df

def _apply_binary_map(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    def map_label(x):
        key = str(x).strip().lower()
        return cfg.binary_map.get(key, None)
    df["label_bin"] = df[cfg.label_col].apply(map_label)
    df = df.dropna(subset=["label_bin"])
    return df[["text","label_bin"]].rename(columns={"label_bin":"label"})

def load_data(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns train/val/test DataFrames with columns: text,label (binary)."""
    # Prefer LIAR folder if all three files exist
    liar_train = os.path.join(cfg.liar_dir, LIAR_TRAIN)
    liar_valid = os.path.join(cfg.liar_dir, LIAR_VALID)
    liar_test  = os.path.join(cfg.liar_dir, LIAR_TEST)

    if all(os.path.exists(p) for p in [liar_train, liar_valid, liar_test]):
        tr = _load_liar_split(liar_train); va = _load_liar_split(liar_valid); te = _load_liar_split(liar_test)
        for df in (tr, va, te):
            df.rename(columns={"label": cfg.label_col, "text": cfg.text_col}, inplace=True)
        tr = _apply_binary_map(tr.rename(columns={"statement":"text"}).rename(columns={cfg.text_col:"text", cfg.label_col:"label"}), cfg)
        va = _apply_binary_map(va.rename(columns={cfg.text_col:"text", cfg.label_col:"label"}), cfg)
        te = _apply_binary_map(te.rename(columns={cfg.text_col:"text", cfg.label_col:"label"}), cfg)
        return tr, va, te

    # Else use a single CSV/TSV file
    if not os.path.exists(cfg.data_file):
        raise FileNotFoundError(
            f"No data found. Provide LIAR folder with tsv splits or a file at {cfg.data_file}"
        )
    df = pd.read_csv(cfg.data_file, sep=cfg.delimiter)
    if cfg.text_col not in df.columns or cfg.label_col not in df.columns:
        raise ValueError(f"Data must have columns '{cfg.text_col}' and '{cfg.label_col}'")
    df = df.dropna(subset=[cfg.text_col, cfg.label_col]).copy()
    # normalize & binarize labels
    df = df.rename(columns={cfg.text_col: "text", cfg.label_col: "label"})
    df = _apply_binary_map(df, cfg)

    # Split: test, then val from remaining
    train_df, test_df = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.random_state, stratify=df["label"]
    )
    val_size_adj = cfg.val_size / (1 - cfg.test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_size_adj, random_state=cfg.random_state, stratify=train_df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
