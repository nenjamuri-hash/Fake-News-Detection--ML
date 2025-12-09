from config import Config
from data_loader import load_data
from train_utils import train_model

if __name__ == "__main__":
    cfg = Config()
    train_df, val_df, _ = load_data(cfg)
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")
    metrics = train_model(cfg.roberta_model, cfg, train_df, val_df)
    print("\n== RoBERTa Validation Metrics ==")
    for k, v in metrics.items():
        print(f"{k}: {v}")
