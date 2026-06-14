import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class StockWindowDataset(Dataset):
    def __init__(self, parquet_path, window=20):
        df = pd.read_parquet(parquet_path).sort_values("date")
        feats = df[["open", "high", "low", "close", "volume"]].values.astype("float32")
        self.feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
        self.returns = df["close"].pct_change().shift(-1).fillna(0).values.astype("float32")
        self.window = window

    def __len__(self):
        return len(self.feats) - self.window

    def __getitem__(self, idx):
        x = self.feats[idx : idx + self.window]
        y = self.returns[idx + self.window - 1]
        return torch.from_numpy(x), torch.tensor(y)


def build_train_val_loaders(parquet_path, window=20, batch_size=32, val_frac=0.2):
    full_ds = StockWindowDataset(parquet_path, window=window)
    n = len(full_ds)
    split = int(n * (1 - val_frac))
    train_ds = Subset(full_ds, range(0, split))
    val_ds = Subset(full_ds, range(split, n))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = build_train_val_loaders("data/aapl_ge_psct.parquet")
    xb, yb = next(iter(train_loader))
    print(xb.shape, yb.shape)
