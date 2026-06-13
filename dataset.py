import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


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


def build_loader(parquet_path, batch_size=32, window=20, shuffle=True):
    ds = StockWindowDataset(parquet_path, window=window)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, drop_last=True)


if __name__ == "__main__":
    loader = build_loader("data/aapl_ge_psct.parquet")
    xb, yb = next(iter(loader))
    print(xb.shape, yb.shape)
