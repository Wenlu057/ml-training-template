import time

import torch
import torch.nn as nn
import yaml

from dataset import build_train_val_loaders
from model import MLP


def benchmark(use_amp, accum_steps, n_batches=50):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = use_amp and torch.cuda.is_available()

    train_loader, _ = build_train_val_loaders(
        "data/aapl_ge_psct.parquet", cfg["window"], cfg["batch_size"]
    )
    model = MLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()

    for warmup, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(model(xb), yb) / accum_steps
        scaler.scale(loss).backward()
        if warmup >= 4:
            break
    optimizer.zero_grad()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.time()

    count = 0
    for i, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(model(xb), yb) / accum_steps
        scaler.scale(loss).backward()
        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        count += 1
        if count >= n_batches:
            break

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    bs = xb.size(0)

    mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    tag = f"amp={use_amp} accum={accum_steps}"
    print(f"[{tag}] {count * bs / elapsed:.0f} samples/sec | peak mem {mem:.2f} GB")


if __name__ == "__main__":
    benchmark(use_amp=False, accum_steps=1)  # (a) baseline fp32
    benchmark(use_amp=True, accum_steps=1)  # (b) +AMP
    benchmark(use_amp=True, accum_steps=4)  # (c) +AMP + gradient accumulation
