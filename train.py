import argparse
import os

import torch
import torch.nn as nn

from dataset import build_train_val_loaders
from model import MLP
from utils import load_checkpoint, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

EPOCHS = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, val_loader = build_train_val_loaders("data/aapl_ge_psct.parquet")
model = MLP().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_fn = nn.MSELoss()

start_epoch, best_val = 0, float("inf")
patience, bad = 50, 0
if args.resume and os.path.exists("last.pt"):
    start_epoch, best_val = load_checkpoint("last.pt", model, optimizer, scheduler)
    start_epoch += 1
    print(f"resumed from epoch {start_epoch}")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += loss_fn(model(xb), yb).item() * len(xb)
    val_loss /= len(val_loader.dataset)
    print(f"epoch {epoch} val_loss {val_loss:.5f} lr {scheduler.get_last_lr()[0]:.2e}")

    save_checkpoint("last.pt", model, optimizer, scheduler, epoch, best_val)

    if val_loss < best_val:
        best_val, bad = val_loss, 0
        save_checkpoint("best.pt", model, optimizer, scheduler, epoch, best_val)
    else:
        bad += 1
        if bad >= patience:
            print(f"early stop at epoch {epoch}")
            break
