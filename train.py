import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml

import wandb
from dataset import build_train_val_loaders
from model import MLP
from utils import load_checkpoint, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--weight_decay", type=float, default=None)
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.safe_load(f)

if args.lr is not None:
    cfg["lr"] = args.lr
if args.weight_decay is not None:
    cfg["weight_decay"] = args.weight_decay

EPOCHS = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, val_loader = build_train_val_loaders(
    "data/aapl_ge_psct.parquet", cfg["window"], cfg["batch_size"]
)
model = MLP().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
loss_fn = nn.MSELoss()


wandb.init(
    project="ml-training-template",
    config={
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "window": cfg["window"],
        "model": "MLP",
    },
)

start_epoch, best_val = 0, float("inf")
patience, bad = cfg["patience"], 0
if args.resume and os.path.exists("last.pt"):
    start_epoch, best_val = load_checkpoint("last.pt", model, optimizer, scheduler)
    start_epoch += 1
    print(f"resumed from epoch {start_epoch}")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(train_loader.dataset)
    scheduler.step()

    model.eval()
    val_loss = 0.0
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += loss_fn(pred, yb).item() * len(xb)
            y_true_list.append(yb.cpu())
            y_pred_list.append(pred.cpu())
    val_loss /= len(val_loader.dataset)
    print(f"epoch {epoch} val_loss {val_loss:.5f} lr {scheduler.get_last_lr()[0]:.2e}")

    if epoch == EPOCHS - 1 or epoch % 10 == 0:
        y_true = torch.cat(y_true_list).numpy()
        y_pred = torch.cat(y_pred_list).numpy()
        fig, ax = plt.subplots()
        ax.plot(y_true[:200], label="true")
        ax.plot(y_pred[:200], label="pred")
        ax.legend()
        wandb.log({"pred_vs_true": wandb.Image(fig)}, commit=False)
        plt.close(fig)

    wandb.log(
        {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "grad_norm": grad_norm,
            "epoch": epoch,
        }
    )

    save_checkpoint("last.pt", model, optimizer, scheduler, epoch, best_val)

    if val_loss < best_val:
        best_val, bad = val_loss, 0
        save_checkpoint("best.pt", model, optimizer, scheduler, epoch, best_val)
    else:
        bad += 1
        if bad >= patience:
            print(f"early stop at epoch {epoch}")
            break
wandb.finish()
