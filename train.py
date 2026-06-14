import torch
import torch.nn as nn

from dataset import build_train_val_loaders
from model import MLP

EPOCHS = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, val_loader = build_train_val_loaders("data/aapl_ge_psct.parquet")
model = MLP().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
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

    best_val, patience, bad = float("inf"), 5, 0
    if val_loss < best_val:
        best_val, bad = val_loss, 0
        torch.save(model.state_dict(), "best.pt")
    else:
        bad += 1
        if bad >= patience:
            print(f"early stop at epoch {epoch}")
            break
