import torch
import torch.nn as nn
import yaml
from torch.profiler import ProfilerActivity, profile

from dataset import build_train_val_loaders
from model import MLP


def run_profile(num_workers, pin_memory, tag):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = cfg["use_amp"] and torch.cuda.is_available()

    train_loader, _ = build_train_val_loaders(
        "data/aapl_ge_psct.parquet",
        cfg["window"],
        cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = MLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    model.train()

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        steps = 0
        for epoch in range(3):
            for xb, yb in train_loader:
                xb, yb = (
                    xb.to(device, non_blocking=pin_memory),
                    yb.to(device, non_blocking=pin_memory),
                )
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                steps += 1
                if steps >= 20:
                    break
            if steps >= 20:
                break
    print(f"profiled {steps} steps")

    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    print(f"\n===== {tag} (num_workers={num_workers}, pin_memory={pin_memory}) =====")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=12))
    prof.export_chrome_trace(f"trace_{tag}.json")  # https://ui.perfetto.dev


if __name__ == "__main__":
    run_profile(num_workers=0, pin_memory=False, tag="baseline")
    run_profile(num_workers=4, pin_memory=True, tag="optimized")
