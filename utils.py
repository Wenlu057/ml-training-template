import random

import numpy as np
import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    torch.set_rng_state(ckpt["rng"]["torch"])
    torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    np.random.set_state(ckpt["rng"]["numpy"])
    random.setstate(ckpt["rng"]["python"])
    return ckpt["epoch"], ckpt["best_val"]
