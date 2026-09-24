"""Training, prediction and save/load for the neural learner.

Training uses AdamW with decoupled weight decay on weight matrices only, a linear warm-up followed by a cosine decay to
`min_lr_fraction` over a FIXED epoch budget, optional gradient-norm clipping and an optional exponential moving average
of the weights. The schedule never depends on the monitoring rows, so a refit with the same budget follows the same
recipe; the monitor only records the per-epoch RMSE (seconds) for the run record. The saved model is the final epoch's
(EMA) weights.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .model import TabularNet
from .preprocess import Preprocessor

ARCHITECTURE_KEYS = ("numeric_mode", "d_num", "cat_cap", "width", "depth", "dropout", "k")
TRAINING_KEYS = ("epochs", "batch_size", "lr", "weight_decay", "warmup_epochs", "min_lr_fraction", "grad_clip", "ema_decay")


def build_model(pre: Preprocessor, architecture: dict, seed: int) -> TabularNet:
    torch.manual_seed(seed)
    return TabularNet(pre.edges, pre.n_bins, pre.cardinalities(), **{k: architecture[k] for k in ARCHITECTURE_KEYS})


def learning_rate(step: int, total: int, warmup: int, peak: float, floor_fraction: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (floor_fraction + (1.0 - floor_fraction) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


@torch.no_grad()
def predict(model: TabularNet, pre: Preprocessor, x_num: np.ndarray, x_cat: np.ndarray, batch: int = 16384) -> np.ndarray:
    """Member-mean prediction in seconds."""
    model.eval()
    out = np.empty(len(x_num), dtype=np.float64)
    for start in range(0, len(x_num), batch):
        stop = start + batch
        z = model(torch.from_numpy(x_num[start:stop]), torch.from_numpy(x_cat[start:stop])).mean(dim=1)
        out[start:stop] = z.double().numpy()
    return pre.unscale(out)


class _Ema:
    def __init__(self, model: TabularNet, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: TabularNet) -> None:
        for n, p in model.named_parameters():
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap(self, model: TabularNet) -> None:
        for n, p in model.named_parameters():
            current = p.detach().clone()
            p.copy_(self.shadow[n])
            self.shadow[n] = current


def train(model: TabularNet, pre: Preprocessor, x_num: np.ndarray, x_cat: np.ndarray, target: np.ndarray, training: dict, seed: int,
          monitor: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None, log=print) -> list[dict]:
    """Fit in place for `training["epochs"]` epochs. `target` is the scaled training target; `monitor` = (x_num, x_cat, label seconds)."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    xn, xc, y = torch.from_numpy(x_num), torch.from_numpy(x_cat), torch.from_numpy(target)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and n != "adapter"]
    plain = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and n != "adapter")]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": training["weight_decay"]}, {"params": plain, "weight_decay": 0.0}], lr=training["lr"])
    n, batch = len(target), int(training["batch_size"])
    steps_per_epoch = math.ceil(n / batch)
    total = steps_per_epoch * int(training["epochs"])
    warmup = max(1, int(round(training["warmup_epochs"] * steps_per_epoch)))
    ema = _Ema(model, training["ema_decay"]) if training.get("ema_decay") else None
    history, step = [], 0
    for epoch in range(int(training["epochs"])):
        started = time.time()
        model.train()
        order = torch.randperm(n, generator=generator)
        running, seen = 0.0, 0
        for s in range(steps_per_epoch):
            index = order[s * batch:(s + 1) * batch]
            rate = learning_rate(step, total, warmup, training["lr"], training["min_lr_fraction"])
            for group in optimizer.param_groups:
                group["lr"] = rate
            out = model(xn[index], xc[index])
            loss = ((out - y[index].unsqueeze(1)) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if training.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            optimizer.step()
            if ema is not None:
                ema.update(model)
            running += float(loss.detach()) * len(index)
            seen += len(index)
            step += 1
        entry = {"epoch": epoch + 1, "train_loss_scaled": running / seen, "lr_end": rate, "seconds": round(time.time() - started, 1)}
        if monitor is not None:
            if ema is not None:
                ema.swap(model)
            values = predict(model, pre, monitor[0], monitor[1])
            if ema is not None:
                ema.swap(model)
            entry["monitor_rmse"] = float(np.sqrt(np.mean((values - monitor[2]) ** 2)))
        history.append(entry)
        log(entry)
    if ema is not None:
        ema.swap(model)  # the saved model carries the averaged weights
    return history


def save(model: TabularNet, pre: Preprocessor, architecture: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pre.save(directory)
    torch.save(model.state_dict(), directory / "model.pt")
    (directory / "architecture.json").write_text(json.dumps({k: architecture[k] for k in ARCHITECTURE_KEYS}, indent=1, sort_keys=True), encoding="utf-8")


def load(directory: Path) -> tuple[TabularNet, Preprocessor]:
    pre = Preprocessor.load(directory)
    architecture = json.loads((directory / "architecture.json").read_text(encoding="utf-8"))
    model = TabularNet(pre.edges, pre.n_bins, pre.cardinalities(), **architecture)
    model.load_state_dict(torch.load(directory / "model.pt", weights_only=True))
    model.eval()
    return model, pre
