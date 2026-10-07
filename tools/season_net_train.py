"""Seasonal-weight lever (fork S; configs/season_net_design_v1.json): prc2026_ctx3.train.train with one addition, a per-row weight on the
main squared-error term.

    main = mean(w[index] * (out - target[index])^2)        instead of   mean((out - target[index])^2)

w is a float32 tensor on the model's device, aligned with the fitting rows, with mean 1 over the fitting rows. Everything else (seeds,
permutation, masking draws, optimiser, schedule, clipping, readouts, logging) is the frozen round-3 loop, copied line for line. With w = 1
everywhere, 1.0 * x == x exactly, so the fitted weights equal prc2026_ctx3.train.train's bit for bit (tools/season_net_selftest.py checks
this on the round-3 test fixture). The running "train_loss_scaled" logs the weighted main loss. The frozen package is not modified.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from prc2026_ctx3.train import MASK_SEED_OFFSET, Context, aux_loss, ema_update, learning_rate, mask_rows, predict  # noqa: F401
from prc2026_ctx3.model import ContextNet3
from prc2026_nn.preprocess import Preprocessor


def train_weighted(model: ContextNet3, pre: Preprocessor, x_num: torch.Tensor, x_cat: torch.Tensor, context: Context, target: torch.Tensor,
                   training: dict, seed: int, row_weight: torch.Tensor, monitor: tuple | None = None, log=print, masking: dict | None = None,
                   aux: dict | None = None, readouts: dict | None = None) -> tuple[list[dict], dict]:
    """prc2026_ctx3.train.train with `row_weight` (len(target) float32 tensor on the device) weighting the main squared error."""
    device = x_num.device
    if row_weight.shape != target.shape or row_weight.device != target.device:
        raise ValueError("row_weight must align with the target on the same device.")
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    p = float(masking["p"]) if masking else 0.0
    if aux is not None and (p <= 0 or not model.n_aux):
        raise ValueError("Auxiliary heads need masked rows and a network with an auxiliary head.")
    mask_generator = torch.Generator(device=device).manual_seed(seed + MASK_SEED_OFFSET) if p > 0 else None
    columns = masking["columns"].to(device) if p > 0 else None
    decay = [p_ for _, p_ in model.named_parameters() if p_.ndim >= 2]
    plain = [p_ for _, p_ in model.named_parameters() if p_.ndim < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": training["weight_decay"]}, {"params": plain, "weight_decay": 0.0}], lr=training["lr"])
    n, batch = len(target), int(training["batch_size"])
    steps_per_epoch = math.ceil(n / batch)
    epochs = int(training["epochs"])
    total = steps_per_epoch * epochs
    warmup = max(1, int(round(training["warmup_epochs"] * steps_per_epoch)))
    names = [name for name, _ in model.named_parameters()]
    params = [p_ for _, p_ in model.named_parameters()]
    ema = [p_.detach().clone() for p_ in params] if readouts else None
    tail_epochs = int(readouts["tail_epochs"]) if readouts else 0
    tail = [torch.zeros_like(p_) for p_ in params] if tail_epochs else None
    history, step, rate = [], 0, training["lr"]
    for epoch in range(epochs):
        started = time.time()
        model.train()
        order = torch.randperm(n, generator=generator).to(device)
        running = torch.zeros((), device=device, dtype=torch.float64)
        running_aux = torch.zeros((), device=device, dtype=torch.float64)
        masked_rows = torch.zeros((), device=device, dtype=torch.int64)
        for s in range(steps_per_epoch):
            index = order[s * batch:(s + 1) * batch]
            rate = learning_rate(step, total, warmup, training["lr"], training["min_lr_fraction"])
            for group in optimizer.param_groups:
                group["lr"] = rate
            tok, tok_nan, wake, valid = context.batch(index, model.max_tokens)
            xb = x_num[index]
            masked = None
            if p > 0:
                draw = torch.rand(len(index), generator=mask_generator, device=device)
                xb, masked = mask_rows(xb, draw, p, columns, int(masking["matched"]))
                masked_rows += masked.sum()
            if aux is not None:
                out, aux_out = model.forward_aux(xb, x_cat[index], tok, tok_nan, wake, valid)
                main = (row_weight[index] * (out[:, 0] - target[index]) ** 2).mean()
                extra = aux_loss(aux_out, aux["targets"][index], masked)
                loss = main + float(aux["weight"]) * extra
                running_aux += extra.detach().double() * len(index)
            else:
                out = model(xb, x_cat[index], tok, tok_nan, wake, valid)
                main = loss = (row_weight[index] * (out[:, 0] - target[index]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if training.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            optimizer.step()
            running += main.detach().double() * len(index)
            step += 1
            if ema is not None:
                ema_update(ema, params, step, readouts["ema_decay"])
        if tail is not None and epoch >= epochs - tail_epochs:
            with torch.no_grad():
                torch._foreach_add_(tail, [p_.detach() for p_ in params])
        entry = {"epoch": epoch + 1, "train_loss_scaled": float(running) / n, "lr_end": rate, "seconds": round(time.time() - started, 1)}
        if p > 0:
            entry["masked_rows"] = int(masked_rows)
        if aux is not None:
            entry["aux_loss"] = float(running_aux) / n
        if monitor is not None:
            values = predict(model, pre, monitor[0], monitor[1], monitor[2])
            entry["monitor_rmse"] = float(np.sqrt(np.mean((values - monitor[3]) ** 2)))
        history.append(entry)
        log(entry)
    averaged = {}
    if readouts:
        base = model.state_dict()
        averaged["ema"] = {k: v.detach().clone() for k, v in base.items()}
        for name, value in zip(names, ema):
            averaged["ema"][name] = value.clone()
        if tail is not None:
            averaged["tail"] = {k: v.detach().clone() for k, v in base.items()}
            for name, value in zip(names, tail):
                averaged["tail"][name] = value / tail_epochs
    return history, averaged
