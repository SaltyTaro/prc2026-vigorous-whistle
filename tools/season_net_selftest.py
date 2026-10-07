"""Seasonal-weight lever (fork S) self-test, CPU, seconds, no competition data:
  1. on the round-3 test fixture (tests/test_ctx3.fixture), tools/season_net_train.train_weighted with w = 1 equals the frozen
     prc2026_ctx3.train.train bit for bit (state dict, histories, EMA / tail readouts), without and with masking;
  2. with non-uniform weights the fit differs (the weight reaches the loss);
  3. the copied loop differs from the frozen source only in the declared lines (signature, docstring, the row_weight check, the two
     main-loss lines);
  4. tools/run_season_net_ctx3.month_weights on the design fold's months: circular distances, mean 1, monotone in distance.
Usage: python -X utf8 -B tools/season_net_selftest.py   (exit 1 on any failure)
"""
from __future__ import annotations

import difflib
import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))


def main():
    import torch
    import prc2026_ctx3.train as T
    from season_net_train import train_weighted
    from run_season_net_ctx3 import month_weights
    from test_ctx3 import ARCH, TRAINING, fixture, tensors
    torch.set_num_threads(2)
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        store, dep_ids, frame, target, pre = fixture(Path(tmp))
        context, xn, xc = tensors(store, dep_ids, frame, pre)
        y = torch.from_numpy(pre.scale_target(target))
        matched_col = ["a", "adsb_matched", "adsb_taxi", "adsb_stopped"].index("adsb_matched")
        masking = {"p": 0.3, "columns": torch.tensor([2, 3]), "matched": matched_col}
        for label, mask in (("plain", None), ("masked", masking)):
            m1 = T.build_model(pre, ARCH, seed=3, device=torch.device("cpu"))
            h1, a1 = T.train(m1, pre, xn, xc, context, y, TRAINING, seed=3, log=lambda e: None, masking=mask,
                             readouts={"ema_decay": 0.99, "tail_epochs": 2})
            m2 = T.build_model(pre, ARCH, seed=3, device=torch.device("cpu"))
            h2, a2 = train_weighted(m2, pre, xn, xc, context, y, TRAINING, 3, torch.ones_like(y), log=lambda e: None, masking=mask,
                                    readouts={"ema_decay": 0.99, "tail_epochs": 2})
            s1, s2 = m1.state_dict(), m2.state_dict()
            same = all(torch.equal(s1[k], s2[k]) for k in s1) and [e["train_loss_scaled"] for e in h1] == [e["train_loss_scaled"] for e in h2]
            same &= all(torch.equal(a1[r][k], a2[r][k]) for r in a1 for k in a1[r])
            print(f"1. w = 1 equals the frozen loop ({label}): {same}")
            if not same:
                failures.append(f"identity {label}")
        w = torch.from_numpy(np.where(np.arange(len(y)) % 3 == 0, 2.0, 0.5).astype(np.float32))
        w = w / w.mean()
        m3 = T.build_model(pre, ARCH, seed=3, device=torch.device("cpu"))
        train_weighted(m3, pre, xn, xc, context, y, TRAINING, 3, w, log=lambda e: None)
        m1 = T.build_model(pre, ARCH, seed=3, device=torch.device("cpu"))
        T.train(m1, pre, xn, xc, context, y, TRAINING, seed=3, log=lambda e: None)
        differs = any(not torch.equal(m1.state_dict()[k], m3.state_dict()[k]) for k in m1.state_dict())
        print(f"2. non-uniform weights change the fit: {differs}")
        if not differs:
            failures.append("weights inert")
    frozen = inspect.getsource(T.train).splitlines()
    import season_net_train as S
    copy = inspect.getsource(S.train_weighted).splitlines()
    diff = [l for l in difflib.unified_diff(frozen, copy, lineterm="", n=0) if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    allowed_tokens = ("def train", "row_weight", "seed: int", "training: dict", "aux: dict", "readouts: dict", '"""', "masking = {", "tensor, NaN",
                      "dicts by readout name", "monitor` =", "prc2026_ctx3.train.train with", "Fit in place", "raise ValueError(\"row_weight")
    replaced = {"-", "-                main = ((out[:, 0] - target[index]) ** 2).mean()",
                "-                main = loss = ((out[:, 0] - target[index]) ** 2).mean()"}  # the frozen main-loss lines and a blank docstring line
    unexpected = [l for l in diff if l not in replaced and not any(t in l for t in allowed_tokens)]
    print(f"3. source differences: {len(diff)} lines, unexpected: {unexpected}")
    if unexpected:
        failures.append("source drift")
    months = np.array([2] * 10 + [3] * 10 + [4] * 10 + [5] * 10 + [6] * 10 + [8] * 10 + [9] * 10)
    wts, table = month_weights(months, {"kernel": "gaussian_circular_month", "a": 3, "sigma_months": 1.5, "target_month": 10})
    dist = {k: v["distance"] for k, v in table.items()}
    ok = abs(wts.mean() - 1) < 1e-12 and dist == {"02": 4, "03": 5, "04": 6, "05": 5, "06": 4, "08": 2, "09": 1}
    ok &= table["09"]["weight"] > table["08"]["weight"] > table["06"]["weight"] >= table["05"]["weight"] >= table["04"]["weight"]
    jan = month_weights(np.array([12, 1, 2, 6, 7]), {"kernel": "gaussian_circular_month", "a": 1, "sigma_months": 1.5, "target_month": 1})[1]
    ok &= jan["12"]["distance"] == 1 and jan["01"]["distance"] == 0 and jan["06"]["distance"] == 5 and jan["07"]["distance"] == 6
    print(f"4. month weights: {ok}; design a = 3: " + ", ".join(f"{k} {v['weight']:.3f}" for k, v in table.items()))
    if not ok:
        failures.append("month weights")
    print("SELFTEST", "FAILED " + str(failures) if failures else "PASSED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
