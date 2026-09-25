import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from prc2026_ctx.model import ContextNet, token_features
from prc2026_ctx.tokens import TokenStore, build_tokens, encode_codes, neighbours, save_file
from prc2026_ctx2.model import ContextNet2
from prc2026_ctx2.tokens import neighbours_window
from prc2026_ctx2.train import Context, Store, build_model, load, predict, save, train
from prc2026_nn.preprocess import Preprocessor
from test_ctx import matches_for, raw_movements

ARCH = {"d_num": 4, "cat_cap": 4, "width": 32, "depth": 2, "dropout": 0.0, "tok_dim": 16, "heads": 4, "max_tokens": 12}


def test_window_rule_reproduces_round_one():
    raw = raw_movements(n=1500, seed=4)
    tokens = encode_codes(build_tokens(raw, matches_for(raw)))
    a = neighbours(tokens, k=16)
    b = neighbours_window(tokens, 16, 3600.0, 1800.0, -900.0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    wide = neighbours_window(tokens, 32, 7200.0, 3600.0, -900.0)[2]
    assert ((wide >= 0).sum(axis=1) >= (b[2] >= 0).sum(axis=1)).all()


def fixture(tmp_path, k=12):
    raw = raw_movements(n=1200, seed=5)
    tokens = encode_codes(build_tokens(raw, matches_for(raw)))
    dep_ids, self_rows, index = neighbours(tokens, k=k)
    index[:3] = -1  # rows without any neighbour
    save_file(tmp_path, "a", tokens, dep_ids, self_rows, index)
    store = TokenStore(tmp_path, ["a"])
    frame = pd.DataFrame({"a": np.random.default_rng(1).normal(size=len(dep_ids)), "c": np.where(np.arange(len(dep_ids)) % 2, "x", "y")})
    target = frame["a"].to_numpy() * 100 + 600
    pre = Preprocessor.fit(frame, ["a"], ["c"], target, bins=8, min_count=1)
    return store, dep_ids, frame, target, pre


def test_no_self_attention_equals_round_one(tmp_path):
    store, dep_ids, frame, target, pre = fixture(tmp_path)
    s, nb = store.lookup(dep_ids)
    feats = token_features(torch.from_numpy(store.times), torch.from_numpy(store.values), torch.from_numpy(store.codes), torch.from_numpy(s), torch.from_numpy(nb))
    xn, xc = (torch.from_numpy(v) for v in pre.transform(frame))
    torch.manual_seed(7)
    one = ContextNet(pre.edges, pre.n_bins, pre.cardinalities(), **ARCH)
    torch.manual_seed(7)
    two = ContextNet2(pre.edges, pre.n_bins, pre.cardinalities(), **ARCH, sa_layers=0)
    one.eval(); two.eval()
    with torch.no_grad():
        assert torch.equal(one(xn, xc, *feats), two(xn, xc, *feats))


def test_self_attention_fit_predict_reload(tmp_path):
    store, dep_ids, frame, target, pre = fixture(tmp_path)
    device = torch.device("cpu")
    dstore = Store.build(store, device)
    context = Context.build(dstore, *store.lookup(dep_ids), {"use_adsb": True})
    arch = {**ARCH, "sa_layers": 2, "sa_heads": 4, "sa_ff": 2, "sa_dropout": 0.0}
    xn, xc = (torch.from_numpy(v) for v in pre.transform(frame))
    model = build_model(pre, arch, seed=3, device=device)
    training = {"epochs": 3, "batch_size": 64, "lr": 3e-3, "weight_decay": 1e-5, "warmup_epochs": 0.5, "min_lr_fraction": 0.05, "grad_clip": 5.0}
    history = train(model, pre, xn, xc, context, torch.from_numpy(pre.scale_target(target)), training, seed=3, monitor=(xn, xc, context, target))
    assert len(history) == 3 and all(np.isfinite(e["monitor_rmse"]) for e in history)
    values = predict(model, pre, xn, xc, context)
    assert np.isfinite(values).all()
    save(model, pre, arch, {"use_adsb": True}, tmp_path / "m")
    again, pre2, settings = load(tmp_path / "m", device)
    assert settings == {"use_adsb": True, "departures_only": False}
    assert np.allclose(predict(again, pre2, xn, xc, context), values, atol=1e-6)
