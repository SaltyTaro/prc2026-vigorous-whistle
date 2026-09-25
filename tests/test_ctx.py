import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from prc2026_ctx.model import TOKEN_FEATURES, token_features
from prc2026_ctx.tokens import ID, RAW_COLUMNS, TokenStore, build_tokens, encode_codes, neighbours, save_file, stand_group
from prc2026_ctx.train import Context, build_model, load, predict, save, train
from prc2026_nn.preprocess import Preprocessor


def raw_movements(n=400, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2025-07-01", tz="UTC")
    phase = rng.choice(["DEP", "ARR"], size=n)
    airport = rng.choice(["EHAM", "LFPG"], size=n)
    mvt = base + pd.to_timedelta(np.sort(rng.uniform(0, 86400, size=n)), unit="s")
    taxi = rng.uniform(300, 1500, size=n)
    return pd.DataFrame({
        ID: np.arange(1, n + 1, dtype=float), "PHASE_mvt": phase,
        "ADEP_mvt": np.where(phase == "DEP", airport, "XXXX"), "ADES_mvt": np.where(phase == "ARR", airport, "YYYY"),
        "MVT_TIME_UTC_mvt": mvt, "SCHED_TIME_UTC_mvt": mvt - pd.to_timedelta(900, unit="s"),
        "BLOCK_TIME_UTC_mvt": np.where(phase == "DEP", mvt - pd.to_timedelta(taxi, unit="s"), mvt + pd.to_timedelta(taxi, unit="s")),
        "TAXITIME_SEC_mvt": taxi, "RUNWAY_mvt": rng.choice(["09", "27"], size=n), "STAND_mvt": rng.choice(["F32", "F34", "C12", None], size=n),
        "AOBT_3_flt": np.where(phase == "DEP", mvt - pd.to_timedelta(taxi + 30, unit="s"), pd.NaT), "EOBT_1_flt": mvt - pd.to_timedelta(1200, unit="s"),
        "WK_TBL_CAT_flt": rng.choice(["M", "H", None], size=n)})[RAW_COLUMNS]


def matches_for(raw):
    dep = raw[raw["PHASE_mvt"].eq("DEP")].iloc[::2]
    takeoff = (dep["MVT_TIME_UTC_mvt"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    return pd.DataFrame({ID: dep[ID], "gate": True, "motion_start": takeoff - 600, "segment_start": takeoff - 700, "taxi_stopped_s": 30.0})


def test_departure_tokens_never_carry_labels():
    raw = raw_movements()
    tokens = build_tokens(raw, matches_for(raw))
    dep = tokens["is_dep"].to_numpy()
    assert tokens.loc[dep, "t_inblock"].isna().all() and tokens.loc[dep, "taxi_in"].isna().all()
    assert tokens.loc[~dep, "t_inblock"].notna().all() and tokens.loc[~dep, "taxi_in"].notna().all()
    assert tokens.loc[~dep, ["t_aobt", "t_motion", "adsb_matched"]].isna().all().all()


def test_neighbours_same_airport_window_no_self():
    raw = raw_movements(n=1500, seed=1)
    tokens = encode_codes(build_tokens(raw, matches_for(raw)))
    dep_ids, self_rows, index = neighbours(tokens, k=16)
    assert (tokens[ID].to_numpy()[self_rows] == dep_ids).all()
    t, airport = tokens["t_mvt"].to_numpy(), tokens["airport"].to_numpy()
    for i in range(len(dep_ids)):
        nb = index[i][index[i] >= 0]
        assert self_rows[i] not in nb
        assert (airport[nb] == airport[self_rows[i]]).all()
        assert ((t[nb] >= t[self_rows[i]] - 3600) & (t[nb] <= t[self_rows[i]] + 1800)).all()
        assert len(nb) == len(set(nb.tolist()))
        # a full list is used whenever the window holds at least k other movements
        window = ((airport == airport[self_rows[i]]) & (t >= t[self_rows[i]] - 3600) & (t <= t[self_rows[i]] + 1800)).sum() - 1
        assert len(nb) == min(16, window)


def test_stand_group():
    assert stand_group("F32") == "F" and stand_group("307") == "3" and stand_group("H44R") == "H" and stand_group(None) is None


def test_store_features_and_fit(tmp_path):
    raw = raw_movements(n=1200, seed=2)
    tokens = encode_codes(build_tokens(raw, matches_for(raw)))
    dep_ids, self_rows, index = neighbours(tokens, k=12)
    save_file(tmp_path, "a", tokens, dep_ids, self_rows, index)
    store = TokenStore(tmp_path, ["a"])
    s, nb = store.lookup(dep_ids[::-1])
    assert (s == self_rows[::-1]).all() and (nb == index[::-1]).all()
    x, nan, wake, valid = token_features(torch.from_numpy(store.times), torch.from_numpy(store.values), torch.from_numpy(store.codes),
                                         torch.from_numpy(s), torch.from_numpy(nb))
    assert x.shape == (len(s), 12, len(TOKEN_FEATURES)) and torch.isfinite(x).all()
    assert (valid.numpy() == (nb >= 0)).all()
    # the focal's own takeoff is t0: a neighbour's rel_mvt is its time difference over 1800 s
    j = TOKEN_FEATURES.index("rel_mvt")
    i0 = int(np.argmax((nb >= 0).sum(axis=1)))
    expect = (store.times[nb[i0, 0], 0] - store.times[s[i0], 0]) / 1800
    assert abs(float(x[i0, 0, j]) - expect) < 1e-4
    x2, nan2, _, _ = token_features(torch.from_numpy(store.times), torch.from_numpy(store.values), torch.from_numpy(store.codes),
                                    torch.from_numpy(s), torch.from_numpy(nb), use_adsb=False)
    assert float(nan2[..., TOKEN_FEATURES.index("adsb_taxi")].min()) == 1.0

    frame = pd.DataFrame({"a": np.random.default_rng(0).normal(size=len(dep_ids)), "c": np.where(np.arange(len(dep_ids)) % 2, "x", "y")})
    target = frame["a"].to_numpy() * 100 + 600
    pre = Preprocessor.fit(frame, ["a"], ["c"], target, bins=8, min_count=1)
    xn, xc = pre.transform(frame)
    arch = {"d_num": 4, "cat_cap": 4, "width": 32, "depth": 2, "dropout": 0.0, "tok_dim": 16, "heads": 4, "max_tokens": 12}
    training = {"epochs": 3, "batch_size": 64, "lr": 3e-3, "weight_decay": 1e-5, "warmup_epochs": 0.5, "min_lr_fraction": 0.05, "grad_clip": 5.0}
    context = Context.build(store, *store.lookup(dep_ids), {"use_adsb": True})
    model = build_model(pre, arch, seed=3)
    history = train(model, pre, xn, xc, context, pre.scale_target(target), training, seed=3, monitor=(xn, xc, context, target))
    assert len(history) == 3 and history[-1]["monitor_rmse"] < history[0]["monitor_rmse"] * 1.5
    values = predict(model, pre, xn, xc, context)
    save(model, pre, arch, {"use_adsb": True}, tmp_path / "m")
    again, pre2, settings = load(tmp_path / "m")
    assert settings == {"use_adsb": True, "departures_only": False}
    assert np.allclose(predict(again, pre2, xn, xc, context), values, atol=1e-6)
