"""Label-free component-disagreement audit, second pass (28 September 2026): the component pairs the first audit
(tools/disagreement_audit.py, fold side vs v9 side) did not look at.

Pairs, D = b - a per row:
  nets_vs_committee   non-LIRF rows      a = committee (tree ensemble table), b = context networks at v34's family weights
  lirf_net_vs_tree    LIRF non-population rows   a = LIRF tree table, b = LIRF network
  residual            matched non-LIRF rows      a = v9, b = v9 + residual stage (D = the residual stage's correction)

Question per pair: is large disagreement more frequent on the ranking months (January / July 2026) than on the four development months,
and in which airport x matched groups and days; on the development months, which side is right when they disagree, split by the SIGN
of D (the first audit's lesson: when the sides disagree strongly the truth follows the higher side).

Ranking rows: runs/adsb_ctx/export_v30/ranking_values.parquet (no labels exist or are read). Development rows: the fair set
(runs/fair_dev/rows.parquet) with committee / nets re-assembled per month exactly as tools/run_mix_gate.py does (checked: the
assembled mix equals the fair rows' mix).

  --report docs/disagreement/audit2.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_v24 import extra_columns  # noqa: E402
from run_lomo_rows import assemble, configs, weighted  # noqa: E402

RANKING = "runs/adsb_ctx/export_v30/ranking_values.parquet"
FAIR = "runs/fair_dev/rows.parquet"
V35 = "runs/submissions/vigorous-whistle_v35.parquet"
DEV_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
THRESHOLDS = [300, 600, 1200]
BINS = [0, 150, 300, 600, 1200, np.inf]
LABELS = ["<150", "150-300", "300-600", "600-1200", ">1200"]


def pair_frame(ids, period, day, airport, matched, lirf_nonpop, v9, residual, committee, nets, lirf_tree, lirf_net, label=None) -> pd.DataFrame:
    parts = []
    non_lirf = airport != "LIRF"
    specs = {"nets_vs_committee": (non_lirf & np.isfinite(committee) & np.isfinite(nets), committee, nets),
             "lirf_net_vs_tree": (lirf_nonpop & np.isfinite(lirf_tree) & np.isfinite(lirf_net), lirf_tree, lirf_net),
             "residual": (non_lirf & matched, v9, v9 + residual)}
    for name, (mask, a, b) in specs.items():
        frame = {ID: ids[mask], "pair": name, "period": period[mask], "day": day[mask], "airport": airport[mask],
                 "group": np.char.add(np.char.add(airport[mask].astype(str), ":"), np.where(matched[mask], "matched", "unmatched")),
                 "a": a[mask], "b": b[mask], "d": (b - a)[mask]}
        if label is not None:
            frame["label"] = label[mask]
        parts.append(pd.DataFrame(frame))
    return pd.concat(parts, ignore_index=True)


def ranking_frame(config, net, lirf_net) -> pd.DataFrame:
    r = pd.read_parquet(ROOT / RANKING)
    nets = weighted(r, net["network"])
    if float(np.nanmax(np.abs(mixed(r["committee"].to_numpy(float), nets, float(net["network_share"])) - r["committee_mix"].to_numpy(float)))) > 1e-6:
        raise ValueError("Ranking committee / nets do not rebuild the recorded committee mix.")
    times = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "MVT_TIME_UTC_mvt"]).drop_duplicates(ID).set_index(ID)["MVT_TIME_UTC_mvt"]
    day = pd.to_datetime(times.reindex(r[ID])).dt.strftime("%Y-%m-%d").to_numpy()
    scope = r["scope"].to_numpy()
    matched = (scope == "matched_no_lirf") | (scope == "lirf_nonpop_matched")
    lirf_nonpop = np.isin(scope, ["lirf_nonpop_matched", "lirf_nonpop_unmatched"])
    return pair_frame(r[ID].to_numpy(), r["month"].to_numpy(), day, r["airport"].to_numpy(), matched, lirf_nonpop,
                      r["v9"].to_numpy(float), np.where(scope == "matched_no_lirf", r["residual"].to_numpy(float), 0.0),
                      r["committee"].to_numpy(float), nets, r[config["lirf_table"]].to_numpy(float) if config["lirf_table"] in r else r["ens_v22_lirf_q"].to_numpy(float),
                      weighted(r, lirf_net))


def dev_frame(config, net, lirf_net, v33) -> pd.DataFrame:
    fair = pd.read_parquet(ROOT / FAIR)
    parts = []
    for month in DEV_MONTHS:
        f0 = fair[fair["month"] == month].reset_index(drop=True)
        f = assemble(f0, f0["v9"].to_numpy(), f0["residual"].to_numpy(), extra_columns(config, month, f0[ID].to_numpy()), config, net, lirf_net, v33)
        if float(np.max(np.abs(f["mix"].to_numpy() - f0["mix"].to_numpy()))) > 1e-9:
            raise ValueError(f"The assembled mix does not equal the fair rows' mix on {month}.")
        airport = f["airport"].to_numpy()
        lirf_nonpop = (airport == "LIRF") & (f["lirf_m"].to_numpy() | f["lirf_u"].to_numpy())
        matched = f["matched"].to_numpy() | (f["lirf_m"].to_numpy() & (airport == "LIRF"))
        parts.append(pair_frame(f[ID].to_numpy(), f["month"].to_numpy(), f["day"].to_numpy(), airport, matched, lirf_nonpop,
                                f["v9"].to_numpy(float), f["residual"].to_numpy(float), f["committee"].to_numpy(float), f["nets"].to_numpy(float),
                                f["lirf_tree"].to_numpy(float), f["lirf_net"].to_numpy(float), f["label"].to_numpy(float)))
    return pd.concat(parts, ignore_index=True)


def rates(frame: pd.DataFrame) -> dict:
    out = {}
    for (pair, period), g in frame.groupby(["pair", "period"]):
        d = g["d"].to_numpy()
        out.setdefault(pair, {})[period] = {"rows": int(len(d)), "rms_d": float(np.sqrt(np.mean(d ** 2))), "mean_d": float(d.mean()),
                                            "p99_abs_d": float(np.quantile(np.abs(d), 0.99)),
                                            **{f"share_up_gt_{t}": float(np.mean(d > t)) for t in THRESHOLDS},
                                            **{f"share_down_gt_{t}": float(np.mean(d < -t)) for t in THRESHOLDS}}
    return out


def excess_groups(dev: pd.DataFrame, rank: pd.DataFrame, t: int = 600) -> list[dict]:
    """Groups whose ranking-month rate of |D| > t (each sign separately) exceeds every development month's rate by 2x and 30+ rows."""
    out = []
    for sign, name in [(1, "up"), (-1, "down")]:
        dv = dev.assign(x=sign * dev["d"] > t).groupby(["pair", "group", "period"])["x"].agg(["mean", "size"])
        rk = rank.assign(x=sign * rank["d"] > t).groupby(["pair", "group", "period"])["x"].agg(["mean", "size", "sum"])
        for (pair, group, period), row in rk.iterrows():
            if (pair, group) not in dv.index.droplevel(2):
                ref, months = float("nan"), 0
            else:
                ref_rows = dv.loc[(pair, group)]
                ref_rows = ref_rows[ref_rows["size"] >= 200]
                ref, months = (float(ref_rows["mean"].max()), len(ref_rows)) if len(ref_rows) else (float("nan"), 0)
            base = 0.0 if np.isnan(ref) else ref
            excess_rows = row["sum"] - base * row["size"]
            if excess_rows >= 30 and (np.isnan(ref) or row["mean"] >= 2 * ref):
                out.append({"pair": pair, "sign": name, "group": group, "period": period, "rows": int(row["size"]), "rows_beyond": int(row["sum"]),
                            "rate": float(row["mean"]), "max_dev_month_rate": ref, "dev_months_with_200_rows": months, "excess_rows": float(excess_rows)})
    return sorted(out, key=lambda e: -e["excess_rows"])


def side_bins(dev: pd.DataFrame) -> dict:
    """Development: per pair x sign x |D| bin, RMSE of each side, the least-squares weight on side b and b's closer share."""
    out = {}
    dev = dev.assign(bin=pd.cut(np.abs(dev["d"]), BINS, labels=LABELS, right=False).astype(str), sign=np.where(dev["d"] >= 0, "up", "down"))
    for (pair, period, sign, b), g in [*dev.groupby(["pair", "period", "sign", "bin"]), *((( p, "pooled", s, bb), gg) for (p, s, bb), gg in dev.groupby(["pair", "sign", "bin"]))]:
        y, a, bb = g["label"].to_numpy(), g["a"].to_numpy(), g["b"].to_numpy()
        d = bb - a
        out.setdefault(pair, {}).setdefault(period, {}).setdefault(sign, {})[b] = {
            "rows": int(len(g)), "rmse_a": float(np.sqrt(np.mean((y - a) ** 2))), "rmse_b": float(np.sqrt(np.mean((y - bb) ** 2))),
            "ls_weight_b": float(np.sum((y - a) * d) / np.sum(d ** 2)) if np.sum(d ** 2) > 0 else float("nan"),
            "b_closer_share": float(np.mean(np.abs(y - bb) < np.abs(y - a)))}
    return out


def top_days(rank: pd.DataFrame, flagged: list[dict], t: int = 600) -> dict:
    out = {}
    for e in flagged[:12]:
        sign = 1 if e["sign"] == "up" else -1
        g = rank[(rank["pair"] == e["pair"]) & (rank["group"] == e["group"]) & (rank["period"] == e["period"])]
        days = g.assign(x=sign * g["d"] > t).groupby("day")["x"].agg(["sum", "size"])
        out[f"{e['pair']}|{e['sign']}|{e['group']}|{e['period']}"] = {k: {"beyond": int(v["sum"]), "rows": int(v["size"])}
                                                                       for k, v in days.sort_values("sum", ascending=False).head(8).iterrows()}
    return out


def main() -> None:
    config, net, lirf_net, v33 = configs()
    rank, dev = ranking_frame(config, net, lirf_net), dev_frame(config, net, lirf_net, v33)
    flagged = excess_groups(dev, rank)
    report = {"created_at_utc": utc_now(), "labels_read": "development months only (fair set); no ranking label exists",
              "inputs": {p: sha256_file(ROOT / p) for p in [RANKING, FAIR]}, "code": sha256_file(Path(__file__)),
              "rates": {"development": rates(dev), "ranking": rates(rank)}, "excess_groups_abs_d_gt_600": flagged,
              "top_days": top_days(rank, flagged), "dev_side_bins": side_bins(dev)}
    out = ROOT / (sys.argv[sys.argv.index("--report") + 1] if "--report" in sys.argv else "docs/disagreement/audit2.json")
    write_json(out, report)
    rank.to_parquet(ROOT / "runs/disagreement2_rank.parquet", index=False)
    dev.to_parquet(ROOT / "runs/disagreement2_dev.parquet", index=False)
    for pair in ["nets_vs_committee", "lirf_net_vs_tree", "residual"]:
        table = pd.DataFrame({**{f"dev {k}": v for k, v in report["rates"]["development"][pair].items()},
                              **{f"rank {k}": v for k, v in report["rates"]["ranking"][pair].items()}})
        print(pair); print(table.round(4).to_string())
    print("groups with ranking-only excess (|D| > 600):")
    print(pd.DataFrame(flagged).round(4).head(25).to_string())


if __name__ == "__main__":
    main()
