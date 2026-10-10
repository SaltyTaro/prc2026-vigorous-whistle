"""Tier B, model part (configs/febjun_tierb_v1.json; readings declared in configs/febjun_tierb_readings_model_v1.json; record
docs/final_file/tierb_model.md): February / June 2025 out-of-sample rows in the development layout of runs/lomo_dev/rows.parquet and the
model part's layers v30 -> v55m on them, through the development code that selected each layer.

Subcommands (outputs under runs/febjun_tierb/model/):
  trees   --months M ... [--check]   tree tables (ens_v20_bag_q_all, ens_v22_lirf_q) from the LOMO leaves with build_lomo_tables' member rule;
                                     --check rebuilds January / July from runs/lomo_dev/learners and compares with runs/lomo_dev/ens
  nets    --months M ... [--check]   network seed means (context families and the three members, unrouted) of the Tier B 4-seed fits;
                                     --check: January / July from the 16-seed LOMO fits against runs/lomo_dev/rows.parquet and the members'
                                     stored routed tables
  rematch --months M ... [--check]   the callsign fix's input (tools/adsb_rematch_nonm.py via prc2026_final.stack.rematch) on the month's
                                     no-NM EDDF / EHAM departures; --check: January 2025 against the declared out_arcw5.parquet rows
  rows    --months M ... [--check]   the development row layout; --check: the frame path on January / July (fold tables) against
                                     runs/fair_dev/rows_shipped.parquet, and the assembly on the LOMO components against runs/lomo_dev/rows.parquet
  chain   --months M ... [--check]   every layer column; --check: on runs/lomo_dev/rows.parquet January / July with the stored member tables,
                                     the stored development RMSEs of v30 / v33 / v45 / v46 / v51m / v52m / v53m / v55m
  read    --months 2025-02 2025-06   the declared readings (refuses unless the readings declaration is committed and unchanged)
No label enters a component; labels are read only by the checks (January / July, already used by every shipped selection) and by `read`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, PHASE, TARGET, TIME  # noqa: E402
from prc2026_december.scoring import ensemble_prediction, leaf_members  # noqa: E402
from prc2026_final import lirf_members, network_member, share_gate  # noqa: E402
from prc2026_final import stack as runner_stack  # noqa: E402
import run_adsb_v24  # noqa: E402,F401  (registers the lirf_nonpop_unmatched scope)
from run_adsb_nn2_dev import weighted  # noqa: E402
from run_adsb_rnn_dev import combine, residual_columns  # noqa: E402
from run_adsb_v11 import SCOPES, rmse  # noqa: E402
from run_adsb_v24 import adsb_features, extra_columns, load_frame, month_table  # noqa: E402
from run_callsign_fix import offsets, scope_table  # noqa: E402
from run_fair_weights import build, candidate_weights  # noqa: E402
from run_lomo_rows import FAMILIES, assemble, configs  # noqa: E402
from run_mix_gate import stack as dev_stack  # noqa: E402
from run_queue_trust import offsets_all, segments  # noqa: E402
from run_share_gate_reread import T_LOWER, T_UPPER, full_stack  # noqa: E402
import run_intruder_bound as R  # noqa: E402

OUT = ROOT / "runs/febjun_tierb/model"
TIERB = ["2025-02", "2025-06"]
DEV = ["2025-01", "2025-07"]
TABLES = {"ens_v20_bag_q_all": "runs/adsb_v20/ens_v20_bag_q_all", "ens_v22_lirf_q": "runs/adsb_v22/ens_v22_lirf_q"}
UMBRELLA = "configs/febjun_tierb_v1.json"
READINGS = "configs/febjun_tierb_readings_model_v1.json"
ADSB_CACHE = ROOT / "runs/adsb_ctx2/inputs/adsb_features_arcw0"  # the learner-directory match flags the member fits read (build_combo_member)
REMATCH_DECL = "configs/callsign_fix_v1.json"
STRETCH = "docs/disagreement/tail_stretch_v2_results.json"
GATE_DECL = "configs/lirf_gate_v1.json"
SHARE_DECL = "configs/thinned_regime_round_v1.json"
SEED = 20261004
LAYERS = ["off_r3", "v30", "v31", "v33", "v34", "v35", "v38", "v40", "v45", "v46", "v51m", "v52m", "v53m", "v54m", "v55m"]
# Tier B network seeds (configs/febjun_tierb_v1.json base.networks_cheap): ensemble -> (directory variant, seeds)
TIERB_NETS = {"ens_ctx_e24": ("ctx_e24", [117, 118, 119, 120]), "ens_ctx_e24b": ("ctx_e24", [133, 134, 135, 136]),
              "ens_ctx_e32": ("ctx_e32", [401, 402, 403, 404]), "ens_ctx_lirf": ("ctx_lirf", [301, 302, 303, 304]),
              "combo_dl32": ("dl32", [601, 602, 603, 604]), "combo_m25dl32": ("m25dl32", [601, 602, 603, 604]),
              "mask_m25_32": ("m25_32", [701, 702, 703, 704]), "lirf_lm25_32": ("lm25_32", [801, 802, 803, 804])}
TIERB_NET_DIR = "runs/febjun_nets/nets"
# the 16-seed LOMO fits behind the January / July development tables
DEV_NETS = {**{name: (f"runs/lomo_dev/nets/{variant}", list(seeds)) for name, (variant, seeds) in FAMILIES.items()},
            "combo_dl32": ("runs/adsb_ctx3_combo/nets/dl32", list(range(601, 617))), "combo_m25dl32": ("runs/adsb_ctx3_combo/nets/m25dl32", list(range(601, 617))),
            "mask_m25_32": ("runs/adsb_ctx3_mask/nets/m25_32", list(range(701, 717))), "lirf_lm25_32": ("runs/adsb_ctx3_lirf_mask/nets/lm25_32", list(range(801, 817)))}


def name_of(leaf: str) -> str:
    return leaf.replace("runs/", "").replace("/", "__")


def raw_month(month: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data/raw" / runner_stack.source_file(month))


def month_departures(month: str, columns: list[str] | None = None) -> pd.DataFrame:
    raw = raw_month(month)
    dep = raw[raw[PHASE].eq("DEP") & raw[TIME].dt.strftime("%Y-%m").eq(month)].reset_index(drop=True)
    return dep if columns is None else dep[columns]


# ---------------------------------------------------------------- trees

def tree_tables(month: str, learners: Path) -> tuple[dict[str, pd.Series], dict]:
    """{table: Series(ID -> value)} for one held month from the LOMO leaves (nanmean with multiplicities, build_lomo_tables' rule)."""
    out, inputs = {}, {}
    for table, run_dir in TABLES.items():
        members = leaf_members(ROOT, run_dir)
        values, ids = {}, None
        for leaf in sorted(set(members)):
            path = learners / name_of(leaf) / f"lomo_{month}" / "predictions.parquet"
            part = pd.read_parquet(path)
            if part[ID].duplicated().any():
                raise ValueError(f"{path}: repeated ids.")
            if ids is None:
                ids = part[ID].to_numpy()
            elif set(part[ID]) != set(ids):
                raise ValueError(f"{path}: the leaves predict different rows.")
            values[leaf] = part.set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
            inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        out[table] = pd.Series(ensemble_prediction(members, values), index=ids)
    return out, inputs


def cmd_trees(months: list[str], check: bool) -> None:
    if check:
        rep = {}
        for month in DEV:
            built, _ = tree_tables(month, ROOT / "runs/lomo_dev/learners")
            for table, series in built.items():
                stored = pd.read_parquet(ROOT / f"runs/lomo_dev/ens/{table}/lomo/predictions.parquet").set_index(ID)["adsb_full"]
                ref = stored.reindex(series.index).to_numpy(dtype=float)
                v = series.to_numpy(dtype=float)
                same_nan = bool((np.isfinite(ref) == np.isfinite(v)).all())
                rep[f"{table}:{month}"] = {"rows": int(len(v)), "finite_pattern_equal": same_nan, "max_abs_s": float(np.nanmax(np.abs(v - ref)))}
                if not (same_nan and rep[f"{table}:{month}"]["max_abs_s"] <= 1e-9):
                    raise ValueError(f"tree table check fails: {table} {month} {rep[f'{table}:{month}']}")
        print("trees check", rep)
        write_json(OUT / "checks" / "trees.json", {"created_at_utc": utc_now(), "check": rep})
        return
    for month in months:
        built, inputs = tree_tables(month, ROOT / "runs/febjun_lomo/learners")
        for table, series in built.items():
            target = OUT / "trees" / f"{table}_{month}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({ID: series.index.to_numpy(), "adsb_full": series.to_numpy(dtype=float)}).to_parquet(target, index=False)
            print(month, table, len(series), "finite", int(np.isfinite(series.to_numpy(dtype=float)).sum()))
        write_json(OUT / "trees" / f"inputs_{month}.json", {"created_at_utc": utc_now(), "inputs": inputs})


# ---------------------------------------------------------------- networks

def net_means(month: str, spec: dict, base: str | None, fold: str) -> tuple[pd.DataFrame, dict]:
    """Seed means per ensemble on the union of predicted rows; spec: name -> (directory prefix or variant, seeds)."""
    cols, inputs, ids = {}, {}, None
    for name, (prefix, seeds) in spec.items():
        tables = []
        for seed in seeds:
            path = ROOT / (f"{base}/{prefix}_s{seed}" if base else f"{prefix}_s{seed}") / fold / "predictions.parquet"
            part = pd.read_parquet(path)
            if ids is None:
                ids = part[ID].to_numpy()
            elif set(part[ID]) != set(ids):
                raise ValueError(f"{path}: predicts different rows.")
            tables.append(part.set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float))
            inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        cols[name] = network_member.seed_mean(tables)
    return pd.DataFrame({ID: ids, **cols}), inputs


def matched_flags() -> pd.Series:
    import build_combo_member
    flags, _ = build_combo_member.matched_flags(ADSB_CACHE)
    return flags


def routed(frame: pd.DataFrame, flags: pd.Series) -> pd.DataFrame:
    """members.members' columns: combo_all (dl32 matched / m25dl32 unmatched), mask_unmatched (m25_32 on unmatched), lirf_mask (unrouted)."""
    matched = flags.reindex(frame[ID])
    if matched.isna().any():
        raise ValueError("Rows without a learner-directory match flag.")
    matched = matched.to_numpy(dtype=bool)
    out = frame[[ID]].copy()
    out["adsb_matched_learner"] = matched
    out["combo_all"] = np.where(matched, frame["combo_dl32"].to_numpy(dtype=float), frame["combo_m25dl32"].to_numpy(dtype=float))
    out["mask_unmatched"] = np.where(matched, np.nan, frame["mask_m25_32"].to_numpy(dtype=float))
    out["lirf_mask"] = frame["lirf_lm25_32"].to_numpy(dtype=float)
    return out


def stored_member(decl_path: str, month: str, ids: np.ndarray) -> np.ndarray:
    decl = read_json(ROOT / decl_path)
    tables = [pd.read_parquet(ROOT / rel).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float) for rel in decl["member"]["tables"][month]]
    return network_member.seed_mean(tables)


def cmd_nets(months: list[str], check: bool) -> None:
    if check:
        rows = pd.read_parquet(ROOT / "runs/lomo_dev/rows.parquet")
        config, net, lirf_net, _ = configs()
        flags, rep = matched_flags(), {}
        for month in DEV:
            f = rows[rows["month"] == month].reset_index(drop=True)
            ids = f[ID].to_numpy()
            ctx = {n: s for n, s in DEV_NETS.items() if n.startswith("ens_")}
            means, _ = net_means(month, {n: (p, s) for n, (p, s) in ctx.items()}, None, f"lomo_{month}")
            means = means.set_index(ID).reindex(ids)
            nets = weighted(means.reset_index(), net["network"])
            lirf = weighted(means.reset_index(), lirf_net)
            mem, _ = net_means(month, {n: s for n, s in DEV_NETS.items() if not n.startswith("ens_")}, None, f"lomo_{month}")
            r = routed(mem, flags).set_index(ID).reindex(ids)
            combo_ref, mask_ref = stored_member("configs/adsb_ctx3_combo_dev_v1.json", month, ids), stored_member("configs/adsb_ctx3_mask_dev_v1.json", month, ids)
            def diff(a, b):
                a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
                return {"finite_pattern_equal": bool((np.isfinite(a) == np.isfinite(b)).all()), "max_abs_s": float(np.nanmax(np.abs(a - b)))}
            rep[month] = {"nets": diff(nets, f["nets"]), "lirf_net": diff(lirf, f["lirf_net"]), "combo_all": diff(r["combo_all"], combo_ref),
                          "mask_unmatched": diff(r["mask_unmatched"], mask_ref)}
            for k, v in rep[month].items():
                if not (v["finite_pattern_equal"] and v["max_abs_s"] <= 1e-6):
                    raise ValueError(f"nets check fails: {month} {k} {v}")
        print("nets check", rep)
        write_json(OUT / "checks" / "nets.json", {"created_at_utc": utc_now(), "check": rep, "adsb_cache": str(ADSB_CACHE.relative_to(ROOT))})
        return
    flags = matched_flags()
    for month in months:
        means, inputs = net_means(month, TIERB_NETS, TIERB_NET_DIR, f"lomo_{month}")
        means = means.merge(routed(means, flags), on=ID, validate="one_to_one")
        target = OUT / "nets" / f"{month}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        means.to_parquet(target, index=False)
        write_json(OUT / "nets" / f"inputs_{month}.json", {"created_at_utc": utc_now(), "seeds": {n: s for n, (_, s) in TIERB_NETS.items()}, "inputs": inputs})
        print(month, len(means), {c: int(np.isfinite(means[c].to_numpy(dtype=float)).sum()) for c in means.columns if c != ID})


# ---------------------------------------------------------------- callsign fix input

def cmd_rematch(months: list[str], check: bool) -> None:
    for month in (["2025-01"] if check else months):
        dep = month_departures(month, runner_stack.REMATCH_COLUMNS)
        deps = runner_stack.rematch_departures(dep)
        folder = OUT / ("checks" if check else "rematch")
        folder.mkdir(parents=True, exist_ok=True)
        deps_path, out_path = folder / f"deps_{month}.parquet", folder / f"rematch_{month}.parquet"
        deps.to_parquet(deps_path, index=False)
        runner_stack.rematch(deps_path, out_path)
        out = pd.read_parquet(out_path)
        days = sorted(deps[TIME].dt.strftime("%Y%m%d").unique().tolist())
        missing = [d for d in days if not (ROOT / "data/external/adsblol/surface" / f"surface_{d}.parquet").exists()]
        rec = {"month": month, "departures": int(len(deps)), "rows_out": int(len(out)), "surface_days_missing": missing, "sha256": sha256_file(out_path)}
        if check:
            ref = pd.read_parquet(ROOT / "runs/adsb_rematch/out_arcw5.parquet")
            ref = ref[ref[ID].isin(out[ID])].set_index(ID).sort_index()
            mine = out.set_index(ID).reindex(ref.index)
            mine_scope = scope_of(out)
            ref_scope = scope_table(read_json(ROOT / REMATCH_DECL))
            ref_scope = ref_scope[ref_scope[ID].isin(deps[ID])]
            cols = [c for c in ref.columns if c in mine.columns]
            diffs = {}
            for c in cols:
                a, b = mine[c], ref[c]
                if pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b):
                    diffs[c] = float(np.nanmax(np.abs(a.to_numpy(dtype=float) - b.to_numpy(dtype=float)))) if len(b) else 0.0
                else:  # missing values compare equal (None / NaN representations differ)
                    diffs[c] = int((~((a.isna() & b.isna()) | (a == b).fillna(False).astype(bool))).sum())
            rec.update({"reference_rows": int(len(ref)), "column_differences": diffs, "scope_rows": int(len(mine_scope)), "reference_scope_rows": int(len(ref_scope)),
                        "scope_ids_equal": bool(set(mine_scope[ID]) == set(ref_scope[ID]))})
            write_json(OUT / "checks" / "rematch.json", {"created_at_utc": utc_now(), "check": rec})
        print(rec)


def scope_of(m: pd.DataFrame) -> pd.DataFrame:
    """run_callsign_fix.scope_table's rule on a rematch output."""
    scope = m["key_icao"].isna() & m["iata_mapped_icao"].notna() & m["iata_mapped_gate"].fillna(False).astype(bool) & m["iata_mapped_motion_start"].notna()
    return m.loc[scope, [ID, "airport", "takeoff", "iata_mapped_motion_start"]].rename(columns={"iata_mapped_motion_start": "motion_start"})


# ---------------------------------------------------------------- rows

def frame_rows(month: str, tables: dict[str, np.ndarray] | None) -> tuple[pd.DataFrame, dict]:
    """The development layout from the cross-fit frame (scratch/v24), the shipped-provenance residual and the given blend tables
    (None: the fold tables of extra_columns, January / July only)."""
    config, net, lirf_net, v33 = configs()
    table = month_table(load_frame(config, month, "cross_fit"), adsb_features(month, ROOT / config["matches_dirs"]["v11"], ROOT / "runs/adsb_v24/features"), month)
    ids = table[ID]
    residual = combine(config["candidates"]["rnet_w90"], residual_columns(config, month, ids, smoke=False), "rnet_w90")
    blend = extra_columns(config, month, ids.to_numpy()) if tables is None else tables
    f = pd.DataFrame({ID: ids.to_numpy(), "month": month, "day": table[TIME].dt.strftime("%Y-%m-%d").to_numpy(), "airport": table["airport"].to_numpy(),
                      "label": table[TARGET].to_numpy(dtype=float), "floor": float(table["floor"].iloc[0]), "matched": table["adsb_matched"].eq(1).to_numpy(),
                      "lirf_m": np.asarray(SCOPES["lirf_nonpop_matched"](table), dtype=bool), "lirf_u": np.asarray(SCOPES["lirf_nonpop_unmatched"](table), dtype=bool)})
    if (np.asarray(f["floor"]) != np.asarray(table["floor"], dtype=float)).any():
        raise ValueError("The floor is not constant within the month.")
    # LIRF rows outside both non-population scopes (the fill population; none in the cross-fit tables of February / June) keep their
    # recipe value, which run_adsb_v11.candidate_values leaves at max(v9, floor) ("fixed" scope)
    fixed = (f["airport"].to_numpy() == "LIRF") & ~f["lirf_m"].to_numpy() & ~f["lirf_u"].to_numpy()
    f["v33"] = np.where(fixed, np.maximum(table["v9"].to_numpy(dtype=float), f["floor"].to_numpy(dtype=float)), np.nan)
    g = assemble(f, table["v9"].to_numpy(dtype=float), residual, blend, config, net, lirf_net, v33)
    extras = {"population_rows": int(np.asarray(table["population"], dtype=bool).sum()), "clock_share": float(np.mean(table["clock_available"]))}
    return g, extras


def cmd_rows(months: list[str], check: bool) -> None:
    if check:
        rep = {}
        shipped = pd.read_parquet(ROOT / "runs/fair_dev/rows_shipped.parquet")
        lomo = pd.read_parquet(ROOT / "runs/lomo_dev/rows.parquet")
        config, net, lirf_net, v33 = configs()
        for month in DEV:
            g, _ = frame_rows(month, None)
            ref = shipped[shipped["month"] == month].set_index(ID).reindex(g[ID])
            out = {}
            for c in ["label", "floor", "v9", "residual", "base", "mix", "lirf_tree", "lirf_net", "v30", "v33"]:
                a, b = g[c].to_numpy(dtype=float), ref[c].to_numpy(dtype=float)
                out[c] = float(np.nanmax(np.abs(a - b))) if (np.isfinite(a) == np.isfinite(b)).all() else float("inf")
            for c in ["matched", "lirf_m", "lirf_u", "day", "airport"]:
                out[c] = int((g[c].to_numpy() != ref[c].to_numpy()).sum())
            out["rows"], out["reference_rows"] = int(len(g)), int((shipped["month"] == month).sum())
            rep[f"frame_path:{month}"] = out
            if not (out["rows"] == out["reference_rows"] and all(v <= 1e-9 for k, v in out.items() if k not in ("rows", "reference_rows"))):
                raise ValueError(f"frame path check fails on {month}: {out}")
            # the assembly on the LOMO components (tree tables rebuilt from the leaves, 16-seed network means) against runs/lomo_dev/rows.parquet
            f = lomo[lomo["month"] == month].reset_index(drop=True)
            ids = f[ID].to_numpy()
            built, _ = tree_tables(month, ROOT / "runs/lomo_dev/learners")
            means, _ = net_means(month, {n: s for n, s in DEV_NETS.items() if n.startswith("ens_")}, None, f"lomo_{month}")
            means = means.set_index(ID).reindex(ids)
            tables = {t: s.reindex(ids).to_numpy(dtype=float) for t, s in built.items()}
            tables.update({n: means[n].to_numpy(dtype=float) for n in means.columns})
            h = assemble(f.drop(columns=["base", "mix", "lirf_tree", "lirf_net", "v30", "committee", "nets"]), f["v9"].to_numpy(dtype=float),
                         f["residual"].to_numpy(dtype=float), tables, config, net, lirf_net, v33)
            out2 = {c: float(np.nanmax(np.abs(h[c].to_numpy(dtype=float) - f[c].to_numpy(dtype=float)))) for c in ["base", "mix", "lirf_tree", "lirf_net", "v30", "v33", "committee", "nets"]}
            rep[f"lomo_assembly:{month}"] = out2
            if not all(v <= 1e-9 for v in out2.values()):
                raise ValueError(f"LOMO assembly check fails on {month}: {out2}")
        print("rows check", rep)
        write_json(OUT / "checks" / "rows.json", {"created_at_utc": utc_now(), "check": rep})
        return
    for month in months:
        trees = {t: pd.read_parquet(OUT / "trees" / f"{t}_{month}.parquet").set_index(ID)["adsb_full"] for t in TABLES}
        nets = pd.read_parquet(OUT / "nets" / f"{month}.parquet").set_index(ID)
        config, *_ = configs()
        frame = load_frame(config, month, "cross_fit")
        ids = frame[ID].to_numpy()
        tables = {t: s.reindex(ids).to_numpy(dtype=float) for t, s in trees.items()}
        tables.update({n: nets[n].reindex(ids).to_numpy(dtype=float) for n in ["ens_ctx_e24", "ens_ctx_e24b", "ens_ctx_e32", "ens_ctx_lirf"]})
        g, extras = frame_rows(month, tables)
        if not np.array_equal(g[ID].to_numpy(), ids):
            raise ValueError("Row order changed.")
        for c in ["combo_all", "mask_unmatched", "lirf_mask", "adsb_matched_learner"]:
            g[c] = nets[c].reindex(ids).to_numpy()
        for n in ["ens_ctx_e24", "ens_ctx_e24b", "ens_ctx_e32", "ens_ctx_lirf"]:
            g[n] = tables[n]
        for c in ["v9", "base", "mix", "lirf_tree", "lirf_net", "v30", "v33", "committee", "nets"]:
            if not np.isfinite(g[c].to_numpy(dtype=float)).all():
                raise ValueError(f"{month}: {c} has non-finite values.")
        dep = month_departures(month, [ID, "ADEP_mvt", "FLIGHT_ID_mvt", "AOBT_3_flt", "LOBT_flt"])
        excluded = dep[~dep[ID].isin(ids)].assign(reason="no out-of-sample v9 (missing NM clock or LIRF fill population; scratch/v24 holds clock-present rows only)")
        excluded[[ID, "ADEP_mvt", "reason"]].to_parquet(OUT / f"excluded_{month}.parquet", index=False)
        target = OUT / f"rows_{month}.parquet"
        g.to_parquet(target, index=False)
        print(month, {"rows": len(g), "excluded": len(excluded), **extras, "sha256": sha256_file(target)})


# ---------------------------------------------------------------- chain

def stack_steps(f: pd.DataFrame, mix: np.ndarray, cs, off_cs, seg, off, k: dict, gate_spec: dict) -> dict[str, np.ndarray]:
    """run_share_gate_reread.full_stack split at its layer boundaries: v35 (run_mix_gate.stack), v38 (two-sided stretch), v45 (LIRF gate)."""
    from prc2026_final import lirf_gate
    values = dev_stack(f, mix, cs, off_cs, seg, off)
    out = {"v35": values.copy()}
    q = seg.reindex(f[ID])
    anchored = cs.reindex(f[ID])["motion_start"].notna().to_numpy() | (q["in_scope"].fillna(False).to_numpy(dtype=bool) & q["source"].eq("regular").to_numpy())
    non_lirf, matched, floor = f["airport"].to_numpy() != "LIRF", f["matched"].to_numpy(), f["floor"].to_numpy(dtype=float)
    for scope, m in (("untracked", non_lirf & ~matched & ~anchored), ("tracked", non_lirf & matched & ~anchored)):
        k1, k2 = k[scope]
        u = values[m] + k1 * np.maximum(values[m] - T_UPPER, 0.0)
        u = u + k2 * np.maximum(T_LOWER - u, 0.0)
        values[m] = np.maximum(u, floor[m])
    out["v38"] = values.copy()
    for flag, scope in (("lirf_u", "unmatched"), ("lirf_m", "matched")):
        m = f[flag].to_numpy() & ~non_lirf
        if not m.any():
            continue
        components = [f.loc[m, c].to_numpy(dtype=float) for c in ["v9", "lirf_net", "lirf_tree"]]
        blend, _ = lirf_gate.gated(*components, scope, None)
        gated, fire = lirf_gate.gated(*components, scope, gate_spec.get(scope))
        idx = np.flatnonzero(m)[fire]
        values[idx] = np.maximum(values[idx] + (gated - blend)[fire], lirf_gate.FLOOR)
    out["v45"] = values
    return out


def chain(f: pd.DataFrame, member: dict[str, np.ndarray], cs, off_cs, seg, off) -> tuple[dict[str, np.ndarray], dict]:
    """Every layer column on one month's rows; member: combo_all, mask_unmatched, lirf_mask in f's order."""
    k = read_json(ROOT / STRETCH)["candidates"]["two_sided_900"]["k_all_months"]
    gate_spec = read_json(ROOT / GATE_DECL)["candidates"]["u600_m600"]
    control = read_json(ROOT / "configs/fair_weights_v1.json")["control"]
    outer34, lirf34 = candidate_weights(control, {"lirf_v9": {"lirf_nonpop_unmatched": 0.0}})
    floor = f["floor"].to_numpy(dtype=float)
    committee, nets = f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float)
    lirf = f["airport"].to_numpy() == "LIRF"
    cols = {"off_r3": np.maximum(f["base"].to_numpy(dtype=float), floor), "v30": f["v30"].to_numpy(dtype=float)}
    cols["v31"] = cols["v30"].copy()  # the EOBT_1 detector's threshold is the 2025 maximum plus span: no 2025 month can be flagged
    cols["v33"] = build(f, control["outer"], control["lirf"])
    if not float(np.max(np.abs(cols["v33"] - f["v33"].to_numpy(dtype=float)))) <= 1e-9:
        raise ValueError("The v33 rebuild does not equal the rows' v33.")
    cols["v34"] = build(f, outer34, lirf34)
    steps = stack_steps(f, f["mix"].to_numpy(dtype=float), cs, off_cs, seg, off, k, gate_spec)
    ref = full_stack(f, f["mix"].to_numpy(dtype=float), cs, off_cs, seg, off, k, gate_spec)
    checks = {"v45_vs_full_stack_max_abs": float(np.max(np.abs(steps["v45"] - ref)))}
    if not checks["v45_vs_full_stack_max_abs"] == 0.0:
        raise ValueError(f"The split stack does not equal full_stack: {checks}")
    cols.update({"v35": steps["v35"], "v38": steps["v38"]})
    cols["v40"] = cols["v38"].copy()  # fill pins off (the user's decision for the final file)
    cols["v45"] = steps["v45"]
    share, spec = share_gate.candidate_spec(read_json(ROOT / SHARE_DECL), "n85_nd1200")
    mix46, _ = share_gate.mix(committee, nets, share, spec)
    cols["v46"] = full_stack(f, mix46, cs, off_cs, seg, off, k, gate_spec)
    t51, in51 = network_member.table_with_member(nets, np.where(lirf, np.nan, member["combo_all"]), 0.40)
    t52, in52 = network_member.table_with_member(t51, np.where(lirf, np.nan, member["mask_unmatched"]), 0.40)
    for name, tab in (("v51m", t51), ("v52m", t52)):
        mix_m, _ = share_gate.mix(committee, tab, share, spec)
        cols[name] = full_stack(f, mix_m, cs, off_cs, seg, off, k, gate_spec)
    scope = np.where(f["lirf_m"].to_numpy() & lirf, "lirf_nonpop_matched", np.where(f["lirf_u"].to_numpy() & lirf, "lirf_nonpop_unmatched", "other"))
    lirf_d, lirf_rec = lirf_members.deltas(f["v9"].to_numpy(dtype=float), f["lirf_net"].to_numpy(dtype=float), f["lirf_tree"].to_numpy(dtype=float),
                                           np.where(np.isin(scope, ["lirf_nonpop_matched", "lirf_nonpop_unmatched"]), member["lirf_mask"], np.nan), scope, f["month"].to_numpy())
    previous = cols["v52m"]
    for name in ["v53m", "v54m", "v55m"]:
        route, delta, ctrl = lirf_d[name]
        values = previous.copy()
        values[route] = np.maximum(previous[route] + delta, lirf_members.lirf_gate.FLOOR)
        lirf_rec[name]["base_minus_control_max_abs_s"] = float(np.max(np.abs(previous[route] - ctrl))) if route.any() else 0.0
        cols[name] = values
        previous = values
    for name, v in cols.items():
        if not np.isfinite(v).all():
            raise ValueError(f"{name} has non-finite values.")
    rows = {"member_rows_v51m": int(in51.sum()), "member_rows_v52m": int((in52 & np.isfinite(member["mask_unmatched"])).sum()),
            **{f"moved_{b}_{a}": int((np.abs(cols[b] - cols[a]) > 1e-9).sum()) for a, b in zip(LAYERS[:-1], LAYERS[1:])}}
    return cols, {"checks": checks, "rows": rows, "lirf": lirf_rec}


def queue_inputs(months: list[str], rematch_files: dict[str, Path] | None):
    """(cs, off_cs, seg, off): the declared ones for the development months, the month's own rematch for Tier B months."""
    if rematch_files is None:
        return scope_table(read_json(ROOT / REMATCH_DECL)).set_index(ID), offsets(), segments(["2025-01", "2025-07", "2025-11", "2025-12"]), offsets_all()
    rem = pd.concat([pd.read_parquet(p) for p in rematch_files.values()], ignore_index=True)
    cs = scope_of(rem)
    regular = pd.concat([pd.read_parquet(ROOT / f"data/external/adsblol/matches_v11/matches2_{m}.parquet") for m in months], ignore_index=True)
    seg = runner_stack.queue_segments(regular, rem, set(cs[ID]))
    return cs.set_index(ID), offsets(), seg, offsets_all()


def cmd_chain(months: list[str], check: bool) -> None:
    if check:
        rows = pd.read_parquet(ROOT / "runs/lomo_dev/rows.parquet")
        cs, off_cs, seg, off = queue_inputs(DEV, None)
        # the runner's segment rule on the declared rematch outputs must equal run_queue_trust.segments on the development months
        rem = pd.concat([pd.read_parquet(ROOT / p) for p in read_json(ROOT / REMATCH_DECL)["rematch"]["outputs"]], ignore_index=True)
        regular = pd.concat([pd.read_parquet(ROOT / f"data/external/adsblol/matches_v11/matches2_{m}.parquet") for m in ["2025-01", "2025-07", "2025-11", "2025-12"]], ignore_index=True)
        seg2 = runner_stack.queue_segments(regular, rem, set(cs.index))
        seg_check = {"rows": int(len(seg)), "rows_runner_rule": int(len(seg2)), "index_equal": bool(seg.sort_index().index.equals(seg2.sort_index().index)),
                     "in_scope_equal": bool((seg.sort_index()["in_scope"] == seg2.sort_index()["in_scope"]).all()),
                     "taxi_max_abs": float(np.nanmax(np.abs(seg.sort_index()["taxi"].to_numpy(dtype=float) - seg2.sort_index()["taxi"].to_numpy(dtype=float))))}
        cs2 = scope_of(rem).set_index(ID)
        seg_check["callsign_scope_equal"] = bool(cs2.sort_index().equals(cs.sort_index()))
        if not (seg_check["index_equal"] and seg_check["in_scope_equal"] and seg_check["taxi_max_abs"] == 0.0 and seg_check["callsign_scope_equal"]):
            raise ValueError(f"segment rule check fails: {seg_check}")
        reread = read_json(ROOT / "docs/thinned_regime/reread_n85_nd1200.json")["months"]
        combo = read_json(ROOT / "docs/adsb_ctx3_combo/results_v1.json")["candidates"]["w40"]["months"]
        mask = read_json(ROOT / "docs/adsb_ctx3_mask/results_v1.json")["candidates"]["w40"]["months"]
        lmask = read_json(ROOT / "docs/adsb_ctx3_lirf_mask/results_v1.json")["results"]["w55"]
        lcombo = read_json(ROOT / "docs/lirf_combo/results_v1.json")["results"]["mw40"]
        summary = read_json(ROOT / "docs/lomo_dev/rows_summary.json")["months"]
        rep = {"segments": seg_check}
        for month in DEV:
            f = rows[rows["month"] == month].reset_index(drop=True)
            ids = f[ID].to_numpy()
            lm, _ = net_means(month, {"lirf_lm25_32": DEV_NETS["lirf_lm25_32"]}, None, f"lomo_{month}")
            member = {"combo_all": stored_member("configs/adsb_ctx3_combo_dev_v1.json", month, ids), "mask_unmatched": stored_member("configs/adsb_ctx3_mask_dev_v1.json", month, ids),
                      "lirf_mask": lm.set_index(ID)["lirf_lm25_32"].reindex(ids).to_numpy(dtype=float)}
            cols, record = chain(f, member, cs, off_cs, seg, off)
            y = f["label"].to_numpy(dtype=float)
            u = f["lirf_u"].to_numpy() & (f["airport"].to_numpy() == "LIRF") & (y < 20000.0)
            mm = f["lirf_m"].to_numpy() & (f["airport"].to_numpy() == "LIRF") & (y < 20000.0)
            got = {"v30": rmse(y, cols["v30"]), "v33": rmse(y, cols["v33"]), "v45": rmse(y, cols["v45"]), "v46": rmse(y, cols["v46"]), "v51m": rmse(y, cols["v51m"]),
                   "v52m": rmse(y, cols["v52m"]), "v53m_route_primary": rmse(y[u], cols["v53m"][u])}
            want = {"v30": summary[month]["v30_lomo"], "v33": summary[month]["v33_lomo"], "v45": reread[month]["control_rmse"], "v46": reread[month]["candidate_rmse"],
                    "v51m": combo[month]["rmse"], "v52m": mask[month]["rmse"], "v53m_route_primary": lmask[f"natural:{month}"]["primary"]["rmse"]}
            if month in record["lirf"]["dark_months"]:  # the dark re-read (v54m) acts on this month: its route RMSE as the dark round read it
                got["v54m_route_primary"] = rmse(y[u], cols["v54m"][u])
                want["v54m_route_primary"] = read_json(ROOT / "docs/lirf_dark/results_v1.json")["results"]["w70_s750"][f"natural:{month}"]["primary"]["rmse"]
            if f"natural:{month}" in lcombo:
                got["v55m_route_primary"] = rmse(y[mm], cols["v55m"][mm])
                want["v55m_route_primary"] = lcombo[f"natural:{month}"]["primary"]["rmse"]
            diffs = {k2: abs(got[k2] - want[k2]) for k2 in want}
            rep[month] = {"rmse": got, "stored": want, "abs_difference": diffs, "record": record, "member_seeds": 16}
            print(month, {k2: f"{got[k2]:.6f} vs {want[k2]:.6f}" for k2 in want}, flush=True)
            if not all(v <= 1e-6 for v in diffs.values()):
                raise ValueError(f"chain check fails on {month}: {diffs}")
        write_json(OUT / "checks" / "chain.json", {"created_at_utc": utc_now(), "check": rep})
        return
    files = {m: OUT / "rematch" / f"rematch_{m}.parquet" for m in months}
    cs, off_cs, seg, off = queue_inputs(months, files)
    out_rec = {}
    for month in months:
        f = pd.read_parquet(OUT / f"rows_{month}.parquet")
        member = {c: f[c].to_numpy(dtype=float) for c in ["combo_all", "mask_unmatched", "lirf_mask"]}
        cols, record = chain(f, member, cs, off_cs, seg, off)
        g = f.copy()
        for name in LAYERS:
            g[name] = cols[name]
        target = OUT / f"chain_{month}.parquet"
        g.to_parquet(target, index=False)
        record["callsign_scope_rows"] = int(f[ID].isin(cs.index).sum())
        record["queue_regular_in_scope_rows"] = int((seg.reindex(f[ID])["in_scope"].fillna(False).to_numpy(dtype=bool) & seg.reindex(f[ID])["source"].eq("regular").to_numpy()).sum())
        record["sha256"] = sha256_file(target)
        out_rec[month] = record
        print(month, record["rows"], record["sha256"])
    write_json(OUT / "chain_record.json", {"created_at_utc": utc_now(), "months": out_rec, "rematch": {m: sha256_file(p) for m, p in files.items()}})


# ---------------------------------------------------------------- readings

def committed_unchanged(path: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path], cwd=ROOT).returncode == 0
    return tracked and clean


def cmd_read(months: list[str]) -> None:
    if not committed_unchanged(READINGS):
        raise SystemExit(f"{READINGS} must be committed and unchanged before any reading.")
    decl = read_json(ROOT / READINGS)
    layers = decl["layers"]
    frames = {m: pd.read_parquet(OUT / f"chain_{m}.parquet") for m in months}
    result = {"created_at_utc": utc_now(), "declaration": READINGS, "declaration_sha256": sha256_file(ROOT / READINGS),
              "declaration_commit": subprocess.run(["git", "log", "-1", "--format=%H", "--", READINGS], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
              "chain_sha256": {m: sha256_file(OUT / f"chain_{m}.parquet") for m in months}, "seed": SEED, "replicates": 2000, "layers": {}}
    for layer in layers:
        on, off = layer["on"], layer["off"]
        rng = np.random.default_rng(SEED)
        entry = {"label": layer["label"], "on": on, "off": off, "months": {}}
        for m in months:
            f = frames[m]
            s = pd.DataFrame({"label": f["label"].to_numpy(dtype=float), "day": f["day"].to_numpy(), "off": f[off].to_numpy(dtype=float)})
            new = f[on].to_numpy(dtype=float)
            pct, lo, hi = R.interval(s, "off", new, rng)
            moved = np.abs(new - s["off"].to_numpy()) > 1e-9
            entry["months"][m] = {"rows": int(len(f)), "rows_moved": int(moved.sum()), "rmse_off": rmse(s["label"].to_numpy(), s["off"].to_numpy()), "rmse_on": rmse(s["label"].to_numpy(), new),
                                  "change_pct": float(pct), "interval_pct": [float(lo), float(hi)], "switch_off": bool(pct > 0 and lo > 0),
                                  "sum_squared_change": float(np.sum((new - s["off"].to_numpy()) ** 2))}
        result["layers"][layer["name"]] = entry
        print(f"{layer['name']:6s} " + " ".join(f"{m}: {e['change_pct']:+.4f}% [{e['interval_pct'][0]:+.4f}, {e['interval_pct'][1]:+.4f}] moved {e['rows_moved']}"
                                                 + (" SWITCH OFF" if e["switch_off"] else "") for m, e in entry["months"].items()), flush=True)
    result["switched_off"] = {m: [n for n, e in result["layers"].items() if e["months"][m]["switch_off"]] for m in months}
    write_json(ROOT / "docs/final_file/tierb_model_results.json", result)
    print("switched off:", result["switched_off"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["trees", "nets", "rematch", "rows", "chain", "read"])
    parser.add_argument("--months", nargs="*", default=TIERB)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check and set(args.months) - set(TIERB):
        raise SystemExit(f"Tier B months are {TIERB}.")
    OUT.mkdir(parents=True, exist_ok=True)
    {"trees": lambda: cmd_trees(args.months, args.check), "nets": lambda: cmd_nets(args.months, args.check), "rematch": lambda: cmd_rematch(args.months, args.check),
     "rows": lambda: cmd_rows(args.months, args.check), "chain": lambda: cmd_chain(args.months, args.check), "read": lambda: cmd_read(args.months)}[args.command]()


if __name__ == "__main__":
    main()
