"""S11-S13: the layers from v30 to the v35 incumbent, each through the frozen code that first built it for the ranking file.

- repair (v31, docs/input_shift/lfpg_eobt_sensitivity.md). A label-free detector flags an airport-month when its share of departures
  with EOBT_1 > IOBT lies more than one 2025 span above the 2025 maximum (reference: the same airport's twelve 2025 months, read from
  data/raw/training_2025-*.parquet; span = maximum - minimum). The audit's finding was LFPG July 2026 at 35% against 4-7%. In a flagged
  airport-month, EOBT_1 := IOBT on the departures with EOBT_1 > IOBT (the sensitivity run's edit) and the chain features..recipe runs
  again on that input; v31 = max(v30 + 0.5 x (v30 on the edited input - v30), -12), as tools/export_lfpg_eobt_blend.py (weight 0.5).
  No flag: v31 = v30.
- rematch (the callsign fix's input): tools/adsb_rematch_nonm.py, unchanged, with its REPO pointed at this workspace, on the file's
  departures without an NM flight record (FLIGHT_ID_mvt missing) at EDDF / EHAM, the declared scope of configs/callsign_fix_v1.json.
- stack: the recipe deltas on the file's own recipe detail, as the exports took them from the dry-run detail:
  v33 = max(v31 + export_v33.recipe(detail, 1.0, 0.6, 0.7) - export_v33.recipe(detail, 0.7, 0.7, 0.7), -12);
  v34 = max(v33 + export_fair_weights.recipe(detail, V33_OUTER, V34_LIRF) - recipe(detail, V33_OUTER, V33_LIRF), -12);
  v35 = tools/export_stack.layers(v34, detail, ..., queue 0.75, regular): + export_disagreement_gate.gate_delta (u300), then the callsign
  rows max(0.5 x observed taxi + 0.5 x value, -12) (run_callsign_fix.scope_table / offsets), then the queue rows
  max(0.75 x observed taxi + 0.25 x value, -12) on regular gate matches (run_queue_trust's segment rule / offsets_all).
  export_stack.layers itself reads the frozen ranking tables, so the layers are re-plumbed here onto the run's own tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prc2026.artifacts import object_hash, read_json, sha256_file
from prc2026.features import ID, PHASE, TIME

from .common import ROOT, save_frame, source_file

sys.path.insert(0, str(ROOT / "tools"))
import export_v33  # noqa: E402
from export_disagreement_gate import V34_LIRF, gate_delta  # noqa: E402
from export_fair_weights import FLOOR, V30_OUTER, V33_LIRF, V33_OUTER, recipe  # noqa: E402
from run_callsign_fix import offsets, scope_table  # noqa: E402
from run_queue_trust import SEGMENT, offsets_all  # noqa: E402

REFERENCE_MONTHS = [f"2025-{m:02}" for m in range(1, 13)]
EOBT_COLUMNS = [PHASE, "ADEP_mvt", TIME, "EOBT_1_flt", "IOBT_flt"]
LFPG_WEIGHT = 0.5  # v31: tools/export_lfpg_eobt_blend.py --weight 0.5 (docs/input_shift/export_v31.json)
REMATCH_AIRPORTS = ["EDDF", "EHAM"]  # configs/callsign_fix_v1.json: no-NM departures at EDDF / EHAM
REMATCH_COLUMNS = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt", "FLIGHT_ID_mvt"]
REMATCH_OUTPUT_COLUMNS = ["MVT_ID_mvt", "airport", "flight", "mapped", "takeoff"] + [f"{m}_{k}" for m in ["iata_mapped", "key"] for k in
                                                                                   ["icao", "gate", "motion_start", "segment_start", "last_ground",
                                                                                    "taxi_path_m", "taxi_stops", "taxi_stopped_s", "at_known_stand"]]
DECLARATIONS = {"gate": "configs/disagreement_gate_v1.json", "callsign": "configs/callsign_fix_v1.json", "queue": "configs/queue_trust_v1.json"}
V35 = {"gate": "u300", "callsign": "u300_cs50", "queue": "qt75", "queue_source": "regular",
       "record": "docs/disagreement/export_v35.json (tools/export_stack.py --queue 0.75 --queue-source regular --version 35, user override)"}


# ---------------------------------------------------------------- repair (v31)

def eobt_shares(frame: pd.DataFrame) -> pd.DataFrame:
    """Per departure airport and movement month: departures, departures with EOBT_1 > IOBT (a missing clock counts as not later), share."""
    dep = frame.loc[frame[PHASE].eq("DEP")]
    table = pd.DataFrame({"airport": dep["ADEP_mvt"].to_numpy(), "month": dep[TIME].dt.strftime("%Y-%m").to_numpy(),
                          "later": (dep["EOBT_1_flt"] > dep["IOBT_flt"]).fillna(False).to_numpy(dtype=bool)})
    out = table.groupby(["airport", "month"]).agg(departures=("later", "size"), later=("later", "sum")).reset_index()
    out["share"] = out["later"] / out["departures"]
    return out


def reference_shares() -> pd.DataFrame:
    """The 2025 reference per airport: minimum, maximum and span of the twelve monthly shares."""
    frames = [pd.read_parquet(ROOT / "data/raw" / source_file(m), columns=EOBT_COLUMNS) for m in REFERENCE_MONTHS]
    shares = eobt_shares(pd.concat(frames, ignore_index=True))
    shares = shares[shares["month"].isin(REFERENCE_MONTHS)]
    ref = shares.groupby("airport")["share"].agg(reference_min="min", reference_max="max", reference_months="size").reset_index()
    ref["span"] = ref["reference_max"] - ref["reference_min"]
    ref["threshold"] = ref["reference_max"] + ref["span"]
    return ref


def detect(frame: pd.DataFrame, reference: pd.DataFrame | None = None) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """Flagged (airport, month) pairs of the file and the full table (share, 2025 reference, threshold, margin = share - threshold)."""
    reference = reference_shares() if reference is None else reference
    table = eobt_shares(frame).merge(reference, on="airport", how="left", validate="many_to_one")
    table["margin"] = table["share"] - table["threshold"]
    table["flagged"] = (table["reference_months"] == len(REFERENCE_MONTHS)) & (table["margin"] > 0)
    return [(a, m) for a, m in table.loc[table["flagged"], ["airport", "month"]].itertuples(index=False)], table


def reset_input(raw: pd.DataFrame, flagged: list[tuple[str, str]]) -> tuple[pd.DataFrame, np.ndarray]:
    """The raw file with EOBT_1 := IOBT on the flagged airport-months' departures whose EOBT_1 > IOBT; (edited frame, edited row mask)."""
    month = raw[TIME].dt.strftime("%Y-%m")
    in_scope = np.zeros(len(raw), dtype=bool)
    for airport, m in flagged:
        in_scope |= (raw[PHASE].eq("DEP") & raw["ADEP_mvt"].eq(airport) & month.eq(m)).to_numpy(dtype=bool)
    edit = in_scope & (raw["EOBT_1_flt"] > raw["IOBT_flt"]).fillna(False).to_numpy(dtype=bool)
    out = raw.copy()
    out.loc[edit, "EOBT_1_flt"] = out.loc[edit, "IOBT_flt"]
    return out, edit


def same_content(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return list(a.columns) == list(b.columns) and len(a) == len(b) and all(a[c].equals(b[c]) for c in a.columns)


# ---------------------------------------------------------------- rematch (v35 callsign fix input)

def rematch_departures(departures: pd.DataFrame) -> pd.DataFrame:
    keep = departures["FLIGHT_ID_mvt"].isna() & departures["ADEP_mvt"].isin(REMATCH_AIRPORTS)
    return departures.loc[keep, REMATCH_COLUMNS].reset_index(drop=True)


def rematch(deps_path: Path, out_path: Path) -> None:
    """tools/adsb_rematch_nonm.py's main, unchanged, with REPO = this workspace (stands: matches_v11; surface: data/external/adsblol/surface)."""
    import adsb_rematch_nonm as frozen
    saved = frozen.REPO, sys.argv
    frozen.REPO, sys.argv = ROOT, ["adsb_rematch_nonm.py", str(deps_path), str(out_path)]
    try:
        frozen.main()
    finally:
        frozen.REPO, sys.argv = saved
    out = pd.read_parquet(out_path)
    if out.empty and not len(out.columns):  # no departure had a surface day: the frozen tool writes a frame without columns
        save_frame(pd.DataFrame({c: pd.Series(dtype=float) for c in REMATCH_OUTPUT_COLUMNS}), out_path)


# ---------------------------------------------------------------- stack (v31 -> v35)

def queue_segments(regular: pd.DataFrame, rematch_out: pd.DataFrame, callsign_ids: set) -> pd.DataFrame:
    """run_queue_trust.segments on the run's own tables: regular gate matches (the residual match dir, matches_v11's stands) plus the
    callsign fix's scope rows with their segment shape; same columns, rule and duplicate guard."""
    regular = regular.loc[regular["gate"].astype(bool), SEGMENT].assign(source="regular")
    rem = rematch_out[rematch_out[ID].isin(callsign_ids)].rename(columns={"iata_mapped_motion_start": "motion_start", "iata_mapped_taxi_stops": "taxi_stops",
                                                                         "iata_mapped_taxi_stopped_s": "taxi_stopped_s", "iata_mapped_taxi_path_m": "taxi_path_m"})
    seg = pd.concat([regular, rem[SEGMENT].assign(source="callsign")], ignore_index=True)
    if seg[ID].duplicated().any():
        raise ValueError("A departure has both a regular and a callsign segment.")
    seg["taxi"] = seg["takeoff"] - seg["motion_start"]
    seg["in_scope"] = (seg["taxi_stops"] >= 5) & (seg["taxi_path_m"] >= 5000) & (seg["taxi_stopped_s"] / seg["taxi"] <= 0.75) & (seg["taxi"] > 3000)
    return seg.set_index(ID)


def weights() -> dict:
    """The shipped layers' parameters, read from their declarations (v35 = the user's override of u300 + cs50 + qt75 on regular matches)."""
    decls = {k: read_json(ROOT / p) for k, p in DECLARATIONS.items()}
    return {"gate": decls["gate"]["candidates"][V35["gate"]], "callsign": float(decls["callsign"]["candidates"][V35["callsign"]]["weight"]),
            "queue": float(decls["queue"]["candidates"][V35["queue"]]["weight"]), "queue_source": V35["queue_source"],
            "declarations": {p: {"sha256": sha256_file(ROOT / p), "object_hash": object_hash(decls[k])} for k, p in DECLARATIONS.items()}}


def compute(detail: pd.DataFrame, repaired_v30: np.ndarray | None, rematch_path: Path, regular: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """`detail`: the recipe stage's detail (v30's columns); `repaired_v30`: v30 on the repaired input in detail order (None: no repair)."""
    ids = detail[ID].to_numpy()
    v30 = detail["v30"].to_numpy(dtype=float)
    # the detail must be v30's recipe under both frozen rebuilds (export_v33 at 0.7; export_fair_weights at v30's outer and LIRF weights)
    checks = {"export_v33_recipe_at_0.7": float(np.max(np.abs(export_v33.recipe(detail, 0.7, 0.7, 0.7) - v30))),
              "fair_weights_recipe_at_v30": float(np.max(np.abs(recipe(detail, V30_OUTER, V33_LIRF) - v30)))}
    if not max(checks.values()) <= 1e-9:
        raise ValueError(f"The detail is not v30's recipe: {checks}")
    effect = np.zeros(len(ids)) if repaired_v30 is None else np.asarray(repaired_v30, dtype=float) - v30
    if not np.isfinite(effect).all():
        raise ValueError("The repair effect has non-finite values.")
    v31 = np.maximum(v30 + LFPG_WEIGHT * effect, FLOOR)
    v33 = np.maximum(v31 + export_v33.recipe(detail, 1.0, 0.6, 0.7) - export_v33.recipe(detail, 0.7, 0.7, 0.7), FLOOR)
    at33, at34 = recipe(detail, V33_OUTER, V33_LIRF), recipe(detail, V33_OUTER, V34_LIRF)
    v34 = np.maximum(v33 + at34 - at33, FLOOR)

    w = weights()
    gdelta, gate_masks = gate_delta(detail, w["gate"], at34)
    if np.any(gdelta[~(gate_masks["unm_up"] | gate_masks["mat_down"] | gate_masks["lirf_up"])] != 0) or not np.isfinite(gdelta).all():
        raise ValueError("The gate delta touches rows outside its masks.")
    values = np.maximum(v34 + gdelta, FLOOR)
    after_gate = values.copy()

    rematch_out = pd.read_parquet(rematch_path)
    cs = scope_table({"rematch": {"outputs": {rematch_path.as_posix(): sha256_file(rematch_path)}}})
    off_cs = offsets()
    s = cs.set_index(ID).reindex(ids)
    cs_in = s["motion_start"].notna().to_numpy()
    cs_taxi = s["takeoff"].to_numpy() - s["motion_start"].to_numpy() + np.array([off_cs.get(a, np.nan) for a in s["airport"].fillna("")])
    values[cs_in] = np.maximum(w["callsign"] * cs_taxi[cs_in] + (1 - w["callsign"]) * values[cs_in], FLOOR)
    after_callsign = values.copy()

    seg = queue_segments(regular, rematch_out, set(cs[ID])).reindex(ids)
    off = offsets_all()
    q_in = seg["in_scope"].fillna(False).to_numpy(dtype=bool)
    if w["queue_source"] == "regular":
        q_in &= seg["source"].eq("regular").to_numpy()
    q_taxi = seg["taxi"].to_numpy() + np.array([off.get(a, np.nan) for a in seg["airport"].fillna("")])
    values[q_in] = np.maximum(w["queue"] * q_taxi[q_in] + (1 - w["queue"]) * values[q_in], FLOOR)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite v35 values.")

    frame = pd.DataFrame({ID: ids, "airport": detail["airport"].to_numpy(), "month": detail["month"].to_numpy(), "scope": detail["scope"].to_numpy(),
                          "v30": v30, "repair_effect": effect, "v31": v31, "v33": v33, "v34": v34, "after_gate": after_gate,
                          "after_callsign": after_callsign, "v35": values, "layer_gate": gdelta != 0, "layer_callsign": cs_in, "layer_queue": q_in,
                          "callsign_taxi": np.where(cs_in, cs_taxi, np.nan), "queue_taxi": np.where(q_in, q_taxi, np.nan)})
    record = {"recipe_checks_max_abs": checks, "lfpg_weight": LFPG_WEIGHT, "floor_seconds": FLOOR,
              "v33_outer": V33_OUTER, "v33_lirf": V33_LIRF, "v34_lirf": V34_LIRF, "layers": {**{k: v for k, v in w.items() if k != "declarations"}, "shipped_as": V35},
              "declarations": w["declarations"], "offsets_callsign_s": off_cs, "offsets_queue_s": off,
              "rows": {"repair_effect_nonzero": int((effect != 0).sum()), "gate": int((gdelta != 0).sum()),
                       **{f"gate_{k}": int(v.sum()) for k, v in gate_masks.items()}, "callsign": int(cs_in.sum()), "queue": int(q_in.sum()),
                       "queue_scope_any_source": int(seg["in_scope"].fillna(False).to_numpy(dtype=bool).sum())},
              "rematch_rows": int(len(rematch_out)), "floored_rows_v35": int((values == FLOOR).sum())}
    return frame, record
