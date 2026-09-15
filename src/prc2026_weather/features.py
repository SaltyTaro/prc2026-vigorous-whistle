"""Label-free, delayed surface observations and runway-relative wind components."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from prc2026.artifacts import read_json, sha256_file
from prc2026.features import ID, TIME
from prc2026_phase5r.features import FEATURE_NAMES as PREVIOUS_NAMES, CATEGORICAL_COLUMNS
from prc2026_phase4.features import FEATURE_NAMES as CORE_NAMES, CATEGORICAL_COLUMNS as CORE_CATEGORIES

POLICY = "iem-delayed-surface-weather-v1"
INPUT_COLUMNS = [ID, TIME, "ADEP_mvt", "RUNWAY_mvt"]
DELAY_SECONDS = 900
MAX_AGE_SECONDS = 5400
STATIONS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
OBS_NAMES = ["temperature_c", "dewpoint_c", "dewpoint_spread_c", "humidity_pct", "pressure_hpa",
             "speed_mps", "gust_mps", "gust_excess_mps", "gust_reported", "direction_missing",
             "direction_sin", "direction_cos", "east_wind_mps", "north_wind_mps", "visibility_km",
             "visibility_below_1km", "visibility_below_5km", "cloud_fraction", "ceiling_m",
             "ceiling_reported", "sky_missing", "rain", "snow", "ice", "fog", "mist", "thunder",
             "freezing_weather", "showers", "heavy_weather", "no_weather_code", "temperature_below_zero"]
CHANGE_NAMES = ["temperature_c", "pressure_hpa", "speed_mps", "visibility_km", "east_wind_mps", "north_wind_mps"]
MEAN_NAMES = ["temperature_c", "speed_mps", "visibility_km", "rain", "snow", "fog", "thunder"]
MAX_NAMES = ["speed_mps", "gust_mps"]
GEOMETRY_NAMES = ["wx_runway_bearing_sin", "wx_runway_bearing_cos", "wx_runway_length_m", "wx_runway_external_bearing"]
WEATHER_NAMES = (["wx_missing", "wx_age_minutes"] + ["wx_"+n for n in OBS_NAMES] + GEOMETRY_NAMES
                 + ["wx_headwind_mps", "wx_tailwind_mps", "wx_crosswind_mps", "wx_abs_crosswind_mps", "wx_gust_crosswind_mps"]
                 + ["wx_lag60_missing", "wx_lag60_observation_gap_minutes"]
                 + ["wx_change60_"+n for n in CHANGE_NAMES] + ["wx_reports_3h"]
                 + ["wx_mean3h_"+n for n in MEAN_NAMES] + ["wx_max3h_"+n for n in MAX_NAMES])
FEATURE_NAMES = list(PREVIOUS_NAMES) + WEATHER_NAMES


def seconds(values: pd.Series) -> np.ndarray:
    if not isinstance(values.dtype, pd.DatetimeTZDtype) or values.isna().any():
        raise ValueError("Complete timezone-aware movement/observation timestamps are required.")
    return values.dt.as_unit("us").astype("int64").to_numpy(dtype=float)/1e6


def verify_external(root: Path, config: dict) -> dict:
    """Verify the frozen source manifests and every source object, without HTTP."""
    result = {}
    for kind in ["weather", "runways"]:
        entry = config["external_data"][kind]
        path = root/entry["manifest"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("External source manifest changed.")
        receipt = read_json(path)
        if kind == "weather" and (receipt["failures"] or len(receipt["objects"]) != 30):
            raise ValueError("Expected all thirty successful weather objects.")
        for item in receipt["objects"]:
            source = root/item["path"]
            if not source.resolve().is_relative_to((root/"data/external").resolve()):
                raise ValueError("External source escaped its cache.")
            if sha256_file(source) != item["sha256"] or source.stat().st_size != item["size_bytes"]:
                raise ValueError("External source bytes changed.")
        result[kind] = receipt
    license_file = root/config["external_data"]["iem_license"]["path"]
    if sha256_file(license_file) != config["external_data"]["iem_license"]["sha256"]:
        raise ValueError("IEM usage evidence changed.")
    return result


def load_observations(root: Path, inventory: dict) -> pd.DataFrame:
    parts = []
    for item in inventory["objects"]:
        raw = pd.read_csv(root/item["path"], na_values=["M"])
        raw["valid"] = pd.to_datetime(raw["valid"], utc=True)
        begin = pd.Timestamp(item["start_inclusive"], tz="UTC")
        end = pd.Timestamp(item["end_exclusive"], tz="UTC")
        raw = raw.loc[raw["valid"].ge(begin) & raw["valid"].lt(end)].copy()
        if len(raw) != item["rows_in_period"] or not raw["station"].eq(item["station"]).all():
            raise ValueError("Weather source station or bounded population changed.")
        parts.append(raw)
    return observation_features(pd.concat(parts, ignore_index=True))


def observation_features(raw: pd.DataFrame) -> pd.DataFrame:
    if raw[["station", "valid"]].isna().any().any() or raw.duplicated(["station", "valid"]).any():
        raise ValueError("Ambiguous observation identity; duplicate reports need an explicit new policy.")
    seconds(raw["valid"])
    number = lambda n: pd.to_numeric(raw[n], errors="coerce").to_numpy(dtype=float)
    temperature, dew = (number("tmpf")-32)*5/9, (number("dwpf")-32)*5/9
    speed, gust, direction = number("sknt")*(1852/3600), number("gust")*(1852/3600), number("drct")
    direction = np.where((direction >= 0) & (direction <= 360), direction, np.nan)
    sin, cos = np.sin(np.deg2rad(direction)), np.cos(np.deg2rad(direction))
    east, north = -speed*sin, -speed*cos  # Meteorological direction is where the wind comes FROM.
    east, north = np.where(speed == 0, 0, east), np.where(speed == 0, 0, north)
    visibility = number("vsby")*1.609344
    cloud = pd.DataFrame({n: raw[n].astype("string").str.strip().str.upper()
                          for n in ["skyc1", "skyc2", "skyc3", "skyc4"]})
    covers = cloud.replace({"FEW": "0.125", "SCT": "0.375", "BKN": "0.75", "OVC": "1", "VV": "1",
                            "CLR": "0", "SKC": "0", "NSC": "0", "NCD": "0"}).apply(pd.to_numeric, errors="coerce")
    ceilings = pd.DataFrame({str(i): np.where(cloud[f"skyc{i}"].isin(["BKN", "OVC", "VV"]), number(f"skyl{i}")*.3048, np.nan)
                             for i in range(1, 5)})
    codes = raw["wxcodes"].astype("string").fillna("").str.upper().str.strip()
    has = lambda pattern: codes.str.contains(pattern, regex=True).to_numpy(dtype=float)
    result = {"temperature_c": temperature, "dewpoint_c": dew, "dewpoint_spread_c": temperature-dew,
              "humidity_pct": number("relh"), "pressure_hpa": number("alti")*33.8638866667,
              "speed_mps": speed, "gust_mps": gust, "gust_excess_mps": gust-speed,
              "gust_reported": np.isfinite(gust).astype(float), "direction_missing": (~np.isfinite(direction)).astype(float),
              "direction_sin": sin, "direction_cos": cos, "east_wind_mps": east, "north_wind_mps": north,
              "visibility_km": visibility,
              "visibility_below_1km": np.where(np.isfinite(visibility), visibility < 1, np.nan),
              "visibility_below_5km": np.where(np.isfinite(visibility), visibility < 5, np.nan),
              "cloud_fraction": covers.max(axis=1).to_numpy(dtype=float, na_value=np.nan),
              "ceiling_m": ceilings.min(axis=1).to_numpy(), "ceiling_reported": ceilings.notna().any(axis=1).to_numpy(dtype=float),
              "sky_missing": covers.isna().all(axis=1).to_numpy(dtype=float),
              "rain": has("RA|DZ"), "snow": has("SN|SG"), "ice": has("PL|GR|GS|IC"),
              "fog": has("FG"), "mist": has("BR"), "thunder": has("TS"), "freezing_weather": has("FZ"),
              "showers": has("SH"), "heavy_weather": has(r"\+"), "no_weather_code": codes.eq("").to_numpy(dtype=float),
              "temperature_below_zero": np.where(np.isfinite(temperature), temperature < 0, np.nan)}
    frame = pd.DataFrame({"station": raw["station"].to_numpy(), "valid": raw["valid"].to_numpy(), **result})
    if np.isinf(frame[OBS_NAMES].to_numpy(dtype=float)).any():
        raise ValueError("Infinite weather values.")
    return frame.sort_values(["station", "valid"], kind="stable").reset_index(drop=True)


def runway_geometry(raw: pd.DataFrame) -> pd.DataFrame:
    """Geodesic bearings from threshold coordinates, including now-closed runways.

    A contemporary static source can differ from historical naming. Exact names
    alone are mapped; unknown designators get a flagged ten-degree approximation.
    """
    rows = []
    for row in raw.loc[raw["airport_ident"].isin(STATIONS)].to_dict("records"):
        for first, second in [("le", "he"), ("he", "le")]:
            name = str(row[first+"_ident"]).strip().upper()
            if not pd.Series([name]).str.fullmatch(r"(?:0?[1-9]|[12][0-9]|3[0-6])[LRC]?").iloc[0]:
                continue
            a, b = np.deg2rad([row[first+"_latitude_deg"], row[second+"_latitude_deg"]])
            delta = np.deg2rad(row[second+"_longitude_deg"]-row[first+"_longitude_deg"])
            x = np.sin(delta)*np.cos(b)
            y = np.cos(a)*np.sin(b)-np.sin(a)*np.cos(b)*np.cos(delta)
            bearing = np.rad2deg(np.arctan2(x, y)) % 360 if np.isfinite([a, b, delta]).all() and (x != 0 or y != 0) else np.nan
            rows.append({"airport": row["airport_ident"], "runway": normalize_runway(pd.Series([name])).iloc[0],
                         "bearing": bearing, "length_m": float(row["length_ft"])*.3048})
    result = pd.DataFrame(rows)
    if result.duplicated(["airport", "runway"]).any():
        raise ValueError("Ambiguous external runway names.")
    return result


def normalize_runway(values: pd.Series) -> pd.Series:
    parts = values.astype("string").str.strip().str.upper().str.extract(r"^((?:0?[1-9]|[12][0-9]|3[0-6]))([LRC]?)$")
    return parts[0].str.zfill(2)+parts[1]


def range_maximum(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Vectorized exact interval maxima, including irregular special reports."""
    levels = [np.where(np.isfinite(values), values, -np.inf)]
    width = 1
    while width*2 <= len(values):
        previous = levels[-1]
        levels.append(np.maximum(previous[:-width], previous[width:]))
        width *= 2
    result = np.full(len(lo), np.nan)
    lengths = hi-lo
    powers = np.floor(np.log2(np.maximum(lengths, 1))).astype(int)
    for level in np.unique(powers[lengths > 0]):
        mask = (lengths > 0) & (powers == level)
        maximum = np.maximum(levels[level][lo[mask]], levels[level][hi[mask]-(1 << int(level))])
        result[mask] = np.where(np.isfinite(maximum), maximum, np.nan)
    return result


def weather_features(queries: pd.DataFrame, observations: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    """Retain query order and every row. Forbidden extra input columns are ignored."""
    query = queries[INPUT_COLUMNS].reset_index(drop=True).copy()
    if not len(query) or query[ID].isna().any() or query[ID].duplicated().any():
        raise ValueError("Unique, nonempty query IDs required.")
    if observations.duplicated(["station", "valid"]).any():
        raise ValueError("Duplicate observation times are not permitted.")
    qt = seconds(query[TIME])
    allowed_months = [f"2025-{m:02}" for m in range(1, 12)]+["2026-01", "2026-07"]
    if not query[TIME].dt.strftime("%Y-%m").isin(allowed_months).all() or not observations["valid"].dt.strftime("%Y-%m").isin(allowed_months).all():
        raise ValueError("Reserved audit or undeclared months cannot enter weather features.")
    result = {n: np.full(len(query), np.nan, dtype=float) for n in WEATHER_NAMES}
    result["wx_missing"][:] = 1
    result["wx_lag60_missing"][:] = 1
    result["wx_reports_3h"][:] = 0
    obs_groups = {name: frame.sort_values("valid") for name, frame in observations.groupby("station", sort=False)}
    for station, group in query.groupby("ADEP_mvt", sort=False):
        obs = obs_groups.get(station)
        if obs is None or not len(obs):
            continue
        idx = group.index.to_numpy()
        time = seconds(obs["valid"])
        cutoff = qt[idx]-DELAY_SECONDS
        hi = np.searchsorted(time, cutoff, side="right")
        pos = np.maximum(hi-1, 0)
        age = qt[idx]-time[pos]
        available = (hi > 0) & (age <= MAX_AGE_SECONDS)
        result["wx_missing"][idx] = (~available).astype(float)
        result["wx_age_minutes"][idx] = np.where(available, age/60, np.nan)
        for name in OBS_NAMES:
            result["wx_"+name][idx] = np.where(available, obs[name].to_numpy()[pos], np.nan)
        laghi = np.searchsorted(time, cutoff-3600, side="right")
        lagpos = np.maximum(laghi-1, 0)
        lagok = (laghi > 0) & (qt[idx]-3600-time[lagpos] <= MAX_AGE_SECONDS)
        pair = available & lagok
        result["wx_lag60_missing"][idx] = (~lagok).astype(float)
        result["wx_lag60_observation_gap_minutes"][idx] = np.where(pair, (time[pos]-time[lagpos])/60, np.nan)
        for name in CHANGE_NAMES:
            values = obs[name].to_numpy()
            result["wx_change60_"+name][idx] = np.where(pair, values[pos]-values[lagpos], np.nan)
        lo = np.searchsorted(time, cutoff-10800, side="left")
        result["wx_reports_3h"][idx] = hi-lo
        for name in MEAN_NAMES:
            values = obs[name].to_numpy(dtype=float)
            count = np.r_[0, np.cumsum(np.isfinite(values))]
            total = np.r_[0, np.cumsum(np.nan_to_num(values, nan=0))]
            n = count[hi]-count[lo]
            result["wx_mean3h_"+name][idx] = np.divide(total[hi]-total[lo], n, out=np.full(len(idx), np.nan), where=n > 0)
        for name in MAX_NAMES:
            result["wx_max3h_"+name][idx] = range_maximum(obs[name].to_numpy(dtype=float), lo, hi)
    keys = pd.DataFrame({"airport": query["ADEP_mvt"], "runway": normalize_runway(query["RUNWAY_mvt"])})
    mapped = keys.merge(geometry, on=["airport", "runway"], how="left", validate="many_to_one", sort=False)
    exact = mapped["bearing"].notna().to_numpy()
    approximate = pd.to_numeric(keys["runway"].str.extract(r"^(\d{2})")[0], errors="coerce").to_numpy(dtype=float, na_value=np.nan)*10
    bearing = np.where(exact, mapped["bearing"], approximate)
    result["wx_runway_external_bearing"] = exact.astype(float)
    result["wx_runway_bearing_sin"] = np.sin(np.deg2rad(bearing))
    result["wx_runway_bearing_cos"] = np.cos(np.deg2rad(bearing))
    result["wx_runway_length_m"] = mapped["length_m"].to_numpy(dtype=float)
    sin, cos = result["wx_runway_bearing_sin"], result["wx_runway_bearing_cos"]
    east, north = result["wx_east_wind_mps"], result["wx_north_wind_mps"]
    head, cross = -(east*sin+north*cos), east*cos-north*sin
    result["wx_headwind_mps"], result["wx_tailwind_mps"] = head, np.maximum(-head, 0)
    result["wx_crosswind_mps"], result["wx_abs_crosswind_mps"] = cross, np.abs(cross)
    result["wx_gust_crosswind_mps"] = result["wx_gust_mps"]*np.abs(result["wx_direction_sin"]*cos-result["wx_direction_cos"]*sin)
    output = pd.DataFrame({ID: query[ID], **{n: result[n].astype("float32") for n in WEATHER_NAMES}})
    if np.isinf(output[WEATHER_NAMES].to_numpy()).any():
        raise ValueError("Infinite derived weather features.")
    return output[[ID, *WEATHER_NAMES]]
