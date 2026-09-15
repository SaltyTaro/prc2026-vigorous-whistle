"""ADS-B surface segments: stand stop, first movement, dark segments, and label-free features."""
import numpy as np
import pandas as pd

from prc2026.features import ID, PHASE, TIME
from prc2026_adsb.features import FEATURE_NAMES, departure_features, learn_stands, match_departures, segment


def track(start: float, stationary_s: float, dark: bool):
    """Synthetic surface track: (optional) stand stop, pushback, taxi, lift-off; ~1 m/s taxi northward."""
    rows, t = [], start
    lat0, lon0 = 52.3100, 4.7700
    if not dark:
        for k in range(int(stationary_s // 30)):
            rows.append([t + 30 * k, lat0, lon0, True, None, 0.0])
        t += stationary_s
    for k in range(60):  # taxi: 20 m per 10 s
        rows.append([t + 10 * k, lat0 + 0.00018 * (k + 1), lon0, True, None, 4.0 if k < 5 else 15.0])
    t += 600
    for k in range(3):
        rows.append([t + 10 * k, lat0 + 0.02 + 0.001 * k, lon0, False, 500.0 + 300 * k, 140.0])
    frame = pd.DataFrame(rows, columns=["time", "lat", "lon", "ground", "alt_baro", "gs"])
    frame["icao"], frame["airport"], frame["callsign"] = "abc123", "EHAM", "KLM1"
    return frame, t  # t = lift-off time


def test_gate_segment_finds_first_movement_and_dark_segment_starts_moving():
    gate, takeoff = track(1_000_000.0, 600.0, dark=False)
    info = segment(gate, takeoff)
    assert info["gate"] is True and info["motion_start"] == 1_000_620.0 and info["segment_start"] == 1_000_000.0
    assert info["stop_seconds"] >= 540 and info["taxi_path_m"] > 1000 and info["last_ground"] == takeoff - 10
    dark, takeoff = track(2_000_000.0, 0.0, dark=True)
    info = segment(dark, takeoff)
    assert info["gate"] is False and info["motion_start"] == info["segment_start"] == 2_000_000.0 and info["first_gs"] == 4.0
    # A known stand at the first point turns a dark segment into a gate one.
    info = segment(dark, takeoff, stand=(52.3100, 4.7700))
    assert info["gate"] is True and info["at_known_stand"] is True


def test_matching_and_features_are_label_free():
    surface, takeoff = track(1_000_000.0, 600.0, dark=False)
    departures = pd.DataFrame({ID: [1.0, 2.0], PHASE: "DEP", TIME: pd.to_datetime([takeoff, takeoff + 7200], unit="s", utc=True),
                               "ADEP_mvt": "EHAM", "CALLSIGN_flt": ["KLM1", "KLM2"], "FLIGHT_mvt": ["KL1", "KL2"], "STAND_mvt": ["D57", "D58"]})
    matches = match_departures(departures, surface)
    assert matches[ID].tolist() == [1.0] and matches["gate"].iloc[0]
    stands = learn_stands(pd.concat([matches] * 3, ignore_index=True))
    assert stands[["airport", "stand"]].values.tolist() == [["EHAM", "D57"]]
    features = departure_features(departures, matches, stands)
    assert list(features.columns) == [ID, *FEATURE_NAMES]
    assert features.loc[0, "adsb_matched"] == 1.0 and features.loc[1, "adsb_matched"] == 0.0 and np.isnan(features.loc[1, "adsb_motion_s"])
    assert features.loc[0, "adsb_motion_s"] == takeoff - 1_000_620.0 and features.loc[0, "adsb_first_dist_stand_m"] < 1.0
