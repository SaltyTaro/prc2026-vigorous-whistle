"""IATA->ICAO callsign rematch of no-NM-record departures at EDDF/EHAM (label-free). usage: rematch_nonm.py DEPS.parquet OUT.parquet
Candidates: (a) the existing key (FLIGHT_mvt; these rows have no CALLSIGN_flt), (b) FLIGHT_mvt with the IATA airline prefix mapped to ICAO.
Segments by the unchanged prc2026_adsb.features.segment with learned stand positions; per method the best transition as match_departures picks it."""
import sys
from pathlib import Path
import pandas as pd
REPO = Path("/home/azureuser/prc-flight-2026")
sys.path.insert(0, str(REPO / "src"))
from prc2026_adsb.features import load_surface, segment  # noqa: E402
IATA = {"KL": "KLM", "HV": "TRA", "DL": "DAL", "UA": "UAL", "CD": "CND", "EK": "UAE", "OR": "TFL", "QR": "QTR", "CZ": "CSN", "AZ": "ITY",
        "AF": "AFR", "MS": "MSR", "FB": "LZB", "SQ": "SIA", "TK": "THY", "LY": "ELY", "CX": "CPA", "BA": "BAW", "VY": "VLG", "WY": "OMA",
        "EI": "EIN", "LH": "DLH", "LX": "SWR", "OS": "AUA", "SK": "SAS", "AY": "FIN", "IB": "IBE", "U2": "EZY", "FR": "RYR", "W6": "WZZ",
        "PC": "PGT", "A3": "AEE", "TP": "TAP", "LO": "LOT", "SN": "BEL", "KE": "KAL", "JL": "JAL", "NH": "ANA", "MU": "CES", "CA": "CCA",
        "HU": "CHH", "EY": "ETD", "GF": "GFA", "RJ": "RJA", "PS": "AUI", "BT": "BTI", "OK": "CSA", "RO": "ROT", "JU": "ASL", "TO": "TVF",
        "BY": "TOM", "LS": "EXS", "MT": "MT", "X3": "TUI", "DY": "NOZ", "D8": "IBK", "SU": "AFL", "ET": "ETH", "KQ": "KQA", "AT": "RAM",
        "MH": "MAS", "GA": "GIA", "CI": "CAL", "BR": "EVA", "AC": "ACA", "WS": "WJA", "AA": "AAL", "B6": "JBU", "PR": "PAL", "TG": "THA",
        "5X": "UPS", "FX": "FDX", "3S": "BOX", "QY": "BCS", "MP": "MPH", "CV": "CLX"}


def main() -> None:
    dep = pd.read_parquet(sys.argv[1])
    dep["takeoff"] = (dep["MVT_TIME_UTC_mvt"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    stands = pd.read_parquet(REPO / "data/external/adsblol/matches_v11/stands.parquet")
    lookup = {(r.airport, r.stand): (r.lat, r.lon) for r in stands.itertuples(index=False)}
    dep["dayk"] = dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y%m%d")
    rows = []
    for dayk, group in dep.groupby("dayk"):
        days = [(pd.Timestamp(dayk) - pd.Timedelta(days=1)).strftime("%Y%m%d"), dayk]
        surface = load_surface(REPO / "data/external/adsblol/surface", days)
        if surface.empty:
            continue
        for airport, g in group.groupby("ADEP_mvt"):
            s = surface[surface["airport"].eq(airport)]
            by_callsign = {c: f["icao"].unique() for c, f in s[s["callsign"].ne("")].groupby("callsign")}
            by_icao = {i: f for i, f in s.groupby("icao")}
            for row in g.itertuples(index=False):
                stand = lookup.get((airport, row.STAND_mvt)) if isinstance(row.STAND_mvt, str) else None
                flight = str(row.FLIGHT_mvt or "").strip().upper()
                mapped = IATA[flight[:2]] + flight[2:] if flight[:2] in IATA and flight[2:3].isdigit() else None
                out = {"MVT_ID_mvt": row.MVT_ID_mvt, "airport": airport, "flight": flight, "mapped": mapped, "takeoff": row.takeoff}
                for method, key in [("key", flight), ("iata_mapped", mapped)]:
                    best = None
                    for icao in (by_callsign.get(key, []) if key else []):
                        track = by_icao[icao]
                        before = track[(track["time"] <= row.takeoff + 30) & (track["time"] >= row.takeoff - 4 * 3600) & track["ground"]]
                        after = track[(track["time"] >= row.takeoff - 30) & (track["time"] <= row.takeoff + 900) & ~track["ground"]]
                        if len(before) >= 3 and len(after) >= 1:
                            info = segment(track, row.takeoff, stand=stand)
                            if info is not None and (best is None or abs(info["last_ground"] - row.takeoff) < abs(best["last_ground"] - row.takeoff)):
                                best = {**info, "icao": icao}
                    if best is not None:
                        for k in ["icao", "gate", "motion_start", "segment_start", "last_ground", "taxi_path_m", "taxi_stops", "taxi_stopped_s", "at_known_stand"]:
                            out[f"{method}_{k}"] = best.get(k)
                rows.append(out)
        print(dayk, len(rows), flush=True)
    pd.DataFrame(rows).to_parquet(sys.argv[2], index=False)
    print("done", len(rows))


if __name__ == "__main__":
    main()
