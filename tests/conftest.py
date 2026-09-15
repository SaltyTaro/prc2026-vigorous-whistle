from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from prc2026.features import BLOCK, CATEGORY_SOURCES, ID, PHASE, TARGET, TIME


@pytest.fixture
def protocol():
    return json.loads((Path(__file__).resolve().parents[1]
                       / "configs/validation_v1.json").read_text(encoding="utf-8"))


@pytest.fixture
def movements():
    # London changes to summer time at 01:00 UTC on 30 March 2025.
    frame = pd.DataFrame({source: ["X"] * 4 for source in set(CATEGORY_SOURCES.values())})
    frame[ID] = [101.0, 102.0, 103.0, 104.0]
    frame["FLIGHT_ID_mvt"] = [1.0, 2.0, 3.0, np.nan]
    frame[PHASE] = ["DEP", "DEP", "ARR", "DEP"]
    frame["ADEP_mvt"] = ["EGLL", "EGLL", "KJFK", "LTFM"]
    frame["ADES_mvt"] = ["KJFK", "LTFM", "EDDF", "EGLL"]
    frame[TIME] = pd.to_datetime(["2025-03-30T00:30:00Z", "2025-03-30T01:30:00Z",
                                 "2025-03-30T01:45:00Z", "2025-03-29T22:30:00Z"])
    frame[TARGET] = [600, 900, 240, -2]
    frame[BLOCK] = frame[TIME] - pd.to_timedelta(frame[TARGET], unit="s")
    frame["SCHED_TIME_UTC_mvt"] = frame[TIME] - pd.Timedelta(minutes=20)
    for name in ["EOBT_1_flt", "IOBT_flt", "AOBT_3_flt", "LOBT_flt"]:
        frame[name] = frame[TIME] - pd.Timedelta(minutes=10)
    frame["ARVT_1_flt"] = frame[TIME] + pd.Timedelta(hours=2)
    frame["ARVT_3_flt"] = frame[TIME] + pd.Timedelta(hours=2, minutes=10)
    frame.loc[3, "ADEP_flt"] = None
    return frame
