"""Build the residual stage's v11 feature cache for the given months, waiting for each month's matches_v11 table if it is still being built."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_adsb_v24 import adsb_features
root = ROOT / "data/external/adsblol/matches_v11"
for m in sys.argv[1:]:
    while not (root / f"matches2_{m}.json").exists():  # the receipt is written after the parquet
        time.sleep(30)
    t = time.time()
    f = adsb_features(m, root, ROOT / "runs/adsb_v24/features")
    print("v11", m, len(f), round(time.time() - t, 1), "s", flush=True)
