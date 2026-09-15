from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026_nm_mask.training import train_job
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    train_job(ROOT, ROOT / args.job)
