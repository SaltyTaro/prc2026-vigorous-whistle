from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026_nm_mask.bundle import prepare
if __name__ == "__main__":
    prepare(ROOT)
