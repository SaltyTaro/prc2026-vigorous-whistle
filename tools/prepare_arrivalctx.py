"""Prepare the separately declared supplied-arrival context bundle."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026_arrivalctx.bundle import main

if __name__ == "__main__":
    main()
