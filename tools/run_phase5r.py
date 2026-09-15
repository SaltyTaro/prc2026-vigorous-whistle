"""Run declared Phase-5R models; December is unavailable to this command."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026_phase5r.training import main

if __name__ == "__main__":
    main()
