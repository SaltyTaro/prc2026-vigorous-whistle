"""Run the separately declared target-scale development batch."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026_ladder_scale.training import main
if __name__ == "__main__":
    main()

