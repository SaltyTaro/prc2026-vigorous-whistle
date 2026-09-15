"""Refit the selected surface recipe after independent development verification."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026_surface_refit.training import preflight, refit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/surface_refit_v1.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/surface_refit"))
    parser.add_argument("--development-runs", type=Path, default=Path("runs/surface_airports"))
    parser.add_argument("--timeline-runs", type=Path, default=Path("runs/timeline"))
    parser.add_argument("--previous-runs", type=Path, default=Path("runs/nmclock"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        result = preflight(ROOT, ROOT / args.config, ROOT / args.development_runs, ROOT / args.timeline_runs, ROOT / args.previous_runs)
        print({"passed": True, "selected_candidate": result["selection"]["selected_candidate"], "new_native_models": 12})
    else:
        refit(ROOT, ROOT / args.config, ROOT / args.runs, ROOT / args.development_runs, ROOT / args.timeline_runs, ROOT / args.previous_runs)


if __name__ == "__main__":
    main()
