"""Refit cat_long at the declared fixed iterations for the selected v10 candidate (arcw0 only)."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026_v10_refit.training import preflight, refit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v10_v2.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/v10/selection_v2.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/v10"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        context = preflight(ROOT, ROOT / args.config, ROOT / args.selection)
        print({"passed": True, "selected_candidate": context["selection"]["selected_candidate"], "iterations": context["spec"]["iterations"],
               "clock_fit_rows": context["spec"]["expected_clock_fit_rows"]})
    else:
        refit(ROOT, ROOT / args.config, ROOT / args.selection, ROOT / args.runs)


if __name__ == "__main__":
    main()
