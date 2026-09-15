"""Plot audited geometry coverage, keeping line-centroid matches separate."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json


def main():
    output = ROOT / "docs/source_geometry/geometry_readiness.png"
    if output.exists() or output.with_suffix(".receipt.json").exists():
        raise ValueError("Preserve the original readiness figure.")
    path = ROOT / "docs/source_geometry/audit_decision.json"
    report = read_json(path)
    stamps = ["2025-01-01T00:00:00Z", "2025-07-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-07-01T00:00:00Z", "current"]
    labels = ["Jan 2025", "Jul 2025", "Jan 2026", "Jul 2026", "Current map"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.4), sharey=True)
    fig.suptitle("Does the map identify usable parking positions?", fontsize=17, fontweight="bold", y=.96)
    for axis, airport, name in zip(axes, ["LFPG", "LIRF"], ["Paris Charles de Gaulle", "Rome Fiumicino"]):
        items = [report["geometry"][airport]["snapshots"][stamp] for stamp in stamps]
        good = np.array([item["supported_connected_fraction"] * 100 for item in items])
        point_only = np.array([(item["supported_parking_fraction"] - item["supported_connected_fraction"]) * 100 for item in items])
        centroids = np.array([item["centroid_only_rows_excluded"] / item["rows"] * 100 for item in items])
        x = np.arange(len(stamps))
        axis.bar(x, good, color="#168575", width=.65)
        axis.bar(x, point_only, bottom=good, color="#ddb24b", width=.65)
        axis.bar(x, centroids, bottom=good + point_only, color="#d9dfe5", hatch="//", edgecolor="#b5bdc6", width=.65)
        for i, value in enumerate(good):
            axis.text(i, value + 1.7 if value else 2, f"{value:.1f}%", ha="center", fontsize=9,
                      fontweight="bold", color="#075c50", bbox={"facecolor":"white", "edgecolor":"none", "pad":1})
        axis.set_title(f"{airport} | {name}", loc="left", fontsize=12, pad=12)
        axis.set_xticks(x, labels, rotation=25, ha="right", fontsize=9)
        axis.set_ylim(0, 112)
        axis.set_yticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
        axis.grid(axis="y", color="#e7ebee")
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Share of departures with an exact stand-reference match", fontsize=10)
    fig.legend(handles=[Patch(color="#168575", label="Supported parking point and mapped runway route"),
        Patch(color="#ddb24b", label="Supported point without mapped runway route"),
        Patch(facecolor="#d9dfe5", edgecolor="#b5bdc6", hatch="//", label="Line-centroid match excluded from usable coverage")],
        loc="lower center", bbox_to_anchor=(.5, .105), ncol=1, frameon=False, fontsize=9)
    fig.text(.5, .028, "Historical maps use the matching month's departure covariates; the current map uses all ranking covariates.\n"
             "Parking endpoints are inferred only where declared controls pass. No model scores or reserved labels were used.\n"
             "Map data: (c) OpenStreetMap contributors, ODbL 1.0. Recorded map state does not prove airport operating dates.",
             ha="center", fontsize=8, color="#46515d")
    fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.35, wspace=.15)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    write_json(output.with_suffix(".receipt.json"), {"created_at_utc": utc_now(),
        "path": output.relative_to(ROOT).as_posix(), "sha256": sha256_file(output),
        "source_report_sha256": sha256_file(path), "code_sha256": sha256_file(Path(__file__)),
        "labels_read": False, "audit_used": False})
    print(output.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
