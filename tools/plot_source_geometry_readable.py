"""Draw the current-map parking/route coverage without accessing departure labels."""

from pathlib import Path
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, utc_now, write_json
from prc2026_source_geometry.geometry_audit import clean_ref, runway_ref


def main():
    output = ROOT / "docs/source_geometry/geometry_pilot_readable.png"
    if output.exists():
        raise ValueError("Preserve the original geometry figure.")
    report_path = ROOT / "docs/source_geometry/current_geometry_mapping_review.json"
    locations_path = report_path.with_name("current_geometry_mapping_review_locations.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if sha256_file(locations_path) != report["location_details_sha256"]:
        raise ValueError("Location details changed.")
    locations = json.loads(locations_path.read_text(encoding="utf-8"))
    rows = pq.read_table(ROOT / "data/raw/ranking.parquet", columns=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"],
        filters=[("ADEP_mvt", "in", ["LFPG", "LIRF"]), ("PHASE_mvt", "=", "DEP")]).to_pandas()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2))
    fig.suptitle("Parking references and mapped taxi routes", fontsize=19, fontweight="bold", y=.97)
    for axis, airport, label in zip(axes, ["LFPG", "LIRF"], ["Paris Charles de Gaulle", "Rome Fiumicino"]):
        source = report["source_receipts"][airport + "/current"]
        source_path = ROOT / source["path"]
        if sha256_file(source_path) != source["sha256"]:
            raise ValueError("Geometry source changed.")
        elements = json.loads(source_path.read_text(encoding="utf-8"))["elements"]
        mapped = locations[airport + "/current"]
        parking = mapped["parking"]
        lat0 = sum(item["lat"] for item in parking.values()) / len(parking)
        lon0 = sum(item["lon"] for item in parking.values()) / len(parking)
        project = lambda lat, lon: (6371.0088 * math.radians(lon - lon0) * math.cos(math.radians(lat0)),
                                    6371.0088 * math.radians(lat - lat0))
        linked = set(mapped["parking_lead_in_way_ids_added_to_graph"])
        for element in elements:
            tags = element.get("tags", {})
            kind = tags.get("aeroway")
            if element["type"] != "way" or tags.get("area") == "yes":
                continue
            if kind not in {"taxiway", "taxilane", "runway"} and element["id"] not in linked:
                continue
            if kind == "runway" and not tags.get("ref"):
                continue
            points = [project(p["lat"], p["lon"]) for p in element.get("geometry", []) if p is not None]
            if len(points) < 2:
                continue
            x, y = zip(*points)
            axis.plot(x, y, color="#405779" if kind == "runway" else "#adb9c4",
                      linewidth=2.7 if kind == "runway" else .65, zorder=1)
        used = rows.loc[rows.ADEP_mvt.eq(airport)]
        used_refs = {clean_ref(ref) for ref in used.STAND_mvt}
        for ref in sorted(used_refs - {None}):
            if ref not in parking:
                continue
            record = parking[ref]
            x, y = project(record["lat"], record["lon"])
            connected = bool(record["runway_minimum_distance_m"])
            axis.scatter([x], [y], s=15, color="#168575" if connected else "#d48228",
                         edgecolors="white", linewidths=.35, zorder=3)
        coverage = report["airports"][airport]["current"]["ranking"]
        axis.set_title(f"{airport} · {label}\n"
            f"Parking matches {coverage['exact_unambiguous_parking_fraction']:.1%}\n"
            f"Runway connections {coverage['parking_to_recorded_runway_connected_fraction']:.1%}",
            fontsize=11.5, loc="left", pad=13)
        axis.set_aspect("equal")
        axis.set_xlabel("East–west distance (km)", fontsize=10)
        axis.set_ylabel("North–south distance (km)", fontsize=10)
        axis.grid(color="#e8edf1", linewidth=.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
    legend = [Line2D([0], [0], color="#405779", lw=3, label="Runway"),
              Line2D([0], [0], color="#adb9c4", lw=1, label="Taxiway / linked parking lead-in"),
              Line2D([0], [0], marker="o", color="white", markerfacecolor="#168575", markersize=6,
                     label="Matched parking with runway access"),
              Line2D([0], [0], marker="o", color="white", markerfacecolor="#d48228", markersize=6,
                     label="Matched parking without mapped access")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .08), ncol=2, frameon=False, fontsize=9)
    fig.text(.5, .023, "Coverage uses supplied ranking covariates only. Routes are undirected geometric lower bounds.\n"
             "Current recorded OSM map, 14 September 2026. © OpenStreetMap contributors · ODbL 1.0.",
             ha="center", fontsize=8.5, color="#46515d")
    fig.subplots_adjust(left=.07, right=.98, top=.84, bottom=.22, wspace=.22)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    write_json(output.with_suffix(".receipt.json"), {"created_at_utc": utc_now(), "path": output.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(output), "source_report_sha256": sha256_file(report_path),
        "code_sha256": sha256_file(Path(__file__)), "labels_read": False, "audit_used": False})
    print(json.dumps({"figure": output.relative_to(ROOT).as_posix(), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()
