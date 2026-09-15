# Reporting and geometry audit after v7

**Rome Fiumicino
(LIRF) has enough supported historical parking and taxiway geometry for a
declared model comparison. Paris Charles de Gaulle (LFPG) does not yet pass
that check.** The reporting audit supplies no supported new clock correction.
No model was fitted, prediction changed, or submission made. V7 remains the
official best at **275.0570 seconds RMSE**; December remains reserved.

The [decision record](audit_decision.json) contains the final coverage and
source findings. The [coverage figure](geometry_readiness.png) separates
supported parking points from approximate line-centroid matches.

## Fitting-only reporting evidence

The audit uses **237,345 LIRF/LFPG departures** from the **1,226,945-ID
intersection of the original fitting folds**, covering February-June and
August-September 2025. Scoring, stopping, December and ranking labels are
excluded. Original negative and very large labels remain present. The
[full source report](fitting_source_audit.json) describes raw timing
relationships; its residual RMSE values are not v7 prediction errors.

- LIRF has 836 missing-NM-clock fitting rows. Of these, 835 lack the airport
  aircraft type and fall under the existing conditional schedule rule. They
  include all 102 missing-clock labels above three hours. The one row outside
  that group is ordinary. Broadening the schedule rule has no support here.
- LFPG has 1,892 missing-clock fitting rows; only 16 have the implied off-block
  time within 60 seconds of schedule. None exceeds three hours. Replacing
  their off-block time with schedule would lack support.
- Neither airport has a schedule-minus-implied-block difference within the
  declared 120-second tolerance of a nonzero whole-day multiple. No missing-NM
  stand group satisfies the declared recurrence minimum of 50 rows in five
  months. The [stand/month table](stand_month_recurrence.csv) preserves the
  groups that do satisfy that minimum; these have NM clocks.

These observations rule out the proposed simple repairs in this sample. They
do not establish that all exceptional reporting errors are irreducible.

## Historical geometry coverage

Coverage below requires an exact full stand reference, a supported parking
point, and a connected mapped route to the recorded runway. A supported point
is an explicit parking node or a parking-way endpoint whose airport passed the
declared controls. Line centroids are excluded. Rows without supported geometry
remain in every denominator.

| Recorded map / matching departure month | LFPG rows | LFPG usable route | LIRF rows | LIRF usable route |
| --- | ---: | ---: | ---: | ---: |
| 1 January 2025 / January 2025 | 17,900 | 0.00% | 11,317 | 100.00% |
| 1 July 2025 / July 2025 | 21,712 | 0.00% | 15,211 | 99.06% |
| 1 January 2026 / January 2026 | 18,129 | 0.00% | 11,061 | 98.32% |
| 1 July 2026 / July 2026 | 21,743 | 61.33% | 15,838 | 98.67% |
| Current map / all ranking departures at airport | 39,872 | 60.53% | 26,899 | 98.53% |

The zero LFPG values are a limitation of the strict parking-point rule and the
available historical map representations. The older maps have many matching
parking lines but only 22 explicit parking references, none used by the
matching-month departures. They are not evidence that the airport lacked
stands or taxiways. A future LFPG study could investigate whole-line geometry
as a separate approximate feature, but this audit does not accept its centroid
as a known nose-wheel position or infer a historical endpoint from the current
map.

LIRF's supported parking positions are stable within 30 metres across the four
period-start snapshots for references covering **98.53% of its ranking rows**.
This checks recorded map coordinates, not the operational dates of physical
airport changes. OSM edit dates and four snapshots cannot establish continuous
layout validity throughout every fitting or scoring month.

## Mapping reviews and controls

Each interpretation is preserved in a separate declaration and output:

1. [Initial declaration](../../configs/source_geometry_audit_v1.json), at
   20:22:01 UTC, and [initial historical report](geometry_coverage.json).
   Node coordinates or way geometry means supplied provisional positions.
   Distant duplicate references were ambiguous. LIRF passed the 50% current
   coverage gate for historical fetching; LFPG initially did not.
2. [Node-membership review](../../configs/source_geometry_mapping_review_v1.json),
   at 20:34:58 UTC, resolves a parking node and its own same-ref lead-in way
   by explicit OSM node identity. This allowed LFPG's historical fetch.
   The [reviewed report](geometry_mapping_review.json) retains approximate
   way-only centroids and therefore is not the final usable-point coverage.
3. [Endpoint-control review](../../configs/source_geometry_endpoint_review_v1.json),
   at 20:47:47 UTC, tests the documented parking-way direction against explicit
   same-ref parking nodes. It requires at least 30 distinct control ways per
   airport and 99% agreement within one metre of the last way node.
   LFPG passes **288/321 distinct control states, 89.72%, across 250 ways**;
   it fails the declared requirement, and no endpoint inference is applied.
   LIRF passes **47/47 controls**, allowing 69 way-only reference endpoints
   per snapshot. Inferred endpoints remain flagged and preserve source way IDs.

The [endpoint report](geometry_endpoint_review.json),
[individual controls](parking_endpoint_controls.json) and
[resolved map records](geometry_endpoint_review_locations.json) preserve every
exception. The final [decision](audit_decision.json) removes remaining line
centroids from usable-point coverage. Earlier apparent LFPG position changes
partly reflect differences between line-centroid and node representations;
they must not be described as actual stand relocations.

Graph edges use mapped taxiways, taxilanes and explicitly linked parking
lead-ins. Shared OSM node identity establishes connectivity. Snapping is limited
to 75 metres; runway traversal is excluded. Distances are undirected minimum
mapped distances to any shared-node access on the recorded runway. They do not
identify an actual taxi clearance, runway-direction threshold, queue order,
runway crossing clearance, or time spent waiting. No fuzzy stand aliases,
zero stripping, or passenger-gate substitution is used.

## Concrete next comparison

A separate LIRF experiment is justified by coverage. It should compare a small
geometry feature set with the frozen v7 recipe at matched model settings,
using stand position, mapped route distance, route connectivity and explicit
missingness. Keep every departure and the existing fallback where geometry is
unavailable. A second declared candidate can add route-sharing queue context
at the permitted off-block clock; generic traffic counts are already in v7.

Snapshot assignment for all January-November fitting months and both original
development folds must be declared before feature preparation or scores. If
learned free-flow baselines are added, fit them on inner flight-group folds.
Score both complete original development folds against v7 and retain the
existing 0.5% improvement margin on each before any promotion. Use only the
authorized arcw0 for a later training campaign and check its limited disk space
before copying artifacts. This audit does not promise a particular score gain.

LFPG needs stronger historical parking-point evidence before the same precise
route features can cover it broadly. Additional clock repair is not selected.
Synthetic NM masking remains a separate, untested fallback hypothesis from the
[v7 gain review](../competition/gain_review_v7_20260915/README.md).

## Sources, verification and reproduction

Map data is **copyright OpenStreetMap contributors, ODbL 1.0**. Preserve that
attribution and the database license when distributing raw or derived map data;
GPLv3 project source and ODbL map data remain separately identified. The OSM
parking-tag documentation is **CC BY-SA 2.0**, as linked in its archived footer;
the collector's generic map-license field is not its content license.

Thirteen successful public objects are cached in
`data/external/osm/source_geometry_v1/`: ten maps, the OSM copyright page,
the ODbL text and the parking-position tagging guide. Public reads were
sequential, bounded and retried only after delays. Three failed attempts are
preserved alongside successful receipts. The alternate service's July 2026
base timestamp for LIRF's January 2026 query is after the requested historical
date; all returned historical element edit timestamps are no later than the
requested snapshot. The [source inventory](source_inventory.json) binds every
successful object and failed-attempt record.

[Local verification](verification.json) checks the frozen inputs, reports,
controls and graph reconstructions. **34 focused local tests pass**, including
protected-period rejection, signed/extreme label retention, exact references,
duplicate ambiguity, graph connectivity and endpoint conflicts. Earlier model
test records remain unchanged; they were not presented as a new training run.
The [archive receipts](archive_receipts.json) identify the separate source/
evidence and openly licensed map-data packages. Prior submissions and raw data
hashes are verified without changing them. No Azure or AWS action was taken.

Install `requirements-source-geometry-audit.txt` in the existing research
environment. Verification reads the frozen cache and original fitting IDs:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools/pull_source_geometry.py --offline
.\.venv\Scripts\python.exe -X utf8 -B -m pytest tests/test_source_geometry_audit.py -q
.\.venv\Scripts\python.exe -X utf8 -B tools/verify_source_geometry.py
```

The audit producers deliberately refuse to overwrite prior outputs. To rebuild
the entire audit, use a separate checkout with the immutable raw data, original
split bundle, archived source receipts and parent artifacts. Run, in order,
`audit_source_reporting.py`, `audit_airport_geometry.py` (current then
`--complete`), `review_airport_geometry.py` (current then `--complete`),
`review_parking_endpoints.py`, and `summarize_source_geometry.py`. The saved
reports preserve declaration timing; a new run has its own creation timestamps
and should be compared on scientific fields and hashes. Do not refresh cached
public sources to reproduce this study.
