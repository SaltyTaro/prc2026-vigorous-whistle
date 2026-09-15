# Audit of the supplied PRC data

Generated from all 14 downloaded Parquet files on **13 September 2026**.
Complete machine-readable results: [data_profile.json](data_profile.json).
Input hashes and source timestamps: [download_manifest.json](download_manifest.json).
No model was fitted and no ranking targets were available to this audit.

## Structure and consistency

- Training contains **4,167,797 movements**, including **2,085,047 departures**
  and 2,082,750 arrivals. Every training departure has a target.
- All 12 training files and the ranking file share the same 30-column schema.
  Timestamps are Arrow `timestamp[us, tz=UTC]`.
- There are **10 reporting airports**, matching the detailed official airport
  table. An arrival's reporting airport is its destination, not its origin.
- Training movement IDs are unique across the full year. Ranking departure IDs
  do not overlap training movement IDs.
- Ranking has **689,534 movements**: 305,590 in January 2026 and 383,944 in July.
  It contains **344,841 departures** and **344,693 arrivals**.
- Every ranking departure's airport block time and taxi-out target are missing.
  Every ranking arrival retains a taxi-in label.
- `submitting.parquet` has exactly 344,841 rows and two columns. Its IDs have
  no duplicates or nulls, are integer-valued doubles, and match ranking departure
  IDs both as a set and in order. Every template prediction is blank.

## Monthly training counts

| Month | Movement rows |
| --- | ---: |
| January | 307,257 |
| February | 287,411 |
| March | 328,622 |
| April | 350,288 |
| May | 370,105 |
| June | 365,986 |
| July | 381,161 |
| August | 382,047 |
| September | 367,679 |
| October | 371,163 |
| November | 324,530 |
| December | 331,548 |
| **Total** | **4,167,797** |

## Target distribution

For **every** training departure,
`TAXITIME_SEC_mvt == (MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt).total_seconds()`.
There were zero mismatches over 2,085,047 comparisons. This consistency checks
the target definition; it does not establish that all underlying timestamps are
operationally accurate.

| Statistic | Seconds |
| --- | ---: |
| Mean | 991.20 |
| Standard deviation | 546.36 |
| Median | 912 |
| 90th percentile | 1,471 |
| 95th percentile | 1,684 |
| 99th percentile | 2,339 |
| 99.9th percentile | 4,501 |
| Minimum | **-12** |
| Maximum | **131,167** |

There are **369 negative**, **19 zero**, **4,126 over-one-hour**, and **202
over-three-hour** labels. Of the 202 over-three-hour labels, 188 occur at LIRF,
six at LFPG, four at EGLL, two at EHAM, and two at LSZH. The largest value is
about 36.4 hours.

These observations matter under squared error. They could reflect reporting
conventions, mismatches, unusual operations, or a combination; their mechanism
has not yet been established. Do not equate deleting them from validation with
improving the competition metric. Keep raw labels intact, compare any fitting
policy on the same complete holdouts, and report airport and tail contributions.

## Available predictors and missing records

`MVT_TIME_UTC_mvt`, `SCHED_TIME_UTC_mvt`, departure airport, and runway have no
missing values in either training or ranking departures. Stand is missing for
28 training departures and four ranking departures. Aircraft type and route
fields have small amounts of missing data.

Network Manager fields, including `AOBT_3_flt`, are missing for **22,470 training
departures (1.08%)** and **5,290 ranking departures (1.53%)**. This leaves
approximately **98.92% training coverage** and **98.47% ranking coverage**.
The difference supports treating missing-flight-record cases as an explicit
validation slice. Some matched flight IDs still lack populated joined fields.

The field names and precise null counts are in [DATA_SCHEMA.md](DATA_SCHEMA.md).
The audit counts nulls; it does not yet classify blank strings, sentinel
categories, unseen categories, or incorrect-but-present values.

## Network Manager timestamp diagnostic

The official data dictionary defines `AOBT_3_flt` as actual off-block time for the
flown M3 trajectory. Its arithmetic taxi-out proxy is
`MVT_TIME_UTC_mvt - AOBT_3_flt`. This field remains populated for most ranking
departures, whereas airport-reported `BLOCK_TIME_UTC_mvt` is blank.

The following is an **unfitted arithmetic diagnostic on training rows with
both values present**, not model validation or an official score:

| Airport | Comparable departures | Raw proxy RMSE, seconds | Median proxy error, seconds |
| --- | ---: | ---: | ---: |
| EDDF | 228,185 | 255.08 | +87 |
| EDDM | 166,324 | 348.92 | -2 |
| EGLL | 238,132 | 418.51 | +54 |
| EHAM | 243,867 | 329.64 | +107 |
| LEBL | 177,843 | 311.36 | -54 |
| LEMD | 211,295 | 276.41 | +2 |
| LFPG | 235,779 | 379.51 | -4 |
| LIRF | 159,216 | 557.36 | -118 |
| LSZH | 132,616 | 268.43 | -33 |
| LTFM | 269,320 | 530.61 | +296 |

The airport-dependent errors show why an available timestamp is not an exact
label replacement. A residual correction is a reasonable experiment, alongside
models that use other available context.

## Practical implications

Preserve all input files and original timestamps. Construct feature tables from
explicitly permitted columns and keep targets out of prediction features. When
testing a ranking-style validation slice, mask held-out departure block times
and targets before any neighbor or traffic aggregation so they cannot enter
features indirectly. Arrival taxi-in and departure taxi-out describe different
processes; preserve the movement phase when building context.

Use stable movement IDs for joins and submission alignment. IDs are not an
aircraft registration or tail number. Repeated flight records across airport
movements should be considered when selecting validation groups. Learn target
encodings and target-derived group statistics only inside each training split.

Seasonal holdouts and a forward-time check are proposed next; neither has been
run. Training hardware and model hyperparameters have not yet been benchmarked.
