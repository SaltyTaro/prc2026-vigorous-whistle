# Observed data schema

Generated from the downloaded Parquet files. Field meanings are documented in [the official data page](pages/data.md).

| Column | Arrow type | Missing in training departures | Missing in ranking departures |
| --- | --- | ---: | ---: |
| `MVT_ID_mvt` | `double` | 0 / 2,085,047 | 0 / 344,841 |
| `FLIGHT_ID_mvt` | `double` | 22,424 / 2,085,047 | 5,283 / 344,841 |
| `FLIGHT_mvt` | `string` | 188 / 2,085,047 | 58 / 344,841 |
| `FLIGHT_RULE_mvt` | `string` | 203 / 2,085,047 | 20 / 344,841 |
| `ADEP_mvt` | `string` | 0 / 2,085,047 | 0 / 344,841 |
| `ADES_mvt` | `string` | 122 / 2,085,047 | 13 / 344,841 |
| `PHASE_mvt` | `string` | 0 / 2,085,047 | 0 / 344,841 |
| `MVT_TIME_UTC_mvt` | `timestamp[us, tz=UTC]` | 0 / 2,085,047 | 0 / 344,841 |
| `BLOCK_TIME_UTC_mvt` | `timestamp[us, tz=UTC]` | 0 / 2,085,047 | 344,841 / 344,841 |
| `SCHED_TIME_UTC_mvt` | `timestamp[us, tz=UTC]` | 0 / 2,085,047 | 0 / 344,841 |
| `AIRCRAFT_TYPE_mvt` | `string` | 1,507 / 2,085,047 | 388 / 344,841 |
| `RUNWAY_mvt` | `string` | 0 / 2,085,047 | 0 / 344,841 |
| `STAND_mvt` | `string` | 28 / 2,085,047 | 4 / 344,841 |
| `TAXITIME_SEC_mvt` | `int32` | 0 / 2,085,047 | 344,841 / 344,841 |
| `LOBT_flt` | `timestamp[us, tz=UTC]` | 22,470 / 2,085,047 | 5,290 / 344,841 |
| `CALLSIGN_flt` | `string` | 22,470 / 2,085,047 | 5,290 / 344,841 |
| `ADEP_flt` | `string` | 22,470 / 2,085,047 | 5,290 / 344,841 |
| `ADES_flt` | `string` | 22,470 / 2,085,047 | 5,290 / 344,841 |
| `ADES_FILED_flt` | `string` | 22,470 / 2,085,047 | 5,290 / 344,841 |
| `MARKET_SEGMENT_flt` | `string` | 22,470 / 2,085,047 | 5,290 / 344,841 |
| `IOBT_flt` | `timestamp[us, tz=UTC]` | 22,470 / 2,085,047 | 5,290 / 344,841 |
| `FLIGHT_RULE_flt` | `string` | 22,544 / 2,085,047 | 5,304 / 344,841 |
| `FLIGHT_TYPE_flt` | `string` | 22,470 / 2,085,047 | 5,290 / 344,841 |
| `AIRCRAFT_TYPE_flt` | `string` | 22,470 / 2,085,047 | 5,290 / 344,841 |
| `WK_TBL_CAT_flt` | `string` | 22,544 / 2,085,047 | 5,304 / 344,841 |
| `AIRCRAFT_OPERATOR_flt` | `string` | 22,470 / 2,085,047 | 5,290 / 344,841 |
| `EOBT_1_flt` | `timestamp[us, tz=UTC]` | 22,470 / 2,085,047 | 5,290 / 344,841 |
| `ARVT_1_flt` | `timestamp[us, tz=UTC]` | 22,470 / 2,085,047 | 5,290 / 344,841 |
| `AOBT_3_flt` | `timestamp[us, tz=UTC]` | 22,470 / 2,085,047 | 5,290 / 344,841 |
| `ARVT_3_flt` | `timestamp[us, tz=UTC]` | 22,470 / 2,085,047 | 5,290 / 344,841 |
