# The final blind dataset (downloaded 8 October 2026)

`prc-2026-datasets/final_ranking.parquet` and `final_submitting.parquet` first appeared at 08:47-08:48 UTC on 8 October and were
**replaced at 11:04 UTC** (new sizes); the versions used here are the 11:04 ones, downloaded with `tools/prc_data.py download` (whose
filename pattern now admits the two names; receipt `docs/competition/download_manifest.json`):

| file | bytes | SHA-256 |
|---|---|---|
| `data/raw/final_ranking.parquet` | 85,853,399 | `4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f` |
| `data/raw/final_submitting.parquet` | 3,003,722 | `b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924` |

- **Four months: January, February, June and July 2026.** 1,341,197 movements; **670,790 departures** (Jan 152,719 / Feb 144,158 / Jun
  181,791 / Jul 192,122), no departure label or block time; every arrival keeps its taxi-in. Same columns and dtypes as `ranking.parquet`;
  the template lists exactly the departures, in file order (float64 ID / float64 empty value, as before).
- **The initial two months are in it again:** all 344,841 ranking departures, plus 325,949 new ones (February, June). So the final file
  scores January / July too ("combined four months").
- **Times jittered.** On the shared movements `MVT_TIME_UTC_mvt` differs on 633,739 of 689,534 rows and arrival `BLOCK_TIME_UTC_mvt` on
  316,835 (so arrival taxi-in values change); the shift is an integer number of seconds with a triangular distribution on [-12, +12]
  (55,795 rows unchanged at 0). Every other column is identical. Whether the hidden departure labels use the jittered or the original
  times is not stated; on rows whose prediction is anchored on the take-off time (ADS-B layers, bounds) the jitter adds noise of about
  12/sqrt(6) = 4.9 s RMS, negligible against ~214 s, but the final file's January / July values will not reproduce v76's.
- **Coverage of our pre-staged inputs:** February and June 2026 are both pre-staged (`prestage.md`: ADS-B days, IEM, adj2 boosters,
  data-link messages); EUROCONTROL's workbook runs to 31 July 2026, so the published layers cover all four months.
- **February 2026 is in winter season W25**, but the W25 layers (season_pass_v1 / v2, key_memory_v1) were declared for January 2026 only and
  change nothing in February; applying them there would need a new declaration with no development month to read it on.
