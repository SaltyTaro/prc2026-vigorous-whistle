# Dig tunnel 9: plumbing audit of the shipped ranking path (3 October 2026) — no new bug of size

Question: after v35's callsign fix (−1.69 s) and today's no-NM rematch (−2.07 s), does the code that produced v60 hide another
plumbing bug that hits 2026 ranking rows? Read-only; label-free on the ranking, 2025 labels only for sizing. Scratch:
`runs/dig/plumbing/` (`boundary_scan.py` and its CSVs, `stand_known.py`). Nothing declared, built, committed or uploaded.

| check | what was looked at | finding | verdict |
|---|---|---|---|
| timestamp units / time zones | every time conversion on the ranking path (ADS-B features and matching, ctx tokens, final runner, published layers, today's layers) | all go through tz-aware `(t - EPOCH).dt.total_seconds()` or `pd.to_datetime(..., utc=True)`; no `int64 / 1e9`, `.view`, `asi8` on the path | clean |
| month start (1 Jan / 1 Jul 2026: no previous-day rows in the ranking file; December 2025 absent from the frozen context) | 336 timeline features, day 1 00–08 UTC vs days 2–31, against Jan / Jul 2025 | look-back features are capped at 21,600 s, so the boundary cannot inflate them: 8–16 day-1 rows exceed the 99.9% quantile of other days; the largest day-1 contrasts are weather and weekday (natural, also in 2025) | clean, < 0.01 s |
| month end (31 Jan / 31 Jul: no next-month rows) | same scan, 18–24 UTC on the last day | contrasts are delay level and weekday (31 July 2026 a heavy Friday evening); no look-ahead feature stands out | clean |
| rows silently dropped to a fallback | runner leaf tables, 9 network ensembles, member tables | no NaN anywhere except `mask_unmatched` on matched rows (225,278 = the matched rows, by design) | clean |
| stand keys for gate detection | ranking stands against the frozen learned-stand table (1,332 stands) | matched rows at unknown stands lose gate detection (EGLL 29–49% → 2–8%, LEMD 29–32% → 1–4%, LSZH 25–47% → 1–3%); 2025 shares are the same (LEMD 78% known, LSZH 80–88%, LIRF 80–85%, LFPG 15%), dev RMSE 228 s on 33,725 rows (5.5% of dev SSE). Already closed 22 September (`docs/adsb_coverage/README.md`): on dark rows the first ADS-B point is a median 273–404 m from the stand (coverage starts after pushback); a 100 m radius made estimates worse (84 → 163 s). 2026-only: EDDF July 2,249 and EDDM July 1,181 matched rows at new stands | not a bug; coverage |
| ADS-B data completeness 2026 | NM-row match rate by day, Jan 2026; local surface files by hour | EDDF NM match falls to 0.31–0.5 on 29–31 Jan (median 0.87) with every hour present and ground points down a third (~58k vs 85–99k): receiver coverage, the known EDDF decline. 23 Jan dips to 0.82–0.87 at four airports with point counts near normal: no missing archive part | coverage, not plumbing |
| weather joins | temperature NaN, report age, ceiling NaN by airport-month, 2025 vs 2026 | identical coverage (age median ~30 min, p99 ~45 min, temperature never missing) | clean |
| published figure coverage | ranking airport-days in the reconciliation | 619 of 620; LTFM 31 Jan 2026 absent (641 rows keep I = 0), on record as a month-end gap worth ~0.003 s; 9 rows assigned to 31 Dec / 30 Jun predicted off-block days | known, negligible |
| unseen categories | stand, aircraft type (APDF / NM), operator, destination, runway vs all of 2025 | new stands EDDM July 1,182 rows and EDDF July 299 (known, read fine); everything else single or double digits; 2 LFPG rows with runway "NA" | known, small |

**Reading.** The shipped path is clean on every bug class checked. The two large plumbing finds of the week (IATA vs ICAO prefixes,
flight number vs callsign) were key mismatches in ADS-B matching; the remaining match gaps (unknown stands, EDDF's 2026 decline,
23 January) are coverage in the archive, not code. No candidate reaches 0.1 s.

Not checked (would need a model re-run, out of scope for a read-only audit): whether the twelve-month refits of the members and
the dev LOMO tables compute every component identically; tunnel 2 reconstructed v55m and the v46-era components to the shipped
values (float noise and one known u300 flip), which covers the stacking layer's inputs.
