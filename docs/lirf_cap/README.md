# LIRF fill-population dominance cap and final candidate 3 (12 October 2026)

Declaration [`configs/lirf_cap_v1.json`](../../configs/lirf_cap_v1.json) (post hoc, disclosed), tool
[`tools/run_lirf_cap.py`](../../tools/run_lirf_cap.py), results [`results_v1.json`](results_v1.json). Built at the user's word
("Build it", 12 October); **not uploaded**.

## Finding

A label-free look at final candidate 2's largest values: every prediction above 25,000 s is a LIRF fill-population row (LIRF departure
without any NM clock and without an airport aircraft type), predicted at its schedule interval s = take-off - SCHED_TIME plus 0 to
12,000 s. The excess is in the model part (runner v55m) and passes unchanged through every later step: population rows with s > 43,200 s
stay at v7, whose models learn label - s. In 2025 every one-day row (label 86,400 s + a taxi) had s < 86,400 s, so its residual was
positive, and the trees carry a positive residual to s > 86,400 s as well.

On these rows the 2025 labels take two values only: the fill (label = s; 18 of the 29 rows with s > 43,200 s) and the one-day record
(label 87,168-88,392 s; 11 rows). A prediction above max(s, 88,392 s) is above both, so capping it there brings it closer to the label
whichever value is true.

## Rule

new = min(base, max(s, U)) on population rows, U = 88,392 s (the largest 2025 one-day label among population rows, recomputed by
`run_lirf_cap.py derive` from the twelve training files). Applied last, after the whole final-file chain. No lower bound: genuine block times
with a stale schedule give labels below s (611 population rows with s in 3,600-10,800 s in 2025).

## Readings

| file | population rows | cap fires on | effect |
|---|---:|---|---|
| board v76 | 383 | 0 | output byte-identical to v76 |

Expected gain on the final file: at least 8.8e7 s^2 if both rows are fills (more if either is a one-day record), about -0.3 s on the four-month
RMSE and -0.6 s on its January / July part.

**Disclosure.** Found by inspection, read before declaring. Both final rows are among v40's nine score-derived fill pins (off in the final file
by the user's 8 October decision; v76 carries them, which is why the cap changes nothing there). The cap reads no official score, moves the
rows to max(s, U) rather than to the pin values, and leaves the other seven pinned rows alone (s < 86,400 s, predictions between the two label
values). It is an approved exception to the final-week rule that new layers leave LIRF population rows unchanged. Guard rows unchanged.

## Final candidate 3

| file | SHA-256 | rows |
|---|---|---:|
| **candidate 3** `runs/final_blind/final_candidate3/chain_out_cap.parquet` = candidate 2 + the cap | `b673791dfdaf8e909175001387183946236075280c9b3457822b52157c9cebc8` | 670,790 |
| candidate 2 (unchanged; fallback) | `f11ca28f7503a98cad9211facafe30d7f4185a589aefdb655372896d59e2669e` | 670,790 |

Receipt: [`final_candidate3_receipt.json`](final_candidate3_receipt.json).

Reproduce: `.venv\Scripts\python.exe -X utf8 -B tools/run_lirf_cap.py apply --base runs/final_blind/final_candidate2/chain_out_km.parquet
--movements data/raw/final_ranking.parquet --out runs/final_blind/final_candidate3/chain_out_cap.parquet`.

Upload (only at the user's word; one upload, never overwritten):

    .venv\Scripts\python.exe -X utf8 -B tools/prc_submit.py upload-final --source runs/final_blind/final_candidate3/chain_out_cap.parquet --expected-sha256 b673791dfdaf8e909175001387183946236075280c9b3457822b52157c9cebc8
