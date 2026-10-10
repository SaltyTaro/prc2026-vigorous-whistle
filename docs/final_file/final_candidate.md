# Final file candidate 1 (10 October 2026) — built, checked, NOT uploaded

The full v76 chain run on the final dataset (`data/raw/final_ranking.parquet`, 8 October 11:04 UTC version, still the live one on
10 October) with the user's 8 October decisions (`final_round_plan.md`): v40's fill pins off, the published-delay layers on (the
workbook covers January-July 2026). Execution record: [`lane_V.md`](lane_V.md) ("Resumed on 10 October" and after).

| file | SHA-256 | rows |
|---|---|---:|
| **candidate** `runs/final_blind/late_9002/key_memory/rank/dry_v76/vigorous-whistle_v76.parquet` (copy: `runs/final_blind/final_candidate/chain_out_km.parquet`) | `30e116a9a0ae4bdf5f2150938272b571bb20b980ac18a7e3bb3f0c48374f381b` | 670,790 |
| early-chain output (step 10) `runs/final_blind/early_9002/vigorous-whistle_v9002.parquet` | `4c3937691bc51b17e5c25379c88f419f7057c94138725ac9aac8f5f5bcde9f79` | 670,790 |
| model part (runner v55m, fill pins off) `runs/final_blind/run_final4m/final/vigorous-whistle_v9001.parquet` | `cdb81e08b845a2436ffb3e6b3b40d4463164c5a234294c5282c71a5ab2ad969b` | 670,790 |

The internal names `v9001` / `v9002` / `..._v76.parquet` are work-directory names, not board versions; nothing was written under
`runs/submissions` except the check log below.

## How it was produced

1. Runner on arcw0 (`tools/final_predict.py`, `--layers v55m --no-fill-pins`): features through recipe on 8 October; repair (LFPG June /
   July 2026 flagged), rematch, members, stack, final on 10 October.
2. Own-points member (32 networks, CPU) and its v19own export (`tools/final_own_member.py`).
3. Early chain steps 0-10 (`tools/final_early_chain.py ... --published-missing auto`) on arcw0.
4. Late chain steps 11-18 (`tools/final_late_chain.py`, `--messages runs/oooi_final/messages_unique.parquet --messages-merge files`)
   on arcw0. Every layer selected what it selected on the ranking file (no `selection_flag`): OUT pin w50, nbr_state w75, nbr_dark,
   second pass sp_hicov7, third pass t3_ro_3, season_pass w25_e3, season_pass_v2 w25b_5, key_memory km_9 (the three W25 layers act on
   January 2026 rows only, by declaration).
5. Everything mirrored here and hash-checked (`runs/final_blind/chain_9002.sha`, 132 files; `mirror_20261010.sha`, 265 files); all eight
   VMs deallocated.

Code changes made for the new file, all optional / no-ops on the ranking file: `prc_submit.py upload-final` (8adaf44),
`final_late_chain --messages-merge files` (0afc15b), `final_own_member export-v19own` reorder (26906b5), `final_early_chain` repair-root
fallback (f3e8152). Scientific settings and declarations unchanged.

## Checks (label-free)

- **Upload rehearsal** `tools/prc_submit.py upload-final --check-only`: 670,790 rows, exact `final_submitting.parquet` order, int32,
  minimum -12, maximum 120,636, 78 negative values; organiser ETags unchanged; no final object in the team bucket.
- **January / July rows against the shipped v76** (`tools/final_vs_shipped.py`; the final file repeats every ranking departure):
  median |difference| 5 s, p90 16 s, p99 46 s (the +-12 s take-off jitter); 92 rows above 300 s, mostly LIRF long-interval rows at
  discrete switches and the nine rows v40 pinned (pins off by decision). EHAM January moves more (RMS 20 s) because nbr_state's declared
  transfer guard is evaluated on the file being predicted: ratio 1.77 on the four-month file (applies) against 2.07 on the ranking file
  (blocked). LEMD sits at the guard's 0.5 edge on both files.
- **Monthly plausibility** (`tools/final_month_means.py`, against EUROCONTROL's published monthly taxi-out means, calibrated on 2025):
  no February or June airport-month outside its 2025 range; the four flags (January EDDF / EDDM / LSZH, LIRF July) are the same on v76.
- **February / June 2025 readings of the chain layers** (`febjun_check.md`, `configs/febjun_check_v1.json`): every readable layer helps
  on both months; the declared rule switches nothing off.

## Upload (only at the user's word; one upload, never overwritten)

    .venv\Scripts\python.exe -X utf8 -B tools/prc_submit.py upload-final --source runs/final_blind/final_candidate/chain_out_km.parquet --expected-sha256 30e116a9a0ae4bdf5f2150938272b571bb20b980ac18a7e3bb3f0c48374f381b

Deadline 14 October 2026 10:00 UTC. A later candidate (a new layer, a changed decision) replaces this one only through the same chain
and the same checks.
