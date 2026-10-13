# The final submission: provenance of `vigorous-whistle_final.parquet` (candidate 6)

The final phase asks for one file on `final_ranking.parquet` (January, February, June, July 2026; 670,790 departures), uploaded once
as `vigorous-whistle_final.parquet`. This page is the index from the supplied
file to the recommended final file. Each step points to its tool, its declaration and its detailed record. The upload itself is
the user's decision; nothing here uploads.

| | file | SHA-256 |
|---|---|---|
| input | `data/raw/final_ranking.parquet` | `4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f` |
| template | `data/raw/final_submitting.parquet` | `b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924` |
| **final file (candidate 6)** | `runs/final_blind/final_candidate6/chain_out_pm.parquet` | **`a7ac1f60c5ddd79ef6927887b50471b3160748c386ed8186d99095bee6b63a81`** |

## The chain

The file is the v76 chain ([`v66_chain.md`](v66_chain.md), [`v69_chain.md`](v69_chain.md)) run on the final file, plus four final-file
steps (19-21 below). v76 scored 214.3353 on the ranking board. The chain is parameterised for a new movement file, and on `ranking.parquet`
it reproduces v76 byte for byte ([`lane_E.md`](lane_E.md), [`lane_L.md`](lane_L.md)).

| # | step | tool | declaration / decision | output (SHA-256) |
|---|---|---|---|---|
| 0 | inputs for the four months: adsb.lol surface days, IEM weather, adj2 boosters, data-link messages; EUROCONTROL workbooks | `tools/final_predict.py inspect / external / extract-adsb / pull-weather / refit-adj2`, `tools/oooi_pin_extract.py` | [`prestage.md`](prestage.md) | receipts in `prestage.md` |
| 1 | **model part**: features, v9, leaves, networks, residual, recipe, repair, rematch, members, stack (layer set v55m), **v40's fill pins off** | `tools/final_predict.py run --layers v55m --no-fill-pins` | [`runner.md`](runner.md), [`lane_V.md`](lane_V.md); pins off by the user's decision of 8 October ([`final_round_plan.md`](final_round_plan.md)) | `runs/final_blind/run_final4m/final/vigorous-whistle_v9001.parquet` `cdb81e08...` |
| 2 | own-aircraft member (32 networks, CPU) | `tools/final_own_member.py features / predict / export-v19own` | `configs/ctx_ownpoints_weight_v1.json` (v66) | `R/members`, `early_9002/features_v19own` |
| 3-10 | early chain: own-points member, v48 daily reconciliation, extreme-day, variance re-allocation, guarded stacker, identity, intruder bounds, own-motion v1 / v2 | `tools/final_early_chain.py --published-missing auto` | [`lane_E.md`](lane_E.md), [`v69_chain.md`](v69_chain.md) | `runs/final_blind/early_9002/vigorous-whistle_v9002.parquet` `4c393769...` |
| 6' | **six-month C6 stacker** (four development months + Tier B February / June 2025), early chain resumed at step 6 | `tools/run_sixmonth.py fit-final --kind stack --override configs/sixmonth_override_v1.json`, then `tools/final_early_resume6.py --stack-preds` | `configs/sixmonth_override_v1.json` (disclosed override); January / July reading `configs/sixmonth_devmonths_v1.json` | preds `48a184a6...`; `runs/final_blind/early_c4/vigorous-whistle_v9002.parquet` `d5834fcb...` |
| 11-18 | late chain: data-link OUT pin, neighbour state (unmatched / dark), **six-month** second pass (override) and third pass (accepted), W25 passes and flight-number memory (January only) | `tools/final_late_chain.py init / step / fit`; `tools/run_sixmonth.py fit-final --kind ss --override ...` and `--kind st` | [`lane_L.md`](lane_L.md), `configs/sixmonth_refit_v1.json` (C15 accepted), `configs/sixmonth_override_v1.json` (C14) | `runs/final_blind/late_c4/key_memory/rank/dry_v76/vigorous-whistle_v76.parquet` `ab25a196...` |
| 19 | LIRF fill-population dominance cap, min(pred, max(s, 88,392 s)) | `tools/run_lirf_cap.py apply` | `configs/lirf_cap_v1.json` ([`../lirf_cap/README.md`](../lirf_cap/README.md)) | candidate 4 `runs/final_blind/final_candidate4/chain_out_cap.parquet` `8fc8828d...` |
| 21 | published monthly taxi-out level shift (EGLL February / June / July, LIRF July non-population) | `tools/run_published_means_v4.py apply` | `configs/published_means_v4.json` (disclosed override) ([`../published_means/final_v4.md`](../published_means/final_v4.md)) | **candidate 6** `a7ac1f60...` |

Step 20 in [`v69_chain.md`](v69_chain.md) is the six-month learners (rows 6' and 11-18 above). The exact VM script of steps 6' to 19 is
[`../sixmonth/c4/run_c4.sh`](../sixmonth/c4/run_c4.sh). It ran on Azure arcw0, after two controls had reproduced candidate 3's early output
and candidate 2 byte for byte. Its record is [`../sixmonth/final_candidate4.md`](../sixmonth/final_candidate4.md).

## Candidates and why candidate 6

| candidate | SHA-256 | = | record |
|---|---|---|---|
| 1 | `30e116a9...` | the v76 chain on the final file, pins off | [`final_candidate.md`](final_candidate.md) |
| 2 | `f11ca28f...` | 1 + six-month third pass (accepted by its rule) | [`../sixmonth/final_candidate2.md`](../sixmonth/final_candidate2.md) |
| 3 | `b673791d...` | 2 + LIRF dominance cap | [`../lirf_cap/README.md`](../lirf_cap/README.md) |
| 4 | `8fc8828d...` | 3 + six-month stacker and second pass (override; January / July 2025 reading better) | [`../sixmonth/final_candidate4.md`](../sixmonth/final_candidate4.md), [`../sixmonth/devmonths.md`](../sixmonth/devmonths.md) |
| **6** | **`a7ac1f60...`** | 4 + published-means level shift (override) | [`../published_means/final_v4.md`](../published_means/final_v4.md) |

No candidate 5 was built: the January / July reading's pre-set rule kept candidate 4 in every month. Each later candidate was checked
against the previous one, without labels. Rows, order and dtype match the template. The 619 LIRF fill-population rows, the guard rows
 and the cap rows behave as declared. The uploader's `--check-only` passes on every candidate.

## Disclosures

- **No official score enters the final file.** v40's nine score-derived LIRF fill pins are switched off (`--no-fill-pins`).
- **The guard rows** keep the chain's own values. Since 2 October, no published-data layer
  moves them, because the v56 score bore on them.
- **Overrides of declared rules in the final steps**, each written down before its file and argued on expected value:
  - the six-month C6 / C14 learners: February 2025's interval included zero; January / July 2025 read better afterwards;
  - the LIRF cap: accepted by a dominance argument, an approved exception to the rule that LIRF population rows stay unchanged;
  - the published-means level shift: on 2025 the procedure missed its rule by +0.0006% in January.
- **External data:**
  - adsb.lol `globe_history` archives (ODbL) and the Aircraft Data Links archive (CC0);
  - IEM ASOS observations (public domain);
  - OpenStreetMap-derived geometry (ODbL);
  - OPDI flight events;
  - EUROCONTROL's published monthly taxi-out and daily pre-departure-delay aggregates, which have no open licence. The organisers
    confirmed on 10 October that they may be used in the final phase.
- **Times in the final file are jittered by up to ±12 s**, as the organisers intended. Predictions anchored on the take-off time carry
  that noise, and the file's January / July values therefore differ slightly from v76's.

## Upload (the user's decision; the last upload counts)

**UPLOADED at the user's word ("upload it") at 16:02:28 UTC on 13 October 2026.** The read-back SHA-256 equals `a7ac1f60...`. The organisers' `vigorous-whistle_final.parquet_result.json` (16:02:41 UTC) reads `Succeeded`, `used_pairs` 670,790 against `final_submitting.parquet`. Record: [`final_upload.json`](final_upload.json).

**Final verification before upload (13 October):** PASS, [`final_verification.md`](final_verification.md) (`tools/verify_final_file.py`).

A re-upload overwrites the previous final file. The organisers take the file in the bucket at
the deadline. The first upload refuses to overwrite anything. A replacement must name the SHA-256 of the file it replaces.

    .venv\Scripts\python.exe -X utf8 -B tools/prc_submit.py upload-final --source runs/final_blind/final_candidate6/chain_out_pm.parquet --expected-sha256 a7ac1f60c5ddd79ef6927887b50471b3160748c386ed8186d99095bee6b63a81

A later replacement:

    .venv\Scripts\python.exe -X utf8 -B tools/prc_submit.py upload-final --source <new export> --expected-sha256 <new sha> --replace-sha256 <sha of the file in the bucket>

Check-only run on 13 October 15:39 UTC: no final object in the bucket, organiser final files unchanged, candidate 6 passes
(`runs/submissions/vigorous-whistle_final.check.json`).
