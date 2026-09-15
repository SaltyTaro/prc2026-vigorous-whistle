# Selected surface recipe refit

**V7 is uploaded and officially scored: 275.0570 seconds RMSE on all 344,841
rows**, improving v6 by **2.3938 seconds / 0.8628%**. It was uploaded once at
20:01:56 UTC on 14 September and processed at 20:02:30 UTC. The
[official submission receipt](submission_v7_official.json) binds the export,
remote readback, official result and dated standings. At 20:04:02 UTC, the
team ranks **19th of 135**, up from the preceding v6 snapshot's 21st. Third
place is 252.9192 seconds, requiring another **8.05%** reduction from v7 to
that dated score; see the [public snapshot](leaderboard_snapshot.json).

The [airport-feature campaign](../surface_airports/README.md) selected
`full_timeline_v6_100` after all twenty new airport fits and four fixed recipes
completed. Complete seasonal/forward RMSE is **322.8193230817628 /
221.13815634167557 seconds**, improving v6 by **0.6456% / 0.9802%**. These reused
development scores remain separate from the official result above.

**The refit and both full verifications are complete.** The submitted file is
`runs/submissions/vigorous-whistle_v7.parquet`, SHA-256
`534e1031a068a743515eeb44b4b8f13676c5626d6c18c9416da092d58b0b924e`. [Remote verification](submission_remote_verification.json)
confirms the complete uploaded bytes and all unchanged prior object signatures.

The [refit declaration](../../configs/surface_refit_v1.json) was recorded at
**18:50:40 UTC on 14 September**, after successful Windows and Azure development
verification. It binds the immutable selection, original native settings and
the ties-to-even rounded medians of the two October stopping counts. No new
feature, selection weight or candidate is introduced.

For departures with an NM clock, the selected prediction is **25% timeline
CatBoost global + 25% timeline LightGBM global + 50% new airport LightGBM**.
Every missing-clock prediction uses the original v6 pipeline unchanged. All
native learners use the frozen 276-feature combined timeline contract and
the original signed NM residual target. No label or prediction is clipped.

| Component | Fixed refit trees |
| --- | ---: |
| CatBoost global | 5,971 |
| LightGBM global | 1,612 |
| EDDF | 902 |
| EDDM | 1,142 |
| EGLL | 1,172 |
| EHAM | 744 |
| LEBL | 773 |
| LEMD | 1,014 |
| LFPG | 1,227 |
| LIRF | 880 |
| LSZH | 748 |
| LTFM | 1,002 |

The twelve components total **17,187 new trees**. The refit assignment must
match v6's 1,919,370 January-November rows and 1,898,549 clock-present fitting
rows exactly. Actual UTC December movements and every flight group appearing
in December remain protected. No stopping or scoring set is used during refit.

Source is isolated in `src/prc2026_surface_refit/`, preserving the completed
airport campaign's model-code hash. The refit model-code SHA-256 is
`4391d8dc1647f32b2175737bf546f72b5418328f4ac14c069319d6bc9827874b`.
The [declaration receipt](declaration.json) binds every source hash. The
ten new guard/routing tests pass on Windows and Azure; the [combined test record](../next_lever/tests_combined.json)
covers all 196 current test identities. Dependencies remain those in
`requirements-surface-airports.txt`.

The one authorized arcw0 refit launched at **18:52:05 UTC**, PID **59022**, log
`runs/surface_refit/refit.log`. It completed as
`20260914T185207842456Z_refit_full_timeline_v6_100`, using **1,800.21 seconds**
and **15.15 GiB** sampled peak RSS. Its run receipt SHA-256 is
`0c1a0fcd1c2aa6034e28eca1b56dadb567aff5269cac823067c4c097a84498e0`.
The initial 7,357-row reload sample reproduced exactly; both final verifiers
then checked every ranking row. Results remain in `runs/surface_refit/`, with
the complete Windows mirror in `runs/azure_arcw0_surface_refit/`.

`tools/verify_surface_refit.py` checks the saved December-protected assignment,
every component's fitting IDs and encoder, all fixed counts and settings, all
copied v6 artifact hashes, and all **344,841** ranking predictions. A separate
calculation reconstructs the 25/25/50 weights from native components and checks
every one of the **5,290** missing-clock predictions against original v6.
The [Azure verifier](verification.json) and [Windows verifier](local_verification.json)
both pass, with **zero difference on all 344,841 predictions** and on the
independent weight reconstruction. All 5,290 missing-clock values also match
the original v6 export after integer rounding. The later upload was explicitly
authorized by the user and introduced no model or prediction changes.

The [refit mirror](refit_mirror.json) restores all **156 files** from a
**103,114,456-byte** archive, SHA-256
`f8fc451ade49df31e4f043774acda46958edbf275b506cacb92a0d0f0729ae9e`.
The archive was streamed directly to Windows, preserving the Azure originals.
The [final process check](final_execution_state.json) found no remaining campaign
training or evaluation jobs; arcw0 remains running with about 1.20 GiB free.
The [Azure allocation check](azure_allocation_check.json) shows arcw1-arcw7
deallocated. Only the existing arcw0 was used.

To independently reproduce the final native verification locally, use a new
report filename:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools\verify_surface_refit.py --runs runs\azure_arcw0_surface_refit --development-runs runs\azure_arcw0_surface_airports --timeline-runs runs\azure_arcw0_timeline --previous-runs runs\azure_arcw0_nmclock --output docs\surface_refit\reproduction_verification.json
```

V1-v7 files are immutable and must not be overwritten or re-uploaded. December remains reserved. No training or
Azure resource changes were needed for the upload.

The [final campaign summary](../next_lever/campaign_summary.json),
[preservation check](../next_lever/preservation_final.json) and
[source archive receipt](../next_lever/final_source_package.json) bind the
completed modeling/export result and its reproducible evidence. They retain their
historical pre-upload state. The separate [official record](submission_v7_official.json)
captures the later submission, while the [integrity check](submission_integrity_20260914.json)
preserves all seven exports and the original source/archive receipts. Earlier
narratives are saved in [the document snapshot](submission_docs_before_20260914/manifest.json).
