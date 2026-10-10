# February / June 2025 check of the chain layers (10 October 2026, survey and design)

**Why.** Every layer of the v76 chain was declared and read on January / July / November / December 2025 (some also on all twelve
2025 months or on August-October 2025). February and June 2025 are the analogues of the final's new months. This page lists, layer by layer,
what is already known there, what is missing and what reading it would take, and proposes a keep / switch-off rule. It is a read-only
survey: no fit was run, no official score was used.

## The one out-of-sample base that covers February / June 2025

`runs/daily_totals/rows12.parquet` (`tools/run_daily_totals.rows12_table`): all twelve 2025 months, `stack` = max(cross-fit v9 + mean of the
twelve-month LOMO residual learners adj2 / cat on matched non-LIRF rows, −12) (the v23 recipe, `scratch/v24/<month>.parquet`,
`runs/adsb_v25/dev/predictions_<month>.parquet`). It also carries 14,092 "extra" departures missing from the cross-fit tables at their
airport-month median prediction. On the twelve-month check's rows its February / June RMSE is 201.007 / 199.788 s (`stack_rmse` in `docs/daily_totals/check12_report.json`).
**No out-of-sample February / June prediction exists for the committee side** (the tree leaves, the context networks, the member networks,
the LIRF tables, the own-points member): the seasonal fold's seven fitting months, the forward fold and both refits
contain them, and the LOMO round (`docs/lomo_dev/README.md`) held out only
January and July. Every layer whose inputs include committee
or network predictions (the recipe weights and gates, the members, the stackers, the late passes) therefore has no February / June reading.

## Layer table

Change = complete-month RMSE change; interval = paired-day 95% interval of the RMSE difference in seconds (as stored). "rows12" = the base
above; each layer's reading is on that base after the earlier accepted layers as recorded in its own result file.

| # | layer (version) | Feb 2025 | Jun 2025 | base | source | status |
|---|---|---|---|---|---|---|
| R1 | v30 recipe, v9 side: residual stage (v23, mean adj2 / cat) on v9, matched non-LIRF scope | scope RMSE 156.173 → 100.038 | 171.101 → 108.342 | cross-fit v9, LOMO residual | `docs/adsb_rnn/development_results.json` (`months/<m>/scope`) | read (component, not a chain layer) |
| R2 | v26 residual network, w90 (inside v30's v9 side) | scope 100.038 → 96.321 (−3.72%) | 108.342 → 105.986 (−2.17%) | same scope, held-month nets | same file, `candidates/rnet_w90` | read (component) |
| R3 | v30 recipe, committee side (trees + context networks) and the mix | — | — | — | — | **no reading** (needs committee OOS) |
| R4 | v31 EOBT_1 repair (detector + 50% edit) | — | — | — | `docs/final_file/runner.md` | no 2025 analogue by construction (2025 shares define the threshold); **fires on LFPG June 2026** in the final file (26.7% vs 10.2%) |
| R5 | v33 / v34 recipe weights | — | — | — | — | no reading |
| R6 | v35 (u300, callsign fix 50%, queue trust 0.75) | — | — | — | — | no reading |
| R7 | v36 (mu600 gate, peer-congestion lifts) | — | — | — | — | no reading |
| R8 | v37 / v38 tail stretch | — | — | — | — | no reading |
| R9 | v39 / v40 fill pins | — | — | — | — | **off in the final file** (user, 8 October) |
| R10 | v45 LIRF disagreement gate | — | — | — | — | no reading |
| R11 | v46 network share / networks-far-below gate | — | — | — | — | no reading |
| R12 | v51m / v52m combination and masked members | — | — | — | — | no reading |
| R13 | v53m / v54m / v55m LIRF members, dark re-read | — | — | — | — | no reading; the dark rule (match share < 5%) **fires on LIRF June 2026** (0% matched), the regime it was declared for (July 2026) |
| C1 | own-points member at 1.00 (v66) | — | — | — | — | no reading |
| C2 | no-NM ADS-B rematch (v59) | −0.0140% [−0.172, −0.020] | −0.0084% [−0.060, −0.020] | rows12 incl. extras (control 467.50 / 426.62 s) | `docs/rematch_nonm/results_v1.json` `twelve_months` | read: helps |
| C3 | v48 daily-totals reconciliation | **−0.7648%** [−2.781, −0.560] | −0.3049% [−0.897, −0.364] | rows12 | `docs/daily_totals/check12_report.json` | read: helps |
| C4 | extreme-day calibration | −0.4498% [−2.251, +0.086] (6 LTFM days) | −0.0343% [−0.211, 0.000] (1 EDDF day) | rows12 after C3 | `docs/scout/extreme_shift_results_v1.json` `twelve` | read: helps |
| C5 | variance re-allocation | −0.4993% [−1.821, −0.327] | −0.1102% [−0.365, −0.093] | rows12 after C4 | `docs/scout/var_alloc_results_v1.json` `twelve` | read: helps |
| C6 | guarded stacker (v3 learner, OPDI at LSZH, weight 0.65, ten seeds) | — | — | — | `docs/stack_resid/README.md` ("no twelve-month check exists") | **no reading** (learner inputs need committee OOS) |
| C7 | intruder stand bound (g3600_w100) | −1.0859% [−3.201, −0.201] | −0.2103% [−0.372, −0.115] | rows12 `stack` | `docs/intruder_bound/results_v1.json` | read: helps |
| C8 | identity extension (same-type intruders) | −0.0390% [−0.242, −0.0003] | −0.0116% [−0.0211, −0.0029] | rows12 | `docs/intruder_identity/results_v1.json` `results/twelve` | read: helps |
| C9 | own-motion bound v1 (om_b2700_w100) | −0.00965% [−0.0373, −0.0027] | −0.01457% [−0.0317, +0.0084] | rows12 after C7-C8 | `docs/own_motion/results_v1.json` | read: helps |
| C10 | own-motion bound v2 (om5_b2700_w50, shipped by override) | −0.03658% [−0.1448, −0.0112] | −0.02333% [−0.0536, −0.0113] | rows12 | `docs/own_motion/results_v2.json` | read: helps |
| C11 | data-link OUT pin at LFPG (w50) | — | — | — | `docs/oooi_pin/results_v1.json` (fresh only: Aug −0.0222, Sep −0.0216, Oct −0.0722%) | **no reading**; cheap (Tier A) |
| C12 | neighbour state (w75) | — | — | — | `docs/nbr_state/results_v1.json` (fresh only: Aug −0.0255, Sep −0.0240, Oct −0.0630%) | **no reading**; cheap (Tier A) |
| C13 | neighbour state on dark rows | — | — | — | `docs/nbr_state/dark_results_v1.json` (fresh only: Aug −0.0043, Sep −0.0087, Oct −0.0216%) | **no reading**; cheap (Tier A) |
| C14 | second guarded pass (sp_hicov7) | — | — | — | `docs/stack_second/README.md` (no fresh-month check possible) | **no reading** (inputs = stacker tables) |
| C15 | third pass with raw ADS-B (t3_ro_3) | — | — | — | `docs/stack_third/README.md` | **no reading** |
| C16-18 | W25 passes v1 / v2, W25 flight-number memory | n/a | n/a | — | `docs/season_pass/`, `docs/season_pass2/`, `docs/key_memory/` | act on January 2026 only; change nothing on February / June 2026 |

**What the table says.** Every layer that can be read on February / June 2025 helps there. February is the strongest month of the
twelve for the published-delay layers C3-C5 (−0.765 / −0.450 / −0.499%), and the intruder bound's February reading (−1.086%) is second
only to July's. The unknowns are the whole committee side of
the model part (R3, R5-R8, R10-R13), the own-points member (C1), the stacker (C6) and the late passes (C14-C15), plus three cheap ones
(C11-C13). Two layers fire on June 2026 for regime reasons with no 2025 analogue: the EOBT_1 repair (R4, LFPG) and the LIRF dark rule
(R13).

**Stakes of the late passes (arithmetic, no score).** A layer with squared change S on N rows can at most add about S / (2 N R) to a month RMSE
R if its moves carry no information. With the recorded ranking-file values (second pass S 1.82e7, third pass 1.04e7, N 344,841) and R ≈ 200 s
(the rows12 February / June scale) that is about 0.13 s and 0.08 s. Their harm on the new months is bounded and small; their gain is the
open question the season-adjacency finding raises (`docs/season_pass/README.md`: late-pass gains came mostly from November / December, and in
the one-month-fit test every non-adjacent month pair lost).

## The W25 layers and February 2026; anything for June

- **February 2026 is in winter season W25** (26 October 2025 - 28 March 2026), like November / December 2025 and January 2026. The three W25
  layers were declared for January 2026 only and are hard-wired to it (`tools/final_late_chain.py`, `docs/final_file/lane_L.md`).
- **There is no clean 2025 analogue for "fit on November + December, apply two months later in the same season".** The season-adjacency test
  only had the four development months; its only same-season pair was November / December (adjacent). Within 2025 the closest analogue is
  W24: fit on January 2025, read March 2025 (one month between, same season), with February → March as the adjacent control.
  - For **key_memory** (flight-number mean residual) this analogue can be read on rows12 residuals (CPU, minutes). The base is weaker than the
    chain, so the memory signal will look larger than on the chain. A null or loss on March 2025 would rule out a February 2026 declaration.
  - The **season passes** need the stacker / late-pass inputs on January-March 2025, which do not exist outside the development months:
    Tier B only.
- **June 2026 is in S26.** No S26 month is labelled; applying S25 months to June 2026 is "the same calendar season a year apart", which the
  adjacency test found to lose (EGLL December → January 2025 +0.016%). No June layer is proposed.

## Proposed declared check (to be committed before any new reading)

Default: the final file keeps the accepted v76 chain (the user's decision of 8 October). A layer is **switched off on the final file's
February 2026 rows** only if its February 2025 reading is worse (change > 0) **and** its paired-day 95% interval lies entirely above zero;
the same for June with June 2025. No partial weights, no re-tuned thresholds, no airport choices; a layer that cannot be read stays on.
This mirrors v57's clauses in the protective direction (clause 1, "no month worse", is the trigger; clause 3, "interval clear of zero",
guards against switching on noise). Readings on rows12 state their base; readings on a Tier B base replace them for the same layer.
Under this rule nothing read so far switches off.

## Compute plan

**Tier A (CPU, today, about 2-4 hours, no GPU; arcw1 / arcw3-5 / arcw7 or this workstation; never arcw0, never arcw2 / arcw6).**
1. C12 / C13 on February / June 2025: `tools/run_nbr_state.py` / `tools/run_nbr_dark.py` with the fresh-month list (`FRESH`, and the `RAW` month-file map) set to
   2025-02 / 2025-06 (rows12 `stack`, v9 from `scratch/v24`, gates from `data/external/adsblol/matches/matches2_2025-0{2,6}.parquet`, all present). Minutes.
2. C11: `tools/oooi_pin_extract.py list/run --months 2025-02,2025-06` and `oooi_pin_pack.py` on one VM, then `oooi_pin_join.py` and the pin's
   reading on rows12 LFPG rows with the declared offsets (from July / November / December 2025). Extraction a few hours (the 2026 pre-staging
   did seven months, `docs/final_file/prestage.md`).
3. The key_memory analogue on rows12 residuals (January → March 2025 and February → March 2025), as a screen for a possible February 2026
   declaration.

**Tier B.** A February / June
2025 out-of-sample base for the committee side, so R3, R5-R8, R10-R13, C1, C6, C14-C15 can be read:
- tree leaves: the LOMO tool (`tools/run_adsb_learner_lomo.py`) for 2025-02 / 2025-06, 29 leaves × 2 = 58 fits; the January / July round took
  about 11 h on seven workers, so about 15 h on arcw1 / arcw3 / arcw4 / arcw5 / arcw7. The cross-fit v9 already exists for both months.
- then the recipe / stack arithmetic (v30 → v55m) on those months and the chain tables (member, stacker inputs, second / third pass inputs),
  and the frozen learners applied to February / June 2025.
Started today this lands on 12 October at the earliest, leaving one day to absorb any switch-off into the final build. Expected value is
modest: everything readable helps, and the late passes' worst case is about 0.1 s each. The committee-side layers' stakes are not bounded
this way.

## Results of Tier A (10 October 2026, `configs/febjun_check_v1.json`, commit 61f904f, declared before these readings)

Tool `tools/run_febjun_check.py`; results [`febjun_results_v1.json`](febjun_results_v1.json). Every layer is read with its own declared
fresh-check code, with the month list as the only parameter. Before reading the new months, the same path reproduced the stored
August-October readings exactly: nbr_state, nbr_dark, and the OUT pin with its July / November / December offsets (2N −62.2606 s, 8D −22.8427 s).
One change to the declared path: take-off times are looked up in all twelve 2025 raw files, not only the month's own file. rows12
holds one February "extra" LTFM row that sits in the March file, and the declared
lookup refused it. The change only adds lookups, as the exact reproduction shows.

| layer | Feb 2025 | Jun 2025 | rows moved Feb / Jun |
|---|---|---|---|
| nbr_state_v1 w75 (rows12 `stack`) | −0.00645% [−0.0246, −0.0024] | −0.00862% [−0.0205, −0.0045] | 6,786 / 8,073 |
| nbr_dark_v1 (on nbr_state) | −0.00259% [−0.0099, −0.0009] | −0.00132% [−0.0055, +0.0020] | 46,998 / 37,068 |
| oooi_pin_v1 w50 (LFPG, rows12 `stack`) | **unreadable** | −0.00939% [−0.0182, −0.0019] | — / 61 |

- **OUT pin, February:** the adsb.lol Aircraft Data Links archive has no asset before 14 June 2025 (listing of 10 October,
  `runs/febjun_check/oooi/assets_all.json`), so February 2025 cannot be read and the layer stays on. The June 2025 reading covers 14-30 June
  only: 61 pinned LFPG rows (57 2N, 4 8D). The final file's February / June 2026 messages are pre-staged for whole months.
- **Extraction:** `tools/oooi_pin_extract.py` (git blob, SHA-256 `8a3f27f2...`) on **arcw1**: 2,071 files, 0.27 GB, 15 s, 0 digest
  mismatches. Packed with `oooi_pin_pack.py` (`e573a2d9...`). Mirrored with all six output hashes checked
  (`messages_unique.parquet` `c8cf467d...`, `runs/febjun_check/oooi/vm/mirror.sha256`). **arcw1 was deallocated right after the mirror.**
  arcw0 (Lane V) was not touched.

**Rule (declared): nothing is switched off.** All eleven readable layers are better on February and June 2025. The eight known before
the declaration are in the table above; the three new ones are in this section. The two intervals that touch zero (nbr_dark June,
own-motion v1 June) do not lie above zero. Every unreadable layer stays on. The final file keeps the full v76 chain on February / June 2026.

### key_memory W24 analogue (screen only, not governed by the rule)

key_memory_v1's arithmetic on rows12 residuals, March 2025 as the target:

| memory → target | km_10 (all ten) | km_9 (no EGLL, shipped set) | gain / squared change |
|---|---|---|---|
| January → March 2025 (one month between, same season W24) | **+0.00918% [+0.0014, +0.0271]** | +0.00603% [−0.0009, +0.0198] | −0.88 / −0.68 |
| February → March 2025 (adjacent control) | −0.01909% [−0.0542, −0.0065] | −0.01902% [−0.0544, −0.0064] | +1.19 / +1.26 |

The flight-number memory carries to the next month but not across a one-month gap inside the same season: with January's memory,
March gets worse (all ten airports, interval above zero). The analogue of "November + December 2025 → February 2026" is a one-month gap,
so this screen gives **no support for a February 2026 declaration of key_memory**. The season passes cannot be screened without Tier B.
Caveat as declared: rows12 is a weaker base than the chain.
