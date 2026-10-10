# Tier B, chain side (lane C): February / June 2025 readings of C1-C15

Status: **complete** (10 October 2026, read 22:51 UTC on arcw6): **nothing switched off**.
Umbrella `configs/febjun_tierb_v1.json`; rule `configs/febjun_check_v1.json` (unchanged); this lane's readings declaration
[`configs/febjun_tierb_readings_chain_v1.json`](../../configs/febjun_tierb_readings_chain_v1.json); model part: lane M,
[`tierb_model.md`](tierb_model.md). Tool `tools/febjun_tierb_chain.py`. No shipped tool was changed.

## Construction

The final file's own chain code is run on a February + June 2025 movement file, so each step's output against its input is that layer's
reading (the chain with the layer against the chain with the layer passed through at that position, every earlier layer on).

- **F** `runs/febjun_tierb/chain/input/febjun2025.parquet` (`5ff8884a...`): every movement of the 2025 training files with MVT time in
  February or June 2025; departures without block time and label, as `data/raw/ranking.parquet`. 143,732 + 183,142 departures (the same
  departures as rows12's two months). Template `febjun2025_template.parquet` (`7d23346e...`, int32 nulls as `submitting.parquet`).
- **Runner root** `runs/febjun_tierb/chain/root` from lane M's `chain_<month>.parquet` (components, v30 ... v55m, members). Departures
  without a lane-M row (no out-of-sample v9) enter at their airport-month's median v55m in every column and are not read.
- **C1** own_all (dl32own matched / m25dl32own unmatched, 4 seeds of the GPU LOMO fits) at 1.00 in place of the combination member,
  then the masked member at 0.40, through lane M's stack functions (step 1's arithmetic of `tools/final_early_chain.py`).
- **C2-C10** `tools/final_early_chain.Chain`'s new-file path (`--published-missing auto`), steps 2-10.
- **Month-blocked fits.** The shipped ranking path fits five layers' offsets on all twelve 2025 months' labels (rematch, intruder bound,
  identity extension, own-motion v1 / v2). For F each of steps 2, 7-8, 9, 10 runs twice, February 2025 (June 2025) removed from every
  twelve-month fit, and keeps February (June) rows from the matching run. In the stand-in run each layer's single twelve-month fit
  dropped the held month and no other call did (`work_standin/chain_tierb.json`, `blocked_fit_calls`).
- **OPDI** features of F from the February / June 2025 event windows (fetched with `tools/opdi_stack_fetch.py --need`); the builder is a
  replica of `run_opdi_stack.build` with the window list given explicitly (the shipped `rank` branch picks the 2026 windows by name).
- **C11-C18** `tools/final_late_chain.py` on F (messages: Tier A's February / June 2025 extraction, `--messages-merge files`). The W25
  layers pass F through.

F also carries **January 2025's arrivals** as context (F `8a0de5bd...`, 806,950 movements): the final file holds January 2026 before its
February rows, so a 1 February departure's inbound turn is visible there; nothing precedes June in the final file, so June gets no May.

## Reproductions before the declaration (copies in [`tierb_chain_checks/`](tierb_chain_checks/))

| check | result |
|---|---|
| own_all routing on the 16 stored lomo_2025-01 fits vs `runs/ctx_ownpoints_dev/member/s6xx/2025-01.parquet` | equal (ids, NaN pattern, max \|diff\| 0.0) |
| OPDI builder replica, shipped `rank` window list, ranking file vs `runs/opdi_stack/features_rank.parquet` | equal (154,475 rows) |
| steps 7-10 of this path (month-blocked) on F with rows12 `stack` vs the stored twelve-month readings (already read before Tier B) | intruder bound Feb -1.0858% (stored -1.0859%), Jun -0.2102% (-0.2103%); identity ext. -0.0390% / -0.0116% (equal); own-motion v1 -0.0096% / -0.0146% (equal); own-motion v2 -0.0366% / -0.0233% (equal) |
| the same without January's arrivals in F | own-motion v2 Feb -0.0283%, intruder bound Feb -1.0793%: the reason for the context arrivals |
| identity table rebuilt for F (relearned key map) vs shipped | 195,600 movements identified by both, 0 different aircraft |
| month blocking | in each blocked run exactly one fit per layer spans all twelve months and loses the held month |
| stand-in (rows12 `stack` in every component), whole chain, first F and again on the current F (after the commit, `work_standin`) | runs end to end; every late layer selects what it selected on the ranking file; W25 layers pass F through |

The only gap in the rows12 reproduction is the integer base (the stored readings used the float `stack`); interval ends differ because
the stored runs drew their bootstrap from other generators.

## Timings (workstation, stand-in)

Early chain 1,399-1,727 s (step 2 90 s, 3-5 209 s, 6 424 s, 7-8 53 s, 9 434 s, 10 518 s; each month-blocked step runs twice); late chain
about 160 s (fits of the second / third pass 62 s / 31 s locally).

## Results (10 October 2026, 22:51 UTC; `docs/final_file/tierb_chain_results.json`)

Run on arcw6 (execution record [`tierb_vm.md`](tierb_vm.md)); the stand-in was first rerun there and reproduced this workstation's stand-in
byte for byte (model part `21f2d65c`, early final `8ca9dbc6`, all eight late outputs). Governing rows: lane M's rows read (February
142,622 of 143,732, June 180,640 of 183,142); intervals in seconds of RMSE.

| reading | Feb 2025 change [95% interval, s] | rows moved | Jun 2025 change [95% interval, s] | rows moved | off? |
|---|---|---:|---|---:|---|
| C1 own-points member | +0.709% [-0.024, +1.427] | 126,144 | -0.159% [-0.288, -0.035] | 159,465 | no |
| C2 no-NM rematch | +0.000% [+0.000, +0.000] | 0 | +0.000% [+0.000, +0.000] | 0 | no |
| C3 v48 reconciliation | -0.713% [-1.304, -0.247] | 132,801 | -0.353% [-0.509, -0.195] | 172,592 | no |
| C4 extreme-day | -0.470% [-1.190, -0.010] | 2,955 | +0.000% [+0.000, +0.000] | 0 | no |
| C5 variance re-allocation | -0.594% [-1.085, -0.177] | 105,142 | -0.188% [-0.278, -0.110] | 126,508 | no |
| C6 guarded stacker | -0.472% [-0.641, -0.332] | 126,618 | -0.410% [-0.587, -0.215] | 158,582 | no |
| C7 intruder bound | -0.518% [-1.101, -0.001] | 28 | -0.383% [-0.650, -0.138] | 21 | no |
| C8 identity extension | -0.268% [-0.833, -0.002] | 4 | -0.021% [-0.065, +0.000] | 2 | no |
| C9 own-motion v1 | -0.002% [-0.007, +0.000] | 6 | +0.045% [-0.016, +0.148] | 11 | no |
| C10 own-motion v2 | +0.034% [-0.014, +0.095] | 768 | -0.006% [-0.026, +0.016] | 646 | no |
| C11 OUT pin | +0.000% [+0.000, +0.000] | 0 | -0.033% [-0.073, -0.007] | 61 | no |
| C12 nbr_state | -0.016% [-0.027, -0.007] | 5,982 | -0.016% [-0.033, -0.000] | 7,412 | no |
| C13 nbr_dark | -0.007% [-0.015, +0.000] | 41,841 | -0.009% [-0.023, +0.003] | 40,435 | no |
| C14 second pass | -0.148% [-0.204, -0.104] | 84,282 | -0.290% [-0.365, -0.216] | 111,345 | no |
| C15 third pass | -0.050% [-0.067, -0.037] | 35,472 | -0.112% [-0.151, -0.074] | 48,126 | no |

**Rule outcome: no chain layer is switched off in either month.** Notes:
- **C1 (own-points member at 1.00) in February: +0.709%, interval [-0.024, +1.427] s** -- the closest call of Tier B; the lower end is just
  below zero, so it stays on. June -0.159%, interval below zero.
- C2 (no-NM rematch) moves no governing row: its rows have no NM clock and are outside the declared reading set (no out-of-sample v9);
  unreadable, stays on. C4 moves nothing in June (no extreme day). C11 in February: no data-link archive before 14 June 2025.
- One code fix during the real run (commit 03c476e): `c1` read the `own_all_w100` weight through `network_member.candidate_weight`,
  which refuses the candidate's extra `member` key; it now reads `member_weight` (1.00) as `tools/final_early_chain.py` does. The stand-in
  never reaches that path. No value or rule changed.
- The `read` step looked for five files absent on arcw6 (a December bundle and two selection records, optional probes); after uploading
  them the readings were recomputed and are identical.
