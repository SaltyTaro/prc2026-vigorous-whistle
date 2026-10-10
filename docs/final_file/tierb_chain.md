# Tier B, chain side (lane C): February / June 2025 readings of C1-C15

Status: **tooling and reproductions done, declaration committed; waiting for lane M's February / June model part** (10 October 2026).
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
| stand-in (rows12 `stack` in every component; first F), whole chain | runs end to end; every late layer selects what it selected on the ranking file; W25 layers pass F through |

The only gap in the rows12 reproduction is the integer base (the stored readings used the float `stack`); interval ends differ because
the stored runs drew their bootstrap from other generators.

## Timings (workstation, stand-in)

Early chain 1,727 s (step 2 90 s, 3-5 209 s, 6 424 s, 7-8 53 s, 9 434 s, 10 518 s; each month-blocked step runs twice); late chain
about 160 s (fits of the second / third pass 62 s / 31 s locally).
