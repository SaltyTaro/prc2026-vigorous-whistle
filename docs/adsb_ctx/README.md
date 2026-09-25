# Context network (adsb_ctx), 25 September 2026

**Question.** Most of the remaining error sits on rows the ADS-B stack cannot see: unmatched LTFM, LFPG and LIRF rows carry 45–60%
of ordinary development SSE. At LTFM, LFPG and EGLL, 2% of the unmatched rows (errors above 10 minutes) carry 39–44% of it, and
in those rows the NM clock is also far from the label (only 7–11% within 5 minutes): holds after pushback that no input observes.
Every model so far sees the traffic around a departure only through hand-made aggregates (timeline surface counts, `nb_*`
medians). The context network reads the individual movements around each departure instead.

## The network (`src/prc2026_ctx/`)

- **Tabular branch:** the committee network's (`prc2026_nn`, unchanged): the 314 x19 columns, piecewise-linear embeddings,
  MLP 512×3.
- **Context branch:** up to 128 movements at the same airport with movement time in [t0 − 60, t0 + 30] min (t0 = the focal
  takeoff), a contiguous time run centred on t0 − 15 min, with the focal row excluded. Each token is described relative to the
  focal departure by 21 features:
  - times relative to the focal takeoff and to its NM off-block clock;
  - its own NM proxy, schedule delay, and arrival in-block time and taxi-in;
  - ADS-B motion start, first point, taxi, stopped time and gate class;
  - same runway / stand / stand group;
  - wake category.
  An MLP encodes each token; attention (query from the focal inputs, 4 heads) and a masked mean pool them.
- **Label contract** (tested in `tests/test_ctx.py`): departure tokens never carry block or taxi time, in any file. Arrival tokens
  carry them, as the supplied ranking file does for every arrival (checked: ranking DEP rows have neither).
- **Tokens** are built by `tools/build_ctx_tokens.py` in about 25 s. All 13 files reproduce bit for bit on all eight workers
  (content hashes in `data/ctx/tokens_v1/manifest.json`).

## Design (October only) — [`configs/adsb_ctx_design_v1.json`](../../configs/adsb_ctx_design_v1.json), commit `a52b012`

Eight fits on arcw0–7 (03:22–04:28 UTC): ctx seeds 1–6, ctx_noadsb and ctx_dep at seed 1, each on the seasonal fit partition
(1,130,307 rows, LIRF excluded) with October monitored. Report:
[`design_report_1.json`](design_report_1.json) (`tools/ctx_design_report.py`).

**R1, standalone, October non-LIRF RMSE (s):**

| seeds | context | plain network (base) |
|---:|---:|---:|
| 1 | 163.558 | 167.741 |
| 2 | 158.891 | 163.784 |
| 4 | 156.458 | 161.809 |
| 6 | **155.396** | 160.975 |

The 6-seed context mean also beats the incumbent's 32-seed plain pool (160.311) by 3.1%.

**R2, incremental on the recipe proxy** (all 183,761 October rows of the proxy table; the committee table
0.5 × committee + 0.5 × [(1 − b) × nnb32 + b × ctx6]):

| b | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---:|---:|---:|---:|
| recipe proxy | −0.356% | −0.673% | −0.952% | **−1.192%** |

At b = 1.0 the paired-day interval is [−2.92, −1.48] s. Unmatched non-LIRF rows move −2.43%, matched −0.45%.
By airport: **LTFM −3.95%**, EGLL −1.13%, LFPG −1.07%, EDDF −0.61%, EDDM −0.60%, LEMD −0.53%, LEBL −0.49%, EHAM −0.34%,
LSZH −0.15%; LIRF is untouched. Seed scaling at b = 1 (1 → 6 seeds): −0.54, −0.93, −1.08, −1.10, −1.17, −1.19%, still
rising. **Decision: GO** (bar −0.10% with the interval below zero).

**R3, ablations at one seed:** ctx s1 163.56 (−0.43% at b = 0.5), ctx_noadsb 161.72 (−0.56%), ctx_dep 163.68 (−0.45%). The
single-seed spread of ctx (160.6–163.6) is larger than these differences, so neither ADS-B in the tokens nor arrival tokens is
shown to matter. The gain is concentrated where ADS-B is absent (LTFM), so it comes from the supplied movements themselves.

A note on row sets: the recipe proxy's October table has 183,761 rows; the bundle's has 185,674, of which 171,532 are
non-LIRF. The per-epoch monitor in the run logs uses all monitored bundle rows, so it reads about 4 s higher for every network
(base s1 171.66 against 167.74 here). The comparison has the same direction and size on both: ctx s1 167.65 against base s1
171.66 on the monitored rows.

## Development round — [`configs/adsb_ctx_v1.json`](../../configs/adsb_ctx_v1.json)

Sixteen fresh seeds (101–116) of variant `ctx` (`configs/adsb_ctx_network_v1.json`, the design settings unchanged). Folds:
seasonal, forward, January–November refit (December leaf) and refit12. Two lanes per worker on arcw0–7.

Candidates change the committee mix's network table against the v26 control:
- `ctx_b50`, `ctx_b75`, `ctx_b100`: the context network takes b of the network table;
- `ctx_b100_s65`: as `ctx_b100`, with the network share raised from 0.5 to 0.65.

The selection rule is the adsb_nn/nn2 rule, unchanged. The control must reproduce v26's recorded development RMSE on all four
recipe months.
