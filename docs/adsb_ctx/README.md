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

**Result (25 September, 10:47 UTC, [`development_results.json`](development_results.json)): `ctx_b100_s65` is selected.**
All four candidates are eligible; the declared rule picks the lowest mean ratio. The control reproduces v26 to 1e-9 s on
every month.

| candidate | 2025-01 | 2025-07 | 2025-11 | 2025-12 | mean |
|---|---:|---:|---:|---:|---:|
| ctx_b50 | −0.201% | −0.514% | −0.423% | −0.701% | −0.460% |
| ctx_b75 | −0.285% | −0.738% | −0.600% | −1.004% | −0.657% |
| ctx_b100 | −0.356% | −0.940% | −0.753% | −1.274% | −0.831% |
| **ctx_b100_s65** | **−0.407%** | **−1.099%** | **−0.859%** | **−1.730%** | **−1.024%** |

For `ctx_b100_s65`:
- **Whole-month RMSE:** 327.71 → 326.38 s (January), 266.03 → 263.10 s (July), 187.88 → 186.27 s (November),
  172.88 → 169.88 s (December).
- **Paired-day 95% intervals:** [−2.55, −0.75], [−3.68, −2.32], [−1.88, −1.37] and [−4.27, −2.06] s. Days better: 27/31,
  31/31, 30/30, 31/31.
- **Every non-LIRF airport improves in every month.** LIRF is untouched by construction. LTFM improves most: −3.6 / −5.1 /
  −3.4 / −5.0%. EGLL and EDDM gain about 1–2% in three of four months; LEMD and LFPG range from −0.1% to −1.8%.
- **Unmatched non-LIRF rows gain most**, e.g. December 215.2 → 208.0 s and July 245.3 → 238.9 s. Matched rows also improve.
- The mean gain rises monotonically with the network's weight. The grid's largest weight was chosen; no weight outside the
  declared grid was scored.

## Coverage-regime ship check — [`configs/adsb_ctx_regime_check_v1.json`](../../configs/adsb_ctx_regime_check_v1.json)

Declared after the selection and before any thinned score (commit `a01a8a8`). The check can only block an export. It asks
whether the gain survives the ranking months' adsb.lol coverage, using the thinned December and July sets of
[`docs/coverage_regime/`](../coverage_regime/README.md).

- **Tokens:** rebuilt from the thinned match tables. The natural rebuild reproduces the stored token file.
- **Networks:** all 16 re-predicted with models that never saw the month.
- **Recipe:** assembled from the coverage campaign's saved tables and residual members.
- **Reproduction of the natural rebuilds:** the recorded v26 and candidate RMSE reproduce within 0.0006 s (December) and
  0.00005 s (July). That campaign's local ADS-B feature rebuild is not bit-identical on a few rows, as documented there.

| month (regime) | natural | thinned | EDDF natural → thinned | paired-day interval (thinned) | gate |
|---|---:|---:|---|---|---|
| 2025-12 (January-2026 coverage) | −1.730% | −1.737% | −1.02% → −1.49% | [−4.31, −2.11] s | pass |
| 2025-07 (July-2026 coverage) | −1.099% | −1.098% | −0.86% → −1.55% | [−3.72, −2.35] s | pass |

The gain survives in full: 100% in December and 99.9% in July. Where coverage thins (EDDF), the context network helps
*more*; LSZH July moves from −1.21% to −1.05%. Reports: [`regime_check_2025-12.json`](regime_check_2025-12.json),
[`regime_check_2025-07.json`](regime_check_2025-07.json).

## Export v27 (25 September, 13:40 UTC) — uploaded, official 230.8057

[`tools/export_adsb_ctx.py`](../../tools/export_adsb_ctx.py) → `runs/submissions/vigorous-whistle_v27.parquet`
(SHA-256 `5a877924b0637f4cb257870410b178901403502131c635984526d23a5a35cd1d`, 1,343,768 bytes). Record:
[`export.json`](export.json).

- **Inputs:** ens_ctx is the mean of the 16 context seeds' twelve-month refits, each a January–December fit on 1,924,343
  rows.
  - Every record carries the declared variant, code, configuration, seed and model hash, the declared token content and one
    ADS-B ranking table (`1dcf5c7f…`).
  - The inherited ens_nn / ens_new / ens_nnl tables equal v25's recorded columns.
- **Control rebuild:** v26 is rebuilt from its recorded components with the development scorer's code path. Committee mix
  and values differ from the recorded v26 by 0.0 s, and the integers equal the uploaded v26.
- **Candidate:** ens_ctx replaces the committee mix's network table at share 0.65.
- **Change against v26:** 298,457 rows change, RMS 22.0 s, mean −0.95 s, median 4 s, p99 86 s, max 779 s.
  - By airport (RMS): LTFM 38.6 s, EHAM 29.4 s, LFPG 26.3 s, EDDF 17.2 s. LIRF is unchanged.
  - By month (RMS): January 26.8 s, July 17.1 s.
  - ens_ctx correlates 0.993 with v26's network table on non-LIRF rows.
- **Independent verification** ([`verification_v27.json`](verification_v27.json), `tools/verify_adsb_ctx.py`):
  - All 16 context networks reload on Windows. From the exact cached ADS-B table and the declared ranking token file, they
    re-predict 5,000 sampled ranking rows within 0.0012 s.
  - Every ensemble column equals the plain mean of its saved seeds.
  - v9, residual and committee equal v26's components and the twelve-month committee table.
  - All 317,942 non-LIRF rows re-derived by plain arithmetic, max(0.3 (v9 + residual) + 0.7 (0.35 committee + 0.65 ens_ctx),
    −12), round to the exported integers. All 26,899 LIRF rows equal the uploaded v26. The file contract holds.
- **Audit** ([`export_audit_v27.json`](export_audit_v27.json), `tools/audit_export.py --version 27 --compare 26 25 9`):
  - v26's shape: the same five January airport-months above their development means (within 0.4 points of v26's ratios),
    the same flagged days, 37 predictions above 4 h (all LIRF, unchanged), and 7 negative predictions (v26: 9), none below
    −12.
  - The EHAM disruption days of early January 2026 rise slightly (4 January mean 2,396 s against 2,346 s).
  - No row moves more than 1,800 s.
- **Expected transfer (not a prediction of the official score):**
  - Development −1.02% (December −1.73%, the closest regime to January 2026; July −1.10%).
  - Under the ranking months' ADS-B coverage the gain was kept in full on both thinned months.
- **Uploaded with the user's "upload it"** (`tools/prc_submit.py upload --calendar-day`: Content-MD5 checked by S3 on write, then a post-upload check of the stored object's size and SHA-256 metadata) and **officially scored 230.8057 on all 344,841 rows** (processed 13:41:03 UTC, [`official_result_v27.json`](official_result_v27.json)): **−3.3381 s (−1.426%) against v26 (234.1438)**, larger than the development gain (−1.02%). v27 is the incumbent; never re-upload v1–v27.
