# Season-weighted network test (27 September 2026) — [`configs/adsb_ctx_season_v1.json`](../../configs/adsb_ctx_season_v1.json), [`report.json`](report.json): no GO, closed

Follow-up to the time-gap test (docs/adsb_ctx_gap/: fitting January–June scored November–December 1.3% better than fitting May–October).
The unchanged `ctx_e32` network was fitted with the scored month's ±1-month fitting rows repeated 3 times, seed-matched to the GPU
development seeds (controls): January from the seasonal fold with target January (Feb ×3), July with target July (Jun, Aug ×3),
December from the refit fold with target December (Nov, Jan ×3). `tools/run_adsb_ctx_season.py`, `tools/ctx_season_chain.sh`; ~14 min
per fit on L40S. Node prc-ctx3c's GPU failed at start (Xid 154, "GPU reset required"; node stopped), so seed 403 was not run: the
readings use seeds 401, 402, 404 (4-seed means of the declaration → 3-seed means; the verdict does not depend on it).

| month (non-LIRF) | control | season-weighted | change | unmatched | matched | day interval (s) |
|---|---:|---:|---:|---:|---:|---|
| January 2025 | 324.50 | 324.95 | +0.14% | +0.23% | −0.74% | [−1.30, +1.53] |
| July 2025 | 206.62 | 211.20 | **+2.22%** | +2.62% | +1.29% | [+3.34, +5.99] |
| December 2025 | 150.56 | 152.38 | **+1.21%** | +1.17% | +1.27% | [+1.30, +2.37] |

- Repeating same-season rows makes the network worse, clearly in July and December. The time-gap test's season effect is about
  which months the fit contains, not something up-weighting reproduces; the twelve-month refit already contains both seasons.
- Unweighted reproduction through the new runner (sw0_s401 vs control s401, January): RMSE 328.80 vs 329.08 s — the runner is
  equivalent — but the two runs of the same seed differ by 84 s RMS per row (GPU nondeterminism), a reminder that single-seed
  comparisons of these networks are noise-dominated.
