"""Writes configs/lirf_own_dev_v1.json (M_routed carried). Run only after the design report carries M_routed."""
import json, datetime, sys
design = json.load(open('docs/lirf_own/design_report.json'))
assert design['carried'] == 'M_routed', design['carried']
import hashlib
dsha = hashlib.sha256(open('docs/lirf_own/design_report.json', 'rb').read()).hexdigest()
proxy = 'runs/ctx_ownpoints_weight/chain/own_all_w100/dev_v66_proxy.parquet'
ident = 'runs/lirf_own/chain/identity_pinned_v66/identity_rebuilt.parquet'
isha = hashlib.sha256(open(ident, 'rb').read()).hexdigest()
psha = hashlib.sha256(open(proxy, 'rb').read()).hexdigest()
now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
oct_ = design['readings']['october']['L2']['M_routed']
disclosed = (f"Known before this declaration: the design batch (configs/lirf_own_design_v1.json, docs/lirf_own/design_report.json sha256 {dsha[:16]}...) "
             f"carried M_routed: October L2 change {oct_['0.35']['change_pct']:+.3f}% (interval {oct_['interval_b0.35_s'][0]:+.2f}..{oct_['interval_b0.35_s'][1]:+.2f} s); "
             "its other readings are in the report. The v66 development proxy and the official scores of v1-v66 exist; nothing here is derived from the scores. "
             "The candidates (full replacement and a half mix on own-route rows), the folds and the rules are fixed before any fit of this round.")
d = {
 "experiment_id": "prc2026-lirf-own-dev-v1",
 "declared_at_utc": now,
 "audit_used": False,
 "session": "final-week campaign, fork B (6 October 2026), lever 2; the user: \"yes start on them now\"; GPU: \"yes you can the GPU\"",
 "purpose": "Development round of the masked LIRF member with the 15 own-aircraft columns: lm25_32own (configs/lirf_own_network_ctx3_v1.json; configs/adsb_ctx3_lirf_mask_network_v1.json's lm25_32 plus the own columns, unmasked) replaces the shipped member M (lm25_32, 16 seeds 801-816) on own-route rows of the LIRF route of v53 / v55, inside v66's chain (the incumbent since 08:20 UTC, 6 October).",
 "disclosed": disclosed,
 "shipped_member": {"declaration": "configs/adsb_ctx3_lirf_mask_dev_v1.json", "variant": "lm25_32", "seeds": "801-816"},
 "fits": {
  "network_config": "configs/lirf_own_network_ctx3_v1.json",
  "variants": ["lm25_32own"],
  "seeds": list(range(801, 817)),
  "folds": {"2025-07": {"fold": "lomo_2025-07", "file": "predictions.parquet"}, "2025-11": {"fold": "forward", "file": "predictions.parquet"},
            "2025-12": {"fold": "refit", "file": "december_predictions.parquet"}, "ranking": {"fold": "refit12", "file": "ranking_predictions.parquet"}},
  "not_fitted": "lomo_2025-01 and seasonal: January 2025 has no own-route LIRF row (0 of 11,254; LIRF naturally dark) and the routing leaves the thinned (dark) July reading unchanged by construction; both months' candidate values equal the control exactly",
  "outputs": "runs/lirf_own/nets/<variant>_s<seed>/<fold>/",
  "count": 64,
  "extra_features": "data/external/adsblol/features_v19own (manifest sha256 0e8811e6...; hash-checked on every node before the fits)",
  "tool": "tools/run_adsb_ctx_lomo.py --device cuda --threads 3 --extra-features data/external/adsblol/features_v19own via tools/combo_queue.sh (ROUND=runs/lirf_own); job lists tools/lirf_own_jobs.py"
 },
 "route": "LIRF non-population rows with own ground motion seen: p_first_mv not null in data/external/adsblol/features_v19own (development: July 13,139, November 4,781, December 7,758, January 0 rows; ranking: January 2026 7,026, July 2026 14). Dark months (LIRF non-population match share below 5%: January 2025, July 2026) are never changed.",
 "member_prime": "M' = (1 - s) M + s Mown on own-route rows, M elsewhere; Mown = the 16-seed mean of lm25_32own on the same fold as M",
 "arithmetic": "tools/build_dev_model_part.py's LIRF layers unchanged except M -> M': v53m on lirf_u rows (v34 unmatched blend with u600 on 0.45 net + 0.55 M'), v55m on lirf_m rows (v34 matched blend with m600 on 0.60 net + 0.40 M'), deltas floored at -12 s",
 "candidates": {"own_full": {"own_share": 1.0}, "own_half": {"own_share": 0.5}},
 "stage_1": {
  "scorer": "tools/run_lirf_own_dev.py --stage1",
  "metric": "LIRF non-population rows of each month (runs/lomo_dev/rows.parquet; lirf_net = ens_ctx_lirf, lirf_tree, v9), primary = labels below 20,000 s; relative change of the route's squared error; paired-day 95% interval (10,000 draws, seed 20261006); also the change as a share of the month's v33 squared error",
  "rule": "the LIRF matched-row member re-read's rule (configs/lirf_combo_reread_v1.json): eligible when December's primary change is at most -0.25% with its interval below zero, July's and November's at most 0, and no month's interval entirely above zero; the lowest mean of the three changes is selected; otherwise nothing",
  "rule_values": {"december_max_relative_change": -0.0025}
 },
 "stage_2": {
  "when": "only if stage 1 selects a candidate",
  "scorer": "tools/run_lirf_own_dev.py --model-part --chain --score --candidate <selected>; before this declaration the scorer reproduced the stored v66 development proxy exactly from v66's own chain files (runs/lirf_own/chain/scorer_identity_v66/identity_score.json)",
  "control": "the pinned identity rebuild of v66's chain (M' = M through this tool with the variance re-allocation pinned; two pinned rebuilds are byte-identical); the stored v66 proxy is reported beside it (the pinned rebuild differs from it by -0.009% to +0.024% per month through the LightGBM row/column choice alone)",
  "identity_file": {"path": ident, "sha256": isha},
  "stored_incumbent_proxy": {"path": proxy, "sha256": psha},
  "candidate": "v66's development model part (runs/ctx_ownpoints_weight/chain/own_all_w100/dmp.parquet) with the LIRF route rows recomputed with M', then the shipped chain with its shipped parameters: no-NM rematch, v48, extreme-day, variance re-allocation, stack_resid_v3 refitted on the rebuilt components (LIRF: refitted v2), intruder bound, identity extension, own-motion v1, own-motion v2",
  "rule": "v57's clauses 1-3 against the pinned identity rebuild of v66: (1) no development month worse than +0.005%; (2) the equal-footing mean (January, July, December 2025; January's change is exactly 0 by construction and enters the mean) at most -0.02%; (3) every equal-footing month whose rows change has its paired-day 95% interval below zero (January has no changed row and no interval)",
  "rule_values": {"equal_footing": ["2025-01", "2025-07", "2025-12"], "max_month_worsening_pct": 0.005, "min_mean_improvement_pct": 0.02}
 },
 "export_plan": "If both stages accept: tools/export_lirf_own.py --dry-run --version N writes runs/lirf_own/rank/dry_vN (v66 plus the pinned candidate chain minus the pinned identity chain: the LIRF own-route rows of January 2026 recomputed from the 16 refit12 fits; July 2026 is dark and unchanged); the pinned export identity (runs/lirf_own/rank/identity_pinned_v66) exists. The coordinator assigns the version; nothing under runs/submissions; no upload.",
 "never_read": "No official score, leaderboard or ranking label enters. LIRF population rows and the guard rows MVT 213142471 / 213142469 never change.",
 "selection": "runs/lirf_own/dev/selection.json",
 "report": "docs/lirf_own/results_v1.json",
 "automatic_submission": False
}
with open('configs/lirf_own_dev_v1.json', 'w', newline='\n') as f:
    json.dump(d, f, indent=1); f.write('\n')
print(now)
