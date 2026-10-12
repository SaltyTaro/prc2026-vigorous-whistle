"""Candidate 4 staging: the input list for arcw0 (repo-relative, LF endings)."""
import glob
import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
lists = ["runs/febjun_tierb/vm/seed_upload.txt"] + sorted(glob.glob("runs/febjun_tierb/vm/up*.txt")) + \
        ["runs/sixmonth/vm/arcw6_dev_inputs.txt", "runs/sixmonth/vm/c2_seed.txt", "runs/sixmonth/vm/control_upload.txt"]
src = [l.strip() for f in lists for l in open(f, encoding="utf-8") if l.strip()]
excl = ("runs/adsb_ctx", "runs/adsb_nn", "runs/adsb_rnn", "runs/adsb_v1", "runs/adsb_v2", "runs/ctx_ownpoints_dev", "runs/azure_arcw0_surface_refit",
        "runs/febjun_tierb/chain/work_repro12", "runs/pubrefit", "runs/december", "runs/febjun_nets", "runs/febjun_lomo")
keep = set()
for p in src:
    if p.startswith(excl):
        continue
    if p.startswith("runs/lomo_dev/") and not p.endswith("rows.parquet"):
        continue
    if p.startswith("runs/febjun_tierb/model/") and not os.path.basename(p).startswith("chain_"):
        continue
    keep.add(p)
R = "runs/final_blind/run_final4m"
pats = ["runs/final_blind/early_9002/**/*", "runs/final_blind/final_candidate2/*", "runs/final_blind/final_candidate3/*",
        f"{R}/input.json", f"{R}/stages/**/*", f"{R}/recipe/**/*", f"{R}/stack/**/*", f"{R}/rematch/**/*", f"{R}/adsb/**/*", f"{R}/final/**/*",
        f"{R}/networks/ensembles.parquet", f"{R}/leaves/tables.parquet", f"{R}/repair/*.json", f"{R}/repair/*.parquet",
        f"{R}/repair/run/input.json", f"{R}/repair/run/stages/**/*", f"{R}/repair/run/recipe/**/*", f"{R}/repair/run/stack/**/*",
        f"{R}/repair/run/networks/ensembles.parquet", f"{R}/repair/run/leaves/tables.parquet",
        "data/raw/*.parquet", "data/external/adsblol/surface/surface_2025*", "data/external/adsblol/surface/surface_202601*",
        "data/external/adsblol/surface/surface_202602*", "data/external/adsblol/surface/surface_202605*",
        "data/external/adsblol/surface/surface_202606*", "data/external/adsblol/surface/surface_202607*",
        "runs/oooi_final/*", "runs/febjun_tierb/chain/work/steps/stack/*", "runs/febjun_tierb/chain/work/late/stack_second/*",
        "runs/febjun_tierb/chain/work/late/stack_third/*", "runs/febjun_tierb/model/chain_*.parquet", "runs/final_nopub/**/*",
        "docs/sixmonth/results_v1.json"]
for pat in pats:
    for p in glob.glob(pat, recursive=True):
        if os.path.isfile(p):
            keep.add(p.replace(os.sep, "/"))
keep = sorted(p for p in keep if os.path.isfile(p))
tot = sum(os.path.getsize(p) for p in keep)
with open("runs/sixmonth/c4/want.txt", "w", newline="\n", encoding="utf-8") as f:
    f.write("\n".join(keep) + "\n")
print(len(keep), round(tot / 1e9, 2), "GB")
