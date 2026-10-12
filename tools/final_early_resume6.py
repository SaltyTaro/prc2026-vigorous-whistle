"""Resume a finished early final-file chain (tools/final_early_chain.py) at step 6 with a given step-6 prediction file.

Steps 0-5 and the caches of FROM are copied byte for byte into a fresh OUT; then the chain's own configure(), step6() (--stack-preds) and
steps_7_to_10() run unchanged. Used for final candidate 4 (configs/sixmonth_override_v1.json: the six-month step-6 bags) and its control
(FROM's own step-6 preds must reproduce FROM's final file byte for byte).

  python tools/final_early_resume6.py --from runs/final_blind/early_9002 --out runs/final_blind/early_c4 --stack-preds P \
         --raw F --template T --runner R --model-part M --version N [--published-missing auto]

Nothing is uploaded; nothing is written under runs/submissions.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import final_early_chain as FE  # noqa: E402
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402

COPY_STEPS = ["rematch", "model_part", "R", "X", "Y"]     # steps 0-5 (the shipped v66 layout)


def main() -> None:
    ap = FE.parser()
    ap.add_argument("--from", dest="src", required=True, help="a finished early chain output directory (chain.json present)")
    a = ap.parse_args()
    if not a.stack_preds:
        raise ValueError("--stack-preds is required (the resume replaces step 6's fit).")
    src, out = ROOT / FE.rel(a.src), ROOT / FE.rel(a.out)
    rec0 = read_json(src / "chain.json")
    if out.exists():
        raise ValueError(f"{out} exists; use a fresh directory.")
    for k in ["raw", "template", "runner", "model_part"]:
        if FE.rel(getattr(a, k)) != rec0[k]:
            raise ValueError(f"--{k.replace('_', '-')} differs from {src}'s run ({rec0[k]}).")
    if a.published_missing != rec0.get("published_missing", "off"):
        raise ValueError("--published-missing differs from the source run.")
    t0 = time.time()
    (out / "steps").mkdir(parents=True)
    for s in COPY_STEPS:
        shutil.copytree(src / "steps" / s, out / "steps" / s)
    shutil.copytree(src / "cache", out / "cache")
    copied = {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}
    for name, h in copied.items():
        if sha256_file(src / name) != h:
            raise ValueError(f"copy of {name} differs")
    c = FE.Chain(a)
    c.record["resumed_from"] = {"dir": FE.rel(src), "chain_json_sha256": sha256_file(src / "chain.json"), "copied_files": len(copied),
                                "steps_copied": COPY_STEPS + ["cache"]}
    c.record["configured"] = c.configure()
    st, info = c.step6()
    c.record["stack"] = info
    final = c.steps_7_to_10(st)
    c.record["steps"] = {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(c.steps.rglob("vigorous-whistle_v*.parquet"))}
    c.record["final"] = {"file": FE.rel(final), "sha256": sha256_file(final)}
    c.record["seconds"]["resume_6_10"] = round(time.time() - t0, 1)
    c.record["code"] = {p: sha256_file(ROOT / p) for p in ["tools/final_early_resume6.py", "tools/final_early_chain.py", "tools/run_stack_weight.py",
                                                           "tools/run_stack_resid.py", "tools/run_stack_resid_v3.py", "tools/run_own_motion_v2.py"]}
    c.record["finished_at_utc"] = utc_now()
    write_json(out / "chain.json", c.record)
    print("final", c.record["final"], flush=True)


if __name__ == "__main__":
    main()
