# Tier B: paused state and how to resume (10 October 2026, ~12:05 UTC)

The user asked to pause everything so that nothing is lost when the session limit is hit. Remote compute keeps running unattended (it does not
depend on the session); the three helper lanes were checkpointed and stopped. Declaration: `configs/febjun_tierb_v1.json`; plan
`docs/final_file/febjun_check.md`. Deadline for Tier B readings: **13 October 12:00 UTC**, else final candidate 1 (SHA-256 `30e116a9...`) stands.
Final file deadline 14 October 10:00 UTC; upload only at the user's word (command in `docs/final_file/final_candidate.md`).

## What keeps running unattended

- **58 tree LOMO fits** (`tools/launch_febjun_lomo.py`, plan `runs/febjun_lomo/plan.json`) as detached `setsid` chains on arcw0, 1, 3, 4, 5, 6, 7.
  At 11:55 UTC 23 of 58 were done (all LightGBMs); the CatBoosts run 2-2.75 h each, on estimate. Expected end: about 21:00 UTC (arcw5 about 22:00).
  Outputs stay on each VM until it is deallocated. **Never deallocate
  arcw0 / arcw1 before their outputs are mirrored and hash-checked: their work dirs are bind mounts of the local NVMe, which deallocation wipes.**
- **80 GPU network fits** on prc-ctx8e (us-east-1b), prc-ctx8f (us-east-1a), prc-ctx8h (us-west-2b), SkyPilot jobs, autodown after 45 idle
  minutes. 42 of 80 done at 11:47 UTC; expected end about 14:00 UTC. A local copy-back loop (scratchpad `g_mirror_loop.sh`, pid file
  `/tmp/g_mirror_loop.pid`, log `runs/febjun_nets/mirror_loop.log`) rsyncs finished fits every 10 minutes, so every fit is local before its node
  autodowns. **The workstation must stay on and online until about 15:00 UTC for that copy-back.** A downed node does not stop the loop from
  mirroring the others; if nodes finish more than 45 minutes apart the loop never sees 3/3 and keeps looping harmlessly (kill it on resume).

## Resume, in order

1. Lane G (networks): `docs/final_file/tierb_resume_G.md`. Hash check (`tools/febjun_net_mirror.py`), confirm the prc-ctx8* clusters are gone
   (`sky status`; never touch k12-p1-open-big or sky-jobs-controller-*), re-run any missing fit, record the spend in `docs/final_file/tierb_networks.md`.
2. Trees: `.venv/Scripts/python.exe -X utf8 -B tools/launch_febjun_lomo.py status`; when every chain log says `chain done` with no FAILED,
   `... mirror`, then compare SHA-256 of every mirrored file with the VM's copy, then deallocate arcw0, 1, 3, 4, 5, 6, 7.
3. Lane M (model part v30-v55m, readings R3, R5-R8, R10-R13): `docs/final_file/tierb_resume_M.md`.
4. Lane C (own-points member and chain, readings C1-C15): `docs/final_file/tierb_resume_C.md`.
