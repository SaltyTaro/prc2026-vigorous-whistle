# Azure pilot

**Only `arcw0` may be started under the current
instruction.** The allocation is recorded in [allocation.json](allocation.json).
Use the existing local SSH alias `arcw0`; no key material belongs in this repo.

The isolated remote directory is `/home/azureuser/prc-flight-2026`. The PRC
credentials stay in the local `.env`: the pilot receives the already downloaded
data and needs no competition, Azure, or AWS credentials inside its directory.
This workflow uses Azure compute and the provided VM disk only.

## Build and run

Starting the authorized VM, if it is deallocated:

```powershell
```

Create a package from an explicit allowlist of Python source, tests, dependency
pins, validation configuration, Azure runner, input receipts, and the 14 raw
Parquet files. `.env`, SSH files, Git metadata, and inherited infrastructure
are excluded by construction.

The phase-1 package runs only the original 26 foundation tests. Phase-2 models,
their additional tests, and dependencies have a separate package below.

```powershell
.\.venv\Scripts\python.exe tools/package_phase1.py
ssh arcw0 'mkdir -p /home/azureuser/prc-flight-2026/incoming'
scp infra/azure/.local/phase1-pilot.tar arcw0:/home/azureuser/prc-flight-2026/incoming/
ssh arcw0 'cd /home/azureuser/prc-flight-2026 && sha256sum incoming/phase1-pilot.tar'
```

Compare the hash with `infra/azure/.local/phase1-pilot.json` before extraction:

```powershell
ssh arcw0 'cd /home/azureuser/prc-flight-2026 && tar -xf incoming/phase1-pilot.tar && bash infra/azure/pilot.sh'
```

The runner uses the VM's installed `uv` with the absolute system Python path,
or standard `venv`/pip when available. The tested image provides `/usr/bin/python3`
3.12.3 and `uv`; it does not have system `ensurepip`. The project dependencies
are isolated in `.venv`. No system package installation is required on this image.
Native math thread limits are 16.

The runner executes the 26 tests, hashes every input, builds the feature/split
bundle, and benchmarks only the seasonal and forward folds. It writes
`runs/phase1/azure_pilot.json` with exact artifact locations. The December audit
is not scored.

## Collect evidence

For the phase-1 bundle created on 13 September:

```powershell
scp arcw0:/home/azureuser/prc-flight-2026/data/phase1/58d621b245de3f5e12dd/manifest.json infra/azure/.local/arcw0_feature_manifest.json
scp -r arcw0:/home/azureuser/prc-flight-2026/runs/phase1 runs/azure_arcw0_phase1
.\.venv\Scripts\python.exe tools/summarize_phase1.py --local-bundle data/phase1/58d621b245de3f5e12dd
```

When collecting again, use a fresh local destination or copy individual new run
directories, so `scp` does not introduce an extra nested `phase1` directory.
The summarizer verifies run artifact hashes and cross-platform predictions.
See the [phase-1 report](../../docs/phase1/README.md) for measured results and
the remaining disk capacity.

The eight VMs are a pool for later independent jobs. This script does not
provision machines, start extra workers, mount/format disks, schedule
submissions, or invoke the inherited `infra/sky/` scripts.

## Phase 2

The CPU baseline workflow is `infra/azure/phase2.sh`, packaged with
`tools/package_phase2.py`. It reuses the verified phase-1 inputs and feature
bundle, installs `requirements-phase2.txt` inside the same isolated project
environment, runs the 48 tests, and trains LightGBM and CatBoost sequentially
on both development folds.

The selected January–November refit and first official submission are complete.
See [phase-2 results and commands](../../docs/phase2/README.md). The 436.8976
official score used all 344,841 rows. The upload was performed locally using
the PRC credentials; none were copied to arcw0.

Only arcw0 is running. All other pool VMs remain deallocated under the current
instruction. There is no multi-VM job scheduling in these baseline runners.

## Phase 3

The Phase-3 feature ablations and v2 submission are complete. Package the code
with `tools/package_phase3.py`; run `infra/azure/phase3.sh` and then
`infra/azure/phase3_followup.sh` on the existing project. The two scripts run
six initial feature combinations and two adaptive follow-ups, respectively,
on both protected development folds. They reuse the original raw inputs and
Phase-1 split bundle, and run one trainer at a time.

The selected 462-tree timing/traffic refit scored **421.3047 seconds RMSE**
officially. See [Phase-3 evidence and reproduction](../../docs/phase3/README.md).
Local copies of all seventeen run directories are under
`runs/azure_arcw0_phase3/`; the 292.7 MB feature cache is under `data/phase3/`.
The original Azure Phase-1 manifest is archived alongside the Phase-3 report
for strict cross-platform verification without changing the local Phase-1
manifest's machine metadata. Credentials remain local.

## Phase 4

Phase-4 development, the selected refit, and official v3 scoring are complete.
Use `tools/package_phase4.py` and `infra/azure/phase4.sh` on the existing project;
the script runs 71 tests, builds the immutable 112-predictor feature bundle,
and tests seven recipes on both frozen folds, one component at a time.

The selected conditional-offset global/airport blend scores **331.9269 seconds
RMSE officially**, using all 344,841 rows. Local model copies are under
`runs/azure_arcw0_phase4/`; Azure originals are under `runs/phase4/`. The new
feature cache is `data/phase4/2ceb3449c26767a97311` (238.17 MB of Parquet).
See [Phase-4 reproduction and receipts](../../docs/phase4/README.md).

Both platform verifiers reproduce all fourteen development scores and the
saved model pipelines with zero prediction difference. The Phase-1 parent
comparison excludes only measured time/host metadata and retains every
scientific field and artifact hash. Original manifests are unchanged.

The refit took 147.46 seconds and about 3.99 GiB peak RSS. Only arcw0 was used;
the other seven allocated VMs remain deallocated. Uploads use local PRC
credentials and the organizer's storage endpoint. December remains reserved.

## Phase 5

Phase 5 completed on the same worker. `tools/package_phase5.py` packages only
the new model package, declarations, helpers, tests, and Azure runners. The
initial workflow is `infra/azure/phase5.sh`; `phase5_remaining.sh` freezes the
initial seed anchor, runs the separately declared CatBoost cap comparison,
fits the two seed replicas, and selects among 28 saved-prediction recipes.
The source package and 112-feature Phase-4 bundle remain distinct artifacts.

The selected recipe combines the longer CatBoost pipeline (75%) and the equal
three-seed LightGBM mean (25%), retaining the global/airport structure. Its
January-November refit has **34 distinct models**, took **1,246.13 seconds**, and
peaked at **7.06 GiB RSS**. Official **v4 scored 326.6895 seconds RMSE** on every
one of the 344,841 template rows. See [Phase-5 evidence](../../docs/phase5/README.md).

All **81 tests** pass on Windows/Azure. Both platform verifiers reproduce 20
complete development evaluations, 56 individual/blend evaluations, and native
reloads for all 21 saved runs. The source and validation receipts are unchanged;
the only verifier correction normalized JSON month-key representation.

Models are under `runs/phase5/` and `runs/phase5_followup/`, mirrored locally as
`runs/azure_arcw0_phase5/` and `runs/azure_arcw0_phase5_followup/`.
`tools/package_phase5_results.py` hashes completed artifacts before archiving;
`tools/restore_phase5_results.py` verifies the transferred archive and preserves
existing differing files. The Phase-5 report has commands using fresh output
directories for reproduction. Do not rerun a refit merely to rerun verification.

The final pool check confirms arcw0 running and arcw1-arcw7 deallocated. The root
disk had about **6.4 GiB free** after the results archive. No additional worker,
AWS resource, or external data was used. PRC credentials remained local, and
December is still reserved for the explicit Phase-6 audit.
