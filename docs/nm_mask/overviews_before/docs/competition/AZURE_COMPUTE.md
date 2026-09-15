# Azure compute plan

User constraint, 13 September 2026: **do not use AWS; use Azure, with up to eight
VMs available**. The user subsequently allocated the existing **arcw0–arcw7**
pool and authorized starting **only arcw0 for now**.
No new VM was provisioned.

The phase-1 pilot ran on arcw0 using the existing SSH alias and an isolated
project directory. Observed capacity: 16 logical CPUs and 62.79 GiB usable RAM.
Full feature preparation took 67.74 seconds and sampled 1.19 GiB peak RSS;
the larger airport-mean evaluation sampled 1.41 GiB. These are foundation
measurements, not boosting-model benchmarks. See [phase-1 results](../phase1/README.md)
and [deployment instructions](../../infra/azure/README.md).

## What this task needs

The supplied dataset has 4,167,797 training movements, including 2,085,047
departure targets, and 30 raw columns. Its compressed training files occupy
284.98 MB; the complete supplied dataset occupies 330.22 MB. Decoding strings,
building traffic windows, and keeping model feature matrices increase memory
use substantially beyond the file size.

This supports starting with CPU-based tabular learning. LightGBM and CatBoost
were benchmarked on arcw0 in phase 2. Development runs including loading and
scoring took 14–17 seconds for LightGBM and 93–117 seconds for CatBoost.
The largest observed peak RSS was 3.95 GiB. The selected LightGBM refit and
ranking prediction took 20.82 seconds and 1.56 GiB peak RSS.
See [phase-2 evidence](../phase2/README.md).
Phase 3 also ran solely on arcw0: feature preparation took 67.68 seconds and
4.61 GiB sampled peak RSS; sixteen LightGBM development jobs took 26.99-51.81
seconds each. The selected refit took 43.38 seconds and 3.38 GiB peak RSS.
Its 86-predictor timing/traffic recipe scored 421.3047 officially. The measured
workload still fits comfortably on this single CPU VM; see
[Phase-3 evidence](../phase3/README.md).
Phase 4 also used only arcw0. Its 112-feature bundle took 39.04 seconds to
prepare at 1.11 GiB peak RSS. Fourteen development recipes took 2.53-77.27
seconds each, and the selected global/airport refit took 147.46 seconds with
3.99 GiB peak RSS. Official v3 scored 331.9269; see
[Phase-4 evidence](../phase4/README.md). At the final 18:10 UTC check,
arcw0 was running and arcw1-arcw7 remained deallocated.
Phase 5 again used only arcw0. The selected 34-model CatBoost/LightGBM refit
took **1,246.13 seconds** at **7.06 GiB peak RSS**, producing official v4's
**326.6895 RMSE** on all 344,841 rows. All model and blend checks passed on
Windows and Azure; see [Phase-5 evidence](../phase5/README.md). At the final
20:29-20:30 UTC pool check, only arcw0 was running. Its root disk had about
**6.4 GiB free** at 20:31 UTC after archiving the completed results. These
measurements remain within the single authorized worker's CPU/RAM capacity.
The initial priority is **RAM, CPU throughput, and reliable local storage**.
GPU use can be considered if measured training time justifies it.

| Role | Initial sizing recommendation | Purpose |
| --- | --- | --- |
| Pilot / feature preparation | 8–16 vCPUs, 32–64 GiB RAM, at least 50 GiB free SSD | Build features once and measure memory/time |
| Independent training worker | 8 vCPUs and 32 GiB RAM as a starting point; 64 GiB gives more headroom | One model or validation experiment per VM |
| Larger feature experiments | 16 vCPUs and 64 GiB RAM, subject to measured need | More feature families or larger CatBoost runs |
| Optional GPU worker | Determine after CPU benchmark | Accelerate supported model variants if worthwhile |

These are planning recommendations, not minimum competition requirements.
The actual arcw0 pilot root disk has about 10.26 GiB free after preparation,
enough for this phase but below the campaign storage recommendation. Two
440 GiB NVMe devices were visible and left unmounted; plan artifact storage
before expanding experiments. Azure D-series offers balanced CPU/RAM configurations;
E-series offers more RAM per vCPU. Select actual SKUs against the user's available
boxes, region, and capacity. Smaller boxes remain useful for lighter experiments.
Memory on separate VMs is not automatically pooled for a single training job.

## Using up to eight boxes

1. Start on one VM to establish a reproducible baseline, disk layout, feature
   schema, and measured peak memory. Pin the input hashes and validation split.
2. Cache common features so every worker does not recompute the same expensive
   traffic joins.
3. Run independent jobs across additional VMs: LightGBM, CatBoost, feature
   ablations, seasonal validation, airport specialists, and seed replicas.
4. Keep one consistent experiment ledger, complete validation predictions, and
   identical row IDs. Compare models on the same untouched evaluation rows.
5. Use a coordinator for result collection, ensemble selection, exact template
   validation, and final reproduction. The pilot VM can serve this role.

An example at full utilization is six training workers, one feature/specialist
worker, and one coordinator. This is a possible allocation, not a reason to keep
eight VMs running when fewer are sufficient. Set model thread counts to the CPU
allocation and avoid running several unrestricted trainers on the same VM.

Use Azure storage or the provided VM disks for campaign artifacts if remote
storage is needed. The mandatory competition input/output endpoint remains
`https://s3.opensky-network.org`, using the PRC credentials. Its S3-compatible
protocol does not require AWS compute or an Amazon S3 account.

The inherited `infra/sky/` scripts point to AWS and unrelated artifact buckets.
They are not the deployment path for this campaign. The tested Azure pilot
workflow is in `infra/azure/`.
