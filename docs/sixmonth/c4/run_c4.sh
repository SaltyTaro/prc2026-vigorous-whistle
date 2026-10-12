#!/bin/bash
# arcw0, final candidate 4 (configs/sixmonth_override_v1.json). Repo overlaid on the NVMe (/mnt/scratch/up).
#   run_c4.sh early_ctl   early chain resumed at step 6 with candidate 3's own step-6 preds  -> must give 4c393769... (early_9002's output)
#   run_c4.sh late_ctl    late chain 11-18 on early_ctl's output, shipped C14 fit + six-month C15 -> must give f11ca28f... (candidate 2)
#   run_c4.sh stack6m     six-month C6 bags (fit-final --kind stack --override)
#   run_c4.sh early_c4    early chain resumed at step 6 with the six-month C6 preds
#   run_c4.sh late_c4     late chain 11-18 on early_c4's output, six-month C14 (--override) + six-month C15, then the LIRF cap
# Every step under runs/sixmonth/c4/vmrun.sh (strace ENOENT list); log runs/sixmonth/c4/logs/<mode>.log.
cd /home/azureuser/prc-flight-2026
PY=".venv/bin/python -X utf8 -B"
F=data/raw/final_ranking.parquet; T=data/raw/final_submitting.parquet; R=runs/final_blind/run_final4m
M=runs/sixmonth/c4/matches_final
LOG=runs/sixmonth/c4/logs/$MODE.log
mkdir -p runs/sixmonth/c4/logs
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
early() {  # early OUT PREDS
      --model-part $R/final/vigorous-whistle_v9001.parquet --version 9002 --published-missing auto
  say "EARLY_DONE $"
}
late() {  # late WORK EARLY SSFIT STFIT
  run init $PY tools/final_late_chain.py init --work $W --raw $F --base $O/vigorous-whistle_v9002.parquet --recipe $R/recipe/detail.parquet \
      --matches-dir $M --features-own-dir runs/final_blind/early_9002/features_v19own --messages runs/oooi_final/messages_unique.parquet \
      --messages-merge files --stacker-rank $O/steps/stack/stacker_rank.parquet
  for s in oooi nbr_state nbr_dark ss_build; do run $s $PY tools/final_late_chain.py step --work $W --name $s; done
  else run fit_ss $PY tools/run_sixmonth.py fit-final --kind ss --override configs/sixmonth_override_v1.json --target $W/stack_second; fi
  for s in ss_score raw st_build; do run $s $PY tools/final_late_chain.py step --work $W --name $s; done
  run fit_st $PY tools/run_sixmonth.py fit-final --kind st --target $W/stack_third
  for s in st_score sp_build; do run $s $PY tools/final_late_chain.py step --work $W --name $s; done
  run fit_sp $PY tools/final_late_chain.py fit --work $W --stage sp
  for s in sp_score sp2_build; do run $s $PY tools/final_late_chain.py step --work $W --name $s; done
  run fit_sp2 $PY tools/final_late_chain.py fit --work $W --stage sp2
  for s in sp2_score km; do run $s $PY tools/final_late_chain.py step --work $W --name $s; done
  KM=$($PY -c "import json; print(json.load(open('$W/late_chain.json'))['outputs']['km']['file'])")
  say "LATE_DONE $KM $(sha256sum $KM | cut -d' ' -f1)"
}
say "start $MODE"
case $MODE in
  early_ctl) early runs/final_blind/early_c4ctl runs/final_blind/early_9002/steps/stack/preds.parquet ;;
  late_ctl)  late runs/final_blind/late_c4ctl runs/final_blind/early_c4ctl shipped ;;
  stack6m)   run fit $PY tools/run_sixmonth.py fit-final --kind stack --override configs/sixmonth_override_v1.json \
                 --target runs/final_blind/early_9002/steps/stack --out runs/sixmonth/c4/stack_preds_6m.parquet
             say "STACK_DONE $(sha256sum runs/sixmonth/c4/stack_preds_6m.parquet)" ;;
  early_c4)  early runs/final_blind/early_c4 runs/sixmonth/c4/stack_preds_6m.parquet ;;
  late_c4)   late runs/final_blind/late_c4 runs/final_blind/early_c4 sixmonth
             KM=$($PY -c "import json; print(json.load(open('runs/final_blind/late_c4/late_chain.json'))['outputs']['km']['file'])")
             run cap $PY tools/run_lirf_cap.py apply --base $KM --movements $F --out runs/final_blind/final_candidate4/chain_out_cap.parquet
             say "CAP_DONE $(sha256sum runs/final_blind/final_candidate4/chain_out_cap.parquet)" ;;
  *) echo "mode?"; exit 2 ;;
esac
say "END $MODE"
