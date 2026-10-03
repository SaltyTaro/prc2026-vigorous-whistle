#!/usr/bin/env bash
# Dig tunnel 21: copy every finished design run (runs/ctx_ownpoints/design/<variant>_s<seed>/design) and the logs back from arcw0-arcw7
# and check the stop-prediction hashes against the workers'. Usage: tools/ctx_ownpoints_mirror.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/ctx_ownpoints/design runs/ctx_ownpoints/logs_mirror
for i in 0 1 2 3 4 5 6 7; do
  h=arcw$i
  ssh -o BatchMode=yes $h "cd /home/azureuser/prc-flight-2026 && tar cf - runs/ctx_ownpoints/design runs/ctx_ownpoints/logs" | tar xf - --transform "s|runs/ctx_ownpoints/logs|runs/ctx_ownpoints/logs_mirror/$h|"
  remote=$(ssh -o BatchMode=yes $h "cd /home/azureuser/prc-flight-2026 && sha256sum runs/ctx_ownpoints/design/*/design/stop_predictions.parquet")
  echo "$remote" | sha256sum -c --quiet && echo "$h ok $(echo "$remote" | wc -l)"
done
