#!/usr/bin/env bash
# Elicit every multi-pass protocol on the same 30 pilot ids -> one cache each.
# The solver benchmark runs separately on Colab/GPU (see run_pilot.py): commit
# and push the caches this produces, then run the Colab payload cell.
#
# Elicitation is resume-safe (each protocol has its own cache); rerunning
# skips done entries. Nothing here loops or re-queries.
#
#   bash metaculus/run_protocol_pilot.sh            # elicit all four caches
#   DRY=1 bash metaculus/run_protocol_pilot.sh      # print selection + budget only
#   RETRY=1 CONC=2 bash metaculus/run_protocol_pilot.sh   # retry 429-failed keys
#
# NOTE on failures: a key that hard-fails is written to <cache>.failures.json
# and is then EXCLUDED from todo on rerun (so a rerun reports todo: 0 rather
# than silently re-burning calls). Provider 429s / quota exhaustion look
# exactly like this. Clear them with RETRY=1, and drop CONC if the quota is
# what you tripped — the last protocol in the loop is the one that starves.
set -euo pipefail

PY="C:/Users/amdic/miniconda3/envs/calp/python.exe"
cd "$(dirname "$0")"                    # -> metaculus/ (runner imports run_elicitation)

# fixed set of 30 resolved ids — every protocol attempts exactly these, so a
# per-protocol failure just drops that id from that arm (the diagnostics step
# scores all arms on their common surviving keys, see run_pilot.py --common).
DATASET="data/full_dataset.json"
IDS="data/pilot_ids_small.txt"
PROTOCOLS=(baseline v1 v1x2 v1_fermi v1_spread)
DRY="${DRY:-0}"
RETRY="${RETRY:-0}"
CONC="${CONC:-8}"

for p in "${PROTOCOLS[@]}"; do
  echo "=== elicit: $p ==="
  args=(run_protocol.py --dataset "$DATASET" --ids-file "$IDS" --protocol "$p"
        --concurrency "$CONC")
  [ "$DRY" = "1" ] && args+=(--dry-run)
  [ "$RETRY" = "1" ] && args+=(--retry-failures)
  "$PY" "${args[@]}"
done

[ "$DRY" = "1" ] && echo "(dry run — nothing elicited)"
echo
echo "done. commit + push the caches, then run the Colab payload to benchmark:"
echo "    git add metaculus/caches/"
echo "    git commit -m 'protocol pilot caches' && git push"
