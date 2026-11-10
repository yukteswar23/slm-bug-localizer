#!/usr/bin/env bash
# Sequential pipeline: one job at a time (14 GB RAM). Each experiment is plotted
# as soon as it finishes. Log: C:/gemma_work/logs/pipeline.log
cd "$(dirname "$0")/.."
LOCK=/c/gemma_work/logs/pipeline.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "pipeline already running ($LOCK exists)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT
L=/c/gemma_work/logs/pipeline.log
H="--struct-answer --repeat-guard --budget-hints --ranked"
T="--split test --ids-file data/test100.txt"
run() {  # name model flags...
  local name=$1 m=$2; shift 2
  echo "=== $name $m $(date +%H:%M)" >> $L
  python scripts/run.py --model $m --run $name "$@" >> $L 2>&1 || echo "FAILED $name $m" >> $L
}
plot() { python scripts/rescore.py > /dev/null 2>&1; python analysis/plot_experiment.py --exp $1 >> $L 2>&1; }

for m in gemma-4-e2b qwen3.5-4b; do run c4g-dev $m --split dev $H --graph-tools --evidence-buffer --unconstrained; done
plot c4g-dev
echo "=== c0-test bm25 $(date +%H:%M)" >> $L
ls results/*_c0-test/bm25.json >/dev/null 2>&1 || { python scripts/run_bm25.py --run c0-test $T >> $L 2>&1; plot c0-test; }
for spec in "c2-test|$H --graph-tools" "c1-test|" "h07-test|$H" \
            "c3-test|$H --graph-tools --evidence-buffer" "c4g-test|$H --graph-tools --unconstrained"; do
  name=${spec%%|*}; flags=${spec#*|}
  for m in gemma-4-e2b qwen3.5-4b; do run $name $m $T $flags; done
  plot $name
done
python analysis/plot_ablation.py --name test-ablation --exps c1-test h07-test c2-test c3-test \
  --labels "C1" "+harness\nfixes" "+graph\ntools (C2)" "+evidence\nbuffer (C3)" \
  --alt c4g-test --alt-at 2 --alt-label "C2 without grammar" >> $L 2>&1
echo "=== ALL DONE $(date +%H:%M)" >> $L
