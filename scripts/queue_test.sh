#!/usr/bin/env bash
# Overnight test-set queue (100 tasks, data/test100.txt). Runs after the dev queue.
# Order puts the main results first. Log: C:/gemma_work/logs/test-queue.log
cd "$(dirname "$0")/.."
L=/c/gemma_work/logs/test-queue.log
H="--struct-answer --repeat-guard --budget-hints --ranked"
IDS="--split test --ids-file data/test100.txt"
echo "=== c0-test bm25 $(date +%H:%M)" >> $L
python scripts/run_bm25.py --run c0-test $IDS >> $L 2>&1 || echo "FAILED c0-test" >> $L
for spec in "c2-test|$H --graph-tools" "c1-test|" "h07-test|$H" \
            "c3-test|$H --graph-tools --evidence-buffer" "c4g-test|$H --graph-tools --unconstrained"; do
  name=${spec%%|*}; flags=${spec#*|}
  for m in gemma-4-e2b qwen3.5-4b; do
    echo "=== $name $m $(date +%H:%M) $flags" >> $L
    python scripts/run.py --model $m --run $name $IDS $flags >> $L 2>&1 || echo "FAILED $name $m" >> $L
  done
done
echo "=== ALL DONE $(date +%H:%M)" >> $L
