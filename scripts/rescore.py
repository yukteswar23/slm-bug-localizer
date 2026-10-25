"""Recompute metrics in results/runs/*.json after a scorer change (no model calls).

    python scripts/rescore.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.score import score  # noqa: E402

for path in sorted((ROOT / "results" / "runs").glob("*.json")):
    summary = json.loads(path.read_text())
    for t in summary["tasks"]:
        t["metrics"] = score(t["answer"], t["gold_files"], t["gold_functions"])
    path.write_text(json.dumps(summary, indent=1))
    print(path.name, [(t["instance_id"], t["metrics"]["file_acc_all@1"], t["metrics"]["func_acc_all@5"])
                      for t in summary["tasks"]])
