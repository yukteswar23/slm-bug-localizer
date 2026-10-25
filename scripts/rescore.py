"""Recompute metrics in every checkpoint's <model>.json after a scorer change
(no model calls), and backfill step actions from the full trajectories.

    python scripts/rescore.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.checkpoint import RESULTS, TRAJ_ROOT  # noqa: E402
from agent.score import score  # noqa: E402

for path in sorted(RESULTS.glob("*/*.json")):
    summary = json.loads(path.read_text())
    if "tasks" not in summary:
        continue  # config.json or non-run results
    for t in summary["tasks"]:
        t["metrics"] = score(t["answer"], t["gold_files"], t["gold_functions"])
        traj = TRAJ_ROOT / path.parent.name / summary["model"] / f"{t['instance_id']}.json"
        if traj.exists():
            full = json.loads(traj.read_text())["steps"]
            for s, f in zip(t["steps"], full):
                s.setdefault("action", f.get("action"))
    path.write_text(json.dumps(summary, indent=1))
    print(path.parent.name, path.name, len(summary["tasks"]), "tasks rescored")
