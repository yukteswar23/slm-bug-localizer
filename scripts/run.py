"""Run localization episodes for one model and configuration.

Full trajectories go to C:/gemma_work/runs/<run>/<model>/<instance_id>.json;
a compact summary (metrics + step statistics, no trajectories) goes to
results/runs/<run>__<model>.json so it can be tracked in git and plotted.

    python scripts/run.py --model gemma-4-e2b --run e2e_smoke --ids astropy__astropy-12907
    python scripts/run.py --model qwen3.5-4b --run c1_dev --split dev
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.llm import LlamaServer  # noqa: E402
from agent.loop import run_episode  # noqa: E402
from agent.score import score  # noqa: E402
from agent.tasks import WORK, add_gold, checkout, load_tasks  # noqa: E402

STEP_KEYS = ("step", "tool", "valid", "error", "prompt_tokens", "completion_tokens",
             "cached_tokens", "prompt_ms", "gen_ms", "observation_chars")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True, help="run name, e.g. c1_dev")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--ids", nargs="*", help="instance ids (default: whole split)")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--unconstrained", action="store_true")
    args = ap.parse_args()

    tasks = load_tasks(args.split)
    if args.ids:
        tasks = [t for t in tasks if t.instance_id in set(args.ids)]
    traj_dir = WORK / "runs" / args.run / args.model
    traj_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / "results" / "runs" / f"{args.run}__{args.model}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"run": args.run, "model": args.model, "split": args.split,
               "max_steps": args.max_steps, "constrained": not args.unconstrained,
               "started": time.strftime("%Y-%m-%d %H:%M"), "tasks": []}

    with LlamaServer(args.model) as llm:
        for n, task in enumerate(tasks, 1):
            repo_dir = checkout(task)
            add_gold(task)
            ep = run_episode(llm, task, repo_dir, args.max_steps,
                             constrained=not args.unconstrained)
            ep.update(gold_files=task.gold_files, gold_functions=task.gold_functions,
                      metrics=score(ep["answer"], task.gold_files, task.gold_functions))
            (traj_dir / f"{task.instance_id}.json").write_text(json.dumps(ep, indent=1))
            summary["tasks"].append({
                k: ep[k] for k in ("instance_id", "answer", "submitted", "n_steps", "wall_s",
                                   "gold_files", "gold_functions", "metrics")}
                | {"steps": [{k: s.get(k) for k in STEP_KEYS} for s in ep["steps"]]})
            summary_path.write_text(json.dumps(summary, indent=1))
            m = ep["metrics"]
            print(f"[{n}/{len(tasks)}] {task.instance_id}: {ep['n_steps']} steps, "
                  f"{ep['wall_s']:.0f} s, file@1={m['file_acc_all@1']:.0f} "
                  f"func@5={m['func_acc_all@5']} answer={ep['answer']}", flush=True)
    print(summary_path)


if __name__ == "__main__":
    main()
