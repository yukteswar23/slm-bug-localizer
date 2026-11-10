"""Run localization episodes for one model and configuration.

Results go to the experiment's checkpoint folder results/<date>_<NN>_<run>/
(see agent/checkpoint.py): <model>.json with metrics and step statistics, and
config.json. Full trajectories go to C:/gemma_work/runs/<same folder>/<model>/.

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
from agent.llm import ApiLLM, LlamaServer  # noqa: E402
from agent.loop import OPTIONS, run_episode  # noqa: E402
from agent.score import score  # noqa: E402
from agent.checkpoint import checkpoint_dir, traj_dir  # noqa: E402
from agent.index import build_index  # noqa: E402
from agent.tasks import add_gold, checkout, load_tasks  # noqa: E402

STEP_KEYS = ("step", "tool", "action", "valid", "error", "repeat_blocked", "evidence", "prompt_tokens", "completion_tokens",
             "cached_tokens", "prompt_ms", "gen_ms", "observation_chars")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True, help="run name, e.g. c1_dev")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--ids", nargs="*", help="instance ids (default: whole split)")
    ap.add_argument("--ids-file", help="file with one instance id per line, e.g. data/test100.txt")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--unconstrained", action="store_true")
    for opt in OPTIONS:
        ap.add_argument("--" + opt.replace("_", "-"), action="store_true")
    args = ap.parse_args()
    options = {k: getattr(args, k) for k in OPTIONS}

    tasks = load_tasks(args.split)
    if args.ids_file:
        args.ids = (args.ids or []) + Path(args.ids_file).read_text().split()
    if args.ids:
        tasks = [t for t in tasks if t.instance_id in set(args.ids)]
    ckpt = checkpoint_dir(args.run)
    trajs = traj_dir(ckpt, args.model)
    summary_path = ckpt / f"{args.model}.json"
    config_path = ckpt / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {
        "experiment": ckpt.name, "split": args.split, "max_steps": args.max_steps,
        "constrained": not args.unconstrained,
        "tools": ("file + graph tools" if options["graph_tools"] else "file tools")
        + (" + Evidence Buffer" if options["evidence_buffer"] else ""), "options": options,
        "trajectories": str(trajs.parent), "models": {}}
    config["models"][args.model] = {"started": time.strftime("%Y-%m-%d %H:%M"),
                                    "n_tasks": len(tasks), "ids": args.ids}
    config_path.write_text(json.dumps(config, indent=1))
    done = {}
    if summary_path.exists():  # resume: keep tasks finished before an interruption
        prev = json.loads(summary_path.read_text())
        done = {t["instance_id"]: t for t in prev.get("tasks", [])}
        tasks = [t for t in tasks if t.instance_id not in done]
    summary = {"run": args.run, "model": args.model, "split": args.split,
               "max_steps": args.max_steps, "constrained": not args.unconstrained,
               "options": options,
               "started": time.strftime("%Y-%m-%d %H:%M"), "tasks": list(done.values())}

    with (ApiLLM() if args.model == "gemma-4-31b" else LlamaServer(args.model)) as llm:
        for n, task in enumerate(tasks, 1):
            repo_dir = checkout(task)
            add_gold(task)
            index = build_index(repo_dir, f"{task.repo}@{task.base_commit}")                 if options["graph_tools"] else None
            ep = run_episode(llm, task, repo_dir, args.max_steps,
                             constrained=not args.unconstrained, options=options, index=index)
            ep.update(gold_files=task.gold_files, gold_functions=task.gold_functions,
                      metrics=score(ep["answer"], task.gold_files, task.gold_functions))
            (trajs / f"{task.instance_id}.json").write_text(json.dumps(ep, indent=1))
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
