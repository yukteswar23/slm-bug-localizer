"""C0 baseline: retrieval only, no language model.

Files are ranked by BM25 between the issue text and each file; functions are
ranked by BM25 against each function/method body, restricted to the top-5
files. Writes the same checkpoint format as scripts/run.py (model "bm25", no
steps), so the same scorer and figures apply.

    python scripts/run_bm25.py --run c0-dev --split dev
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.checkpoint import checkpoint_dir  # noqa: E402
from agent.index import build_index  # noqa: E402
from agent.loop import MAX_ISSUE_CHARS  # noqa: E402
from agent.score import score  # noqa: E402
from agent.tasks import add_gold, checkout, load_tasks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--ids-file", help="file with one instance id per line, e.g. data/test100.txt")
    args = ap.parse_args()

    tasks = load_tasks(args.split)
    if args.ids_file:
        args.ids = (args.ids or []) + Path(args.ids_file).read_text().split()
    if args.ids:
        tasks = [t for t in tasks if t.instance_id in set(args.ids)]
    ckpt = checkpoint_dir(args.run)
    (ckpt / "config.json").write_text(json.dumps({
        "experiment": ckpt.name, "split": args.split, "tools": "none (C0: BM25 retrieval only)",
        "files": "BM25(issue, file text), top 5",
        "functions": "BM25(issue, function body), top 10 within the top-5 files",
        "models": {"bm25": {"started": time.strftime("%Y-%m-%d %H:%M"), "n_tasks": len(tasks)}}},
        indent=1))
    summary = {"run": args.run, "model": "bm25", "split": args.split, "max_steps": 0,
               "constrained": None, "started": time.strftime("%Y-%m-%d %H:%M"), "tasks": []}
    for n, task in enumerate(tasks, 1):
        repo_dir = checkout(task)
        add_gold(task)
        t0 = time.time()
        index = build_index(repo_dir, f"{task.repo}@{task.base_commit}")
        t_index = time.time() - t0
        query = task.problem_statement[:MAX_ISSUE_CHARS]
        files = [f for f, _ in index.rank_files(query, 5)]
        funcs = [f"{s['file']}::{s['qualname']}" for s, _ in index.rank_symbols(query, 10, files=set(files))]
        answer = {"files": files, "functions": funcs}
        summary["tasks"].append({
            "instance_id": task.instance_id, "answer": answer, "submitted": True, "n_steps": 0,
            "wall_s": time.time() - t0, "index_s": t_index, "gold_files": task.gold_files,
            "gold_functions": task.gold_functions, "steps": [],
            "metrics": score(answer, task.gold_files, task.gold_functions)})
        (ckpt / "bm25.json").write_text(json.dumps(summary, indent=1))
        m = summary["tasks"][-1]["metrics"]
        print(f"[{n}/{len(tasks)}] {task.instance_id}: file@1={m['file_acc_all@1']:.0f} "
              f"file@5={m['file_acc_all@5']:.0f} func@5={m['func_acc_all@5']} ({t_index:.1f} s index)",
              flush=True)


if __name__ == "__main__":
    main()
