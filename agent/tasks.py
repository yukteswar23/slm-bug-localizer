"""SWE-bench Lite tasks: loading, repo checkout, and gold locations.

Gold locations come from the reference patch: every file it touches, and every
function or method that encloses a changed line in the ORIGINAL file (parsed
with `ast` at the task's base commit). Names look like
"path/to/file.py::Class.method". Lines outside any function count toward the
file only.

    python -m agent.tasks           # print gold locations for the first 5 tasks
"""

import ast
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

WORK = Path("C:/gemma_work")
DATA = WORK / "data"
# A separate clone folder lets two runs (e.g. the API reference and a GPU run)
# check out different commits of the same repo at the same time.
REPOS = Path(os.environ.get("SLM_REPOS_DIR", WORK / "repos"))
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


@dataclass
class Task:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str
    gold_files: list = field(default_factory=list)
    gold_functions: list = field(default_factory=list)

    @property
    def repo_dir(self) -> Path:
        return REPOS / self.repo.replace("/", "__")


def load_tasks(split: str = "test") -> list:
    """split: 'test' (300 tasks, evaluation) or 'dev' (23 tasks, tuning)."""
    path = DATA / f"swebench_lite_{split}.jsonl"
    if not path.exists():
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
        with open(path, "w", encoding="utf-8") as f:
            for r in ds:
                f.write(json.dumps(r) + "\n")
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        tasks.append(Task(r["instance_id"], r["repo"], r["base_commit"],
                          r["problem_statement"], r["patch"]))
    return tasks


def git(repo_dir: Path, *args, check=True) -> str:
    out = subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def checkout(task: Task) -> Path:
    """Clone the repo once (blobless) and check out the task's base commit."""
    d = task.repo_dir
    if not d.exists():
        REPOS.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout",
                        "-c", "core.longpaths=true", f"https://github.com/{task.repo}.git", str(d)],
                       check=True)
    git(d, "checkout", "-q", "-f", task.base_commit)
    git(d, "clean", "-q", "-fdx")
    return d


def patch_changes(patch: str) -> dict:
    """{file: set of ORIGINAL line numbers touched}. Pure insertions count the
    line just before the insertion point."""
    changes, path, old_line = {}, None, 0
    for line in patch.splitlines():
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            path = line[4:].strip().removeprefix("b/")
            changes.setdefault(path, set())
        elif (m := HUNK.match(line)):
            old_line = int(m.group(1))
        elif path is None:
            continue
        elif line.startswith("-"):
            changes[path].add(old_line)
            old_line += 1
        elif line.startswith("+"):
            changes[path].add(max(old_line - 1, 1))
        elif line.startswith(" "):
            old_line += 1
    return changes


def enclosing_functions(source: str, lines: set) -> set:
    """Qualified names of the innermost functions/methods containing `lines`."""
    found = set()

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                start = min([d.lineno for d in child.decorator_list] + [child.lineno])
                span = {ln for ln in lines if start <= ln <= child.end_lineno}
                if span:
                    if isinstance(child, ast.ClassDef):
                        visit(child, name + ".")
                    else:
                        found.add(name)  # innermost function wins; nested defs stay with the outer one
            elif hasattr(child, "body"):
                visit(child, prefix)  # if/try blocks at module or class level

    visit(ast.parse(source), "")
    return found


def add_gold(task: Task) -> Task:
    changes = patch_changes(task.patch)
    task.gold_files = sorted(changes)
    funcs = set()
    for path, lines in changes.items():
        if path.endswith(".py") and lines:
            try:
                src = git(task.repo_dir, "show", f"{task.base_commit}:{path}")
            except RuntimeError:
                continue  # file created by the patch
            funcs |= {f"{path}::{q}" for q in enclosing_functions(src, lines)}
    task.gold_functions = sorted(funcs)
    return task


if __name__ == "__main__":
    for t in load_tasks()[:5]:
        checkout(t)
        add_gold(t)
        print(t.instance_id, t.gold_files, t.gold_functions)
