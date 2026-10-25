"""Localization metrics against gold locations from the reference patch.

acc_all@k: every gold item is in the top-k predictions (main metric, as in
           Agentless / LocAgent style evaluations).
hit@k:     at least one gold item is in the top-k predictions.
Function names are compared as "path::Qualified.name"; a prediction without a
class prefix also matches a gold method of the same name in the same file
(small models often drop the class). Module-style paths are normalized to
file paths ("pkg.mod.py" or "pkg.mod" -> "pkg/mod.py") because the intent is
unambiguous; small models often write them that way.
"""

import re

MODULE_PATH = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)+")  # pkg.mod.sub, at least one dot
NON_PY_EXT = {"rst", "txt", "md", "cfg", "toml", "ini", "json", "yml", "yaml", "in", "c", "h", "pyx"}


def norm_path(p: str) -> str:
    p = p.strip().replace("\\", "/")
    while p.startswith(("./", "/")):
        p = p[1:] if p.startswith("/") else p[2:]
    if "/" not in p and MODULE_PATH.fullmatch(p.removesuffix(".py")) \
            and p.rsplit(".", 1)[-1] not in NON_PY_EXT:
        p = p.removesuffix(".py").replace(".", "/") + ".py"
    return p


def norm_func(f: str) -> str:
    path, _, name = f.partition("::")
    return f"{norm_path(path)}::{name.strip().removesuffix('()')}"


def func_match(pred: str, gold: str) -> bool:
    if pred == gold:
        return True
    pp, pn = pred.split("::", 1) if "::" in pred else ("", pred)
    gp, gn = gold.split("::", 1)
    return pp == gp and "." not in pn and gn.split(".")[-1] == pn


def dedup(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def score(answer: dict, gold_files: list, gold_functions: list) -> dict:
    files = dedup(norm_path(f) for f in answer.get("files", []))
    funcs = dedup(norm_func(f) for f in answer.get("functions", []) if f.strip())
    res = {}
    for k in (1, 3, 5):
        top = files[:k]
        res[f"file_acc_all@{k}"] = float(all(g in top for g in gold_files))
        res[f"file_hit@{k}"] = float(any(g in top for g in gold_files))
    for k in (5, 10):
        top = funcs[:k]
        if gold_functions:
            res[f"func_acc_all@{k}"] = float(all(any(func_match(p, g) for p in top) for g in gold_functions))
            res[f"func_hit@{k}"] = float(any(func_match(p, g) for p in top for g in gold_functions))
        else:  # change is outside any function: only file metrics apply
            res[f"func_acc_all@{k}"] = res[f"func_hit@{k}"] = None
    return res
