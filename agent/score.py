"""Localization metrics against gold locations from the reference patch.

acc_all@k: every gold item is in the top-k predictions (main metric, as in
           Agentless / LocAgent style evaluations).
hit@k:     at least one gold item is in the top-k predictions.
Function names are compared as "path::Qualified.name"; a prediction without a
class prefix also matches a gold method of the same name in the same file
(small models often drop the class). Module-style paths are normalized to
file paths ("pkg.mod.py" or "pkg.mod" -> "pkg/mod.py") because the intent is
unambiguous; small models often write them that way.

Function entries not in the requested "path.py::Name" format are resolved
against the model's own submitted files: a dotted name "pkg.mod.Class.f" is
matched to the submitted file pkg/mod.py, and a bare "Class.f" is assigned to
the top submitted file. Entries that still cannot be resolved (line numbers,
file names in place of functions) count as wrong. `func_malformed` records
whether ANY entry needed repair, so format failures stay visible as their own
failure mode.
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


CANONICAL = re.compile(r"^[\w./-]+\.py::[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")
QUALNAME = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")


def is_canonical(f: str) -> bool:
    """Exactly the requested format: path/to/file.py::Qual.name"""
    f = f.strip()
    return bool(CANONICAL.match(f)) and is_qualname(f.split("::", 1)[1])


def is_qualname(name: str) -> bool:
    return bool(QUALNAME.match(name)) and name.rsplit(".", 1)[-1] != "py"


def module_of(path: str) -> list:
    """Import names a file can appear under: pkg/mod.py -> pkg.mod (and without src/)."""
    mod = path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
    return [mod, mod.removeprefix("src.")] if mod.startswith("src.") else [mod]


def norm_func(f: str, files: list):
    """Canonical "path.py::Qual.name", or None if the entry cannot be resolved."""
    f = f.strip().removesuffix("()")
    if "::" in f:
        path, _, name = f.partition("::")
        path, name = norm_path(path), name.strip().removesuffix("()")
        return f"{path}::{name}" if path.endswith(".py") and is_qualname(name) else None
    if not is_qualname(f):
        return None
    for path in files:  # dotted "pkg.mod.Class.f": find the submitted file it lives in
        for mod in module_of(path):
            if f.startswith(mod + "."):
                return f"{path}::{f[len(mod) + 1:]}"
    # bare "func" or "Class.method" belongs to the top submitted file; longer dotted
    # names are module paths we could not match to a submitted file
    return f"{files[0]}::{f}" if files and f.count(".") <= 1 else None


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
    raw = [f for f in answer.get("functions", []) if f.strip()]
    funcs = dedup(n for n in (norm_func(f, files) for f in raw) if n)
    res = {"func_malformed": float(any(not is_canonical(f) for f in raw))}
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
