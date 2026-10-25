"""Code index for one repository at one commit: functions, classes, calls, BM25.

Built with Python's `ast` (all SWE-bench Lite repos are Python), cached as
gzipped JSON in C:/gemma_work/indexes/<repo>@<commit>.json.gz. Used by the C0
retrieval baseline and by the graph tools (C2).

Symbols: every function, method, and class, with file, qualified name, line
span, signature, first docstring line, and the names it calls. Call edges are
resolved by name only (a call to `foo` or `x.foo` links to every definition
named `foo`), so the call graph over-approximates; tools report this.
"""

import ast
import gzip
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

INDEX_DIR = Path("C:/gemma_work/indexes")
SKIP_DIRS = {".git", "build", "dist", ".tox", "node_modules", "__pycache__"}
MAX_FILE_BYTES = 1_000_000
TOKEN = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")
STOP = {"the", "and", "for", "self", "this", "that", "with", "from", "not", "return", "none",
        "true", "false", "import", "def", "class", "is", "in", "if", "to", "of", "a", "be",
        "it", "or", "as", "on", "an", "by", "at", "are", "was", "but", "when"}


def tokenize(text: str) -> list:
    """Split identifiers too: getSeparabilityMatrix / separability_matrix -> separability, matrix."""
    return [t.lower() for t in TOKEN.findall(text) if len(t) > 1 and t.lower() not in STOP]


def _signature(node) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    return f"def {node.name}({ast.unparse(node.args)})"


def _calls(node) -> list:
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return sorted(names)


def _symbols(path: str, source: str) -> list:
    tree = ast.parse(source)
    lines = source.splitlines()
    out = []

    def visit(node, prefix, parent_is_class):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = f"{prefix}{child.name}"
                doc = (ast.get_docstring(child) or "").strip().split("\n")[0][:160]
                kind = "class" if isinstance(child, ast.ClassDef) else ("method" if parent_is_class else "function")
                body = "\n".join(lines[child.lineno - 1:child.end_lineno])
                out.append({"file": path, "qualname": qual, "name": child.name, "kind": kind,
                            "start": child.lineno, "end": child.end_lineno, "signature": _signature(child),
                            "doc": doc, "calls": [] if kind == "class" else _calls(child),
                            "tokens": tokenize(qual + " " + body)})
                visit(child, qual + ".", isinstance(child, ast.ClassDef))
            elif hasattr(child, "body"):
                visit(child, prefix, parent_is_class)

    visit(tree, "", False)
    return out


class BM25:
    def __init__(self, docs: list, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list) -> list:
        q = [t for t in set(query) if t in self.idf]
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


class CodeIndex:
    def __init__(self, data: dict):
        self.files = data["files"]          # {path: [tokens]}
        self.symbols = data["symbols"]      # list of symbol dicts
        self.by_name = defaultdict(list)
        for i, s in enumerate(self.symbols):
            self.by_name[s["name"]].append(i)
        self.callers = defaultdict(set)     # symbol index -> indices of symbols calling it
        for i, s in enumerate(self.symbols):
            for name in s["calls"]:
                for j in self.by_name.get(name, []):
                    if j != i:
                        self.callers[j].add(i)
        self._file_bm25 = self._sym_bm25 = None

    @property
    def file_paths(self) -> list:
        return list(self.files)

    def rank_files(self, query: str, k: int = 5) -> list:
        if self._file_bm25 is None:
            self._file_bm25 = BM25(list(self.files.values()))
        sc = self._file_bm25.scores(tokenize(query))
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        return [(self.file_paths[i], sc[i]) for i in order[:k] if sc[i] > 0]

    def rank_symbols(self, query: str, k: int = 10, kinds=("function", "method"), files=None) -> list:
        if self._sym_bm25 is None:
            self._sym_bm25 = BM25([s["tokens"] for s in self.symbols])
        sc = self._sym_bm25.scores(tokenize(query))
        idx = [i for i in range(len(sc)) if sc[i] > 0 and self.symbols[i]["kind"] in kinds
               and (files is None or self.symbols[i]["file"] in files)]
        idx.sort(key=lambda i: -sc[i])
        return [(self.symbols[i], sc[i]) for i in idx[:k]]


def build_index(repo_dir: Path, key: str) -> CodeIndex:
    """key: "<repo>@<commit>" used for the cache file name."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    cache = INDEX_DIR / f"{key.replace('/', '__')}.json.gz"
    if cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            return CodeIndex(json.load(f))
    files, symbols = {}, []
    for p in sorted(repo_dir.rglob("*.py")):
        rel = p.relative_to(repo_dir).as_posix()
        if any(part in SKIP_DIRS for part in p.parts) or p.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
            syms = _symbols(rel, src)
        except (SyntaxError, ValueError, RecursionError):
            continue  # python 2 files, generated code
        files[rel] = tokenize(rel.replace("/", " ") + " " + src)
        symbols.extend(syms)
    data = {"files": files, "symbols": symbols}
    tmp = cache.with_suffix(f".{os.getpid()}.tmp")  # atomic: never expose a half-written cache
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, cache)
    return CodeIndex(data)
