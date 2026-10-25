"""File tools for the C1 configuration: list_dir, read_file, search_code, submit.

Every tool output is capped so one step fits the ~8K-token context budget.
Paths are relative to the repo root and may not escape it.
"""

import subprocess
from pathlib import Path

MAX_OUT_CHARS = 4000  # ~1K tokens
MAX_READ_LINES = 120

TOOL_DOCS = """\
list_dir(path): list files and folders in a directory of the repo ("." is the root).
search_code(pattern, path): find lines containing the exact text `pattern` (case-sensitive). `path` limits the search to a folder or file; use "." for the whole repo.
read_file(path, start_line, end_line): show numbered lines of a file, at most 120 lines per call.
"""
SUBMIT_DOC = """submit(files, functions): final answer. `files`: up to 5 file paths, most likely first. `functions`: up to 10 entries "path/to/file.py::function" or "path/to/file.py::Class.method", most likely first."""
SUBMIT_DOC_RANKED = """submit(files, functions): final answer as ranked candidate lists, most likely first. `files`: 3 to 5 file paths. `functions`: 3 to 10 entries "path/to/file.py::function" or "path/to/file.py::Class.method". Put your best guess first, then the next most likely candidates."""
SUBMIT_DOC_STRUCT = """submit(files, functions): final answer. `files`: up to 5 file paths, most likely first. `functions`: up to 10 objects {"file": "path/to/file.py", "name": "function" or "Class.method"}, most likely first."""
SUBMIT_DOC_STRUCT_RANKED = """submit(files, functions): final answer as ranked candidate lists, most likely first. `files`: 3 to 5 file paths. `functions`: 3 to 10 objects {"file": "path/to/file.py", "name": "function" or "Class.method"}. Put your best guess first, then the next most likely candidates."""

# "path/to/file.py::Name" or "path/to/file.py::Class.method". Bounded repetition
# matters: with unbounded parts, a model that does not want to write ".py::" can
# keep extending the string until it hits the token limit.
FUNC_PATTERN = r"^[A-Za-z0-9_/.-]{1,100}[.]py::[A-Za-z_][A-Za-z0-9_.]{0,60}$"
FILE_PATTERN = r"^[A-Za-z0-9_/.-]{1,120}$"  # a repo path: no spaces, no "::name"
NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_.]{0,80}$"  # dots allowed, so "pkg.mod.func" can end


def tool_docs(ranked: bool = False, struct_answer: bool = False) -> str:
    if struct_answer:
        return TOOL_DOCS + (SUBMIT_DOC_STRUCT_RANKED if ranked else SUBMIT_DOC_STRUCT)
    return TOOL_DOCS + (SUBMIT_DOC_RANKED if ranked else SUBMIT_DOC)

TOOL_ARGS = {
    "list_dir": {"path": {"type": "string"}},
    "search_code": {"pattern": {"type": "string", "minLength": 2, "maxLength": 100},
                    "path": {"type": "string"}},
    "read_file": {"path": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1},
                  "end_line": {"type": "integer", "minimum": 1}},
    "submit": {"files": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
               "functions": {"type": "array", "items": {"type": "string"}, "maxItems": 10}},
}


def submit_args(func_format: bool = False, ranked: bool = False, struct_answer: bool = False) -> dict:
    """Arguments of submit.
    func_format:   regex forces "path.py::Name" strings (experiment 04; hurts Qwen,
                   which starts dotted names and is then forced into garbage).
    struct_answer: each function is an object {"file", "name"}; no regex has to
                   hold a path and a name in one string.
    ranked:        at least 3 candidates per list."""
    constrained = func_format or struct_answer
    file_item = {"type": "string", "pattern": FILE_PATTERN} if constrained else {"type": "string"}
    files = {"type": "array", "items": file_item, "maxItems": 5}
    if struct_answer:
        item = {"type": "object", "properties": {"file": {"type": "string", "pattern": FILE_PATTERN},
                                                 "name": {"type": "string", "pattern": NAME_PATTERN}},
                "required": ["file", "name"], "additionalProperties": False}
    elif func_format:
        item = {"type": "string", "pattern": FUNC_PATTERN}
    else:
        item = {"type": "string"}
    funcs = {"type": "array", "items": item, "maxItems": 10}
    if ranked:
        files["minItems"] = funcs["minItems"] = 3
    return {"files": files, "functions": funcs}


def action_schema(tools: list, func_format: bool = False, ranked: bool = False,
                  struct_answer: bool = False, evidence: bool = False) -> dict:
    """JSON schema for one agent step: a short thought plus exactly one tool call."""
    args = {**TOOL_ARGS, "submit": submit_args(func_format, ranked, struct_answer)}
    options = [{"type": "object",
                "properties": {"tool": {"const": name}, **args[name]},
                "required": ["tool", *args[name]], "additionalProperties": False}
               for name in tools]
    props = {"thought": {"type": "string", "maxLength": 400}}
    if evidence:  # Evidence Buffer (C3): one short fact per step, may be empty
        props["evidence"] = {"type": "string", "maxLength": 200}
    props["action"] = {"anyOf": options}
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def _cap(text: str) -> str:
    if len(text) <= MAX_OUT_CHARS:
        return text
    return text[:MAX_OUT_CHARS] + f"\n... [truncated, {len(text) - MAX_OUT_CHARS} more characters]"


class FileTools:
    def __init__(self, repo_dir: Path):
        self.root = repo_dir.resolve()

    def _resolve(self, path: str) -> Path:
        p = (self.root / path.strip().lstrip("/\\")).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"path outside the repository: {path}")
        return p

    def list_dir(self, path: str = ".") -> str:
        p = self._resolve(path)
        if not p.is_dir():
            raise ValueError(f"not a directory: {path}")
        entries = sorted((e.name + "/" if e.is_dir() else e.name) for e in p.iterdir()
                         if not e.name.startswith("."))
        return _cap("\n".join(entries) or "(empty)")

    def read_file(self, path: str, start_line: int = 1, end_line: int = None) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ValueError(f"not a file: {path}")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, start_line)
        end = min(len(lines), end_line or start + MAX_READ_LINES - 1, start + MAX_READ_LINES - 1)
        body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        return _cap(f"{path} (lines {start}-{end} of {len(lines)})\n{body}")

    def search_code(self, pattern: str, path: str = ".") -> str:
        self._resolve(path)
        out = subprocess.run(["git", "-C", str(self.root), "grep", "-n", "-I", "-F", "--",
                              pattern, path.strip() or "."],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        hits = out.stdout.splitlines()
        if not hits:
            return "No matches."
        shown = [h[:200] for h in hits[:40]]
        more = f"\n... {len(hits) - 40} more matches; narrow `path` or `pattern`." if len(hits) > 40 else ""
        return _cap(f"{len(hits)} matches\n" + "\n".join(shown) + more)

    def call(self, action: dict) -> str:
        name = action["tool"]
        args = {k: v for k, v in action.items() if k != "tool"}
        return getattr(self, name)(**args)


# ---------------------------------------------------------------- graph tools (C2)

GRAPH_TOOL_DOCS = """\
find_definition(name): where a function, method, or class is defined: file, lines, signature. `name` is a plain name ("fuentes") or qualified ("Dataset.to_json_dict").
get_code_neighbors(name): the call graph around a function: the functions it calls and the functions that call it, with file and line. Matched by name, so it can include same-named functions elsewhere.
search_similar_code(query): the functions most related to a short description or error message, ranked by word overlap with their code.
get_file_outline(path): every class, method, and function in a file, with line ranges.
"""

GRAPH_TOOL_ARGS = {
    "find_definition": {"name": {"type": "string", "minLength": 2, "maxLength": 80}},
    "get_code_neighbors": {"name": {"type": "string", "minLength": 2, "maxLength": 80}},
    "search_similar_code": {"query": {"type": "string", "minLength": 3, "maxLength": 200}},
    "get_file_outline": {"path": {"type": "string"}},
}
TOOL_ARGS.update(GRAPH_TOOL_ARGS)
MAX_LIST = 15


def _loc(s: dict) -> str:
    return f"{s['file']}:{s['start']}-{s['end']}"


class GraphTools(FileTools):
    """File tools plus code-graph tools over a CodeIndex (agent/index.py)."""

    def __init__(self, repo_dir: Path, index):
        super().__init__(repo_dir)
        self.index = index

    def _match(self, name: str) -> list:
        name = name.strip().removesuffix("()").split("::")[-1]
        last = name.split(".")[-1]
        cands = [i for i in self.index.by_name.get(last, [])]
        if "." in name:  # qualified: prefer exact qualname suffix matches
            exact = [i for i in cands if self.index.symbols[i]["qualname"].endswith(name)]
            cands = exact or cands
        return cands

    def find_definition(self, name: str) -> str:
        hits = self._match(name)
        if not hits:
            return f"No definition named {name!r}. Try search_similar_code or search_code."
        rows = []
        for i in hits[:MAX_LIST]:
            s = self.index.symbols[i]
            rows.append(f"{_loc(s)}  {s['kind']} {s['qualname']}: {s['signature']}"
                        + (f"\n    \"{s['doc']}\"" if s["doc"] else ""))
        more = f"\n... {len(hits) - MAX_LIST} more definitions" if len(hits) > MAX_LIST else ""
        return _cap(f"{len(hits)} definition(s)\n" + "\n".join(rows) + more)

    def get_code_neighbors(self, name: str) -> str:
        hits = [i for i in self._match(name) if self.index.symbols[i]["kind"] != "class"]
        if not hits:
            return f"No function named {name!r}. Try find_definition or search_similar_code."
        out = []
        for i in hits[:3]:
            s = self.index.symbols[i]
            callees = []
            for c in s["calls"]:
                for j in self.index.by_name.get(c, [])[:3]:
                    callees.append(f"{self.index.symbols[j]['qualname']} ({_loc(self.index.symbols[j])})")
            callers = [f"{self.index.symbols[j]['qualname']} ({_loc(self.index.symbols[j])})"
                       for j in sorted(self.index.callers.get(i, []))]
            out.append(f"{s['qualname']} ({_loc(s)})\n"
                       f"  calls ({len(callees)}): " + ("; ".join(callees[:MAX_LIST]) or "none in this repo") + "\n"
                       f"  called by ({len(callers)}): " + ("; ".join(callers[:MAX_LIST]) or "none found"))
        more = f"\n({len(hits) - 3} more functions share this name)" if len(hits) > 3 else ""
        return _cap("\n".join(out) + more)

    def search_similar_code(self, query: str) -> str:
        hits = self.index.rank_symbols(query, MAX_LIST, kinds=("function", "method", "class"))
        if not hits:
            return "No related code found. Try different words."
        return _cap("\n".join(f"{_loc(s)}  {s['kind']} {s['qualname']}: {s['signature'][:120]}"
                              + (f"  \"{s['doc'][:80]}\"" if s["doc"] else "") for s, _ in hits))

    def get_file_outline(self, path: str) -> str:
        path = path.strip().lstrip("./").replace("\\", "/")
        syms = [s for s in self.index.symbols if s["file"] == path]
        if not syms:
            self._resolve(path)
            return f"No functions or classes indexed in {path} (not a Python file, or empty)."
        rows = [f"{'    ' * s['qualname'].count('.')}{s['start']}-{s['end']}  {s['kind']} {s['qualname']}"
                for s in sorted(syms, key=lambda s: s["start"])]
        return _cap(f"{path}: {len(syms)} definitions\n" + "\n".join(rows))
