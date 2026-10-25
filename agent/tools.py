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
submit(files, functions): final answer. `files`: up to 5 file paths, most likely first. `functions`: up to 10 entries "path/to/file.py::function" or "path/to/file.py::Class.method", most likely first."""

TOOL_ARGS = {
    "list_dir": {"path": {"type": "string"}},
    "search_code": {"pattern": {"type": "string", "minLength": 2, "maxLength": 100},
                    "path": {"type": "string"}},
    "read_file": {"path": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1},
                  "end_line": {"type": "integer", "minimum": 1}},
    "submit": {"files": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
               "functions": {"type": "array", "items": {"type": "string"}, "maxItems": 10}},
}


def action_schema(tools: list) -> dict:
    """JSON schema for one agent step: a short thought plus exactly one tool call."""
    options = [{"type": "object",
                "properties": {"tool": {"const": name}, **TOOL_ARGS[name]},
                "required": ["tool", *TOOL_ARGS[name]], "additionalProperties": False}
               for name in tools]
    return {"type": "object",
            "properties": {"thought": {"type": "string", "maxLength": 400},
                           "action": {"anyOf": options}},
            "required": ["thought", "action"], "additionalProperties": False}


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
