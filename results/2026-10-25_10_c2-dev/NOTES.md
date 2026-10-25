# 10 · C2: graph tools (2026-10-25)

**Question:** do code-graph tools help small models localize bugs?

**Setup:** experiment 07 (structured answers, repeat guard, budget hints, ranked lists) plus `--graph-tools`, which adds four tools over a code index (`agent/index.py`, Python `ast`):

| Tool | Returns |
|---|---|
| `find_definition(name)` | file, lines, signature, docstring line of matching definitions |
| `get_code_neighbors(name)` | callees and callers (name-matched call graph) |
| `search_similar_code(query)` | top functions by BM25 over their code |
| `get_file_outline(path)` | classes/methods/functions in a file with line ranges |

| Metric | Gemma 4 E2B: 07 → C2 | Qwen3.5-4B: 07 → C2 |
|---|---|---|
| File@1 | 65% → **74%** | 78% → 78% |
| File@5 | 78% → 83% | 83% → 87% |
| Function@5 (n = 21) | 43% → **52%** | 71% → 71% |
| Median steps / time | 16 / 47 s → 16 / 40 s | 10 / 39 s → 10 / 33 s |
| Share of calls to graph tools | — → 8% (26 of 329) | — → 5% (12 of 247) |
| Tasks with ≥ 1 graph-tool call | — → 14 of 23 | — → 8 of 23 |

Paired tests (07 → C2): Gemma gained 3 and lost 1 task on both File@1 and Function@5 (exact McNemar p = 0.63); Qwen gained 1 and lost 1 (p = 1.0).

**Findings:**
1. **Gemma reaches its best accuracy so far** (Function@5 52%, File@1 74%) and is ~15% faster per task; Qwen is unchanged. Neither change is statistically significant at n = 23.
2. **The models barely use the new tools.** Over 90% of calls still go to `search_code` and `read_file`. `find_definition` is the only graph tool used regularly; `search_similar_code` and `get_file_outline` are almost never called, and Qwen never calls `search_similar_code`.
3. Tool *availability* is not tool *adoption*: for small models, how tools are presented (or whether they are offered at all) may matter more than how good they are. A natural follow-up is to run graph queries automatically (e.g. show `find_definition` results for names in the issue) instead of waiting for the model to call them.

**Files:** `<model>.json`, `metrics.json`, `tool_usage.json`, `config.json`, `figures/summary.*` (panel e: tool usage), `figures/outcomes.*`. Trajectories: `C:\gemma_work\runs\2026-10-25_10_c2-dev\`.
**Regenerate:** `python scripts/rescore.py && python analysis/plot_experiment.py --exp c2-dev`.
