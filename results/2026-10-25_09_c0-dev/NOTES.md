# 09 · C0 retrieval baseline, no language model (2026-10-25)

**Question:** how far does plain keyword retrieval get without any model? This is the floor every agent configuration has to beat.

**Setup:** for each dev task (23), files are ranked by BM25 between the issue text and each file's text (identifiers split into words); functions are ranked by BM25 against each function/method body, restricted to the top-5 files. Code index: `agent/index.py` (Python `ast`, cached in `C:\gemma_work\indexes\`). About 2.5 s per task, CPU only.

| Metric (95% Wilson CI) | BM25 | Gemma 4 E2B, C1 (03) | Qwen3.5-4B, C1 (03) |
|---|---|---|---|
| File@1 | 35% [19, 55] | 83% | 83% |
| File@5 | 57% [37, 74] | 83% | 83% |
| Function@5 (n = 21) | 43% [24, 63] | 43% | 62% |
| Function@10 | 52% [32, 72] | 43% | 62% |

**Findings:**
1. **The agents' advantage is in finding the file**: +48 points File@1 over BM25 for both models.
2. **At function level, BM25 matches Gemma's agent** (43% vs 43%), and with 10 candidates beats it (52%). Once BM25 has the right file, its function ranking is competitive, because it ranks every function in 5 files while the agents name one or three.
3. This motivates combining the two: retrieval candidates as a starting point for the agent (a possible later configuration), and it is the baseline row for the paper's main table.

**Files:** `bm25.json`, `metrics.json`, `config.json`, `figures/summary.*`, `figures/outcomes.*`.
**Regenerate:** `python scripts/run_bm25.py --run c0-dev --split dev && python analysis/plot_experiment.py --exp c0-dev`.
