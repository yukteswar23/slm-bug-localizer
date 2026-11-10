# 11 · C3: Evidence Buffer (2026-10-25/26) — negative result

**Question:** does an Evidence Buffer (the model records one short fact per step, and the collected facts survive when old tool outputs are hidden) improve localization?

**Setup:** C2 (experiment 10) plus `--evidence-buffer`:
- every reply has a required `evidence` field (≤ 200 chars, may be empty);
- the collected facts are appended to the latest tool output as "Evidence so far";
- old tool outputs are hidden in one chunk, down to half the history budget, instead of one at a time (intended to keep the KV cache valid longer).

| Metric | Gemma 4 E2B: C2 → C3 | Qwen3.5-4B: C2 → C3 |
|---|---|---|
| File@1 | 17 → 18 of 23 | 18 → 18 |
| Function@5 | 11 → 9 of 21 | 15 → 14 |
| Median time per task | 40 → 57 s (+43%) | 33 → 50 s (+52%) |
| Generated tokens (all tasks) | 41.9K → 54.2K | 20.7K → 29.0K |
| KV cache reuse | 80% → 75% | 74% → 76% |
| Steps carrying evidence | — | — |

Paired tests (C2 → C3): Gemma File@1 +2/−1 (p = 1.0), Function@5 +1/−3 (p = 0.63); Qwen File@1 +1/−1, Function@5 +1/−2 (p = 1.0).

**Findings:**
1. **No accuracy gain for either model**, at a 43–52% time cost: writing evidence every step adds ~30–40% generated tokens, and generation is the slow part on a 4 GB GPU.
2. **Chunked hiding did not improve cache reuse** (75–76% vs 74–80%): moving the buffer to each new output, and hiding a big chunk at once, cost about as much cache as the old one-at-a-time hiding saved.
3. Within a 20-step, ~8K-token budget, the conversation rarely grows long enough for lost context to matter; the buffer solves a problem these episodes mostly do not have. It might matter for longer budgets.

**Decision:** C2 (structured answers + guard + budget hints + ranked lists + graph tools) is the best configuration and is used as the "full system", including for the Gemma 4 31B reference. C3 is kept in the test-set runs as an ablation row.

**Files:** `<model>.json` (steps include `evidence`), `metrics.json`, `tool_usage.json`, `config.json`, `figures/summary.*`, `figures/outcomes.*`. Trajectories: `C:\gemma_work\runs\2026-10-25_11_c3-dev\`.
**Regenerate:** `python scripts/rescore.py && python analysis/plot_experiment.py --exp c3-dev`.
