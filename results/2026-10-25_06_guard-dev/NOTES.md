# 06 · Repeated-call guard and budget hints (2026-10-25)

**Question:** Gemma ran into the step limit in 70% of episodes and repeated earlier calls in ~40% of steps (experiment 03). Do a repeated-call guard and turn-budget hints fix this, and does fixing it improve accuracy?

**Setup:** experiment 05 plus `--repeat-guard --budget-hints`:
- *Repeat guard:* an exact repeat of an earlier call is not executed; the model is told it already made that call and to choose something else or submit.
- *Budget hints:* every tool output ends with `[Turn i of 20]`; in the last 5 turns it adds "submit as soon as you have a candidate location".

| Metric | Gemma 4 E2B: 05 → 06 | Qwen3.5-4B: 05 → 06 |
|---|---|---|
| Forced submit (tasks) | 57% → **9%** | 43% → 35% |
| Repeated calls (share of steps, attempted) | 40% → 21% | 17% → **3%** |
| Median steps / time per task | 20 / 57 s → 16 / 42 s | 11 / 46 s → 11 / 38 s |
| File@1 | 78% → 70% [49, 84] | 87% → 83% [63, 93] |
| Function@5 (n = 21) | 29% → 33% [17, 55] | 76% → 71% [50, 86] |

**Findings:**
1. **The guard and hints fix the behavior.** Gemma now answers on its own in 91% of episodes (was 43%) and finishes 26% faster; Qwen's repeats nearly vanish.
2. **They do not improve accuracy.** Changes in File@1 and Function@5 are within noise for both models. Gemma's looping was a symptom of not knowing where to look, not the cause of its misses: stopping it earlier saves time but does not find the bug.
3. The "Repeated call" rate counts *attempted* repeats, including blocked ones: Gemma still tries to repeat a call in 21% of steps even after being told it already made it.

**Files:** `<model>.json`, `metrics.json`, `config.json`, `figures/summary.*`, `figures/outcomes.*`. Trajectories: `C:\gemma_work\runs\2026-10-25_06_guard-dev\`.
**Regenerate:** `python scripts/rescore.py && python analysis/plot_experiment.py --exp guard-dev`.
