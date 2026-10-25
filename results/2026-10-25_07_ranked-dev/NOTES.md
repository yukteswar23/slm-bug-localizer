# 07 · Ranked candidate lists (2026-10-25)

**Question:** in experiments 03–06 both models submit only ~1 file and ~1 function, so @5 metrics equal @1. Does requiring ranked candidate lists turn the extra slots into extra hits?

**Setup:** experiment 06 plus `--ranked`: `submit` must contain 3–5 files and 3–10 functions (`minItems: 3` in the grammar), and the prompt asks for a ranked list with the best guess first.

| Metric (95% Wilson CI) | Gemma 4 E2B: 06 → 07 | Qwen3.5-4B: 06 → 07 |
|---|---|---|
| File@1 | 70% → 65% [45, 81] | 83% → 78% [58, 90] |
| File@5 | 70% → 78% [58, 90] | 83% → 83% [63, 93] |
| Function@5 (n = 21) | 33% → 43% [24, 63] | 71% → 71% [50, 86] |
| Candidates submitted (files / functions) | 3.0 / 3.0 | 3.2 / 3.1 |
| Forced submit | 9% → 9% | 35% → 22% |

**Findings:**
1. Both models submit exactly the minimum (3), so the ranked format is padding to the floor rather than real ranking.
2. **Gemma gains from the extra slots** (File@5 +8, Function@5 +10 points): its right answer is sometimes its 2nd or 3rd guess.
3. **The top slot gets slightly worse for both** (File@1 −5 points): asking for alternatives sometimes moves the best guess down.
4. Qwen's accuracy does not change: when it is right, its first guess is right.
5. None of these changes is statistically significant at n = 23 (see 08).

**Files:** `<model>.json`, `metrics.json`, `config.json`, `figures/summary.*`, `figures/outcomes.*`. Trajectories: `C:\gemma_work\runs\2026-10-25_07_ranked-dev\`.
**Regenerate:** `python scripts/rescore.py && python analysis/plot_experiment.py --exp ranked-dev`.
