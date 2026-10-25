# 08 · Dev-split ablation ladder: 03 → 05 → 06 → 07 (+ 04) (2026-10-25)

**What this is:** a comparison of experiments run on the same 23 dev tasks, no new model runs. Each step adds one harness change to the previous one; experiment 04 (regex format) is shown as an alternative to step 05.

| Step | Change | Gemma 4 E2B File@1 / Func@5 | Qwen3.5-4B File@1 / Func@5 |
|---|---|---|---|
| 03 | C1: file tools, free-text function names | 83 / 43 | 83 / 62 |
| (04) | regex-forced `path.py::Name` | 83 / 43 | 83 / **33*** |
| 05 | structured `{"file", "name"}` answers | 78 / 29 | 87 / **76** |
| 06 | + repeated-call guard + budget hints | 70 / 33 | 83 / 71 |
| 07 | + ranked lists (≥3 candidates) | 65 / 43 | 78 / 71 |

\* the only statistically significant change: Qwen lost 6 tasks and gained none from 03 to 04 (exact McNemar, p = 0.031). Every ladder step has p > 0.25 for both models.

**Findings across the ladder:**
1. **Format matters most, and in both directions.** How the answer is written moved Qwen's function accuracy from 33% (regex) to 76% (structured) with identical exploration. A grammar can guarantee the *form* of an answer while destroying its *content* (04).
2. **Output-format changes are model-dependent.** The structured format helped Qwen (+14) and hurt Gemma (−14), which then left function slots empty.
3. **Behavior fixes are not accuracy fixes.** The guard and budget hints cut Gemma's forced submits from 57% to 9% and Qwen's repeats from 17% to 3%, with no accuracy gain: looping was a symptom of not knowing where to look.
4. **Ranked lists help a model whose first guess is unreliable** (Gemma Func@5 33 → 43) and slightly hurt the top slot of both.
5. **Qwen3.5-4B is the stronger localizer** at every step (Func@5 62–76% vs 29–43%), at similar wall time per task (36–46 s vs 42–57 s).
6. **Statistical power is the main limitation.** With 23 tasks, a change needs to flip about 5–6 tasks one way to be detectable. Before drawing conclusions for the paper, the most promising configurations should run on the 100-task test set.

**Recommended configuration going forward:** structured answers + guard + budget hints (06) as the base for both models, ranked lists on for Gemma only if we allow per-model settings; otherwise 07 for both.

**Files:** `stats.json` (per-step metrics, Wilson intervals, McNemar tests, alternative branch), `figures/ablation.{pdf,png}`.
**Regenerate:**
```
python analysis/plot_ablation.py --name dev-ablation --exps c1-dev struct-dev guard-dev ranked-dev \
  --labels "C1\n(03)" "+struct.\nanswer (05)" "+guard\n+budget (06)" "+ranked\n(07)" \
  --alt fmt-dev --alt-at 1 --alt-label "regex format, 04"
```
