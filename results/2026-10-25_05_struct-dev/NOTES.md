# 05 · Structured answer format (2026-10-25)

**Question:** experiment 04 showed that forcing `path.py::Name` with a regex corrupts Qwen's answers. Does a *structured* format work instead, where each function is an object `{"file": "path/to/file.py", "name": "Class.method"}`?

**Setup:** C1 (experiment 03) plus `--struct-answer`: file paths and the two fields are grammar-constrained with bounded patterns; the name may contain dots, so a model that starts a dotted name can still finish it. The scorer strips a module prefix repeated inside the name (`pvlib/iam.py` + `iam.physical` → `physical`). Unlike 04, this also changes the `submit` description in the system prompt, so whole trajectories differ from 03, not only the final step.

| Metric (95% Wilson CI) | Gemma 4 E2B: 03 → 05 | Qwen3.5-4B: 03 → 05 |
|---|---|---|
| File@1 | 83% → 78% [58, 90] | 83% → 87% [68, 95] |
| Function@5 (n = 21) | 43% → **29%** [14, 50] | 62% → **76%** [55, 89] |
| Malformed names | 13% → 4% | 43% → 0% |
| Forced submit | 70% → 57% | 43% → 43% |

**Findings:**
1. **The structured format helps Qwen** (Function@5 62% → 76%, malformed names 43% → 0%). It recovers the right answers that experiment 04's regex destroyed (04: 33%).
2. **It hurts Gemma** (43% → 29%). Of Gemma's changes against 03: 2 tasks returned an *empty* function list (the object format seems to make it skip the field), 2 named a wrong or hallucinated function (`infer_str_format_call` for `_infer_str_format_call`), and 1 was gained (`reindl`, found in 9 steps instead of 20).
3. The effect of an output format is model-dependent: the same change moves the two models in opposite directions. Intervals are wide (n = 21), so treat the size of each change as indicative only; see the paired tests in the ablation checkpoint.

**Files:** `<model>.json`, `metrics.json`, `config.json`, `figures/summary.*`, `figures/outcomes.*`. Trajectories: `C:\gemma_work\runs\2026-10-25_05_struct-dev\`.
**Regenerate:** `python scripts/rescore.py && python analysis/plot_experiment.py --exp struct-dev`.
