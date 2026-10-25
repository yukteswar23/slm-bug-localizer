# 03 · C1 baseline on the dev split (2026-10-25)

**Question:** how well do the two small models localize bugs with only file tools (configuration C1: `list_dir`, `search_code`, `read_file`, `submit`)?

**Setup:** SWE-bench Lite dev split (23 tasks, 6 repos: sqlfluff, marshmallow, pvlib, astroid, pyvista, pydicom), 20-step limit, grammar-constrained JSON steps, temperature 0, no thinking. RTX 2050, about 20 min per model.

| Metric (95% Wilson CI) | Gemma 4 E2B | Qwen3.5-4B |
|---|---|---|
| File@1 | 83% [63, 93] | 83% [63, 93] |
| Function@5 (n = 21) | 43% [24, 63] | 62% [41, 79] |
| Median steps / time per task | 20 / 55 s | 11 / 36 s |
| Forced submit (hit the step limit) | 70% | 43% |
| Repeated identical calls (share of steps) | 39% | 18% |
| Malformed function names (tasks) | 13% | 43% |
| Invalid replies | 0% | 0% |

**Findings:**
1. **Finding the file is not the bottleneck; naming the function is.** Both models reach 83% File@1 with file tools alone. Function accuracy is 20–40 points lower.
2. **Qwen often finds the right function but writes it in the wrong format** (`pydicom.dataset.Dataset.to_json_dict` instead of `pydicom/dataset.py::Dataset.to_json_dict`): 43% of tasks. Strict scoring gave Qwen 33% Function@5; resolving names against its submitted files gives 62%. Both numbers are kept (see `agent/score.py`); the format failure is reported separately.
3. **Gemma does not know when to stop.** 70% of episodes ran into the step limit, and 39% of its steps repeated an earlier call. Qwen answers earlier (median 11 steps), but its runs are more variable (time spread 11–137 s).
4. Grammar-constrained decoding holds: 0 invalid replies on both models.
5. Both models submit only ~1 file and ~1 function, so @5 and @10 add nothing over @1. The prompt asks for up to 5, ranked; small models do not use the extra slots.

**Caveats:** n = 23 and 6 repos, so the confidence intervals are wide (the two models' Function@5 intervals overlap). The dev split is for tuning, not for final claims.

**Next steps suggested by this run:** (a) constrain function names to `path.py::Name` in the grammar (schema `pattern`); (b) a repeated-call guard or budget policy for Gemma; (c) ask for and constrain ranked lists of 3–5 candidates.

**Files:** `<model>.json` (per-task metrics + step stats), `metrics.json`, `config.json`, `figures/summary.{pdf,png}`, `figures/outcomes.{pdf,png}`. Trajectories: `C:\gemma_work\runs\2026-10-25_03_c1-dev\`.
**Regenerate:** `python scripts/rescore.py && python analysis/plot_experiment.py --exp c1-dev`.
