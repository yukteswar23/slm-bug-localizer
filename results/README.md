# Experiments

One folder per experiment: `<date>_<NN>_<name>/` with `NOTES.md` (question, setup, results, findings), `config.json`, per-model results, and `figures/` (PDF for the paper, PNG for viewing). Full agent trajectories live outside the repo in `C:\gemma_work\runs\<same folder name>\`.

| # | Experiment | Question | Headline |
|---|---|---|---|
| 01 | [hw-bench](2026-10-25_01_hw-bench/NOTES.md) | Do ≤4B models run fast enough on a 4 GB GPU? | Yes: Gemma E2B 58 tok/s in 2.1 GB, Qwen3.5-4B 31 tok/s in 3.7 GB |
| 02 | [e2e-smoke](2026-10-25_02_e2e-smoke/NOTES.md) | Does the pipeline work end to end? | Yes, after fixing grammar enforcement for Gemma 4 |
| 03 | [c1-dev](2026-10-25_03_c1-dev/NOTES.md) | File tools only, dev split (23 tasks) | File@1 83% for both; Function@5 Gemma 43%, Qwen 62% |

Regenerate everything: `python scripts/rescore.py`, then the `analysis/plot_*.py` command listed in each NOTES.md.
