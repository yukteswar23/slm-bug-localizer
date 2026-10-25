# Experiments

One folder per experiment: `<date>_<NN>_<name>/` with `NOTES.md` (question, setup, results, findings), `config.json`, per-model results, and `figures/` (PDF for the paper, PNG for viewing). Full agent trajectories live outside the repo in `C:\gemma_work\runs\<same folder name>\`.

| # | Experiment | Question | Headline |
|---|---|---|---|
| 01 | [hw-bench](2026-10-25_01_hw-bench/NOTES.md) | Do ≤4B models run fast enough on a 4 GB GPU? | Yes: Gemma E2B 58 tok/s in 2.1 GB, Qwen3.5-4B 31 tok/s in 3.7 GB |
| 02 | [e2e-smoke](2026-10-25_02_e2e-smoke/NOTES.md) | Does the pipeline work end to end? | Yes, after fixing grammar enforcement for Gemma 4 |
| 03 | [c1-dev](2026-10-25_03_c1-dev/NOTES.md) | File tools only, dev split (23 tasks) | File@1 83% for both; Function@5 Gemma 43%, Qwen 62% |
| 04 | [fmt-dev](2026-10-25_04_fmt-dev/NOTES.md) | Regex-forced function names | Negative: Qwen Function@5 62% → 33% (p = 0.031); grammar corrupts answers |
| 05 | [struct-dev](2026-10-25_05_struct-dev/NOTES.md) | Structured `{file, name}` answers | Qwen 62% → 76%, Gemma 43% → 29% |
| 06 | [guard-dev](2026-10-25_06_guard-dev/NOTES.md) | + repeated-call guard + budget hints | Gemma forced submits 57% → 9%; accuracy unchanged |
| 07 | [ranked-dev](2026-10-25_07_ranked-dev/NOTES.md) | + ranked candidate lists | Gemma Function@5 33% → 43%; Qwen unchanged |
| 08 | [dev-ablation](2026-10-25_08_dev-ablation/NOTES.md) | Ladder comparison of 03–07 | Format matters most; no ladder step significant at n = 23 |
| 09 | [c0-dev](2026-10-25_09_c0-dev/NOTES.md) | C0: BM25 retrieval, no model | File@1 35%, Function@5 43% (= Gemma's C1 agent) |
| 10 | [c2-dev](2026-10-25_10_c2-dev/NOTES.md) | C2: + graph tools | Gemma Function@5 43% → 52%; models use graph tools for only 5–8% of calls |

Regenerate everything: `python scripts/rescore.py`, then the `analysis/plot_*.py` command listed in each NOTES.md.
