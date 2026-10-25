# 02 · End-to-end smoke test (2026-10-25)

**Question:** does the full pipeline work (task → agent → answer → score → figures) on both models?

**Setup:** 1 task (`astropy__astropy-12907`), C1 configuration (file tools only), 20-step limit, grammar-constrained JSON steps, temperature 0.

| Model | File@1 | Function@5 | Steps | Time |
|---|---|---|---|---|
| Gemma 4 E2B | correct | wrong (`_separable`, gold `_cstack`) | 20 | 70 s |
| Qwen3.5-4B | correct | correct | 10 | 35 s |

n = 1: this checks the pipeline, it says nothing about model quality.

**Found and fixed:**
- llama-server's chat endpoint applies the grammar only after a thinking block, and Gemma 4 opened its thought channel by itself → invalid replies. Switched to `/apply-template` + `/completion`: 0 invalid replies.
- Qwen wrote module-style paths (`astropy.modeling.separable.py::_cstack`); the scorer now normalizes them.

**Observed (to measure at scale):** repeated identical tool calls (Gemma read the same 4 lines 4 times); KV cache lost once old outputs are compressed, so later steps reprocess ~6K tokens.

**Files:** `<model>.json` (metrics + per-step stats), `figures/episode_astropy__astropy-12907.{pdf,png}`. Trajectories: `C:\gemma_work\runs\2026-10-25_02_e2e-smoke\`.
**Regenerate:** `python analysis/plot_episode.py --exp e2e-smoke --instance astropy__astropy-12907`.
