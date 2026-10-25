# slm-bug-localizer

Bug localization with small language models (≤ 4B parameters) running fully offline on a 4 GB consumer GPU, with no training.

**Research question:** how far can scaffolding alone (code-graph tools, an Evidence Buffer, a tool-budget policy, and grammar-constrained tool calls) push off-the-shelf small models at finding where a bug lives in a real codebase?

- Models: Gemma 4 E2B and Qwen3.5-4B (4-bit GGUF, llama.cpp), with Gemma 4 31B as a large-model reference
- Benchmark: SWE-bench Lite (localization as the main result, fixing as a secondary experiment)
- Hardware: NVIDIA RTX 2050 laptop GPU, 4 GB VRAM

See [PLAN.md](PLAN.md) for the full plan, experiments, and timeline.

## Layout

| Path | Contents |
|---|---|
| `agent/` | Agent harness and model clients |
| `docs/competition/` | Kaggle rules and overview (local only, gitignored) |
| `PLAN.md` | Research plan |

Status: early development.
