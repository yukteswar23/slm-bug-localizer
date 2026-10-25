# 01 · Hardware benchmark (2026-10-25)

**Question:** which ≤4B models fit and run fast enough on the RTX 2050 (4 GB) for agent episodes?

**Setup:** llama.cpp b11873 (CUDA 13.4), `llama-bench`, all layers on GPU, flash attention, 2 repetitions. Prompt lengths 4K/8K, generation of 128 tokens, each on an empty context and after 6K tokens of existing context. VRAM sampled at 1 Hz with `nvidia-smi`.

| Model | Prefill @ 6K ctx | Generation @ 6K ctx | Peak VRAM |
|---|---|---|---|
| Gemma 4 E2B (QAT Q4_0) | ~2,200 tok/s | ~58 tok/s | 2.1 GB |
| Qwen3.5-4B (Q4_K_M) | ~1,430 tok/s | ~31 tok/s | 3.7 GB |

**Takeaways:** both fit fully on the GPU; Gemma uses about half the VRAM of its 3.1 GB file. Qwen at 14K context is close to the 4 GB limit. The laptop can run all experiments itself.

**Files:** `results.json` (all numbers + raw VRAM trace), `figures/hw_bench.{pdf,png}`.
**Regenerate:** `python analysis/plot_hw_bench.py` (raw benchmark output in `C:\gemma_work\bench\`).
