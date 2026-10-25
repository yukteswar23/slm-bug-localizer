# Plan: Scaffolding vs. Scale for Small Code Agents

**Plan date:** October 25, 2026. Replaces the distillation plan (archived in commit `5806683`).
**Target:** research paper (arXiv); optional Kaggle Gemma 4 Paper Track entry (deadline **Nov 12, 2026, 11:59 PM UTC**, we submit Nov 11).

---

## 1. Research question

> **How far can scaffolding alone push off-the-shelf language models of ≤ 4B parameters at bug localization, running fully offline on a 4 GB consumer GPU, with no training?**

Scaffolding means the agent harness around the model: code-graph tools, an Evidence Buffer, a tool-budget policy, and grammar-constrained tool calls. The models are used as released. No fine-tuning.

Hypotheses:

- **H1.** Each scaffolding component improves localization accuracy over a plain ReAct agent with file tools.
- **H2.** Grammar-constrained tool calls matter more for small models than for large ones: invalid tool calls are a major failure mode below 4B.
- **H3.** The gains hold across model families (Gemma, Qwen), so the result is about small models in general, not one model.
- **H4.** Fixing (writing a passing patch) remains far harder than localization at this scale.

A negative result on any hypothesis is still reported.

---

## 2. Scope

| | Role |
|---|---|
| **Bug localization** | **Main result.** Given a GitHub issue and the repo, name the files and functions that need to change. |
| Bug fixing | Small secondary experiment: run the best configuration end to end and report how far it gets. |

---

## 3. Hardware and measured speed

Laptop: RTX 2050 **4 GB**, 14 GB RAM, Ryzen 5 5500H. No GPU server.
Runtime: llama.cpp b11873 (CUDA 13.4), all layers on GPU, flash attention.

Measured 2026-10-25 with `llama-bench` (2 runs each, raw output in `C:\gemma_work\bench\`):

| Model | File | Prefill @ 6K ctx | Generation @ 6K ctx | Peak VRAM (≤ 14K ctx) |
|---|---|---|---|---|
| Gemma 4 E2B (Google QAT Q4_0) | 3.1 GB | ~2,200 tok/s | ~58 tok/s | 2.1 GB |
| Qwen3.5-4B (Q4_K_M) | 2.6 GB | ~1,430 tok/s | ~31 tok/s | 3.7 GB |

Estimated cost of a 25-step episode (~6K context, ~150 output tokens per step): **~2 min (Gemma E2B), ~4 min (Qwen3.5-4B)**. Models run in **non-thinking mode**, because thinking tokens at 30–60 tok/s would multiply episode time.

Benchmark numbers are a ceiling; replace them with timings from real episodes once the harness runs.

---

## 4. Models

| Model | Params | Why |
|---|---|---|
| **Gemma 4 E2B** | ~2B effective | Smallest, fastest; keeps the Kaggle entry possible |
| **Qwen3.5-4B** | 4B | Different family, upper end of the size range |
| Third model (stretch) | ≤ 4B | Only if time allows (e.g. Phi-4-mini or Llama 3.2 3B) |
| Gemma 4 31B (reference) | 31B | Free Gemini API (`agent/gemma_client.py`). Shows the gap to a large model. Not offline |

---

## 5. Agent harness

One harness for all models, served by `llama-server` (OpenAI-compatible API).

| Component | What it does |
|---|---|
| File tools | `list_dir`, `read_file` (line ranges), `search_code` (ripgrep) |
| **Graph tools** | Tree-sitter parse + call graph + embeddings: `get_code_neighbors`, `get_code_subgraph`, `search_similar_code` |
| **Evidence Buffer** | Short persistent notes of findings; replaces raw tool history so the prompt stays ≤ 8K tokens |
| **Budget policy** | Tool-call budget per episode; switches to "commit to an answer" mode near the limit |
| **Constrained tool calls** | llama.cpp grammar (GBNF / JSON schema) forces every reply to be a valid tool call |
| Final answer | Ranked list of files and functions |

Tool output is capped (~1K tokens per call) so each step fits the 8K context budget.

Implementation notes (from the first end-to-end runs, 2026-10-25):
- Requests go through llama-server's `/apply-template` + `/completion`, not the chat endpoint. The chat endpoint applies the grammar lazily after a thinking block, and Gemma 4 opened its thought channel by itself, which produced invalid replies. With `/completion` the grammar holds from the first token (0 invalid replies).
- Compressing old tool outputs changes the prompt prefix, so the KV cache is lost and every later step reprocesses ~6K tokens (visible in `figures/episode_e2e_smoke_*`). The Evidence Buffer should be append-only to keep the cache.
- Small models repeat the same tool call (Gemma read the same lines up to 4 times). Count repeated calls as a failure-mode metric.
- The scorer normalizes module-style paths (`pkg.mod.py::f` → `pkg/mod.py::f`); `scripts/rescore.py` recomputes metrics without rerunning models.

---

## 6. Evaluation

**Data:** SWE-bench Lite (300 Python tasks; localization results from Agentless and LocAgent use it, so our numbers can be compared).

- Dev set: the official SWE-bench Lite **dev split** (23 tasks, from repos that do not appear in the test split), used for prompt and harness tuning.
- Test set: 100 tasks from the test split, fixed before any test run. Scale up to the full 300 if time allows.

**Metrics:**

| Metric | Description |
|---|---|
| File Acc@1 / @5 | A gold-patch file is in the top-1 / top-5 predicted files |
| Function Acc@5 / @10 | A gold-patch function is in the top-5 / top-10 predicted functions |
| Tool-call validity | Share of model replies that parse as a valid tool call |
| Cost | Steps, tokens, wall time per task, peak VRAM |
| Fix rate (secondary) | Resolved tasks on a small subset, via `sb-cli` |

**Configurations** (each model):

| ID | Configuration |
|---|---|
| C0 | Retrieval only (BM25 over files, no agent) |
| C1 | ReAct + file tools |
| C2 | C1 + graph tools |
| C3 | C2 + Evidence Buffer |
| C4 | C3 + budget policy (**full system**) |
| C4−G | C4 without constrained tool calls |
| Ref | C4 with Gemma 4 31B via API |

2 models × 6 local configurations × 100 tasks ≈ 30–40 laptop hours, i.e. about 4–5 overnight runs.

---

## 7. Related work to position against

- **Agentless:** localization and repair without an agent loop, using large models.
- **LocAgent:** graph-guided code localization with LLM agents.

Our difference: models ≤ 4B, a 4 GB consumer GPU, fully offline, no training, and an ablation of which scaffolding parts small models actually need.

---

## 8. Timeline

| Phase | Dates | Output |
|---|---|---|
| **Build** | Oct 25–28 (one task end to end on both models: done Oct 25) | `llama-server` wrapper, file tools, task loader, localization scorer, constrained tool calls; one task end to end on both models |
| **Scaffolding** | Oct 29–31 | Graph tools, Evidence Buffer, budget policy; tune on dev set |
| **Main runs** | Nov 1–5 | C0–C4 and C4−G on the test set, overnight |
| **Secondary** | Nov 4–6 | 31B reference; fix experiment on a subset |
| **Paper** | Nov 3–10 | Method from Nov 3; results and figures last |
| **Submit** | Nov 11 | Paper + public code; Kaggle entry with the Gemma results (optional) |

Decision dates: **Oct 28** (both models complete one task end to end? else cut to Gemma E2B only); **Nov 3** (test runs on track? else shrink the test set to 50).

---

## 9. Storage

Bulk data lives in `C:\gemma_work\`, outside OneDrive. This repo holds only code, plans, and the paper.

| Path | Contents |
|---|---|
| `C:\gemma_work\llama.cpp\` | llama.cpp binaries |
| `C:\gemma_work\models\` | GGUF model files |
| `C:\gemma_work\repos\` | Task repositories (`git clone --filter=blob:none`) |
| `C:\gemma_work\indexes\` | Per-task graphs and embeddings |
| `C:\gemma_work\runs\` | Episode logs and results |

---

## 10. Risks

| Risk | Fallback |
|---|---|
| Small models can't follow the tool protocol | Constrained decoding; simpler tool set; fewer tools per step |
| 8K context too small | Tighter Evidence Buffer; smaller tool-output caps; quantized KV cache |
| Qwen3.5-4B runs out of VRAM | Q8 KV cache; shorter context; smaller quant |
| Runs slower than estimated | Shrink test set to 50; drop C4−G or the third model |
| Scaffolding shows no gain | Report it as a negative result, with the failure analysis |
| Fix rate near zero | Expected; report honestly as the secondary result |
