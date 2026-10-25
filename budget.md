# GraphAgent: Offline Distillation Plan (31B teacher → 4 GB student)

**Competition:** Google × Kaggle Gemma 4 Developer Agent Paper Track
**Deadline:** November 12, 2026, 11:59 PM UTC (we submit Nov 11)
**Plan revised:** October 21, 2026 (22 days left). Replaces the A5000-based plan and the API-only draft.

> Throughput figures are estimates from one smoke-test call. Replace them with numbers from `logs/gemma_usage.jsonl` after the first full episode.

---

## 1. Goal

An agent that **runs fully offline on a 4 GB laptop GPU** and localizes (and where possible fixes) real bugs, built by distilling **Gemma 4 31B** into **Gemma 4 E2B**.

Headline claim we test: *a 2B Gemma agent on a 4 GB GPU retains X% of a 31B teacher's bug-localization accuracy, using graph tools and an Evidence Buffer.*

The API is used only for development: the 31B teacher generates training data and is the upper-bound comparison. The submitted agent makes no API calls.

---

## 2. Resources

| Resource | Details | Role |
|---|---|---|
| Laptop | RTX 2050 **4 GB** (Ampere, cc 8.6), 14 GB RAM, Ryzen 5 5500H, **33.6 GB free on C:** | Offline student inference + evaluation; graph building |
| Gemini API free tier | Gemma 4 31B and 26B-A4B; 30 RPM · 16K TPM · 14.4K RPD per model; free; no tuning | Teacher trajectories, 31B baseline |
| Kaggle free notebooks | T4 16 GB, weekly GPU quota; datasets for storage | QLoRA training of the student; public notebook; trajectory dataset |
| Nebius Token Factory | Paid | Overflow teacher inference only if behind schedule |
| Friend's Gemini Pro | Gemini app / CLI | Annotation help, LLM-as-judge. **Not** a source of training data |
| `sb-cli` | Free SWE-bench cloud evaluation | Resolve-rate check on a subset |

Measured API behaviour (2026-10-21): 31B latency ~22–32 s/call, frequent 500s (client retries), always "thinks" (~4× the answer length), thinking cannot be disabled.

---

## 3. Models

| Role | Model | Where | Size |
|---|---|---|---|
| Teacher | Gemma 4 31B | Gemini API | — |
| Dev / fast teacher | Gemma 4 26B-A4B | Gemini API | — |
| **Student (main)** | **Gemma 4 E2B**, 4-bit GGUF | Laptop, llama.cpp, fully on GPU | ~2.9 GB weights, ~1 GB for KV cache → ~8K context |
| Student (stretch) | Gemma 4 E4B, 4-bit GGUF | Laptop, partial CPU offload | ~4.5 GB, slower |

---

## 4. Method

1. **Agent harness** (shared by teacher and student): graph tools (Tree-sitter + call graph + embeddings: `get_code_neighbors`, `search_similar_code`, `get_code_subgraph`), file tools, **Evidence Buffer**, **Budget Management Policy**.
2. **Small context by design:** prompt ≤ 8K tokens per step. This satisfies both the 16K TPM API limit and the ~8K context that fits in 4 GB VRAM.
3. **Teacher data:** ~300 teacher episodes on SWE-bench-style tasks (SWE-Gym / SWE-bench train-split repos, **disjoint from the eval set**).
4. **Filtering (RLVR-lite):** keep episodes that reached the gold location (and passed tests where checked); prefer fewer tool calls.
5. **Step-level distillation:** each example = issue + Evidence Buffer + last k tool calls → teacher's next action (with a short rationale; teacher thoughts included if the API exposes them). Expected ~4–5K examples.
6. **Student training:** QLoRA SFT of E2B on Kaggle T4 (Unsloth), seq len ≤ 8K, ~6–10 GPU hours. DR-LoRA vs. uniform LoRA as an ablation if quota allows.
7. **Export:** merge LoRA → GGUF Q4 → llama.cpp on the RTX 2050.

---

## 5. Evaluation

**Primary metric:** localization accuracy (file / function hit vs. gold patch) and tool calls used. **Secondary:** resolve rate via `sb-cli` on a subset. **Efficiency:** tokens/episode, latency, VRAM on the RTX 2050.

| System | Where it runs |
|---|---|
| 31B teacher (upper bound) | API |
| E2B base, no graph tools | Laptop |
| E2B base + graph tools + Evidence Buffer | Laptop |
| **E2B distilled + graph tools + Evidence Buffer** | Laptop |
| Ablations: distilled without graph tools; without Evidence Buffer; without budget policy | Laptop |
| Stretch: E4B distilled | Laptop (offload) |

Eval set: 50 SWE-bench Verified tasks (30 if behind schedule).

---

## 6. Storage (target ≤ 20 GB on the laptop)

**Bulk data lives outside OneDrive** (`C:\gemma_work\`), otherwise OneDrive will try to sync gigabytes. The OneDrive project folder holds only code, `budget.md`, and the paper.

| Item | Size | Notes |
|---|---|---|
| Python env, llama.cpp | ~5 GB | |
| Student GGUFs (E2B base + distilled; E4B optional) | ~6 GB (+5) | Delete E4B if space is tight |
| Repo clones for eval tasks | ~3 GB | `git clone --filter=blob:none`, check out per task |
| Graphs + embedding indexes | ~3–5 GB | Build per task, fp16 embeddings, delete after the run; keep only eval-set indexes |
| Trajectories, logs, GraphNav-Bench | ~1–2 GB | Mirrored to a private Kaggle dataset (needed for training anyway) |
| **Total** | **~18–21 GB** | Leaves ~12 GB free on C: |

Teacher-side repos for training data are cloned and deleted in batches. Training data and LoRA checkpoints stay on Kaggle. No paid storage needed; free up a few GB on C: as extra margin.

---

## 7. API Budget

| Job | Episodes | Free-tier days |
|---|---|---|
| Teacher episodes (also the 31B baseline on its eval slice) | 300 | ~5–6 |
| 31B on the 50-task eval set | 50 | ~1 |
| **Total** | **350** | **~6–7 days**, overnight; 26B-A4B for development |

Student evaluation is local and costs no API quota. RTX 2050 estimate for E2B: ~20–40 tokens/s → a 40-step episode in ~10–20 min → 50 tasks per configuration ≈ 1 night. **7 configurations ≈ 5–7 nights** (start with the 4 core rows).

---

## 8. Timeline

| Phase | Dates | Output |
|---|---|---|
| **Build** | Oct 21–27 | Agent loop, Evidence Buffer, budget policy, graph tools, localization scorer; llama.cpp + E2B running on the RTX 2050; 1-task end-to-end on API and locally |
| **Teacher data** | Oct 27–Nov 2 | ~300 teacher episodes (overnight); E2B base evals run locally in parallel |
| **Distill** | Nov 1–4 | Filter → step-level dataset → QLoRA on Kaggle → GGUF |
| **Evaluate** | Nov 4–8 | Distilled student + ablations on the laptop; `sb-cli` subset |
| **GraphNav-Bench** | Oct 27–Nov 8 (daytime) | Auto paths from gold patch + call graph, Gemini-assisted check, hand-verify ~150 |
| **Paper** | Nov 3–10 | Intro / method from Nov 3; results and figures last; ≤ 3,000 words |
| **Submit** | Nov 11 | Public Kaggle notebook, dataset (CC-BY 4.0), LoRA + GGUF |

**Build must finish by Oct 27.** Hard decision dates: **Oct 30** (teacher data on track? else Nebius overflow or smaller set); **Nov 4** (student trained? else report base E2B + tools as the offline result).

---

## 9. Risks & Fallbacks

| Risk | Fallback |
|---|---|
| E2B too weak to follow the tool protocol | Constrained output (llama.cpp grammar / JSON schema); more step-level examples; E4B with offload |
| 8K context too small on 4 GB | Q8/Q4 KV cache; tighter Evidence Buffer; cap tool output at ~1K tokens |
| Teacher data too slow (API 500s / limits) | Use 26B-A4B as a second teacher with its own quota; Nebius overflow; fewer episodes (200) |
| Kaggle GPU quota runs out | Train E2B only, 1 epoch, shorter sequences |
| Disk full | Delete per-task indexes after runs; drop E4B; move logs to Kaggle dataset |
| Student shows no gain over base | Report graph tools + Evidence Buffer as the offline result; distillation as a negative finding |
| Resolve rate near zero | Expected for 2B; lead with localization, report resolve rate honestly |
