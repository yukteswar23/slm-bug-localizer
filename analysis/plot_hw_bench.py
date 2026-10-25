"""Figure: inference speed and VRAM of the candidate models on the RTX 2050.

Input:  llama-bench markdown + 1 Hz nvidia-smi VRAM logs in C:/gemma_work/bench/
Output: results/hw_bench.json, figures/hw_bench.{pdf,png}

    python analysis/plot_hw_bench.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (MODEL_COLORS, MODEL_LABELS, PALETTE, RESULTS_DIR,  # noqa: E402
                      apply_publication_style, finalize_figure, panel_label, plt)

BENCH_DIR = Path("C:/gemma_work/bench")
RUNS = {  # model key -> file stem used by the benchmark run
    "gemma-4-e2b": "gemma-4-E2B_q4_0-it",
    "qwen3.5-4b": "Qwen3.5-4B-Q4_K_M",
}
VRAM_LIMIT_MIB = 4096
ROW = re.compile(r"\|\s*(pp|tg)(\d+)(?: @ d(\d+))?\s*\|\s*([\d.]+) ± ([\d.]+)\s*\|")


def ingest():
    results = {}
    for key, stem in RUNS.items():
        tests = []
        for m in ROW.finditer((BENCH_DIR / f"bench_{stem}.md").read_text(encoding="utf-8")):
            kind, n, depth, mean, std = m.groups()
            tests.append({"kind": kind, "n_tokens": int(n), "depth": int(depth or 0),
                          "tok_per_s": float(mean), "std": float(std)})
        vram = [int(v) for v in (BENCH_DIR / f"vram_{stem}.txt").read_text().split()]
        results[key] = {"file": f"{stem}.gguf", "tests": tests,
                        "vram_mib_1hz": vram, "peak_vram_mib": max(vram)}
    out = {"date": "2026-10-25", "gpu": "RTX 2050 4 GB", "runtime": "llama.cpp b11873 CUDA 13.4",
           "settings": "ngl=99, flash-attn on, 2 repetitions", "models": results}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "hw_bench.json").write_text(json.dumps(out, indent=1))
    return out


def get(tests, kind, n, depth):
    t = next(t for t in tests if (t["kind"], t["n_tokens"], t["depth"]) == (kind, n, depth))
    return t["tok_per_s"], t["std"]


def bars(ax, groups, getter, ylabel):
    keys = list(RUNS)
    width = 0.8 / len(keys)
    x = np.arange(len(groups))
    for i, key in enumerate(keys):
        vals = [getter(key, g) for g in groups]
        ax.bar(x + (i - (len(keys) - 1) / 2) * width, [v[0] for v in vals], width,
               yerr=[v[1] for v in vals], capsize=3, color=MODEL_COLORS[key],
               edgecolor="black", linewidth=0.8, label=MODEL_LABELS[key])
    ax.set_xticks(x, [g[0] for g in groups])
    ax.set_ylabel(ylabel)


def main():
    data = ingest()["models"]
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.3, 1, 1.4]})

    pp_groups = [("4K\nempty", 4096, 0), ("8K\nempty", 8192, 0),
                 ("4K\n+6K ctx", 4096, 6000), ("8K\n+6K ctx", 8192, 6000)]
    bars(axes[0], pp_groups, lambda k, g: get(data[k]["tests"], "pp", g[1], g[2]),
         "Prompt processing (tokens/s)")
    axes[0].set_xlabel("Prompt length and existing context")

    tg_groups = [("Empty", 128, 0), ("+6K ctx", 128, 6000)]
    bars(axes[1], tg_groups, lambda k, g: get(data[k]["tests"], "tg", g[1], g[2]),
         "Generation (tokens/s)")
    axes[1].set_xlabel("Existing context")

    ax = axes[2]
    for key in RUNS:
        v = np.array(data[key]["vram_mib_1hz"]) / 1024
        ax.plot(np.arange(len(v)), v, color=MODEL_COLORS[key], lw=1.6, label=MODEL_LABELS[key])
    ax.axhline(VRAM_LIMIT_MIB / 1024, color=PALETTE["red_strong"], ls="--", lw=1.2)
    ax.text(0.99, VRAM_LIMIT_MIB / 1024, "4 GB device limit", color=PALETTE["red_strong"],
            transform=ax.get_yaxis_transform(), ha="right", va="bottom")
    ax.set_ylim(0, 4.6)
    ax.set_xlabel("Time during benchmark (s)")
    ax.set_ylabel("GPU memory used (GB)")
    ax.grid(axis="x", visible=False)

    for a, letter in zip(axes, "abc"):
        panel_label(a, letter)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    for p in finalize_figure(fig, "hw_bench"):
        print(p)


if __name__ == "__main__":
    main()
