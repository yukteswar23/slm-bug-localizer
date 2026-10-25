"""Figure: step-level diagnostics of agent episodes, one row per model.

(a) prompt tokens per step, split into reused KV cache and newly processed
(b) wall time per step, split into prompt processing and generation
(c) tool calls per episode by tool, with invalid replies and tool errors

Input:  results/<checkpoint>/<model>.json
Output: results/<checkpoint>/figures/episode_<instance>.{pdf,png}

    python analysis/plot_episode.py --exp e2e-smoke --instance astropy__astropy-12907
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (MODEL_COLORS, MODEL_LABELS, PALETTE,  # noqa: E402
                      apply_publication_style, finalize_figure, panel_label, plt)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.checkpoint import checkpoint_dir  # noqa: E402

TOOL_ORDER = ["list_dir", "search_code", "read_file", "submit", "invalid", "tool error"]


def load(ckpt, instance):
    out = {}
    for path in sorted(ckpt.glob("*.json")):
        s = json.loads(path.read_text())
        if "tasks" not in s:
            continue
        task = next(t for t in s["tasks"] if instance in (None, t["instance_id"]))
        out[s["model"]] = task
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="checkpoint name, e.g. e2e-smoke")
    ap.add_argument("--instance", required=True)
    args = ap.parse_args()
    ckpt = checkpoint_dir(args.exp, create=False)
    data = load(ckpt, args.instance)

    apply_publication_style()
    max_steps = max(len(t["steps"]) for t in data.values())
    fig, axes = plt.subplots(len(data), 3, figsize=(13, 3.0 * len(data)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.4, 1.4, 1]})
    for row, (model, task) in enumerate(data.items()):
        steps = task["steps"]
        x = np.array([s["step"] + 1 for s in steps])
        color = MODEL_COLORS[model]
        cached = np.array([s["cached_tokens"] for s in steps])
        total = np.array([s["prompt_tokens"] for s in steps])
        # /completion reports tokens_evaluated as the full prompt length; cache_n of it was reused
        new = np.clip(total - cached, 0, None)

        ax = axes[row, 0]
        ax.bar(x, cached, color=PALETTE["neutral"], edgecolor="black", lw=0.6, label="Reused (KV cache)")
        ax.bar(x, new, bottom=cached, color=color, edgecolor="black", lw=0.6, label="Newly processed")
        ax.set_ylabel("Prompt tokens")
        ax.set_ylim(0, total.max() * 1.4)
        ax.legend(loc="upper left", ncol=2)

        ax = axes[row, 1]
        pre = np.array([s["prompt_ms"] for s in steps]) / 1000
        gen = np.array([s["gen_ms"] for s in steps]) / 1000
        ax.bar(x, pre, color=PALETTE["neutral"], edgecolor="black", lw=0.6, label="Prompt processing")
        ax.bar(x, gen, bottom=pre, color=color, edgecolor="black", lw=0.6, label="Generation")
        ax.set_ylabel("Time per step (s)")
        ax.set_ylim(0, (pre + gen).max() * 1.4)
        ax.legend(loc="upper left", ncol=2)
        for a in axes[row, :2]:
            a.set_xlim(0.3, max_steps + 0.7)
            a.set_xticks(range(1, max_steps + 1, 2 if max_steps > 12 else 1))

        ax = axes[row, 2]
        counts = Counter(s["tool"] if s["valid"] else "invalid" for s in steps)
        counts["tool error"] = sum(1 for s in steps if s["error"] and s["valid"])
        vals = [counts.get(t, 0) for t in TOOL_ORDER]
        colors = [color] * 4 + [PALETTE["red_strong"], PALETTE["red_2"]]
        ax.barh(TOOL_ORDER[::-1], vals[::-1], color=colors[::-1], edgecolor="black", lw=0.6)
        ax.set_xlabel("Calls")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        m = task["metrics"]
        ok = lambda v: "n/a" if v is None else ("correct" if v else "wrong")  # noqa: E731
        ax.set_title(f"file@1: {ok(m['file_acc_all@1'])}    function@5: {ok(m['func_acc_all@5'])}",
                     fontsize=10, loc="right", pad=14)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axes[row, 0].set_title(MODEL_LABELS[model], loc="left", fontsize=11, fontweight="bold",
                               pad=14)

    for a in axes[-1, :2]:
        a.set_xlabel("Agent step")
    for a, letter in zip(axes[0], "abc"):
        panel_label(a, letter)
    fig.tight_layout()
    for p in finalize_figure(fig, f"episode_{args.instance}", ckpt / "figures"):
        print(p)


if __name__ == "__main__":
    main()
