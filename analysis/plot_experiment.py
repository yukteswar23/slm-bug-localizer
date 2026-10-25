"""Aggregate figures for one localization experiment (one checkpoint folder).

summary.{pdf,png}
  (a) localization accuracy per model, 95% Wilson intervals
  (b) agent steps per task      (c) wall time per task
  (d) failure modes: share of tasks / steps affected
outcomes.{pdf,png}
  per-task outcome for each model (miss / file only / file + function)
Also writes metrics.json with the numbers behind the figures.

    python analysis/plot_experiment.py --exp c1-dev
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (MODEL_COLORS, MODEL_LABELS, PALETTE,  # noqa: E402
                      apply_publication_style, finalize_figure, panel_label, plt)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.checkpoint import checkpoint_dir  # noqa: E402

METRICS = [("file_acc_all@1", "File\n@1"), ("file_acc_all@5", "File\n@5"),
           ("func_acc_all@5", "Function\n@5"), ("func_acc_all@10", "Function\n@10")]
MODEL_ORDER = ["bm25", "gemma-4-e2b", "qwen3.5-4b", "gemma-4-31b"]
TOOL_ORDER = ["list_dir", "search_code", "read_file", "find_definition", "get_code_neighbors",
              "search_similar_code", "get_file_outline", "submit"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def failure_modes(task, max_steps):
    steps = task["steps"]
    actions = [json.dumps(s.get("action"), sort_keys=True) for s in steps if s.get("action")]
    repeats = sum(1 for i, a in enumerate(actions) if a in actions[:i])
    return {
        "invalid_reply": any(not s["valid"] for s in steps),
        "tool_error": any(s["error"] and s["valid"] for s in steps),
        "forced_submit": max_steps > 0 and task["submitted"] and task["n_steps"] == max_steps,
        "no_answer": not task["answer"]["files"],
        "malformed": bool(task["metrics"].get("func_malformed")),
        "repeat_steps": repeats, "n_steps": len(steps),
    }


def load(ckpt):
    runs = {}
    for path in ckpt.glob("*.json"):
        s = json.loads(path.read_text())
        if "tasks" in s and s["tasks"]:
            runs[s["model"]] = s
    return {m: runs[m] for m in MODEL_ORDER if m in runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()
    ckpt = checkpoint_dir(args.exp, create=False)
    runs = load(ckpt)
    models = list(runs)

    stats = {}
    for m, run in runs.items():
        tasks = run["tasks"]
        st = {"n_tasks": len(tasks)}
        for key, _ in METRICS:
            vals = [t["metrics"][key] for t in tasks if t["metrics"][key] is not None]
            p, lo, hi = wilson(sum(vals), len(vals))
            st[key] = {"mean": p, "ci95": [lo, hi], "n": len(vals)}
        fm = [failure_modes(t, run["max_steps"]) for t in tasks]
        st["failure"] = {
            "Invalid reply (tasks)": np.mean([f["invalid_reply"] for f in fm]),
            "Tool error (tasks)": np.mean([f["tool_error"] for f in fm]),
            "Forced submit (tasks)": np.mean([f["forced_submit"] for f in fm]),
            "Empty answer (tasks)": np.mean([f["no_answer"] for f in fm]),
            "Malformed function name (tasks)": np.mean([f["malformed"] for f in fm]),
            "Repeated call (steps)": sum(f["repeat_steps"] for f in fm) / max(1, sum(f["n_steps"] for f in fm)),
        }
        st["steps"] = [t["n_steps"] for t in tasks]
        st["wall_s"] = [t["wall_s"] for t in tasks]
        stats[m] = st
    (ckpt / "metrics.json").write_text(json.dumps(stats, indent=1, default=float))

    apply_publication_style()
    fig = plt.figure(figsize=(13, 6.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1, 1], height_ratios=[1.25, 1],
                          hspace=0.5, wspace=0.42)
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_steps, ax_time = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    agent_models = [m for m in models if runs[m]["max_steps"] > 0]
    ax_fail = fig.add_subplot(gs[1, :2] if agent_models else gs[1, :])
    ax_tools = fig.add_subplot(gs[1, 2]) if agent_models else None

    # (a) accuracy with Wilson intervals
    x = np.arange(len(METRICS))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        means = np.array([stats[m][k]["mean"] for k, _ in METRICS]) * 100
        ci = np.array([stats[m][k]["ci95"] for k, _ in METRICS]).T * 100
        pos = x + (i - (len(models) - 1) / 2) * w
        ax_acc.bar(pos, means, w, yerr=[means - ci[0], ci[1] - means], capsize=3,
                   color=MODEL_COLORS[m], edgecolor="black", lw=0.8,
                   label=f"{MODEL_LABELS[m]} (n={stats[m]['n_tasks']})")
        for px, v, top in zip(pos, means, ci[1]):
            ax_acc.text(px, top + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax_acc.set_xticks(x, [lbl for _, lbl in METRICS])
    ax_acc.set_ylabel("Tasks localized (%)")
    ax_acc.set_ylim(0, 112)
    ax_acc.set_yticks(range(0, 101, 20))

    # (b), (c) distributions: box + jittered points
    rng = np.random.default_rng(0)
    for ax, key, label in ((ax_steps, "steps", "Agent steps per task"),
                           (ax_time, "wall_s", "Wall time per task (s)")):
        data = [stats[m][key] for m in models]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": "black", "lw": 1.5})
        for patch, m in zip(bp["boxes"], models):
            patch.set(facecolor=MODEL_COLORS[m], alpha=0.35, edgecolor="black")
        for i, (vals, m) in enumerate(zip(data, models), 1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=14,
                       color=MODEL_COLORS[m], edgecolor="black", lw=0.4, zorder=3)
        ax.set_xticks(range(1, len(models) + 1), [MODEL_LABELS[m] for m in models], fontsize=9)
        ax.set_ylabel(label)
    max_steps = max(r["max_steps"] for r in runs.values())
    if max_steps > 0:  # retrieval-only runs (C0) have no agent steps
        ax_steps.axhline(max_steps, color=PALETTE["red_strong"], ls="--", lw=1)
        ax_steps.text(0.5, max_steps, " step limit", color=PALETTE["red_strong"], va="bottom", fontsize=8)
    ax_steps.set_ylim(0, max_steps + 3)
    ax_time.set_ylim(0, max(max(stats[m]["wall_s"]) for m in models) * 1.1)

    # (d) failure modes
    names = list(next(iter(stats.values()))["failure"])
    y = np.arange(len(names))
    h = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = np.array([stats[m]["failure"][n] for n in names]) * 100
        pos = y + (i - (len(models) - 1) / 2) * h
        ax_fail.barh(pos, vals, h, color=MODEL_COLORS[m], edgecolor="black", lw=0.8,
                     label=MODEL_LABELS[m])
        for py, v in zip(pos, vals):
            ax_fail.text(v + 0.8, py, f"{v:.0f}%", va="center", fontsize=8)
    ax_fail.set_yticks(y, names)
    ax_fail.invert_yaxis()
    ax_fail.set_xlim(0, 105)
    ax_fail.set_xlabel("Share of tasks or steps affected (%)")
    ax_fail.grid(axis="y", visible=False)
    ax_fail.grid(axis="x", visible=True)
    handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor="black",
                     label=f"{MODEL_LABELS[m]} (n={stats[m]['n_tasks']})") for m in models]
    fig.legend(handles=handles, loc="upper center", ncol=len(models), bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, -0.04 if ax_tools is None else -0.13, "Error bars: 95% Wilson intervals. Forced submit: the model answered only "
             "when the step limit made it. Repeated call: identical to an earlier call in the episode.\n"
             "Malformed function name: not in the requested path.py::Name format (resolved against the "
             "model's submitted files for scoring when unambiguous).",
             ha="center", fontsize=8.5, color="#555555")

    # (e) tool usage: share of each model's calls going to each tool
    if ax_tools is not None:
        tools = [t for t in TOOL_ORDER if any(s.get("tool") == t for m in agent_models
                                              for task in runs[m]["tasks"] for s in task["steps"])]
        cmap = plt.get_cmap("tab10")
        left = np.zeros(len(agent_models))
        for k, tool in enumerate(tools):
            share = np.array([
                sum(s.get("tool") == tool for task in runs[m]["tasks"] for s in task["steps"])
                / max(1, sum(bool(s.get("tool")) for task in runs[m]["tasks"] for s in task["steps"]))
                for m in agent_models]) * 100
            ax_tools.barh(range(len(agent_models)), share, left=left, color=cmap(k), edgecolor="white",
                          lw=0.6, label=tool)
            left += share
        ax_tools.set_yticks(range(len(agent_models)), [MODEL_LABELS[m] for m in agent_models], fontsize=9)
        ax_tools.invert_yaxis()
        ax_tools.set_xlim(0, 100)
        ax_tools.set_xlabel("Share of tool calls (%)")
        ax_tools.grid(axis="y", visible=False)
        ax_tools.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2, fontsize=7.5,
                        handlelength=1.2, columnspacing=0.8)
        stats_tools = {m: {t: int(sum(s.get("tool") == t for task in runs[m]["tasks"] for s in task["steps"]))
                           for t in tools} for m in agent_models}
        (ckpt / "tool_usage.json").write_text(json.dumps(stats_tools, indent=1))

    for a, letter in zip([ax_acc, ax_steps, ax_time, ax_fail, ax_tools], "abcde"):
        if a is not None:
            panel_label(a, letter)
    for p in finalize_figure(fig, "summary", ckpt / "figures"):
        print(p)

    # per-task outcome matrix
    ids = sorted({t["instance_id"] for r in runs.values() for t in r["tasks"]})
    grid = np.full((len(ids), len(models)), np.nan)
    for j, m in enumerate(models):
        for t in runs[m]["tasks"]:
            mt = t["metrics"]
            func_ok = mt["func_acc_all@5"] if mt["func_acc_all@5"] is not None else mt["file_acc_all@5"]
            grid[ids.index(t["instance_id"]), j] = 2 if (mt["file_acc_all@5"] and func_ok) else \
                1 if mt["file_acc_all@5"] else 0
    colors = [PALETTE["red_1"], PALETTE["green_2"], PALETTE["blue_main"]]
    fig, ax = plt.subplots(figsize=(2.2 + 1.3 * len(models), 0.28 * len(ids) + 1.4))
    ax.imshow(np.ma.masked_invalid(grid), cmap=ListedColormap(colors), vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models], rotation=20, ha="right")
    ax.set_yticks(range(len(ids)), ids, fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(models)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ids)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in
                       zip(colors, ["Miss", "File only (top 5)", "File + function (top 5)"])],
              loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    for p in finalize_figure(fig, "outcomes", ckpt / "figures"):
        print(p)


if __name__ == "__main__":
    main()
