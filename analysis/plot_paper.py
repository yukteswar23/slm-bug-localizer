"""Paper-only figures and tables, built from the saved experiment results.

Writes into paper/figures/ (PDF + PNG) and paper/tables/ (LaTeX fragments), plus
paper/data/paper_numbers.json with every number quoted in the text. The
grammar-off failure breakdown reads raw replies from the trajectories in
C:/gemma_work/runs/ once and caches the counts in paper/data/, so the paper
can be rebuilt without the trajectories.

    python analysis/plot_paper.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.checkpoint import TRAJ_ROOT, checkpoint_dir  # noqa: E402
from analysis.figstyle import (MODEL_COLORS, MODEL_LABELS, PALETTE,  # noqa: E402
                               apply_publication_style, finalize_figure, panel_label)
from analysis.plot_ablation import mcnemar  # noqa: E402
from analysis.plot_experiment import wilson  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
FIGS, TABLES, DATA = PAPER / "figures", PAPER / "tables", PAPER / "data"

MODELS = ["gemma-4-e2b", "qwen3.5-4b"]
# Test-set configurations in ladder order; C2-G is C2 without grammar-constrained decoding.
CONFIGS = [("c1-test", "C1"), ("h07-test", "C1+H"), ("c2-test", "C2"), ("c3-test", "C3"),
           ("c4g-test", "C2$-$G")]
DEV_TEST = [("c1-dev", "c1-test", "C1"), ("ranked-dev", "h07-test", "C1+H"),
            ("c2-dev", "c2-test", "C2"), ("c3-dev", "c3-test", "C3")]
METRICS = [("file_acc_all@1", "File@1"), ("file_acc_all@5", "File@5"), ("func_acc_all@5", "Function@5")]
# Label offsets (points) where scatter labels would overlap.
LABEL_OFFSETS = {("gemma-4-e2b", "c1-test"): (6, -12), ("gemma-4-e2b", "c3-test"): (-20, -15),
                 ("gemma-4-e2b", "c2-test"): (6, -12), ("qwen3.5-4b", "c3-test"): (6, -12)}
TOOLS = ["list_dir", "search_code", "read_file", "find_definition", "get_code_neighbors",
         "search_similar_code", "get_file_outline", "submit"]


def load(exp: str, model: str) -> list:
    path = checkpoint_dir(exp, create=False) / f"{model}.json"
    return json.loads(path.read_text(encoding="utf-8"))["tasks"]


def outcomes(tasks: list, metric: str) -> dict:
    """{instance_id: 0/1}, or None where the metric is undefined (no gold function)."""
    return {t["instance_id"]: t["metrics"].get(metric) for t in tasks}


def rate(out: dict):
    vals = [v for v in out.values() if v is not None]
    return wilson(sum(vals), len(vals)), len(vals)


def repo_of(instance_id: str) -> str:
    return instance_id.split("__")[0].replace("-dev", "").replace("scikit-learn", "sklearn")


# ---------------------------------------------------------------- table

def main_table(runs, bm25, numbers):
    """LaTeX rows of the main results table (test split, 100 tasks)."""
    lines = []

    def cells(tasks):
        out = []
        for metric, _ in METRICS:
            (p, lo, hi), _n = rate(outcomes(tasks, metric))
            out.append(f"{100 * p:.0f} {{\\scriptsize[{100 * lo:.0f},{100 * hi:.0f}]}}")
        return out

    bm = cells(bm25)
    lines.append("C0 & BM25 retrieval, no LLM & " + " & ".join(bm) + " & " + " & ".join(bm)
                 + " & \\multicolumn{2}{c}{2} \\\\")
    lines.append("\\midrule")
    desc = {"c1-test": "ReAct + file tools", "h07-test": "+ harness fixes (H)",
            "c2-test": "+ graph tools", "c3-test": "+ Evidence Buffer",
            "c4g-test": "C2 without grammar"}
    for exp, label in CONFIGS:
        row = [label.replace("$-$", "$-$"), desc[exp]]
        for m in MODELS:
            row += cells(runs[exp][m])
        row += [f"{np.median([t['wall_s'] for t in runs[exp][m]]):.0f}" for m in MODELS]
        if exp == "c4g-test":
            lines.append("\\midrule")
        lines.append(" & ".join(row) + " \\\\")
    (TABLES / "main_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for exp, label in CONFIGS:
        for m in MODELS:
            for metric, name in METRICS:
                (p, lo, hi), n = rate(outcomes(runs[exp][m], metric))
                numbers[f"{exp}/{m}/{name}"] = [round(100 * p, 1), round(100 * lo, 1), round(100 * hi, 1), n]
    for metric, name in METRICS + [("func_acc_all@10", "Function@10")]:
        (p, lo, hi), n = rate(outcomes(bm25, metric))
        numbers[f"c0-test/bm25/{name}"] = [round(100 * p, 1), round(100 * lo, 1), round(100 * hi, 1), n]


# ---------------------------------------------------------------- figures

def fig_repos(runs, bm25, numbers):
    """Per-repository accuracy: where each system succeeds."""
    systems = [("BM25", bm25), ("Gemma C1", runs["c1-test"]["gemma-4-e2b"]),
               ("Gemma C2", runs["c2-test"]["gemma-4-e2b"]), ("Qwen C1", runs["c1-test"]["qwen3.5-4b"]),
               ("Qwen C2", runs["c2-test"]["qwen3.5-4b"])]
    repos = sorted(Counter(repo_of(t["instance_id"]) for t in bm25).items(), key=lambda kv: -kv[1])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), constrained_layout=True)
    for ax, (metric, name), letter in zip(axes, [METRICS[0], METRICS[2]], "ab"):
        grid = np.full((len(repos), len(systems)), np.nan)
        counts = []
        for i, (repo, _) in enumerate(repos):
            for j, (_, tasks) in enumerate(systems):
                vals = [v for k, v in outcomes(tasks, metric).items() if repo_of(k) == repo and v is not None]
                if vals:
                    grid[i, j] = 100 * np.mean(vals)
                if j == 0:
                    counts.append(len(vals))
        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                            color="white" if grid[i, j] > 60 else "black")
        ax.set_xticks(range(len(systems)), [s for s, _ in systems], rotation=30, ha="right")
        ax.set_yticks(range(len(repos)), [f"{r} (n={c})" for (r, _), c in zip(repos, counts)])
        ax.grid(False)
        ax.set_title(f"{name} (%)")
        for s in ax.spines.values():
            s.set_visible(False)
        panel_label(ax, letter)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Tasks localized (%)")
    numbers["repos"] = {r: c for r, c in repos}
    finalize_figure(fig, "repo_heatmap", FIGS)


def fig_cost(runs, numbers):
    """Accuracy against time and generated tokens per task."""
    markers = dict(zip([e for e, _ in CONFIGS], ["o", "s", "D", "^", "X"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    ax = axes[0]
    for m in MODELS:
        xs, ys = [], []
        for exp, label in CONFIGS:
            tasks = runs[exp][m]
            w = np.array([t["wall_s"] for t in tasks])
            (p, lo, hi), _ = rate(outcomes(tasks, "func_acc_all@5"))
            x, xe = w.mean(), 1.96 * w.std(ddof=1) / np.sqrt(len(w))
            ax.errorbar(x, 100 * p, xerr=xe, yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt=markers[exp],
                        color=MODEL_COLORS[m], mec="black", mew=0.6, ms=7, capsize=2, lw=1)
            ax.annotate(label, (x, 100 * p), textcoords="offset points", fontsize=8.5,
                        xytext=LABEL_OFFSETS.get((m, exp), (6, 4)))
            xs.append(x)
            ys.append(100 * p)
        ax.plot([], [], "o", color=MODEL_COLORS[m], mec="black", label=MODEL_LABELS[m])
    ax.set_xlabel("Mean wall time per task (s)")
    ax.set_ylabel("Function@5 (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    panel_label(ax, "a")

    ax = axes[1]
    width = 0.38
    x = np.arange(len(CONFIGS))
    for k, m in enumerate(MODELS):
        means, errs = [], []
        for exp, _ in CONFIGS:
            tok = np.array([sum(s.get("completion_tokens") or 0 for s in t["steps"]) for t in runs[exp][m]])
            means.append(tok.mean())
            errs.append(1.96 * tok.std(ddof=1) / np.sqrt(len(tok)))
            numbers[f"{exp}/{m}/gen_tokens_mean"] = round(float(tok.mean()))
            numbers[f"{exp}/{m}/wall_mean_s"] = round(float(np.mean([t["wall_s"] for t in runs[exp][m]])), 1)
        ax.bar(x + (k - 0.5) * width, means, width, yerr=errs, capsize=2, color=MODEL_COLORS[m],
               edgecolor="black", lw=0.6, label=MODEL_LABELS[m])
    ax.set_xticks(x, [l for _, l in CONFIGS])
    ax.set_ylabel("Generated tokens per task")
    ax.legend(loc="upper left")
    panel_label(ax, "b")
    finalize_figure(fig, "cost_accuracy", FIGS)


def fig_overlap(runs, bm25, numbers):
    """Which tasks each system solves, and how much a union of configurations would solve."""
    metric = "func_acc_all@5"
    g = outcomes(runs["c2-test"]["gemma-4-e2b"], metric)
    q = outcomes(runs["c2-test"]["qwen3.5-4b"], metric)
    b = outcomes(bm25, metric)
    ids = [i for i in g if g[i] is not None]
    cats = Counter()
    for i in ids:
        cats[(bool(g[i]), bool(q[i]))] += 1
    bm_rescue = sum(1 for i in ids if b[i] and not g[i] and not q[i])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    labels = ["Both", "Qwen only", "Gemma only", "Neither"]
    vals = [cats[(True, True)], cats[(False, True)], cats[(True, False)], cats[(False, False)]]
    colors = [PALETTE["blue_secondary"], MODEL_COLORS["qwen3.5-4b"], MODEL_COLORS["gemma-4-e2b"], PALETTE["neutral"]]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", lw=0.6)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 1, str(v), ha="center", fontsize=9)
    ax.set_ylabel(f"Tasks (of {len(ids)})")
    ax.set_title("Function@5, C2: who solves which task")
    panel_label(ax, "a")

    ax = axes[1]
    rows = []
    union_all = {i: 0 for i in ids}
    for m in MODELS:
        best = max(rate(outcomes(runs[e][m], metric))[0][0] for e, _ in CONFIGS)
        union = {i: int(any(outcomes(runs[e][m], metric)[i] for e, _ in CONFIGS)) for i in ids}
        for i in ids:
            union_all[i] |= union[i]
        rows.append((MODEL_LABELS[m], 100 * best, 100 * np.mean(list(union.values())), MODEL_COLORS[m]))
    both_union = 100 * np.mean(list(union_all.values()))
    with_bm = 100 * np.mean([union_all[i] or (b[i] or 0) for i in ids])
    y = np.arange(len(rows))
    for k, (lab, best, uni, c) in enumerate(rows):
        ax.barh(k, uni, color=c, alpha=0.35, edgecolor="black", lw=0.6)
        ax.barh(k, best, color=c, edgecolor="black", lw=0.6)
        ax.text(uni + 1, k, f"{best:.0f} → {uni:.0f}", va="center", fontsize=9)
    ax.barh(len(rows), both_union, color=PALETTE["blue_secondary"], alpha=0.35, edgecolor="black", lw=0.6)
    ax.text(both_union + 1, len(rows), f"{both_union:.0f}", va="center", fontsize=9)
    ax.barh(len(rows) + 1, with_bm, color=PALETTE["red_2"], alpha=0.6, edgecolor="black", lw=0.6)
    ax.text(with_bm + 1, len(rows) + 1, f"{with_bm:.0f}", va="center", fontsize=9)
    ax.set_yticks(list(y) + [len(rows), len(rows) + 1],
                  [r[0] for r in rows] + ["Both models", "Both models + BM25"])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Function@5 (%)")
    ax.set_title("Best single configuration (solid) vs. any configuration (light)", fontsize=10)
    ax.grid(axis="x")
    panel_label(ax, "b")
    numbers["overlap_c2_func5"] = {"both": vals[0], "qwen_only": vals[1], "gemma_only": vals[2],
                                   "neither": vals[3], "bm25_rescues": bm_rescue, "n": len(ids)}
    numbers["oracle_func5"] = {r[0]: [round(r[1], 1), round(r[2], 1)] for r in rows}
    numbers["oracle_func5"]["both"] = round(both_union, 1)
    numbers["oracle_func5"]["both+bm25"] = round(with_bm, 1)
    finalize_figure(fig, "overlap_oracle", FIGS)


def grammar_failures():
    """Classify invalid replies of the grammar-off run from the raw trajectories (cached)."""
    cache = DATA / "grammar_failures.json"
    if cache.exists():
        return json.loads(cache.read_text())
    exp_dir = checkpoint_dir("c4g-test", create=False)
    out = {}
    for m in MODELS:
        c = Counter()
        n_steps = 0
        for f in sorted((TRAJ_ROOT / exp_dir.name / m).glob("*.json")):
            for s in json.loads(f.read_text(encoding="utf-8"))["steps"]:
                n_steps += 1
                if s.get("valid"):
                    continue
                err, raw = str(s.get("error") or ""), str(s.get("raw") or "")
                if "missing or unknown action.tool" in err:
                    try:
                        obj = json.JSONDecoder().raw_decode(raw[raw.find("{"):])[0]
                        act = obj.get("action")
                    except ValueError:
                        act = None
                    if isinstance(act, dict) and ("files" in act or "functions" in act):
                        c["Answer without tool name"] += 1
                    elif not isinstance(act, dict):
                        c["No action object"] += 1
                    else:
                        c["Unknown tool name"] += 1
                elif "no JSON" in err:
                    c["No JSON at all"] += 1
                else:
                    c["Malformed JSON"] += 1
        out[m] = {"steps": n_steps, "invalid": dict(c)}
    cache.write_text(json.dumps(out, indent=1))
    return out


def fig_grammar(runs, numbers):
    fails = grammar_failures()
    cats = ["Malformed JSON", "Answer without tool name", "No action object", "Unknown tool name", "No JSON at all"]
    colors = [PALETTE["red_strong"], PALETTE["violet"], PALETTE["red_2"], PALETTE["neutral"], "#8c8c8c"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    for k, m in enumerate(MODELS):
        left = 0.0
        for cat, col in zip(cats, colors):
            v = 100 * fails[m]["invalid"].get(cat, 0) / fails[m]["steps"]
            ax.barh(k, v, left=left, color=col, edgecolor="black", lw=0.5, label=cat if k == 0 else None)
            left += v
        ax.text(left + 0.5, k, f"{left:.0f}%", va="center", fontsize=9)
        numbers[f"c4g-test/{m}/invalid_step_share"] = round(left, 1)
        numbers[f"c4g-test/{m}/invalid_by_type"] = fails[m]["invalid"]
    ax.set_yticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlabel("Invalid replies (% of all steps)")
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=8.5)
    panel_label(ax, "a")

    ax = axes[1]
    labels = ["Any invalid\nreply", "Empty final\nanswer"]
    keys = ["Invalid reply (tasks)", "Empty answer (tasks)"]
    width = 0.38
    for k, m in enumerate(MODELS):
        tasks = runs["c4g-test"][m]
        inv = np.mean([any(not s.get("valid") for s in t["steps"]) for t in tasks])
        empty = np.mean([not (t["answer"] or {}).get("files") for t in tasks])
        vals = [100 * inv, 100 * empty]
        r = ax.bar(np.arange(2) + (k - 0.5) * width, vals, width, color=MODEL_COLORS[m], edgecolor="black",
                   lw=0.6, label=MODEL_LABELS[m])
        for rect, v in zip(r, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8.5)
        numbers[f"c4g-test/{m}/tasks_with_invalid"] = round(100 * inv, 1)
        numbers[f"c4g-test/{m}/empty_answer"] = round(100 * empty, 1)
    del keys
    ax.set_xticks(range(2), labels)
    ax.set_ylabel("Tasks (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8.5)
    panel_label(ax, "b")

    ax = axes[2]
    for m in MODELS:
        for exp, ls in [("c2-test", "-"), ("c4g-test", "--")]:
            steps = np.sort([t["n_steps"] for t in runs[exp][m]])
            ax.step(steps, np.arange(1, len(steps) + 1) / len(steps) * 100, where="post", ls=ls,
                    color=MODEL_COLORS[m], lw=1.6,
                    label=f"{MODEL_LABELS[m].split()[0]}, {'grammar' if exp == 'c2-test' else 'no grammar'}")
    ax.set_xlabel("Steps per task")
    ax.set_ylabel("Tasks finished within (%)")
    ax.set_xlim(0, 20.5)
    ax.legend(fontsize=8, loc="upper left")
    panel_label(ax, "c")
    finalize_figure(fig, "grammar_failures", FIGS)


def fig_tools(runs, numbers):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True, sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, m, letter in zip(axes, MODELS, "ab"):
        for k, (exp, label) in enumerate(CONFIGS):
            usage = json.loads((checkpoint_dir(exp, create=False) / "tool_usage.json").read_text())[m]
            total = sum(usage.values())
            left = 0.0
            for j, tool in enumerate(TOOLS):
                v = 100 * usage.get(tool, 0) / total
                ax.barh(k, v, left=left, color=cmap(j), edgecolor="white", lw=0.4,
                        label=tool if k == 0 and m == MODELS[0] else None)
                left += v
            graph = sum(usage.get(t, 0) for t in TOOLS[3:7])
            numbers[f"{exp}/{m}/graph_tool_share"] = round(100 * graph / total, 1)
        ax.set_yticks(range(len(CONFIGS)), [l for _, l in CONFIGS])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of tool calls (%)")
        ax.set_title(MODEL_LABELS[m])
        ax.grid(axis="x")
        panel_label(ax, letter)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=8, fontsize=8.5)
    finalize_figure(fig, "tool_usage", FIGS)


def fig_rank(runs, numbers):
    """Where the correct file sits in a ranked answer."""
    exps = [("c1-test", "C1"), ("h07-test", "C1+H"), ("c2-test", "C2"), ("c3-test", "C3")]
    cats = ["Rank 1", "Rank 2-5", "Not in top 5"]
    colors = [PALETTE["blue_main"], PALETTE["blue_secondary"], PALETTE["neutral"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), constrained_layout=True, sharey=True)
    for ax, m, letter in zip(axes, MODELS, "ab"):
        for k, (exp, label) in enumerate(exps):
            tasks = runs[exp][m]
            h1 = np.mean([t["metrics"]["file_hit@1"] for t in tasks])
            h5 = np.mean([t["metrics"]["file_hit@5"] for t in tasks])
            vals = 100 * np.array([h1, h5 - h1, 1 - h5])
            left = 0.0
            for v, c, cat in zip(vals, colors, cats):
                ax.barh(k, v, left=left, color=c, edgecolor="black", lw=0.5,
                        label=cat if k == 0 and m == MODELS[0] else None)
                if v >= 6:
                    ax.text(left + v / 2, k, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                            color="white" if c in colors[:2] else "black")
                left += v
            numbers[f"{exp}/{m}/file_rank"] = [round(float(v), 1) for v in vals]
            numbers[f"{exp}/{m}/n_files_submitted"] = round(float(np.mean(
                [len((t["answer"] or {}).get("files") or []) for t in tasks])), 2)
        ax.set_yticks(range(len(exps)), [l for _, l in exps])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Tasks (%)")
        ax.set_title(MODEL_LABELS[m])
        ax.grid(False)
        panel_label(ax, letter)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    finalize_figure(fig, "file_rank", FIGS)


def fig_dev_test(numbers):
    """Does the 23-task dev split predict the 100-task test split?"""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), constrained_layout=True)
    markers = ["o", "s", "D", "^"]
    for ax, (metric, name), letter in zip(axes, [METRICS[0], METRICS[2]], "ab"):
        for m in MODELS:
            for (dev, test, label), mk in zip(DEV_TEST, markers):
                (pd, lod, hid), _ = rate(outcomes(load(dev, m), metric))
                (pt, lot, hit), _ = rate(outcomes(load(test, m), metric))
                ax.errorbar(100 * pd, 100 * pt, xerr=[[100 * (pd - lod)], [100 * (hid - pd)]],
                            yerr=[[100 * (pt - lot)], [100 * (hit - pt)]], fmt=mk, color=MODEL_COLORS[m],
                            mec="black", mew=0.6, ms=6.5, capsize=1.5, lw=0.7, alpha=0.9,
                            label=label if m == MODELS[0] else None)
                numbers[f"devtest/{m}/{label}/{name}"] = [round(100 * pd, 1), round(100 * pt, 1)]
        ax.plot([0, 100], [0, 100], color="#999999", lw=0.8, ls=":")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Dev split, n=23 ({name}, %)")
        ax.set_ylabel(f"Test split, n=100 ({name}, %)")
        panel_label(ax, letter)
    handles = [plt.Line2D([], [], marker=mk, ls="", color="white", mec="black", label=l)
               for (_, _, l), mk in zip(DEV_TEST, markers)]
    handles += [plt.Line2D([], [], marker="o", ls="", color=MODEL_COLORS[m], mec="black", label=MODEL_LABELS[m])
                for m in MODELS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=6, fontsize=9)
    finalize_figure(fig, "dev_vs_test", FIGS)


def paired_tests(runs, numbers):
    """C1 -> C2 and C1 -> C3 tests that the ladder figure does not show (it tests adjacent steps)."""
    for m in MODELS:
        for a, b in [("c1-test", "c2-test"), ("c1-test", "c3-test"), ("c2-test", "c3-test"),
                     ("c2-test", "c4g-test")]:
            for metric, name in METRICS:
                numbers[f"mcnemar/{m}/{a}->{b}/{name}"] = mcnemar(outcomes(runs[a][m], metric),
                                                                 outcomes(runs[b][m], metric))
        bm = load("c0-test", "bm25")
        for metric, name in METRICS:
            numbers[f"mcnemar/{m}/bm25->c2-test/{name}"] = mcnemar(outcomes(bm, metric),
                                                                   outcomes(runs["c2-test"][m], metric))
    total_h = sum(t["wall_s"] for e, _ in CONFIGS for m in MODELS for t in runs[e][m]) / 3600
    numbers["test_compute_hours"] = round(total_h, 1)
    numbers["test_episodes"] = sum(len(runs[e][m]) for e, _ in CONFIGS for m in MODELS)


def main():
    for d in (FIGS, TABLES, DATA):
        d.mkdir(parents=True, exist_ok=True)
    apply_publication_style()
    runs = {exp: {m: load(exp, m) for m in MODELS} for exp, _ in CONFIGS}
    bm25 = load("c0-test", "bm25")
    numbers = {}
    main_table(runs, bm25, numbers)
    fig_repos(runs, bm25, numbers)
    fig_cost(runs, numbers)
    fig_overlap(runs, bm25, numbers)
    fig_grammar(runs, numbers)
    fig_tools(runs, numbers)
    fig_rank(runs, numbers)
    fig_dev_test(numbers)
    paired_tests(runs, numbers)
    (DATA / "paper_numbers.json").write_text(json.dumps(numbers, indent=1))
    print(f"wrote {len(list(FIGS.glob('*.pdf')))} figures, tables and {len(numbers)} numbers to {PAPER}")


if __name__ == "__main__":
    main()
