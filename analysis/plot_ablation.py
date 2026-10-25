"""Figure: an ablation ladder across several experiments run on the same tasks.

Each experiment adds one harness change to the previous one. Panels show, per
model, how each metric moves along the ladder:
  (a) File@1   (b) Function@5   (c) Function@10      (95% Wilson intervals)
  (d) steps per task (median, IQR)   (e) forced submits   (f) repeated calls
Consecutive configurations are compared with an exact McNemar test on the
paired per-task outcomes (same tasks in every experiment); the results are
written to stats.json, and significant changes (p < 0.05) are starred.

    python analysis/plot_ablation.py --name dev-ablation \
        --exps c1-dev fmt-dev guard-dev ranked-dev \
        --labels "C1" "+format" "+guard\n+budget" "+ranked"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (MODEL_COLORS, MODEL_LABELS,  # noqa: E402
                      apply_publication_style, finalize_figure, panel_label, plt)
from plot_experiment import MODEL_ORDER, failure_modes, load, wilson  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.checkpoint import checkpoint_dir  # noqa: E402

# Function@10 is left out: the models never submit more than ~3 functions, so it equals Function@5
ACC = [("file_acc_all@1", "File@1 (%)"), ("file_acc_all@5", "File@5 (%)"),
       ("func_acc_all@5", "Function@5 (%)")]
MARKERS = {"gemma-4-e2b": "o", "qwen3.5-4b": "s", "gemma-4-31b": "^"}


def mcnemar(a: dict, b: dict):
    """Exact McNemar test on paired binary outcomes {instance_id: 0/1}."""
    ids = [i for i in a if i in b and a[i] is not None and b[i] is not None]
    gained = sum(1 for i in ids if not a[i] and b[i])
    lost = sum(1 for i in ids if a[i] and not b[i])
    p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
    return {"gained": gained, "lost": lost, "p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="checkpoint to write, e.g. dev-ablation")
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--alt", help="experiment that is an alternative to one ladder step")
    ap.add_argument("--alt-at", type=int, help="index of the ladder step it replaces")
    ap.add_argument("--alt-label", default="alternative")
    args = ap.parse_args()
    alt = load(checkpoint_dir(args.alt, create=False)) if args.alt else None
    labels = [lbl.replace("\\n", "\n") for lbl in args.labels]
    runs = [load(checkpoint_dir(e, create=False)) for e in args.exps]
    models = [m for m in MODEL_ORDER if all(m in r for r in runs)]
    out = checkpoint_dir(args.name)

    stats = {"experiments": args.exps, "models": {}}
    for m in models:
        ms = {"metrics": {}, "mcnemar": {}}
        for key, _ in ACC:
            per_exp, outcomes = [], []
            for r in runs:
                vals = {t["instance_id"]: t["metrics"][key] for t in r[m]["tasks"]}
                ok = [v for v in vals.values() if v is not None]
                p, lo, hi = wilson(sum(ok), len(ok))
                per_exp.append({"mean": p, "ci95": [lo, hi], "n": len(ok)})
                outcomes.append(vals)
            ms["metrics"][key] = per_exp
            ms["mcnemar"][key] = [mcnemar(outcomes[i], outcomes[i + 1]) for i in range(len(runs) - 1)]
        fms = [[failure_modes(t, r[m]["max_steps"]) for t in r[m]["tasks"]] for r in runs]
        ms["steps"] = [[t["n_steps"] for t in r[m]["tasks"]] for r in runs]
        ms["forced_submit"] = [float(np.mean([f["forced_submit"] for f in fm])) for fm in fms]
        ms["repeat_steps"] = [sum(f["repeat_steps"] for f in fm) / sum(f["n_steps"] for f in fm)
                              for fm in fms]
        ms["malformed"] = [float(np.mean([f["malformed"] for f in fm])) for fm in fms]
        stats["models"][m] = ms
    if alt:
        stats["alternative"] = {"experiment": args.alt, "replaces_step": args.alt_at, "models": {}}
        for m in models:
            tasks = alt[m]["tasks"]
            fm = [failure_modes(t, alt[m]["max_steps"]) for t in tasks]
            am, tests = {}, {}
            for key, _ in ACC:
                ok = [t["metrics"][key] for t in tasks if t["metrics"][key] is not None]
                p, lo, hi = wilson(sum(ok), len(ok))
                am[key] = {"mean": p, "ci95": [lo, hi], "n": len(ok)}
                prev = {t["instance_id"]: t["metrics"][key] for t in runs[args.alt_at - 1][m]["tasks"]}
                tests[key] = mcnemar(prev, {t["instance_id"]: t["metrics"][key] for t in tasks})
            stats["alternative"]["models"][m] = {
                "metrics": am, "mcnemar_vs_previous": tests,
                "steps": [t["n_steps"] for t in tasks],
                "forced_submit": float(np.mean([f["forced_submit"] for f in fm])),
                "repeat_steps": sum(f["repeat_steps"] for f in fm) / sum(f["n_steps"] for f in fm)}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    ALT_DX = 0.22

    def alt_marker(ax, m, y, yerr=None):
        ax.errorbar(args.alt_at + ALT_DX + offs[m], y, yerr=yerr, color=MODEL_COLORS[m], marker=MARKERS[m],
                    ms=6, mfc="white", mec=MODEL_COLORS[m], mew=1.4, lw=0, elinewidth=1.2, capsize=3)

    apply_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    axes = axes.ravel()
    x = np.arange(len(runs))
    offs = {m: (i - (len(models) - 1) / 2) * 0.08 for i, m in enumerate(models)}

    for ax, (key, ylabel) in zip(axes[:3], ACC):
        for m in models:
            ms = stats["models"][m]["metrics"][key]
            mean = np.array([e["mean"] for e in ms]) * 100
            ci = np.array([e["ci95"] for e in ms]).T * 100
            ax.errorbar(x + offs[m], mean, yerr=[mean - ci[0], ci[1] - mean], color=MODEL_COLORS[m],
                        marker=MARKERS[m], ms=6, mec="black", mew=0.6, lw=1.6, capsize=3,
                        label=MODEL_LABELS[m])
            for i, t in enumerate(stats["models"][m]["mcnemar"][key]):
                if t["p"] < 0.05:
                    ax.text(i + 0.5, (mean[i] + mean[i + 1]) / 2, "*", color=MODEL_COLORS[m],
                            ha="center", va="bottom", fontsize=14, fontweight="bold")
            if alt:
                a = stats["alternative"]["models"][m]["metrics"][key]
                y, lo, hi = a["mean"] * 100, a["ci95"][0] * 100, a["ci95"][1] * 100
                alt_marker(ax, m, y, [[y - lo], [hi - y]])
                if stats["alternative"]["models"][m]["mcnemar_vs_previous"][key]["p"] < 0.05:
                    ax.text(args.alt_at + ALT_DX + offs[m] + 0.09, y, "*", color=MODEL_COLORS[m],
                            ha="left", va="center", fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 105)

    ax = axes[3]
    for m in models:
        st = stats["models"][m]["steps"]
        med = np.array([np.median(s) for s in st])
        q1, q3 = np.array([np.percentile(s, 25) for s in st]), np.array([np.percentile(s, 75) for s in st])
        ax.errorbar(x + offs[m], med, yerr=[med - q1, q3 - med], color=MODEL_COLORS[m],
                    marker=MARKERS[m], ms=6, mec="black", mew=0.6, lw=1.6, capsize=3)
        if alt:
            s_ = stats["alternative"]["models"][m]["steps"]
            md = np.median(s_)
            alt_marker(ax, m, md, [[md - np.percentile(s_, 25)], [np.percentile(s_, 75) - md]])
    max_steps = max(r[m]["max_steps"] for r in runs for m in models)
    ax.axhline(max_steps, color="#B64342", ls="--", lw=1)
    ax.set_ylim(0, max_steps + 2)
    ax.set_ylabel("Steps per task (median, IQR)")

    for ax, key, ylabel in ((axes[4], "forced_submit", "Forced submits (% of tasks)"),
                            (axes[5], "repeat_steps", "Repeated calls (% of steps)")):
        for m in models:
            ax.plot(x + offs[m], np.array(stats["models"][m][key]) * 100, color=MODEL_COLORS[m],
                    marker=MARKERS[m], ms=6, mec="black", mew=0.6, lw=1.6)
            if alt:
                alt_marker(ax, m, stats["alternative"]["models"][m][key] * 100)
        ax.set_ylim(0, 105)
        ax.set_ylabel(ylabel)

    for i, ax in enumerate(axes):
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_xlim(-0.4, len(runs) - 0.6)
        panel_label(ax, "abcdef"[i])
    handles, lbls = axes[0].get_legend_handles_labels()
    lbls = [f"{lb} (n={stats['models'][m]['metrics']['file_acc_all@1'][0]['n']} tasks)"
            for lb, m in zip(lbls, models)]
    if alt:
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], color="black", marker="o", mfc="white", lw=0))
        lbls.append(f"Hollow: {args.alt_label} (alternative to step "
                    f"\"{labels[args.alt_at].split(chr(10))[0]}\")")
    fig.legend(handles, lbls, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, -0.03, "Each configuration adds one change to the one on its left; all run on the same "
             "tasks. Error bars: 95% Wilson intervals (a-c), interquartile range (d).\n* exact McNemar test "
             "on paired per-task outcomes against the previous configuration, p < 0.05 "
             "(the hollow alternative is tested against the same previous configuration).",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for p in finalize_figure(fig, "ablation", out / "figures"):
        print(p)


if __name__ == "__main__":
    main()
