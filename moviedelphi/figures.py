"""Figures for the report and the slides, drawn from the saved result tables and run logs.

    python -m moviedelphi.figures --main DELPHI

Writes PNG (slides) and PDF (report) files to results/figures/.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .common import day_to_date, load_split  # noqa: E402

# one colour per method, the same in every figure (validated categorical palette, first four slots)
COLOR = {"Delphi": "#2a78d6", "RECENT-POP": "#eb6834", "POP": "#1baf7a", "ITEM-KNN": "#eda100"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CURVES = ["#86b6ef", "#2a78d6", "#898781"]  # training curves: first run, final run, a run not kept


SCALE, SUFFIX = 1.0, ""  # slide variants: smaller canvas (so text is larger once placed) and a file suffix


def fs(w, h):
    return (w * SCALE, h * SCALE)


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Noto Sans", "DejaVu Sans"], "font.size": 8,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.titlesize": 9, "axes.titlelocation": "left", "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False, "legend.fontsize": 7.5,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}{SUFFIX}.{ext}"))
    plt.close(fig)


def dot_ci(ax, labels, means, cis, colors, xlabel):
    """Horizontal dot-and-whisker plot of means with 95% intervals, value printed beside each dot."""
    y = np.arange(len(labels))[::-1]
    for yi, m, c, col in zip(y, means, cis, colors):
        ax.errorbar(m, yi, xerr=c, fmt="o", color=col, ecolor=col, elinewidth=1.2, capsize=0, markersize=6,
                    markeredgecolor="white", markeredgewidth=1.2)
        ax.text(m + c + 0.002, yi, f"{m:.3f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)


def fig_data(processed, out):
    """Users by number of active days, and ratings per month among training users."""
    users = pd.read_csv(os.path.join(processed, "users_train.csv"))
    bins = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3–4"), (5, 9, "5–9"), (10, 49, "10–49"), (50, 10**9, "50+")]
    share = [((users.n_days >= lo) & (users.n_days <= hi)).mean() for lo, hi, _ in bins]
    train = load_split("train", processed)
    month = pd.Series(day_to_date(train.days).astype("datetime64[M]")).value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=fs(7.0, 2.3), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    ax.bar(range(len(bins)), share, width=0.6, color=INK2)
    for i, s in enumerate(share):
        ax.text(i, s + 0.01, f"{100 * s:.0f}%", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(bins)), [b[2] for b in bins])
    ax.set_xlabel("active days (days with at least one rating)")
    ax.set_ylabel("share of training users")
    ax.set_ylim(0, max(share) * 1.18)
    ax.grid(axis="x", visible=False)
    ax.set_title("Most users rate everything on one day")
    ax = axes[1]
    ax.plot(month.index, month.values / 1e3, color=INK2, linewidth=1.2)
    ax.set_ylabel("training users' ratings per month (1000s)")
    ax.set_xlabel("month the rating was entered (UTC)")
    ax.set_title("Ratings come in bursts over 1996–2019")
    save(fig, out, "data_overview")


def fig_path(path_rows, val_main, out):
    """Validation recall@20 for each step from the baselines to the final model."""
    labels = [f"{r['label']}: {r['short']}" for r in path_rows]
    means = [r["recall@20"] for r in path_rows]
    cis = [r["ci"] for r in path_rows]
    colors = [COLOR.get(r["color"], r["color"]) for r in path_rows]
    fig, ax = plt.subplots(figsize=fs(5.2, 0.32 * len(labels) + 0.7))
    dot_ci(ax, labels, means, cis, colors, "validation recall@20 (mean over users, 95% interval)")
    ax.set_xlim(min(means) - 0.02, max(m + c for m, c in zip(means, cis)) + 0.02)
    save(fig, out, "validation_path")


def fig_training_curves(runs, refs, out):
    """Validation recall@20 during training, with the two popularity baselines as reference lines."""
    fig, ax = plt.subplots(figsize=fs(5.0, 2.6))
    for (label, run), col in zip(runs, CURVES):
        log = pd.read_csv(os.path.join("runs", run, "log.csv"))
        ax.plot(log["iter"] / 1000, log["recall@20"], color=col, marker="o", markersize=3.5,
                markeredgecolor="white", markeredgewidth=0.8, label=label)
    for name, value in refs.items():
        ax.axhline(value, color=COLOR[name], linewidth=1.2)
        ax.text(ax.get_xlim()[1], value, f" {name}", va="center", fontsize=7.5, color=INK2)
    ax.set_xlabel("training iterations (thousands)")
    ax.set_ylabel("validation recall@20")
    ax.legend(loc="lower right")
    save(fig, out, "training_curves")


def panel_headline(ax, main, main_name, seed_names=()):
    order = ["POP", "ITEM-KNN", "RECENT-POP", main_name]
    labels = ["POP", "ITEM-KNN", "RECENT-POP", "Delphi"]
    dot_ci(ax, labels, [main.loc[m, "recall@20"] for m in order],
           [main.loc[m, "recall@20_ci95"] for m in order], [COLOR[label] for label in labels],
           "test recall@20 (95% interval)")
    lo = min(main.loc[m, "recall@20"] - main.loc[m, "recall@20_ci95"] for m in order)
    hi = max(main.loc[m, "recall@20"] + main.loc[m, "recall@20_ci95"] for m in order)
    ax.set_xlim(lo - 0.01, hi + 0.025)


def fig_results(main_csv, hist_csv, main_name, out, seed_names=()):
    """(a) test recall@20 per method; (b) recall@20 against the number of visible history events.
    Saved as one two-panel figure (report) and as two single panels (slides)."""
    main = pd.read_csv(main_csv).set_index("method")
    hist = pd.read_csv(hist_csv)
    fig, ax = plt.subplots(figsize=fs(4.6, 1.9))
    panel_headline(ax, main, main_name, seed_names)
    save(fig, out, "test_headline")
    fig, ax = plt.subplots(figsize=fs(5.2, 2.8))
    panel_history(ax, main, hist, main_name)
    save(fig, out, "history_length")

    fig, axes = plt.subplots(1, 2, figsize=fs(7.0, 2.0), gridspec_kw={"width_ratios": [1, 1.25]})
    panel_headline(axes[0], main, main_name, seed_names)
    axes[0].set_title("a  Next-active-day ranking, held-out users")
    panel_history(axes[1], main, hist, main_name)
    axes[1].set_title("b  Recall@20 against visible history")
    save(fig, out, "test_results")


def panel_history(ax, main, hist, main_name):
    for name, label in ((main_name, "Delphi"), ("ITEM-KNN", "ITEM-KNN")):
        h = hist[hist.method == name]
        if len(h):
            ax.fill_between(h.k, h["recall@20"] - h["recall@20_ci95"], h["recall@20"] + h["recall@20_ci95"],
                            color=COLOR[label], alpha=0.12, linewidth=0)
            ax.plot(h.k, h["recall@20"], color=COLOR[label], marker="o", markersize=4,
                    markeredgecolor="white", markeredgewidth=0.8, label=label)
    for name in ("RECENT-POP", "POP"):
        ax.axhline(main.loc[name, "recall@20"], color=COLOR[name], linewidth=1.2, label=f"{name} (popularity only)")
    ax.set_xscale("log", base=2)
    ks = sorted(hist.k.unique())
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("visible history: most recent ratings given to the model")
    ax.set_ylabel("test recall@20")
    ax.set_ylim(0.03, ax.get_ylim()[1])
    ax.legend(loc="lower right", ncol=1)  # below the POP line, where no data is drawn


def bucket_recall(pc, col, edges, labels):
    b = pd.cut(pc[col], edges, labels=labels, right=True)
    g = pc.groupby(b, observed=False)["recall@20"]
    return pd.DataFrame({"mean": g.mean(), "ci": 1.96 * g.std(ddof=1) / np.sqrt(g.count()), "n": g.count()})


GAP = ([0, 1, 7, 30, 365, 10**6], ["1 d", "2–7 d", "8–30 d", "1–12 mo", ">1 y"])
HIST = ([0, 15, 50, 150, 500, 10**7], ["1–15", "16–50", "51–150", "151–500", "500+"])
TARGET = ([0, 1, 5, 20, 100, 10**6], ["1", "2–5", "6–20", "21–100", "100+"])


def fig_errors(per_case_dir, split, main_name, out):
    """Recall@20 by days since the last active day, by history length and by next-day basket size."""
    pcs = {label: pd.read_csv(os.path.join(per_case_dir, f"{split}_{name}.csv"))
           for label, name in (("Delphi", main_name), ("RECENT-POP", "RECENT-POP"), ("ITEM-KNN", "ITEM-KNN"))}
    panels = (("gap", GAP, "days since the last active day"),
              ("n_hist", HIST, "ratings in the history"),
              ("n_target", TARGET, "movies rated on the target day"))
    fig, axes = plt.subplots(1, 3, figsize=fs(7.0, 1.9), sharey=True)
    width = 0.26
    for ax, (col, (edges, labels), xlabel) in zip(axes, panels):
        x = np.arange(len(labels))
        for j, (label, pc) in enumerate(pcs.items()):
            r = bucket_recall(pc, col, edges, labels)
            ax.errorbar(x + (j - 1) * width, r["mean"], yerr=r["ci"], fmt="o", color=COLOR[label],
                        elinewidth=1.0, markersize=4.5, markeredgecolor="white", markeredgewidth=0.8, label=label)
        n = bucket_recall(pcs["Delphi"], col, edges, labels)["n"]
        ax.set_xticks(x, [f"{lab}\nn={c}" for lab, c in zip(labels, n)], fontsize=5.8)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(f"{split} recall@20")
    axes[0].legend(loc="upper right")
    save(fig, out, f"{split}_errors")


def fig_controls(controls_csv, main_csv, main_name, out):
    c = pd.read_csv(controls_csv)
    main = pd.read_csv(main_csv).set_index("method")
    names = {"real history": "own history", "shuffle_order": "own movies, order shuffled",
             "other_user": "another user's history"}
    labels = [names[x] for x in c.control] + ["RECENT-POP (popularity only)"]
    means = list(c["recall@20"]) + [main.loc["RECENT-POP", "recall@20"]]
    cis = list(c["recall@20_ci95"]) + [main.loc["RECENT-POP", "recall@20_ci95"]]
    colors = [COLOR["Delphi"]] * len(c) + [COLOR["RECENT-POP"]]
    fig, ax = plt.subplots(figsize=fs(5.0, 1.8))
    dot_ci(ax, labels, means, cis, colors, "test recall@20 (95% interval)")
    ax.set_xlim(min(means) - 0.02, max(m + ci for m, ci in zip(means, cis)) + 0.025)
    save(fig, out, "controls")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--results", default="results")
    ap.add_argument("--main", default="DELPHI")
    ap.add_argument("--seeds", nargs="*", default=[])
    ap.add_argument("--path", default="results/tables/validation_path.json")
    ap.add_argument("--curves", nargs="*", default=[], help="LABEL=run_name for the training-curve figure")
    ap.add_argument("--slides", action="store_true", help="write *_slide variants with larger text for the slides")
    args = ap.parse_args()
    global SCALE, SUFFIX
    if args.slides:
        SCALE, SUFFIX = 0.72, "_slide"
    out = os.path.join(args.results, "figures")
    os.makedirs(out, exist_ok=True)
    style()
    t = lambda name: os.path.join(args.results, "tables", name)  # noqa: E731
    fig_data(args.processed, out)
    if os.path.exists(args.path):
        with open(args.path) as f:
            fig_path(json.load(f), None, out)
    if args.curves:
        val = pd.read_csv(t("val_main_path.csv")).set_index("method")
        fig_training_curves([c.split("=", 1) for c in args.curves],
                            {k: val.loc[k, "recall@20"] for k in ("RECENT-POP", "POP")}, out)
    if os.path.exists(t("test_main.csv")):
        fig_results(t("test_main.csv"), t("test_history_length.csv"), args.main, out, args.seeds)
        fig_errors(os.path.join(args.results, "per_case"), "test", args.main, out)
        fig_controls(t("test_controls.csv"), t("test_main.csv"), args.main, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
