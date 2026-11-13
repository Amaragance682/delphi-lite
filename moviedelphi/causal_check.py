"""Causal-candidate check: how much does each method gain from movies that no training user had rated
by the prediction day (films released or added to MovieLens later)?

    python -m moviedelphi.causal_check --split test --model DELPHI=models/delphi_final.pt

Users are split, not time: the model and the baselines are fitted on training users' whole histories,
including ratings made after a held-out user's prediction day. Here every method's candidates are
restricted to movies some training user had rated on or before the prediction day. Targets are
unchanged, so a "future" target counts as a miss for every method. Writes
results/tables/{split}_causal.csv (overall and by days since the last active day) and
results/tables/{split}_causal_shares.csv (how often future movies are targets, recommendations, hits).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from .baselines import ItemKNN, Popularity
from .batches import target_tokens
from .common import load_eval_cases, load_split, vocab_size
from .evaluation import PRIMARY, evaluate_topk, model_topk, paired_difference
from .train import load_model

GAPS = ([0, 30, 365, 10**6], ["1-30 days", "31-365 days", ">1 year"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", action="append", required=True, help="NAME=path/to/ckpt.pt (first = main)")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--results", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    train = load_split("train", args.processed)
    split, cases = load_split(args.split, args.processed), load_eval_cases(args.split, args.processed)
    targets = target_tokens(split, cases)
    V = vocab_size(args.processed)
    with open(os.path.join(args.results, "tables", "baseline_choice.json")) as f:
        choice = json.load(f)
    pop = Popularity(train, V, "all")
    first_day = pop.first_rated_day()  # first day any training user rated each movie

    topks = {"POP": pop.topk(split, cases, "static", first_day=first_day),
             "RECENT-POP": pop.topk(split, cases, choice["window"], first_day=first_day),
             "ITEM-KNN": ItemKNN(train, V, args.device).topk(split, cases, 256, first_day=first_day)}
    main_name = args.model[0].split("=", 1)[0]
    for spec in args.model:
        name, path = spec.split("=", 1)
        topks[name] = model_topk(load_model(path, args.device)[0], split, cases, device=args.device,
                                 first_day=first_day)

    per_case, rows = {}, []
    for name, topk in topks.items():
        per_case[name], s = evaluate_topk(topk, split, cases, name)
        rows.append(s)
    gap = pd.cut(cases.gap, GAPS[0], labels=GAPS[1], right=True)
    for s in rows:
        if s["method"] != "RECENT-POP":
            s["diff_vs_RECENT-POP"], s["diff_vs_RECENT-POP_ci95"] = paired_difference(
                per_case[s["method"]], per_case["RECENT-POP"])
        for g in GAPS[1]:
            sel = (gap == g).values
            s[f"recall@20 gap {g}"] = per_case[s["method"]][PRIMARY].values[sel].mean()
        print(f"{s['method']:<12} causal recall@20 {s[PRIMARY]:.4f} ± {s[PRIMARY + '_ci95']:.4f}  "
              + (f"vs RECENT-POP {s['diff_vs_RECENT-POP']:+.4f} ± {s['diff_vs_RECENT-POP_ci95']:.4f}  "
                 if "diff_vs_RECENT-POP" in s else "")
              + "  ".join(f"{g}: {s[f'recall@20 gap {g}']:.3f}" for g in GAPS[1]))
    pd.DataFrame(rows).to_csv(os.path.join(args.results, "tables", f"{args.split}_causal.csv"), index=False)

    # how often "future" movies are targets, and how often the unrestricted model recommends and hits them
    last = cases.last_day.values.astype(np.float64)
    fd = first_day.numpy()
    future_target = np.concatenate([fd[t] > d for t, d in zip(targets, last)])
    top20 = np.load(os.path.join(args.results, "per_case", f"{args.split}_{main_name}_top100.npy"))[:, :20]
    future_top = fd[top20] > last[:, None]
    hit = np.array([np.isin(row, t) for row, t in zip(top20, targets)])
    shares = {
        "future_share_of_targets": float(future_target.mean()),
        "cases_with_a_future_target": float(np.mean([(fd[t] > d).any() for t, d in zip(targets, last)])),
        "future_share_of_top20": float(future_top.mean()),
        "future_share_of_hits": float(future_top[hit].mean()),
        "cases_gap_over_30_days": float((cases.gap > 30).mean()),
    }
    pd.DataFrame([shares]).to_csv(os.path.join(args.results, "tables", f"{args.split}_causal_shares.csv"), index=False)
    print({k: round(v, 3) for k, v in shares.items()})


if __name__ == "__main__":
    main()
