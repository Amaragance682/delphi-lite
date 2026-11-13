"""Evaluate the baselines and trained models on the validation or the test cases.

    python -m moviedelphi.evaluate --split val --model NAME=runs/NAME/ckpt.pt [...] [--main NAME --analyses]
    python -m moviedelphi.evaluate --split test ...      # once, for the final numbers

Outputs (results/):
  tables/{split}_main{tag}.csv            one row per method: metrics with 95% intervals and the
                                          paired recall@20 difference against RECENT-POP and POP
  tables/{split}_recall_curve{tag}.csv    mean recall@k, k = 1..100, per method
  per_case/{split}_{method}.csv           per-case metrics with case features, for error analysis
  per_case/{split}_{method}_top100.npy    the 100 highest-ranked movie tokens of every case
  with --analyses, for the --main model (and ITEM-KNN where it applies):
  tables/{split}_history_length{tag}.csv  recall@20 when only the last k events are visible
  tables/{split}_controls{tag}.csv        shuffled order and another user's history
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .baselines import ItemKNN, Popularity, window_name
from .batches import target_tokens
from .common import load_eval_cases, load_split, vocab_size
from .evaluation import PRIMARY, evaluate_topk, model_topk, paired_difference, recall_curve
from .train import load_model

HISTORY_KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CASE_COLUMNS = ["user", "userId", "target_day", "last_day", "prior_days", "n_hist", "n_target", "gap"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--model", action="append", default=[], help="NAME=path/to/ckpt.pt (repeatable)")
    ap.add_argument("--main", default=None, help="model used for the history-length and control analyses")
    ap.add_argument("--analyses", action="store_true")
    ap.add_argument("--no_baselines", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "per_case"), exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False  # full float32 ranking, same as on a CPU
    torch.set_float32_matmul_precision("highest")

    train = load_split("train", args.processed)
    split = load_split(args.split, args.processed)
    cases = load_eval_cases(args.split, args.processed)
    targets = target_tokens(split, cases)
    V = vocab_size(args.processed)
    models = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        models[name] = load_model(path, args.device)[0]
    block_size = models[args.main].config.block_size if args.main else 256
    print(f"{args.split}: {len(cases)} cases, models: {list(models)}")

    topks, seconds = {}, {}
    if not args.no_baselines:
        with open(os.path.join(args.out, "tables", "baseline_choice.json")) as f:
            choice = json.load(f)  # chosen on validation by moviedelphi.baselines
        t = time.time()
        pop_all = Popularity(train, V, "all")
        recent = pop_all if choice["event_set"] == "all" else Popularity(train, V, choice["event_set"])
        topks["POP"] = pop_all.topk(split, cases, "static")
        topks["RECENT-POP"] = recent.topk(split, cases, choice["window"])
        seconds["POP+RECENT-POP"] = time.time() - t
        print("RECENT-POP =", window_name(choice["event_set"], choice["window"]))
        t = time.time()
        knn = ItemKNN(train, V, args.device)
        topks["ITEM-KNN"] = knn.topk(split, cases, block_size)
        seconds["ITEM-KNN"] = time.time() - t
    for name, model in models.items():
        t = time.time()
        topks[name] = model_topk(model, split, cases, device=args.device)
        seconds[name] = time.time() - t

    rows, per_case = [], {}
    for name, topk in topks.items():
        per_case[name], s = evaluate_topk(topk, split, cases, name)
        rows.append(s)
        pd.concat([cases[CASE_COLUMNS], per_case[name]], axis=1).to_csv(
            os.path.join(args.out, "per_case", f"{args.split}_{name}.csv"), index=False)
        np.save(os.path.join(args.out, "per_case", f"{args.split}_{name}_top100.npy"), topk.astype(np.int32))
    for s in rows:  # paired differences once every method has been scored
        name = s["method"]
        for ref in ("RECENT-POP", "POP"):
            if ref in per_case and ref != name:
                s[f"diff_vs_{ref}"], s[f"diff_vs_{ref}_ci95"] = paired_difference(per_case[name], per_case[ref])
        print(f"{name:<28} recall@20 {s[PRIMARY]:.4f} ± {s[PRIMARY + '_ci95']:.4f}  "
              f"recall@10 {s['recall@10']:.4f}  recall@50 {s['recall@50']:.4f}  ndcg@20 {s['ndcg@20']:.4f}  "
              + (f"vs RECENT-POP {s['diff_vs_RECENT-POP']:+.4f} ± {s['diff_vs_RECENT-POP_ci95']:.4f}"
                 if "diff_vs_RECENT-POP" in s else ""))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "tables", f"{args.split}_main{args.tag}.csv"), index=False)
    curves = pd.DataFrame({name: recall_curve(topk, targets) for name, topk in topks.items()})
    curves.insert(0, "k", np.arange(1, len(curves) + 1))
    curves.to_csv(os.path.join(args.out, "tables", f"{args.split}_recall_curve{args.tag}.csv"), index=False)
    print("seconds:", {k: round(v, 1) for k, v in seconds.items()})

    if args.analyses:
        model = models[args.main]
        hist_rows = []
        for k in HISTORY_KS:
            for name, fn in ((args.main, lambda: model_topk(model, split, cases, k=k, device=args.device)),
                             ("ITEM-KNN", lambda: knn.topk(split, cases, block_size, k=k))):
                if name == "ITEM-KNN" and args.no_baselines:
                    continue
                _, s = evaluate_topk(fn(), split, cases, name)
                hist_rows.append({"k": k, **s})
                print(f"history k={k:<4} {name:<12} recall@20 {s[PRIMARY]:.4f} ± {s[PRIMARY + '_ci95']:.4f}")
        pd.DataFrame(hist_rows).to_csv(
            os.path.join(args.out, "tables", f"{args.split}_history_length{args.tag}.csv"), index=False)

        ctrl_rows = []
        real = per_case[args.main]
        for control in (None, "shuffle_order", "other_user"):
            pc, s = evaluate_topk(model_topk(model, split, cases, control=control, seed=0, device=args.device),
                                  split, cases, args.main)
            d, ci = paired_difference(pc, real)
            ctrl_rows.append({"control": control or "real history", **s, "diff_vs_real": d, "diff_vs_real_ci95": ci})
            print(f"control {str(control):<14} recall@20 {s[PRIMARY]:.4f} ± {s[PRIMARY + '_ci95']:.4f}  "
                  f"diff vs real {d:+.4f} ± {ci:.4f}")
        pd.DataFrame(ctrl_rows).to_csv(
            os.path.join(args.out, "tables", f"{args.split}_controls{args.tag}.csv"), index=False)


if __name__ == "__main__":
    main()
