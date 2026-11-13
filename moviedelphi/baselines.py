"""Popularity baselines, fitted on training users only.

    python -m moviedelphi.baselines            # tune on validation, write results/tables/baselines_val.csv

POP            all-time number of training users who rated the movie (the brief's popularity baseline)
POP-to-date    training ratings on or before the prediction day (calendar-aware, uses no future)
RECENT-POP(W)  training ratings in the W days up to the prediction day; ties broken by POP-to-date

Each can count all ratings ("all") or only ratings made after a user's first active day
("returning"), which have the same kind as the evaluation targets. Already-rated movies are
excluded exactly as for the model.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from .batches import history_tokens
from .common import load_eval_cases, load_split, vocab_size
from .evaluation import PRIMARY, TOPK, evaluate_topk, exclude_future, exclude_history

WINDOWS = (1, 7, 30, 90, 365, None)  # None = everything up to the prediction day


class Popularity:

    def __init__(self, train, n_vocab, event_set="all", n_days=9200):
        tokens, days = train.tokens, train.days
        if event_set == "returning":
            first_day = np.repeat(train.data[train.users.start.values, 1], train.users.length.values)
            keep = days != first_day
            tokens, days = tokens[keep], days[keep]
        elif event_set != "all":
            raise ValueError(event_set)
        assert days.max() < n_days
        counts = np.bincount(days * n_vocab + tokens, minlength=n_days * n_vocab).reshape(n_days, n_vocab)
        self.cum = torch.from_numpy(counts.cumsum(0).astype(np.int32))  # ratings on or before each day
        self.total = counts.sum(0)
        self.n_events = int(len(tokens))
        self.n_vocab = n_vocab
        self.event_set = event_set

    def scores(self, last_day, window):
        """(n, V) float64 scores at the end of each prediction day; window=None means up to that day."""
        last_day = torch.from_numpy(np.array(last_day, dtype=np.int64))
        to_date = self.cum[last_day].double()
        if window is None:
            return to_date + (self.n_vocab - torch.arange(self.n_vocab)) / (self.n_vocab + 1.0)
        lo = last_day - window
        before = torch.where((lo >= 0)[:, None], self.cum[lo.clamp(min=0)].double(), torch.zeros_like(to_date))
        return (to_date - before) + to_date / (self.n_events + 1.0)

    def static_scores(self):
        """All-time counts (the plain POP baseline), ties broken by token order."""
        total = torch.from_numpy(self.total).double()
        return total + (self.n_vocab - torch.arange(self.n_vocab)) / (self.n_vocab + 1.0)

    def topk(self, split, cases, window="static", K=TOPK, batch_size=1024, first_day=None):
        hist = history_tokens(split, cases)
        out = np.empty((len(cases), K), dtype=np.int64)
        for i in range(0, len(cases), batch_size):
            c = cases.iloc[i:i + batch_size]
            if window == "static":
                s = self.static_scores()[None, :].repeat(len(c), 1)
            else:
                s = self.scores(c.last_day.values, window)
            s[:, :2] = float("-inf")  # reserved tokens
            exclude_history(s, hist[i:i + batch_size])
            if first_day is not None:
                exclude_future(s, first_day, c.last_day.values.astype(np.float64))
            out[i:i + batch_size] = s.topk(K, dim=1).indices.numpy()
        return out

    def first_rated_day(self):
        """(V,) float tensor: first day any training user (in the counted event set) rated each movie."""
        rated = self.cum > 0
        first = rated.int().argmax(0).double()
        first[~rated.any(0)] = float("inf")
        return first


def recent_log_counts(train, n_vocab, window=30, n_days=9200):
    """(n_days, V) float16 table: log(1 + training ratings of each movie in the `window` days up to
    and including each day), the same counts as RECENT-POP. Used as Delphi's popularity offset."""
    cum = Popularity(train, n_vocab, "all", n_days).cum
    before = torch.cat([torch.zeros(window, n_vocab, dtype=cum.dtype), cum[:-window]])
    return torch.log1p((cum - before).float()).half()


class ItemKNN:
    """Personalised non-neural reference: cosine similarity between movies over training users.

    sim(i, j) = (#training users who rated both) / sqrt(#rated i * #rated j). A case's score for a
    movie is the sum of its similarities to the movies in the visible history (the same last
    block_size events the model sees).
    """

    def __init__(self, train, n_vocab, device="cuda", chunk=16384):
        users = train.data[:, 0].astype(np.int64)
        tokens = torch.from_numpy(train.tokens)
        bounds = np.searchsorted(users, np.arange(0, users.max() + 1 + chunk, chunk))
        co = torch.zeros(n_vocab, n_vocab, device=device)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi == lo:
                continue
            u = torch.from_numpy(users[lo:hi] - users[lo])
            x = torch.zeros(int(u.max()) + 1, n_vocab, device=device)
            x[u.to(device), tokens[lo:hi].to(device)] = 1.0
            co += x.T @ x  # counts < 2**24, exact in float32
        norm = co.diagonal().sqrt().clamp(min=1.0)
        self.sim = co / norm[:, None] / norm[None, :]
        self.sim.fill_diagonal_(0.0)
        self.n_vocab = n_vocab
        self.device = device

    @torch.no_grad()
    def topk(self, split, cases, block_size, k=None, K=TOPK, batch_size=512, control=None, seed=0, first_day=None):
        from .batches import history_inputs
        tok, _ = history_inputs(split, cases, k, block_size, control, seed)
        hist = history_tokens(split, cases)
        tiebreak = (self.n_vocab - torch.arange(self.n_vocab, device=self.device)).double() * 1e-12
        out = np.empty((len(cases), K), dtype=np.int64)
        for i in range(0, len(cases), batch_size):
            t = torch.from_numpy(tok[i:i + batch_size]).to(self.device)
            h = torch.zeros(len(t), self.n_vocab, device=self.device)
            h.scatter_(1, t, 1.0)
            s = (h @ self.sim).double() + tiebreak
            s[:, :2] = float("-inf")
            exclude_history(s, hist[i:i + batch_size])
            if first_day is not None:
                exclude_future(s, first_day, cases.last_day.values[i:i + batch_size].astype(np.float64))
            out[i:i + batch_size] = s.topk(K, dim=1).indices.cpu().numpy()
        return out


def window_name(event_set, window):
    if window == "static":
        return f"POP[{event_set}]"
    if window is None:
        return f"POP-to-date[{event_set}]"
    return f"RECENT-POP[{event_set},W={window}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train = load_split("train", args.processed)
    split = load_split(args.split, args.processed)
    cases = load_eval_cases(args.split, args.processed)
    V = vocab_size(args.processed)
    rows = []
    for event_set in ("all", "returning"):
        pop = Popularity(train, V, event_set)
        for window in ("static",) + WINDOWS:
            _, s = evaluate_topk(pop.topk(split, cases, window), split, cases, window_name(event_set, window))
            s.update(event_set=event_set, window=str(window))
            rows.append(s)
            print(f"{s['method']:<36} recall@20 {s['recall@20']:.4f} ± {s['recall@20_ci95']:.4f}  "
                  f"recall@10 {s['recall@10']:.4f}  ndcg@20 {s['ndcg@20']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"baselines_{args.split}.csv"), index=False)
    if args.split == "val":
        best = df.loc[df[df.window != "static"][PRIMARY].idxmax()]
        choice = {"event_set": best.event_set, "window": None if best.window == "None" else int(best.window),
                  "method": best.method, "val_" + PRIMARY: float(best[PRIMARY])}
        with open(os.path.join(args.out, "baseline_choice.json"), "w") as f:
            json.dump(choice, f, indent=2)
        print("chosen calendar-time baseline on validation:", choice)


if __name__ == "__main__":
    main()
