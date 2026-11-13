"""Helpers for notebooks/demo.ipynb: inspect a saved Delphi model on held-out (test) histories.

Uses exactly the preprocessing of the experiments: day index from 1995-01-01, the training
vocabulary, the stored order of same-day ratings, the last block_size events as input, and
already-rated movies excluded from the candidates. RECENT-POP is shown next to the model.
"""

import json

import numpy as np
import pandas as pd
import torch

from .baselines import Popularity
from .common import day_to_date, load_split, load_vocab, vocab_size
from .train import load_model


class Explorer:

    def __init__(self, checkpoint, processed="data/processed", split="test", device="cpu",
                 baseline_choice="results/tables/baseline_choice.json"):
        torch.set_float32_matmul_precision("highest")
        self.model, ckpt = load_model(checkpoint, device)
        self.device = device
        self.block_size = self.model.config.block_size
        self.vocab = load_vocab(processed).set_index("token")
        self.split = load_split(split, processed)
        users = self.split.users
        self.users = users[users.eligible].reset_index(drop=True)
        with open(baseline_choice) as f:
            choice = json.load(f)
        self.window = choice["window"]
        train = load_split("train", processed)
        self.pop = Popularity(train, vocab_size(processed), choice["event_set"])
        self.first_day, self.last_day = int(train.days.min()), int(train.days.max())
        self.tokens, self.days = self.split.tokens, self.split.days

    # ----- data access ---------------------------------------------------------------------
    def user_days(self, i):
        """Active days of eligible user i (index into self.users) and the row where each starts."""
        u = self.users.iloc[i]
        d = self.days[u.start:u.start + u.length]
        first = np.r_[0, np.flatnonzero(np.diff(d)) + 1]
        return d[first], u.start + first, u.start + u.length

    def movies(self, tokens, days=None, extra=None):
        v = self.vocab.loc[tokens]
        df = pd.DataFrame({"title": v.display_title.values, "genres": v.genres.fillna("—").values})
        if days is not None:
            df.insert(0, "date", day_to_date(days).astype(str))
        if extra is not None:
            for k, col in extra.items():
                df[k] = col
        return df

    # ----- predictions ---------------------------------------------------------------------
    def rank_model(self, tokens, days, k):
        n = min(len(tokens), self.block_size)
        t = torch.tensor(tokens[-n:], dtype=torch.long, device=self.device)[None]
        a = torch.tensor(days[-n:], dtype=torch.float32, device=self.device)[None]
        logits = self.model.predict_last(t, a)[0]
        logits[torch.tensor(tokens, device=self.device)] = float("-inf")  # already rated
        p = torch.softmax(logits, 0)
        top = torch.topk(p, k)
        return top.indices.cpu().numpy(), top.values.cpu().numpy()

    def rank_baseline(self, tokens, last_day, k):
        s = self.pop.scores(np.array([last_day]), self.window)[0]
        s[:2] = float("-inf")
        s[torch.from_numpy(np.asarray(tokens))] = float("-inf")
        return torch.topk(s, k).indices.numpy()

    def predict_next_day(self, i, n_days, k=20):
        """Reveal the first n_days active days of user i and rank movies for the next active day."""
        active, starts, end = self.user_days(i)
        if not 1 <= n_days < len(active):
            raise ValueError(f"user {i} has {len(active)} active days: choose n_days in 1..{len(active) - 1}")
        u_start = starts[0]
        hist_end, target_end = starts[n_days], (starts[n_days + 1] if n_days + 1 < len(active) else end)
        h_tok, h_day = self.tokens[u_start:hist_end], self.days[u_start:hist_end]
        target = set(self.tokens[hist_end:target_end].tolist())
        m_top, m_p = self.rank_model(h_tok, h_day, k)
        b_top = self.rank_baseline(h_tok, h_day[-1], k)
        denom = min(k, len(target))
        return {
            "history": self.movies(h_tok, h_day),
            "last_day": str(day_to_date(h_day[-1])), "target_day": str(day_to_date(active[n_days])),
            "target": self.movies(np.array(sorted(target))),
            "model": self.movies(m_top, extra={"p": m_p.round(4), "hit": [t in target for t in m_top]}),
            "baseline": self.movies(b_top, extra={"hit": [t in target for t in b_top]}),
            f"model_recall@{k}": sum(t in target for t in m_top) / denom,
            f"baseline_recall@{k}": sum(t in target for t in b_top) / denom,
        }

    def timeline(self, i, k=20):
        """Recall@k of the model and RECENT-POP for every day-to-next-day step of user i."""
        active, _, _ = self.user_days(i)
        rows = []
        for n in range(1, len(active)):
            r = self.predict_next_day(i, n, k)
            rows.append({"revealed days": n, "events seen": len(r["history"]), "next day": r["target_day"],
                         "movies that day": len(r["target"]), f"Delphi recall@{k}": r[f"model_recall@{k}"],
                         f"RECENT-POP recall@{k}": r[f"baseline_recall@{k}"]})
        return pd.DataFrame(rows)

    # ----- your own history ----------------------------------------------------------------
    def find(self, text):
        """Vocabulary titles containing `text` (case-insensitive)."""
        hit = self.vocab[self.vocab.display_title.str.contains(text, case=False, regex=False)]
        return hit[["display_title", "genres", "train_count"]].sort_values("train_count", ascending=False)

    def predict_for_titles(self, titles, date, k=20):
        """Rank movies for the next active day after rating `titles` on `date` (YYYY-MM-DD)."""
        index = pd.Series(self.vocab.index.values, index=self.vocab.display_title.values)
        missing = [t for t in titles if t not in index]
        if missing:
            raise KeyError(f"not in the vocabulary (use find()): {missing}")
        tokens = index.loc[titles].values.astype(np.int64)
        day = int((np.datetime64(date) - np.datetime64("1995-01-01")).astype(int))
        if not self.first_day <= day <= self.last_day:
            day = min(max(day, self.first_day), self.last_day)
            print(f"note: the data covers {day_to_date(self.first_day)} to {day_to_date(self.last_day)}; "
                  f"using {day_to_date(day)}")
        days = np.full(len(tokens), day)
        m_top, m_p = self.rank_model(tokens, days, k)
        b_top = self.rank_baseline(tokens, day, k)
        return (self.movies(m_top, extra={"p": m_p.round(4)}), self.movies(b_top))
