"""Batches in Delphi's layout: left-padded (token 0, age -10000), sorted by day, tokens shifted by +1."""

import numpy as np
import torch

PAD_AGE = -10000.0


def windows(tokens, days, row_start, n_real, width, rng=None):
    """Rows [row_start, row_start + n_real) of each history, right-aligned in `width` columns.

    With rng, events that share a day are put in a new random order (ties are unordered).
    Returns int64 tokens and float32 ages, both (B, width).
    """
    rel = np.arange(width)[None, :] - (width - n_real)[:, None]
    real = rel >= 0
    rows = row_start[:, None] + np.maximum(rel, 0)
    tok = np.where(real, tokens[rows], 0)
    day = np.where(real, days[rows], 0)
    if rng is not None:
        key = np.where(real, day + rng.random(day.shape), -1.0)  # days are integers >= 0
        order = np.argsort(key, axis=1, kind="stable")
        tok = np.take_along_axis(tok, order, 1)
        day = np.take_along_axis(day, order, 1)
        real = np.take_along_axis(real, order, 1)
    age = np.where(real, day, PAD_AGE).astype(np.float32)
    return tok, age


class TrainBatches:
    """Random windows of block_size + 1 events from eligible histories (adapted from Delphi's get_batch)."""

    def __init__(self, split, block_size, user_fraction=1.0, seed=0):
        self.tokens = split.tokens
        self.days = split.days
        users = split.users[split.users.eligible]
        if user_fraction < 1.0:
            keep = np.random.default_rng(seed + 99).permutation(len(users))[:int(round(user_fraction * len(users)))]
            users = users.iloc[np.sort(keep)]
        self.start = users.start.values.astype(np.int64)
        self.length = users.length.values.astype(np.int64)
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)

    @property
    def n_users(self):
        return len(self.start)

    def get(self, batch_size, device):
        u = self.rng.integers(0, self.n_users, batch_size)
        width = self.block_size + 1
        n_real = np.minimum(self.length[u], width)
        offset = self.rng.integers(0, self.length[u] - n_real + 1)  # uniform window position
        tok, age = windows(self.tokens, self.days, self.start[u] + offset, n_real, width, self.rng)
        tok, age = torch.from_numpy(tok), torch.from_numpy(age)
        x, a, y, b = tok[:, :-1], age[:, :-1], tok[:, 1:], age[:, 1:]
        if device.startswith("cuda"):
            return [t.pin_memory().to(device, non_blocking=True) for t in (x, a, y, b)]
        return [t.to(device) for t in (x, a, y, b)]


def history_inputs(split, cases, k, block_size, control=None, seed=0):
    """Model inputs for evaluation cases: the last min(k, block_size) events before the target day.

    The stored order of same-day events (fixed random) is used as is. Controls:
      "shuffle_order": the visible movies are randomly permuted over the visible time stamps.
      "other_user":    the visible history is taken from another case whose prediction day is the
                       closest one not after this case's own (see partner_index).
    Returns numpy (tokens, ages) of shape (n_cases, width).
    """
    k = block_size if k is None else min(k, block_size)
    src = cases
    if control == "other_user":
        src = cases.iloc[partner_index(cases.last_day.values, seed)]
    n_vis = np.minimum(src.n_hist.values, k)
    width = int(n_vis.max())
    tok, age = windows(split.tokens, split.days, src.hist_end.values - n_vis, n_vis, width)
    if control == "shuffle_order":
        rng = np.random.default_rng(seed)
        for i in range(len(tok)):
            vis = slice(width - n_vis[i], width)
            tok[i, vis] = rng.permutation(tok[i, vis])
    elif control not in (None, "other_user"):
        raise ValueError(control)
    return tok, age


def partner_index(last_day, seed):
    """For every case, a different case whose prediction day is the closest one not after its own.

    Cases are sorted by prediction day (random order within a day) and each takes the case just
    before it, so the borrowed history ends before the case's own target day. The earliest case,
    which has no predecessor, takes the second one.
    """
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(last_day)), last_day))
    partner = np.empty(len(order), dtype=np.int64)
    partner[order[1:]] = order[:-1]
    partner[order[0]] = order[1]
    return partner


def history_tokens(split, cases):
    """All movies rated before the target day (the exclusion set), as a list of arrays."""
    tokens = split.tokens
    return [tokens[s:e] for s, e in zip(cases.start.values, cases.hist_end.values)]


def target_tokens(split, cases):
    tokens = split.tokens
    return [tokens[s:e] for s, e in zip(cases.hist_end.values, cases.target_end.values)]
