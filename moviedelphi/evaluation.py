"""Next-active-day ranking evaluation.

Task, for one evaluation case (one held-out user):
  prediction time  the end of the user's last active day before the target day (last_day);
  input            the user's events on days before the target day (the model sees the most
                   recent block_size of them);
  target           the set of movies the user rated on their next active day (target_day);
  candidates       every movie in the vocabulary except those the user rated before the target
                   day (MovieLens users rate a movie at most once, so those can never be targets).
Metrics per case, then averaged over cases (users):
  recall@k = hits in top k / min(k, |target|)   (Liang et al., 2018), so 1.0 is attainable
  ndcg@k, hit@k (at least one target in the top k).
"""

import numpy as np
import pandas as pd
import torch

from .batches import history_inputs, history_tokens, target_tokens

KS = (10, 20, 50)
TOPK = 100
PRIMARY = "recall@20"


def exclude_history(scores, hist):
    """Set the scores of already-rated movies to -inf (in place). scores: (B, V) tensor."""
    rows = np.repeat(np.arange(len(hist)), [len(h) for h in hist])
    cols = np.concatenate(hist)
    scores[torch.from_numpy(rows).to(scores.device), torch.from_numpy(cols).to(scores.device)] = float("-inf")
    return scores


def exclude_future(scores, first_day, last_day):
    """Set to -inf every movie no training user had rated by the prediction day (in place).

    first_day: (V,) tensor, first day any training user rated each movie (inf if never);
    last_day: (B,) prediction days. Used by the causal-candidate check only.
    """
    future = first_day.to(scores.device)[None, :] > torch.as_tensor(last_day, device=scores.device)[:, None]
    scores[future] = float("-inf")
    return scores


@torch.no_grad()
def model_topk(model, split, cases, k=None, control=None, seed=0, batch_size=512, device="cuda", K=TOPK,
               first_day=None):
    """Top-K movie tokens for each case from the model's prediction after the last visible event.
    With first_day, movies first rated by training users after the prediction day are excluded."""
    model.eval()
    tok, age = history_inputs(split, cases, k, model.config.block_size, control, seed)
    hist = history_tokens(split, cases)
    out = np.empty((len(cases), K), dtype=np.int64)
    for i in range(0, len(cases), batch_size):
        t = torch.from_numpy(tok[i:i + batch_size]).to(device)
        a = torch.from_numpy(age[i:i + batch_size]).to(device)
        first = int((t > 0).any(0).nonzero()[0])  # drop columns that are padding for every row
        logits = model.predict_last(t[:, first:], a[:, first:])
        exclude_history(logits, hist[i:i + batch_size])
        if first_day is not None:
            exclude_future(logits, first_day, cases.last_day.values[i:i + batch_size].astype(np.float64))
        out[i:i + batch_size] = logits.topk(K, dim=1).indices.cpu().numpy()
    return out


def hits_matrix(topk, targets):
    """hits[i, r] is True if the r-th ranked movie for case i is in its target set."""
    m = int(max(topk.max(), max(t.max() for t in targets))) + 1
    codes = np.concatenate([i * m + t for i, t in enumerate(targets)])
    return np.isin(np.arange(len(targets))[:, None] * m + topk, codes)


def ranking_metrics(topk, targets, ks=KS):
    """Per-case metrics as a DataFrame (one row per case)."""
    hits = hits_matrix(topk, targets)
    n_t = np.array([len(t) for t in targets])
    disc = 1.0 / np.log2(np.arange(2, topk.shape[1] + 2))
    res = {}
    for k in ks:
        denom = np.minimum(k, n_t)
        res[f"recall@{k}"] = hits[:, :k].sum(1) / denom
        res[f"ndcg@{k}"] = (hits[:, :k] * disc[:k]).sum(1) / np.cumsum(disc)[denom - 1]
        res[f"hit@{k}"] = hits[:, :k].any(1).astype(float)
    return pd.DataFrame(res)


def recall_curve(topk, targets):
    """Mean recall@k for k = 1..K."""
    hits = hits_matrix(topk, targets)
    n_t = np.array([len(t) for t in targets])[:, None]
    k = np.arange(1, topk.shape[1] + 1)[None, :]
    return (hits.cumsum(1) / np.minimum(k, n_t)).mean(0)


def summarize(per_case, name=None):
    """Mean and normal-approximation 95% interval of each metric over cases."""
    n = len(per_case)
    out = {"method": name, "n_cases": n}
    for c in per_case.columns:
        m, se = per_case[c].mean(), per_case[c].std(ddof=1) / np.sqrt(n)
        out[c] = m
        out[c + "_ci95"] = 1.96 * se
    return out


def paired_difference(a, b, metric=PRIMARY):
    """Mean of a - b over the same cases with a 95% interval."""
    d = a[metric].values - b[metric].values
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))


def evaluate_topk(topk, split, cases, name=None):
    per_case = ranking_metrics(topk, target_tokens(split, cases))
    return per_case, summarize(per_case, name)
