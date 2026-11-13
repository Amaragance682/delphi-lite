"""Checks for the parts of the pipeline that could silently invalidate the results.

    python -m tests.test_pipeline        (also runs under pytest)

The data tests need data/processed (python -m moviedelphi.prepare_data) and are skipped otherwise.
"""

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from moviedelphi.batches import PAD_AGE, history_inputs, history_tokens, partner_index, target_tokens, windows
from moviedelphi.common import Split, load_eval_cases, load_split, load_vocab
from moviedelphi.evaluation import ranking_metrics
from moviedelphi.model import Delphi, DelphiConfig, next_day_pairs, source_positions

PROCESSED = "data/processed"
HAVE_DATA = os.path.exists(os.path.join(PROCESSED, "summary.json"))


def hand_batch():
    """Two padding positions, two events on day 10, two on day 12, one on day 15."""
    tok = torch.tensor([[0, 0, 5, 6, 7, 8, 9]])
    age = torch.tensor([[PAD_AGE, PAD_AGE, 10.0, 10.0, 12.0, 12.0, 15.0]])
    return tok[:, :-1], age[:, :-1], tok[:, 1:], age[:, 1:]


def small_model(vocab=50):
    torch.manual_seed(0)
    return Delphi(DelphiConfig(block_size=16, vocab_size=vocab, n_layer=2, n_head=2, n_embd=32)).eval()


def test_source_positions_by_hand():
    x, a, y, b = hand_batch()
    src, valid = source_positions(x, a, y, b)
    # targets 1-2 are the first day in the window (no earlier day); 3-4 (day 12) come from position 3,
    # the last day-10 event; target 5 (day 15) comes from position 5
    assert valid.tolist() == [[False, False, False, True, True, True]]
    assert src[valid].tolist() == [3, 3, 5]


def test_prediction_ignores_same_day_events():
    model = small_model()
    x, a, y, b = hand_batch()
    base, _, valid = model.target_logits(x, a, y, b)
    x2 = x.clone()
    x2[0, 4] = 20  # another movie on day 12, the target day of targets 3 and 4
    same_day, _, _ = model.target_logits(x2, a, y, b)
    assert torch.allclose(base[:2], same_day[:2]), "a same-day event changed a prediction"
    x3 = x.clone()
    x3[0, 2] = 20  # a day-10 movie: earlier history must matter
    earlier, _, _ = model.target_logits(x3, a, y, b)
    assert not torch.allclose(base[:2], earlier[:2])


def delphi_mask_ties_logits(model, idx, age, targets_age):
    """Logits at every position with Delphi's original mask_ties attention mask (Delphi model.py,
    lines 225-231) instead of reading predictions from the last earlier-day position."""
    b, t = idx.shape
    diag = torch.diag(torch.ones(t))
    tril = torch.tril(torch.ones(t, t))[None, None] > 0
    attn_mask = (idx > 0).view(b, 1, 1, t) * (idx > 0).view(b, 1, t, 1)
    attn_mask *= tril
    attn_mask *= age.view(b, 1, 1, t) != targets_age.view(b, 1, t, 1)  # hide keys on the target's day
    attn_mask += (attn_mask.sum(-1, keepdim=True) == 0) * diag > 0
    attn_mask = attn_mask + (idx == 0).view(b, 1, 1, t) * diag > 0
    attn_mask *= tril
    x = model.transformer.wte(idx) + model.transformer.wae(age.unsqueeze(-1))
    for block in model.transformer.h:
        x = block(x, attn_mask)
    return model.lm_head(model.transformer.ln_f(x))


def test_original_delphi_mask_ties_leaks_current_token():
    """Delphi's mask_ties hides same-day keys, but the current token still reaches its own output."""
    model = small_model()
    x, a, y, b = hand_batch()
    x2 = x.clone()
    x2[0, 4] = 20  # same day (12) as the target of position 4
    with torch.no_grad():
        l1 = delphi_mask_ties_logits(model, x, a, b)[0, 4]
        l2 = delphi_mask_ties_logits(model, x2, a, b)[0, 4]
    assert not torch.allclose(l1, l2), "expected the original tie masking to let the same-day token through"


def test_attention_matches_delphi_explicit_softmax():
    model = small_model()
    attn = model.transformer.h[0].attn
    x = torch.randn(2, 6, 32)
    real = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    t = 6
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool))[None, None]
    eye = torch.eye(t, dtype=torch.bool)[None, None]
    mask = (real[:, None, None, :] & real[:, None, :, None] & causal) | (~real[:, None, None, :] & eye)
    ours = attn(x, mask)
    q, k, v = attn.c_attn(x).split(32, dim=2)
    q, k, v = [z.view(2, t, 2, 16).transpose(1, 2) for z in (q, k, v)]
    att = (q @ k.transpose(-2, -1)) / math.sqrt(16)
    att = F.softmax(att.masked_fill(mask == 0, float("-inf")), dim=-1)
    ref = attn.c_proj((att @ v).transpose(1, 2).reshape(2, t, 32))
    assert torch.allclose(ours, ref, atol=1e-5)


def test_basket_weighting_by_hand():
    """Targets 3 and 4 share one source (a 2-movie day), target 5 has its own: weights 1/4, 1/4, 1/2."""
    model = small_model()
    x, a, y, b = hand_batch()
    logits, src, valid = model.target_logits(x, a, y, b)
    ce = F.cross_entropy(logits, y[valid], reduction="none")
    plain = model(x, a, y, b)["loss_ce"]
    assert torch.allclose(plain, ce.mean())
    model.config.basket_weighting = True
    weighted = model(x, a, y, b)["loss_ce"]
    assert torch.allclose(weighted, 0.25 * ce[0] + 0.25 * ce[1] + 0.5 * ce[2])


def test_every_position_pairs_by_hand():
    tok = torch.tensor([[0, 0, 5, 6, 7, 8, 9]])
    age = torch.tensor([[PAD_AGE, PAD_AGE, 10.0, 10.0, 12.0, 12.0, 15.0]])
    pair, has_next, next_day = next_day_pairs(tok, age)
    assert has_next.tolist() == [[False, False, True, True, True, True, False]]
    assert pair[0, 2].nonzero().flatten().tolist() == [4, 5] and pair[0, 3].nonzero().flatten().tolist() == [4, 5]
    assert pair[0, 4].nonzero().flatten().tolist() == [6] and pair[0, 5].nonzero().flatten().tolist() == [6]
    assert pair[0, 0].sum() == 0 and pair[0, 6].sum() == 0
    # a position only sees days up to its own, and its targets are on a strictly later day
    for i in range(7):
        if has_next[0, i]:
            assert next_day[0, i] > age[0, :i + 1].max()


def test_every_position_loss_by_hand():
    """Day 10 has three events, day 12 one, day 15 one."""
    model = small_model()
    model.config.objective = "every_position"
    tok = torch.tensor([[0, 5, 6, 7, 8, 9]])
    age = torch.tensor([[PAD_AGE, 10.0, 10.0, 10.0, 12.0, 15.0]])
    x, a, y, b = tok[:, :-1], age[:, :-1], tok[:, 1:], age[:, 1:]
    logp = F.log_softmax(model.logits_from_hidden(model.hidden(tok, age)), -1)[0]
    ce = torch.stack([-logp[1, 8], -logp[2, 8], -logp[3, 8], -logp[4, 9]])
    assert torch.allclose(model(x, a, y, b)["loss_ce"], ce.mean())
    model.config.basket_weighting = True  # the three day-10 positions share one transition
    assert torch.allclose(model(x, a, y, b)["loss_ce"], (ce[:3].sum() / 3 + ce[3]) / 2)


def test_popularity_offset_uses_prediction_day():
    """The offset added to the log-rates is the row of the day the prediction is made on (the source
    position's day), never the target's day."""
    plain = small_model()
    torch.manual_seed(0)
    model = Delphi(DelphiConfig(block_size=16, vocab_size=50, n_layer=2, n_head=2, n_embd=32,
                                popularity_offset=True)).eval()
    state = plain.state_dict()
    state["pop_beta"] = torch.tensor(0.5)
    model.load_state_dict(state)
    table = torch.zeros(20, 50)
    table[10, 9], table[12, 7], table[15, 3] = 3.0, 2.0, 4.0
    model.set_popularity(table)
    x, a, y, b = hand_batch()
    with torch.no_grad():
        assert torch.allclose(model.predict_last(x, a), plain.predict_last(x, a) + 0.5 * table[12])
        lo, _, _ = model.target_logits(x, a, y, b)
        lp, _, _ = plain.target_logits(x, a, y, b)
    # targets on day 12 are predicted from a day-10 position, the day-15 target from a day-12 position
    assert torch.allclose(lo, lp + 0.5 * table[[10, 10, 12]])


def test_reserved_tokens_never_predicted():
    model = small_model()
    x, a, _, _ = hand_batch()
    logits = model.predict_last(x, a)
    assert torch.isinf(logits[:, :2]).all() and torch.isfinite(logits[:, 2:]).all()


def test_windows_right_aligned_and_shuffled_within_days():
    tokens = np.arange(100, 120)
    days = np.repeat([3, 5, 9, 11], 5)
    rng = np.random.default_rng(0)
    tok, age = windows(tokens, days, np.array([0, 12]), np.array([20, 4]), 22, rng)
    assert (tok[0, :2] == 0).all() and (age[0, :2] == PAD_AGE).all()
    assert (tok[1, :18] == 0).all()
    assert (np.diff(age[0, 2:]) >= 0).all(), "days must stay sorted"
    for d in (3, 5, 9, 11):  # same movies on each day, possibly in a new order
        assert sorted(tok[0][age[0] == d]) == sorted(tokens[days == d])
    assert sorted(tok[1, 18:]) == [112, 113, 114, 115]


def test_partner_index_never_later():
    for n in (2, 3, 7, 10):
        days = np.arange(n) % 4
        p = partner_index(days, seed=0)
        assert (p != np.arange(n)).all()
        assert (days[p] <= days).sum() >= n - 1  # every case but the earliest borrows an earlier history


def test_ranking_metrics_by_hand():
    topk = np.array([[5, 6, 7], [9, 8, 7]])
    targets = [np.array([6, 9]), np.array([9])]
    m = ranking_metrics(topk, targets, ks=(1, 2, 3))
    assert np.allclose(m["recall@1"], [0.0, 1.0])
    assert np.allclose(m["recall@2"], [0.5, 1.0])
    assert np.allclose(m["recall@3"], [0.5, 1.0])
    assert np.allclose(m["ndcg@2"], [(1 / math.log2(3)) / (1 + 1 / math.log2(3)), 1.0])
    assert np.allclose(m["hit@1"], [0.0, 1.0])


def test_popularity_window_counts():
    from moviedelphi.baselines import Popularity
    data = np.array([[0, 0, 1], [0, 2, 2], [1, 2, 2], [1, 3, 3], [1, 5, 1]], dtype=np.uint32)  # raw tokens
    users = pd.DataFrame({"user": [0, 1], "userId": [10, 11], "start": [0, 2], "length": [2, 3],
                          "n_days": [2, 3], "eligible": [True, True]})
    pop = Popularity(Split("toy", data, users), n_vocab=5, n_days=8)
    s = pop.scores(np.array([3]), window=2)  # days 2-3: model token 3 twice, token 4 once
    assert s[0, 3].floor() == 2 and s[0, 4].floor() == 1 and s[0, 2].floor() == 0
    s_all = pop.scores(np.array([3]), window=None)  # days 0-3
    assert s_all[0, 2].floor() == 1 and s_all[0, 3].floor() == 2
    ret = Popularity(Split("toy", data, users), n_vocab=5, event_set="returning", n_days=8)
    assert ret.total.tolist() == [0, 0, 1, 1, 1]  # first-day events (days 0 and 2) dropped


def test_causal_candidates_exclude_movies_first_rated_later():
    from moviedelphi.baselines import Popularity
    from moviedelphi.evaluation import exclude_future
    data = np.array([[0, 0, 1], [0, 2, 2], [1, 2, 2], [1, 3, 3], [1, 5, 1]], dtype=np.uint32)  # raw tokens
    users = pd.DataFrame({"user": [0, 1], "userId": [10, 11], "start": [0, 2], "length": [2, 3],
                          "n_days": [2, 3], "eligible": [True, True]})
    first = Popularity(Split("toy", data, users), n_vocab=5, n_days=8).first_rated_day()
    assert first.tolist() == [float("inf"), float("inf"), 0.0, 2.0, 3.0]
    s = torch.zeros(2, 5)
    exclude_future(s, first, np.array([2.0, 3.0]))
    assert torch.isinf(s[0]).tolist() == [True, True, False, False, True]  # model token 4 first rated on day 3
    assert torch.isinf(s[1]).tolist() == [True, True, False, False, False]


def test_splits_disjoint_and_vocab_from_training_users():
    if not HAVE_DATA:
        print("  skipped: no processed data")
        return
    ids = {s: set(load_split(s, PROCESSED).users.userId) for s in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    vocab = load_vocab(PROCESSED)
    train = load_split("train", PROCESSED)
    counts = np.bincount(train.tokens, minlength=len(vocab) + 2)
    assert (counts[vocab.token.values] == vocab.train_count.values).all()
    assert (np.diff(vocab.train_count.values) <= 0).all()


def test_eval_cases_use_strictly_earlier_days():
    if not HAVE_DATA:
        print("  skipped: no processed data")
        return
    for name in ("val", "test"):
        split, cases = load_split(name, PROCESSED), load_eval_cases(name, PROCESSED)
        days = split.days
        tok, age = history_inputs(split, cases, None, 256)
        real = tok > 0
        assert (np.where(real, age, -np.inf).max(1) == cases.last_day.values).all()
        assert (np.where(real, age, -np.inf) < cases.target_day.values[:, None]).all()
        assert (real.sum(1) == np.minimum(cases.n_hist.values, 256)).all()
        hist, targ = history_tokens(split, cases), target_tokens(split, cases)
        for i in range(len(cases)):
            s, h, t = cases.start.values[i], cases.hist_end.values[i], cases.target_end.values[i]
            assert (days[s:h] < cases.target_day.values[i]).all() and (days[h:t] == cases.target_day.values[i]).all()
            assert not np.isin(targ[i], hist[i]).any()
        assert cases.user.is_unique


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print("ok ", name)
    print(f"{len(tests)} tests passed")
