"""Turn MovieLens-25M ratings into Delphi-format event histories.

    python -m moviedelphi.prepare_data --raw data/raw/ml-25m --out data/processed

Decisions (every choice that looks at the data uses training users only):
 1. Users are split first: 80% train / 10% validation / 10% test (seed SPLIT_SEED).
 2. Vocabulary = the N movies rated by the most training users (ties broken by lower movieId).
    Validation and test use the same mapping; ratings of other movies are dropped.
 3. Every rating is an event, whatever its star value. The time unit is the UTC calendar day,
    counted from 1995-01-01. Ratings on the same day count as simultaneous: the time of day is
    discarded and same-day rows are stored in a fixed random order (seed ORDER_SEED).
 4. A history is eligible if it has at least 2 active days (days with an in-vocabulary rating).
    With fewer there is no earlier day to predict from.
 5. Evaluation cases (val/test): for every eligible user one target day T, drawn uniformly from
    their active days after the first (seed TARGET_SEED). History = all events on days before T,
    target = the set of movies rated on day T.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .common import DAY0_UNIX_DAYS, ORDER_SEED, SECONDS_PER_DAY, SPLIT_SEED, SPLITS, TARGET_SEED


def split_users(user_ids, seed=SPLIT_SEED, fractions=(0.8, 0.1, 0.1)):
    perm = np.random.default_rng(seed).permutation(np.sort(user_ids))
    n_train = int(round(fractions[0] * len(perm)))
    n_val = int(round(fractions[1] * len(perm)))
    parts = {"train": perm[:n_train], "val": perm[n_train:n_train + n_val], "test": perm[n_train + n_val:]}
    return {k: np.sort(v) for k, v in parts.items()}


def build_vocab(train_ratings, movies, n_movies):
    counts = train_ratings.groupby("movieId").size().rename("train_count").reset_index()
    counts = counts.sort_values(["train_count", "movieId"], ascending=[False, True]).head(n_movies)
    vocab = counts.merge(movies, on="movieId", how="left", validate="one_to_one")
    assert vocab.title.notna().all(), "every rated movie should be in movies.csv"
    vocab.insert(0, "raw_token", np.arange(1, len(vocab) + 1))
    vocab.insert(0, "token", vocab.raw_token + 1)
    vocab["year"] = vocab.title.str.strip().str.extract(r"\((\d{4})\)$")[0].astype("Int64")
    vocab["genres"] = vocab.genres.where(vocab.genres != "(no genres listed)")
    duplicated = vocab.title.duplicated(keep=False)
    vocab["display_title"] = vocab.title.where(~duplicated, vocab.title + " [movieId " + vocab.movieId.astype(str) + "]")
    return vocab


def to_events(ratings, vocab, seed):
    """Ratings of one split -> uint32 rows (user, day, raw_token) and a per-user table."""
    raw_token = ratings.movieId.map(pd.Series(vocab.raw_token.values, index=vocab.movieId.values))
    r = ratings[raw_token.notna()]
    raw_token = raw_token[raw_token.notna()].astype(np.int64).values
    day = (r.timestamp.values // SECONDS_PER_DAY) - DAY0_UNIX_DAYS
    assert day.min() >= 0
    user_ids = np.sort(r.userId.unique())
    user = np.searchsorted(user_ids, r.userId.values)
    order = np.lexsort((np.random.default_rng(seed).random(len(r)), day, user))  # last key sorts first
    data = np.stack([user[order], day[order], raw_token[order]], axis=1).astype(np.uint32)

    starts = np.r_[0, np.flatnonzero(np.diff(data[:, 0].astype(np.int64))) + 1]
    lengths = np.diff(np.r_[starts, len(data)])
    new_day = np.r_[True, (np.diff(data[:, 0].astype(np.int64)) != 0) | (np.diff(data[:, 1].astype(np.int64)) != 0)]
    n_days = np.bincount(data[new_day, 0].astype(np.int64), minlength=len(user_ids))
    users = pd.DataFrame({"user": np.arange(len(user_ids)), "userId": user_ids, "start": starts,
                          "length": lengths, "n_days": n_days})
    users["eligible"] = users.n_days >= 2
    return data, users


def make_eval_cases(data, users, seed):
    """One evaluation case per eligible user: a random active day after the first is the target."""
    user = data[:, 0].astype(np.int64)
    day = data[:, 1].astype(np.int64)
    day_start = np.flatnonzero(np.r_[True, (np.diff(user) != 0) | (np.diff(day) != 0)])  # first row of each (user, day)
    day_end = np.r_[day_start[1:], len(data)]
    first_group = np.searchsorted(user[day_start], users.user.values)

    elig = users[users.eligible].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    k = rng.integers(1, elig.n_days.values)  # index of the target day among the user's active days, >= 1
    g = first_group[elig.user.values] + k
    cases = pd.DataFrame({
        "user": elig.user.values,
        "userId": elig.userId.values,
        "start": elig.start.values,             # history = rows [start, hist_end)
        "hist_end": day_start[g],
        "target_end": day_end[g],               # target = rows [hist_end, target_end), all on target_day
        "target_day": day[day_start[g]],
        "last_day": day[day_start[g] - 1],      # last day of history = prediction time (end of that day)
        "prior_days": k,                        # active days in the history
    })
    cases["n_hist"] = cases.hist_end - cases.start
    cases["n_target"] = cases.target_end - cases.hist_end
    cases["gap"] = cases.target_day - cases.last_day

    # sanity checks: target rows share one day and belong to the user; history ends strictly earlier
    for row in cases.itertuples():
        assert (user[row.start:row.target_end] == row.user).all()
        assert (day[row.hist_end:row.target_end] == row.target_day).all()
        assert day[row.hist_end - 1] < row.target_day and row.n_hist > 0
        assert not np.isin(data[row.hist_end:row.target_end, 2], data[row.start:row.hist_end, 2]).any()
    return cases


def quantiles(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)):
    return {str(q): float(np.quantile(x, q)) for q in qs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/ml-25m")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--n_movies", type=int, default=10000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    ratings = pd.read_csv(os.path.join(args.raw, "ratings.csv"), engine="pyarrow",
                          usecols=["userId", "movieId", "timestamp"])
    movies = pd.read_csv(os.path.join(args.raw, "movies.csv"))
    assert not ratings.duplicated(["userId", "movieId"]).any(), "a user should rate a movie at most once"

    split_ids = split_users(ratings.userId.unique())
    split_of = pd.Series(np.concatenate([np.full(len(v), k) for k, v in split_ids.items()]),
                         index=np.concatenate(list(split_ids.values())))
    ratings["split"] = ratings.userId.map(split_of)

    vocab = build_vocab(ratings[ratings.split == "train"], movies, args.n_movies)
    vocab.to_csv(os.path.join(args.out, "vocab.csv"), index=False)

    summary = {
        "raw": {
            "ratings": int(len(ratings)), "users": int(ratings.userId.nunique()),
            "movies_rated": int(ratings.movieId.nunique()), "movies_in_movies_csv": int(len(movies)),
            "first_rating_utc": str(pd.to_datetime(ratings.timestamp.min(), unit="s")),
            "last_rating_utc": str(pd.to_datetime(ratings.timestamp.max(), unit="s")),
        },
        "vocab": {"movies": int(len(vocab)), "model_vocab_size": int(len(vocab) + 2),
                  "min_train_count": int(vocab.train_count.min())},
        "splits": {},
    }

    for name in SPLITS:
        r = ratings[ratings.split == name]
        data, users = to_events(r, vocab, ORDER_SEED + SPLITS.index(name))
        data.tofile(os.path.join(args.out, f"{name}.bin"))
        users.to_csv(os.path.join(args.out, f"users_{name}.csv"), index=False)
        first_day = np.repeat(data[users.start.values, 1], users.length.values)
        s = {
            "users": int(len(split_ids[name])), "ratings": int(len(r)),
            "in_vocab_share": float(len(data) / len(r)),
            "users_with_events": int(len(users)), "events": int(len(data)),
            "eligible_users": int(users.eligible.sum()),
            "eligible_share": float(users.eligible.mean()),
            "events_of_eligible_users": int(users.length[users.eligible].sum()),
            "share_events_on_first_day": float((data[:, 1] == first_day).mean()),
            "events_per_user": quantiles(users.length),
            "active_days_per_user": quantiles(users.n_days),
        }
        if name != "train":
            cases = make_eval_cases(data, users, TARGET_SEED + SPLITS.index(name))
            cases.to_csv(os.path.join(args.out, f"eval_{name}.csv"), index=False)
            s["eval_cases"] = {
                "n": int(len(cases)),
                "n_hist": quantiles(cases.n_hist), "n_target": quantiles(cases.n_target),
                "gap_days": quantiles(cases.gap),
                "share_gap_1_day": float((cases.gap == 1).mean()),
                "share_target_is_second_active_day": float((cases.prior_days == 1).mean()),
            }
        summary["splits"][name] = s
        print(name, json.dumps(s)[:300], "...")

    # metadata checks reported in the hand-in
    years = movies.title.str.strip().str.extract(r"\((\d{4})\)$")[0].astype(float)
    rated_year = pd.to_datetime(ratings.timestamp, unit="s").dt.year
    release = ratings.movieId.map(pd.Series(years.values, index=movies.movieId.values))
    summary["metadata"] = {
        "movies_without_year": int(years.isna().sum()),
        "movies_without_genres": int((movies.genres == "(no genres listed)").sum()),
        "duplicate_titles": int(movies.title.duplicated().sum()),
        "vocab_without_year": int(vocab.year.isna().sum()),
        "vocab_without_genres": int(vocab.genres.isna().sum()),
        "vocab_duplicate_titles": int(vocab.title.duplicated().sum()),
        "ratings_before_release_year": int((rated_year < release).sum()),
        "ratings_before_release_year_minus_1": int((rated_year < release - 1).sum()),
    }
    day_counts = ((ratings.timestamp // SECONDS_PER_DAY) - DAY0_UNIX_DAYS).value_counts()
    summary["busiest_days"] = {str(np.datetime64("1995-01-01") + np.timedelta64(int(d), "D")): int(c)
                               for d, c in day_counts.head(5).items()}
    summary["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["vocab"]), json.dumps(summary["metadata"]), f"done in {summary['seconds']}s")


if __name__ == "__main__":
    main()
