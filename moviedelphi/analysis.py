"""Error analysis on the per-case results of one split.

    python -m moviedelphi.analysis --split test --main DELPHI

Writes, in results/tables/:
  {split}_buckets_long.csv     recall@20 (mean, 95% interval, n) per method and bucket of: days since
                               the last active day, history length, next-day basket size, number of
                               earlier active days, calendar year of the target day
  {split}_target_popularity.csv  share of target movies found in the top 20, by how popular the
                               target movie is among training users (vocabulary rank)
  {split}_buckets.csv          key/value numbers quoted in the report
"""

import argparse
import os

import numpy as np
import pandas as pd

from .batches import target_tokens
from .common import day_to_date, load_eval_cases, load_split

METHODS = ("RECENT-POP", "ITEM-KNN", "POP")
BUCKETS = {
    "gap": ([0, 1, 7, 30, 365, 10**6], ["1 day", "2-7 days", "8-30 days", "31-365 days", ">1 year"]),
    "n_hist": ([0, 15, 50, 150, 500, 10**7], ["<=15", "16-50", "51-150", "151-500", ">500"]),
    "n_target": ([0, 1, 5, 20, 100, 10**6], ["1", "2-5", "6-20", "21-100", ">100"]),
    "prior_days": ([0, 1, 4, 19, 10**6], ["1 (onboarding only)", "2-4", "5-19", "20+"]),
}
POPULARITY = ([0, 100, 1000, 5000, 10002], ["top 100", "101-1000", "1001-5000", "5001-10000"])


def ci(x):
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--main", default="DELPHI")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    pc_dir, tables = os.path.join(args.results, "per_case"), os.path.join(args.results, "tables")
    names = (args.main,) + METHODS
    pcs = {n: pd.read_csv(os.path.join(pc_dir, f"{args.split}_{n}.csv")) for n in names}
    year = pd.Series(day_to_date(pcs[args.main].target_day.values).astype("datetime64[Y]").astype(int) + 1970)

    rows = []
    for n, pc in pcs.items():
        groups = {col: pd.cut(pc[col], e, labels=lab, right=True) for col, (e, lab) in BUCKETS.items()}
        groups["year"] = pd.cut(year, [1995, 1999, 2004, 2009, 2014, 2019],
                                labels=["1996-99", "2000-04", "2005-09", "2010-14", "2015-19"], right=True)
        for col, g in groups.items():
            for b, x in pc.groupby(g, observed=False)["recall@20"]:
                rows.append({"method": n, "feature": col, "bucket": str(b), "recall@20": x.mean(),
                             "ci95": ci(x), "n": len(x)})
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(tables, f"{args.split}_buckets_long.csv"), index=False)

    # which target movies are found: by popularity of the target movie (token order = training count rank)
    split, cases = load_split(args.split, args.processed), load_eval_cases(args.split, args.processed)
    targets = target_tokens(split, cases)
    case_of = np.repeat(np.arange(len(targets)), [len(t) for t in targets])
    tok = np.concatenate(targets)
    rank = tok - 1  # token 2 = most-rated movie -> rank 1
    pop_bucket = pd.cut(rank, POPULARITY[0], labels=POPULARITY[1], right=True)
    weight = 1.0 / np.repeat([len(t) for t in targets], [len(t) for t in targets])  # each case counts once
    pop_rows = []
    for n in names:
        top20 = np.load(os.path.join(pc_dir, f"{args.split}_{n}_top100.npy"))[:, :20]
        found = (top20[case_of] == tok[:, None]).any(1)
        df = pd.DataFrame({"bucket": pop_bucket, "found": found, "w": weight})
        for b, x in df.groupby("bucket", observed=False):
            pop_rows.append({"method": n, "target_popularity": str(b), "share_found": x.found.mean(),
                             "targets": len(x), "share_of_targets": len(x) / len(df),
                             "share_of_targets_case_weighted": x.w.sum() / weight.sum()})
    pd.DataFrame(pop_rows).to_csv(os.path.join(tables, f"{args.split}_target_popularity.csv"), index=False)

    # numbers quoted in the report and the demo
    key = []
    d, r = pcs[args.main], pcs["RECENT-POP"]
    key.append(("BothZero", ((d["recall@20"] == 0) & (r["recall@20"] == 0)).mean()))
    key.append(("DelphiZero", (d["recall@20"] == 0).mean()))
    key.append(("DelphiBetter", (d["recall@20"] > r["recall@20"]).mean()))
    key.append(("RecentBetter", (d["recall@20"] < r["recall@20"]).mean()))
    year_labels = ["1996-99", "2000-04", "2005-09", "2010-14", "2015-19"]
    for col, labels in [(c, lab) for c, (_, lab) in BUCKETS.items()] + [("year", year_labels)]:
        for lab in labels:
            sel = long[(long.feature == col) & (long.bucket == lab)].set_index("method")["recall@20"]
            tag = col.replace("_", "") + "".join(ch for ch in lab if ch.isalnum())
            key.append((f"{tag}Delphi", sel[args.main]))
            key.append((f"{tag}Recent", sel["RECENT-POP"]))
    pop = pd.DataFrame(pop_rows)
    for lab in POPULARITY[1]:
        sel = pop[pop.target_popularity == lab].set_index("method")
        tag = "pop" + "".join(ch for ch in lab if ch.isalnum())
        key.append((f"{tag}Delphi", sel.loc[args.main, "share_found"]))
        key.append((f"{tag}Recent", sel.loc["RECENT-POP", "share_found"]))
        key.append((f"{tag}Share", sel.loc[args.main, "share_of_targets"]))
    # what the cases where neither Delphi nor RECENT-POP finds anything look like
    both_zero = ((d["recall@20"] == 0) & (r["recall@20"] == 0)).values
    tail = np.array([(t - 1 > 1000).mean() for t in targets])  # share of a case's targets outside the top 1000
    key.append(("BothZeroSmallShare", (cases.n_target.values[both_zero] <= 5).mean()))
    key.append(("BothZeroTailShare", tail[both_zero].mean()))
    key.append(("HitTailShare", tail[~both_zero].mean()))
    # how old the target films are when rated (target year - release year in the title), by period
    vocab_year = pd.read_csv(os.path.join(args.processed, "vocab.csv")).set_index("token")["year"].astype(float)
    film_age = np.array([np.nan if np.isnan(vocab_year.loc[t].values).all() else
                         np.nanmedian(y - vocab_year.loc[t].values) for y, t in zip(year.values, targets)])
    key.append(("LateFilmAge", np.nanmedian(film_age[year.values >= 2015])))
    key.append(("EarlyFilmAge", np.nanmedian(film_age[year.values < 2015])))
    pd.DataFrame(key, columns=["key", "value"]).to_csv(os.path.join(tables, f"{args.split}_buckets.csv"), index=False)
    print(long[long.method.isin([args.main, "RECENT-POP"])].pivot_table(
        index=["feature", "bucket"], columns="method", values="recall@20", sort=False).round(3).to_string())
    print(pop.pivot_table(index="target_popularity", columns="method", values="share_found", sort=False).round(3))
    print(pd.DataFrame(key, columns=["key", "value"]).head(4).to_string())


if __name__ == "__main__":
    main()
