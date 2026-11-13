"""Write notebooks/demo.ipynb (run from the repository root: python scripts/build_demo_notebook.py).

The example users are chosen by fixed rules from the test results (see pick_examples), so the
notebook shows one case where the model does well and one where it fails, both explained.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root

import nbformat as nbf  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from moviedelphi.baselines import Popularity  # noqa: E402
from moviedelphi.batches import target_tokens  # noqa: E402
from moviedelphi.common import load_eval_cases, load_split, vocab_size  # noqa: E402

CHECKPOINT = "models/delphi_final.pt"

# Explanations written after looking at the cases the rules below pick on the test set (keyed by the
# test user, so a different pick falls back to a neutral sentence instead of a wrong explanation).
SUCCESS_TEXT = {1729: (
    "This user's history is mainstream blockbusters (*Jaws*, *Star Wars*, *Men in Black*, *Jurassic "
    "Park*). Five weeks later they rated both *Kill Bill* films. Delphi ranks both in its top 20, while "
    "RECENT-POP, which only knows what everyone rated that month, offers *Little Miss Sunshine*, "
    "*Children of Men* and *Casino Royale*. Both *Kill Bill* films came out years before the "
    "prediction day, so this success does not come from knowing later releases.")}
FAILURE_TEXT = {12336: (
    "One day of history: 17 ratings, mostly of 1995 releases (*GoldenEye*, *Heat*, *Twelve Monkeys*). A week "
    "later the user rated films that everyone was rating then (*Pulp Fiction*, *Apollo 13*, *Batman*, "
    "*True Lies*), and RECENT-POP has five of the six in its top 20. Delphi followed the user's apparent "
    "taste for action and thrillers (*The Rock*, *Mission: Impossible*, *Speed*) and found none. On "
    "average Delphi does better than RECENT-POP for histories of this length, so this is a miss of "
    "personalisation, not the typical case.")}


def pick_examples(per_case_csv, users_csv, name, split="test"):
    """Success: largest Delphi - RECENT-POP recall@20 among users with 6-40 active days and a 3-30 movie
    target day, whose hits are all films some training user had rated by the prediction day (so the
    success does not come from knowing later releases). Failure: a user with the same filters where
    Delphi finds nothing although RECENT-POP does."""
    model = pd.read_csv(per_case_csv)
    base = pd.read_csv(per_case_csv.replace(name, "RECENT-POP"))
    users = pd.read_csv(users_csv)
    eligible = users[users.eligible].reset_index(drop=True)
    pos = pd.Series(eligible.index.values, index=eligible.user.values)
    df = model[["user", "prior_days", "n_target", "n_hist", "gap", "recall@20"]].copy()
    df["base"] = base["recall@20"].values
    df["n_days"] = df.user.map(users.set_index("user").n_days)
    first_day = Popularity(load_split("train"), vocab_size(), "all").first_rated_day().numpy()
    top20 = np.load(per_case_csv.replace(".csv", "_top100.npy"))[:, :20]
    cases = load_eval_cases(split)
    targets = target_tokens(load_split(split), cases)
    df["future_hit"] = [bool((first_day[np.intersect1d(t20, t)] > d).any())
                        for t20, t, d in zip(top20, targets, cases.last_day.values)]
    ok = df[df.n_days.between(6, 40) & df.n_target.between(3, 30)]
    success = ok[~ok.future_hit].assign(d=lambda x: x["recall@20"] - x.base)
    success = success.sort_values(["d", "user"], ascending=[False, True]).iloc[0]
    fail = ok[(ok["recall@20"] == 0) & (ok.base > 0)].sort_values(["base", "user"], ascending=[False, True]).iloc[0]
    return {k: {"index": int(pos[int(r.user)]), "n_days": int(r.prior_days), "user": int(r.user)}
            for k, r in (("success", success), ("failure", fail))}


def main(split="test", name="DELPHI", out="notebooks/demo.ipynb"):
    ex = pick_examples(f"results/per_case/{split}_{name}.csv", f"data/processed/users_{split}.csv", name, split)
    print(json.dumps(ex))
    with open("results/tables/numbers.json") as f:
        n = json.load(f)
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("# Delphi-lite demo: which movies will a user rate on their next active day?\n\n"
           "Everything below uses **held-out test users**: their histories were never used for training, "
           "for choosing the model, or for tuning the baselines.\n\n"
           "For one user we reveal their ratings up to the end of an active day and rank the 10,000 movies "
           "for the user's **next active day**. The model sees only days before that day (its most recent "
           "256 ratings); movies the user already rated are never suggested. Next to Delphi we show "
           "**RECENT-POP**, the strongest baseline: what training users rated most in the last 30 days.\n\n"
           "Rows marked `hit = True` are movies the user really rated on that next day. "
           "`p` is the model's probability for the movie among all candidates."),
        code("import os, sys\n"
             "if os.path.basename(os.getcwd()) == 'notebooks':\n"
             "    os.chdir('..')  # run from the repository root\n"
             "sys.path.insert(0, os.getcwd())\n"
             "import pandas as pd\n"
             "from IPython.display import HTML, display\n"
             "from moviedelphi.demo import Explorer\n\n"
             f"ex = Explorer('{CHECKPOINT}', split='{split}', device='cpu')\n"
             "print(f'{len(ex.users)} held-out test users with at least two active days')"),
        code("def highlight(df):\n"
             "    if 'hit' not in df:\n"
             "        return df.style\n"
             "    return df.style.apply(lambda r: ['background-color:#dcefdc' if r.hit else '' for _ in r], axis=1)\n\n"
             "def side_by_side(*frames):\n"
             "    html = ''.join(f'<div style=\"margin-right:24px\"><b>{t}</b>{highlight(f).to_html()}</div>' for t, f in frames)\n"
             "    display(HTML(f'<div style=\"display:flex;align-items:flex-start\">{html}</div>'))\n\n"
             "def show(r, n_history=10):\n"
             "    print(f\"history: {len(r['history'])} ratings up to {r['last_day']}  ->  next active day {r['target_day']}\"\n"
             "          f\" ({len(r['target'])} movies)\")\n"
             "    print(f\"recall@20   Delphi {r['model_recall@20']:.2f}   RECENT-POP {r['baseline_recall@20']:.2f}\")\n"
             "    side_by_side((f'Last {n_history} ratings seen', r['history'].tail(n_history)),\n"
             "                 ('What the user rated next', r['target'].head(20)))\n"
             "    side_by_side(('Delphi top 20', r['model']), ('RECENT-POP top 20', r['baseline']))"),
        md("## 1. Step through one user's timeline\n\n"
           "Each row reveals one more active day and predicts the next one. This is the comparison tool: "
           "how predictions change with more visible history, against the baseline."),
        code(f"USER = {ex['success']['index']}  # index into ex.users; try any number below len(ex.users)\n"
             "ex.timeline(USER)"),
        md("## 2. Inspect one prediction\n\n" + SUCCESS_TEXT.get(ex["success"]["user"],
           "A case where Delphi finds more of the next day's movies than RECENT-POP. Compare the "
           "user's history with the two top-20 lists.")),
        code(f"show(ex.predict_next_day(USER, n_days={ex['success']['n_days']}))"),
        md("## 3. A failure, explained\n\n" + FAILURE_TEXT.get(ex["failure"]["user"],
           "Here Delphi finds none of the next-day movies although RECENT-POP finds some. Compare the "
           "history with what the user rated next.")
           + f" Misses are common: in {n['BucketBothZero']} of test cases neither list contains a single "
           "movie the user rated next."),
        code(f"show(ex.predict_next_day({ex['failure']['index']}, n_days={ex['failure']['n_days']}))"),
        md("## 4. Your own history (new data)\n\n"
           "Find titles with `ex.find(...)`, then give a list of exact titles and the date you rated them "
           "(the data covers 1995-01-09 to 2019-11-21; later dates are moved to the last day)."),
        code("ex.find('matrix').head(5)"),
        code("my_movies = ['Matrix, The (1999)', 'Inception (2010)', 'Interstellar (2014)', 'Arrival (2016)']\n"
             "delphi_top, recent_pop_top = ex.predict_for_titles(my_movies, '2019-06-01', k=10)\n"
             "side_by_side(('Delphi top 10', delphi_top), ('RECENT-POP top 10', recent_pop_top))"),
        md("### How to read this\n\n"
           "* A hit is only a movie the user *rated* that day; MovieLens dates are rating dates, not "
           "watch dates, so a sensible suggestion can still count as a miss.\n"
           "* The model never sees the next day's date, only the dates of earlier ratings.\n"
           "* Users were split, not time: Delphi learned from training users' later ratings, so it "
           f"sometimes suggests a film that appeared after the prediction day ({n['FutureShareTopTwenty']} "
           f"of its top 20, {n['FutureShareHits']} of its hits). Without such films its gain over "
           f"RECENT-POP stays {n['CausalDelphiVsRecent']}.\n"
           "* Test-set averages, not single cases, are the evidence: see the report."),
    ]
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    # python scripts/build_demo_notebook.py [split] [model name in results/per_case] [output]
    sys.exit(main(*sys.argv[1:]))
