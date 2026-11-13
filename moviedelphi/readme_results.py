"""Write the results section of README.md (between the RESULTS markers) from the result tables.

    python -m moviedelphi.readme_results
"""

import json

import pandas as pd

START, END = "<!-- RESULTS -->", "<!-- /RESULTS -->"
LATEX = {"$\\times$": "×", "$\\beta\\log(1+c_{30})$": "β·log(1 + c₃₀)"}


def plain(s):
    for k, v in LATEX.items():
        s = s.replace(k, v)
    assert "$" not in s and "\\" not in s, s
    return s


def main(readme="README.md"):
    with open("results/tables/numbers.json") as f:
        n = json.load(f)
    test = pd.read_csv("results/tables/test_main.csv").set_index("method")
    with open("results/tables/validation_path.json") as f:
        path = json.load(f)

    lines = [START, "## Results", "",
             f"Test set: {n['CasesTest']} held-out users, scored once after every decision was made on "
             f"{n['CasesVal']} validation users. Recall@20 is averaged over users; ± is a 95% interval; "
             "Δ is the paired difference to RECENT-POP on the same users.", "",
             "| Method | Recall@20 | Recall@10 | Recall@50 | NDCG@20 | Δ recall@20 vs RECENT-POP |",
             "|---|---|---|---|---|---|"]
    for label, m in (("**Delphi (final)**", "DELPHI"), ("RECENT-POP (30-day popularity)", "RECENT-POP"),
                     ("ITEM-KNN (co-occurrence)", "ITEM-KNN"), ("POP (all-time popularity)", "POP")):
        r = test.loc[m]
        diff = "—" if m == "RECENT-POP" else f"{r['diff_vs_RECENT-POP']:+.3f} ± {r['diff_vs_RECENT-POP_ci95']:.3f}"
        lines.append(f"| {label} | {r['recall@20']:.3f} ± {r['recall@20_ci95']:.3f} | {r['recall@10']:.3f} | "
                     f"{r['recall@50']:.3f} | {r['ndcg@20']:.3f} | {diff} |")
    lines += ["", f"Three training seeds of the final model: recall@20 {n['SeedMin']}–{n['SeedMax']}; each seed's paired "
              f"95% interval against RECENT-POP lies above zero (lowest bound {n['SeedDiffMinLow']}).",
              f"Leakage check for the user split: if no method may suggest a film that no training user had rated by "
              f"the prediction day, Delphi scores {n['CausalDelphi']} and its gain over RECENT-POP is "
              f"{n['CausalDelphiVsRecent']} ± {n['CausalDelphiVsRecentCI']} (`results/tables/test_causal.csv`).", "",
              "Validation path (one change per step; struck-through steps were not kept):", "",
              "| Step | Change | Val recall@20 |", "|---|---|---|"]
    for r in path:
        step = r["label"] if r["kept"] else f"~~{r['label']}~~"
        lines.append(f"| {step} | {plain(r['change'])} | {r['recall@20']:.3f} ± {r['ci']:.3f} |")
    lines += ["", "Figures: `results/figures/`; every table: `results/tables/`; the report explains each number.", END]

    with open(readme) as f:
        text = f.read()
    a = text.index(START)
    b = text.index(END) + len(END) if END in text else a + len(START)
    with open(readme, "w") as f:
        f.write(text[:a] + "\n".join(lines) + text[b:])
    print("README results section updated")


if __name__ == "__main__":
    main()
