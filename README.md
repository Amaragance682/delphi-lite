# Delphi-lite: which movies will a user rate on their next active day?

Final project by **Ragnar Björn and Ólafur Sær**, Introduction to Deep Learning (Fall 2026). We apply **Delphi** (Shmatko et al., 2025),
a GPT-2-style transformer for event histories with continuous time, to **MovieLens-25M** rating
histories, and ask one focused question:

> Given everything a user rated up to the end of an active day, which movies will they rate on
> their **next active day**, and does their own history help beyond knowing what is popular then?

* Report: [`report/report.pdf`](report/report.pdf) (two pages, references on page three)
* Slides: [`slides/delphi_lite.pptx`](slides/delphi_lite.pptx) (PDF copy alongside); timed speaker notes in
  [`slides/speaker_notes.md`](slides/speaker_notes.md)
* Live demo: [`notebooks/demo.ipynb`](notebooks/demo.ipynb); an executed copy is in `notebooks/demo.html` (backup)

<!-- RESULTS -->
## Results

Test set: 7,129 held-out users, scored once after every decision was made on 7,067 validation users. Recall@20 is averaged over users; ± is a 95% interval; Δ is the paired difference to RECENT-POP on the same users.

| Method | Recall@20 | Recall@10 | Recall@50 | NDCG@20 | Δ recall@20 vs RECENT-POP |
|---|---|---|---|---|---|
| **Delphi (final)** | 0.158 ± 0.005 | 0.137 | 0.223 | 0.132 | +0.009 ± 0.005 |
| RECENT-POP (30-day popularity) | 0.149 ± 0.005 | 0.130 | 0.211 | 0.124 | — |
| ITEM-KNN (co-occurrence) | 0.136 ± 0.005 | 0.122 | 0.189 | 0.120 | -0.013 ± 0.005 |
| POP (all-time popularity) | 0.117 ± 0.005 | 0.107 | 0.154 | 0.104 | -0.032 ± 0.005 |

Three training seeds of the final model: recall@20 0.156–0.158; each seed's paired 95% interval against RECENT-POP lies above zero (lowest bound +0.002).
Leakage check for the user split: if no method may suggest a film that no training user had rated by the prediction day, Delphi scores 0.158 and its gain over RECENT-POP is +0.009 ± 0.005 (`results/tables/test_causal.csv`).

Validation path (one change per step; struck-through steps were not kept):

| Step | Change | Val recall@20 |
|---|---|---|
| POP | all-time popularity | 0.117 ± 0.005 |
| RECENT-POP | popularity in the last 30 days | 0.143 ± 0.005 |
| ITEM-KNN | co-occurrence with the history, no time | 0.133 ± 0.005 |
| M1 Delphi | 4 layers × 128, age in years, 10k it. | 0.145 ± 0.005 |
| ~~M2~~ | M1, age encoding in days | 0.142 ± 0.005 |
| ~~M3~~ | M1, every position predicts its next day | 0.145 ± 0.005 |
| ~~M4~~ | M1, day transitions weighted equally | 0.145 ± 0.006 |
| M5 final | M1, trained for 30k iterations | 0.155 ± 0.005 |
| ~~M6~~ | M1, 6 layers × 256 (3.5× parameters) | 0.143 ± 0.005 |
| ~~M7~~ | M1 + popularity offset β·log(1 + c₃₀) | 0.147 ± 0.005 |
| ~~M8~~ | M5 + popularity offset | 0.153 ± 0.005 |

Figures: `results/figures/`; every table: `results/tables/`; the report explains each number.
<!-- /RESULTS -->

## Repository layout

| Path | What it is |
|---|---|
| `moviedelphi/prepare_data.py` | MovieLens → Delphi's format: user split, vocabulary, day index, evaluation cases |
| `moviedelphi/model.py` | Delphi model adapted from the original repository (the changes are listed at the top) |
| `moviedelphi/batches.py` | training windows and evaluation inputs (Delphi's left-padded layout) |
| `moviedelphi/train.py` | training loop adapted from Delphi's `train.py`, with run logging |
| `moviedelphi/baselines.py` | POP, RECENT-POP (calendar-time popularity) and ITEM-KNN |
| `moviedelphi/evaluation.py` | candidates, exclusion of already-rated movies, recall@k / NDCG@k, 95% intervals |
| `moviedelphi/evaluate.py` | one evaluation pass on validation or test, history-length curve, controls |
| `moviedelphi/analysis.py` | error analysis by bucket and by how popular the target movies are |
| `moviedelphi/causal_check.py` | leakage check for the user split: no method may suggest a film that no training user had rated by the prediction day |
| `moviedelphi/efficiency.py` | parameters, training time, CPU and GPU inference cost |
| `moviedelphi/figures.py`, `report_numbers.py`, `path_table.py`, `readme_results.py` | every figure, every number in the report and slides, and the results section above, generated from the logged results |
| `moviedelphi/demo.py` | the history explorer behind the demo notebook |
| `tests/test_pipeline.py` | 16 checks: leakage, tie handling, batching, metrics, baselines, popularity offset, causal candidates |
| `configs/base.json` | hyper-parameters of M1; other runs override single values (see `scripts/run_experiments.sh`) |
| `runs/<name>/` | one folder per training run: `config.json`, `log.csv` (validation curve), `summary.json`, `ckpt.pt` |
| `models/delphi_final.pt` | the final model (`runs/m5_long/ckpt.pt`), used for the test results and the demo |
| `results/tables/`, `results/figures/` | every table and figure in the report and slides; `results/per_case/` has per-user results |
| `scripts/` | data download, all training runs, the final evaluation, the demo-notebook builder, `make_handin.py` (the zip) |

## Setup

```bash
conda create -n delphi-movies python=3.11 -y && conda activate delphi-movies
pip install -r requirements.txt          # pick the CUDA build of torch from pytorch.org if you have a GPU
```

Tested with Python 3.11.16, torch 2.11.0 (CUDA 12.8), numpy 2.4.6 and pandas 3.0.6 on an RTX 5080 Laptop GPU.
Evaluation and the demo also run on a CPU; training on a CPU works but is much slower.

## Data (not included)

MovieLens-25M may not be redistributed, so the hand-in contains no MovieLens files. Download it and
check the published MD5:

```bash
bash scripts/download_data.sh            # -> data/raw/ml-25m/ (ratings.csv, movies.csv, ...)
python -m moviedelphi.prepare_data       # -> data/processed/ (about 15 s)
```

`prepare_data` splits users 80/10/10 *before* anything else (seed 42), keeps the 10,000 movies rated
by the most **training** users, converts timestamps to UTC days and writes Delphi's binary format
(`uint32` rows of user, day, token). It also draws one evaluation case per eligible validation and test
user. `data/processed/summary.json` holds every dataset statistic quoted in the report.

## Reproducing the results

```bash
python -m tests.test_pipeline            # 16 checks, about 30 s
bash scripts/run_experiments.sh          # baselines and every training run (12 runs, 3.1 h of training on one laptop GPU)
bash scripts/final_evaluation.sh         # validation table, then the test set once; analyses, figures, numbers
```

* `scripts/run_experiments.sh` lists every run on the validation path, with the one change it makes.
  The final model is M5: `configs/base.json` trained for 30k iterations (seeds 1-3: `m5_long`,
  `m5_long_s2`, `m5_long_s3`).
* Every run is logged in `runs/<name>/`: `log.csv` has the validation curve, `summary.json` the best
  checkpoint (chosen on validation recall@20).
* `final_evaluation.sh` is the only script that reads test users.
* The tables, figures and report numbers are regenerated from these files. Then build the report with
  `tectonic -X compile report/report.tex` (or upload `report/` and `results/figures/` to Overleaf) and the
  slides with `npm install pptxgenjs && node slides/build_slides.js` (run from the repository root).
* One logging detail: runs started before the fix in `train.py` computed the logged validation *loss* on one
  batch of 64 windows instead of 20 batches. Checkpoints were always chosen on validation recall@20, so
  no result depends on it.

## Demo

```bash
jupyter lab notebooks/demo.ipynb
```

The notebook loads `models/delphi_final.pt` on the CPU and applies it to held-out test users. It can:

* step through a user's timeline;
* inspect a success and a failure;
* rank movies for a history you type in yourself.

It calls the experiment code (`moviedelphi/demo.py`), so the preprocessing is identical.
`python scripts/build_demo_notebook.py` rebuilds it from the test results.

## Task definition (short version)

* **Event**: one rating, whatever its star value. **Time unit**: UTC calendar day since 1995-01-01.
* **Ties**: ratings on the same day are simultaneous. They are stored in a fixed random order and reshuffled during training.
* **Eligible histories**: at least two active days (days with a rating of a vocabulary movie).
* **Evaluation case**: for each eligible validation or test user, one target day drawn at random among
  their active days after the first.
  * **Prediction time**: the end of the last active day before the target day.
  * **Input**: ratings from earlier days only (the latest 256).
  * **Target**: the movies rated on the target day.
  * **Candidates**: all 10,000 movies except those the user has already rated.
* **Metric**: recall@20 = hits in the top 20 / min(20, number of target movies), averaged over users,
  with 95% intervals and paired differences between methods on the same users.

## Licences and citations

* MovieLens-25M: F. M. Harper and J. A. Konstan. The MovieLens Datasets: History and Context.
  ACM TiiS 5(4), 2015. https://doi.org/10.1145/2827872. Research use, citation required, no redistribution.
* Delphi: A. Shmatko, A. W. Jung, K. Gaurav et al. Learning the natural history of human disease with
  generative transformers. Nature, 2025. https://doi.org/10.1038/s41586-025-09529-3. The code is MIT-licensed
  and the adapted files keep the notice (see `THIRD_PARTY_NOTICES.md`). No Delphi weights are used.
