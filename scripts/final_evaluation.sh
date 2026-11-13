#!/usr/bin/env bash
# Validation table for every run, then the test set ONCE, then analyses, figures, report numbers and the demo.
# This is the only script that reads test users. Run after scripts/run_experiments.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
ck() { echo "runs/$1/ckpt.pt"; }

# 1. validation: every run on the path, with 95% intervals and paired differences (decisions were made here)
$PY -m moviedelphi.evaluate --split val --tag _path \
  --model M1="$(ck m1_base_years)" --model M2="$(ck m2_age_days)" --model M3="$(ck m3_every_position)" \
  --model M4="$(ck m4_basket_weighting)" --model M5="$(ck m5_long)" --model M6="$(ck m6_big)" \
  --model M7="$(ck m7_popularity_offset)" --model M8="$(ck m8_offset_30k)"
$PY -m moviedelphi.path_table

# 2. test, once: the final model (M5's configuration, 3 seeds), the data-efficiency runs and the baselines
mkdir -p models
cp "$(ck m5_long)" models/delphi_final.pt
SEEDS=(--model DELPHI=models/delphi_final.pt --model DELPHI_s2="$(ck m5_long_s2)" --model DELPHI_s3="$(ck m5_long_s3)")
$PY -m moviedelphi.evaluate --split test --main DELPHI --analyses "${SEEDS[@]}" \
  --model DATA10="$(ck data_10pct)" --model DATA30="$(ck data_30pct)" --model DATA100="$(ck m1_base_years)"
$PY -m moviedelphi.analysis --split test --main DELPHI
$PY -m moviedelphi.causal_check --split test "${SEEDS[@]}"   # leakage check for the user split
$PY -m moviedelphi.efficiency --model DELPHI=models/delphi_final.pt

# 3. figures (report and slide variants), the numbers in the report and slides, the README results section
CURVES=("M1 (10k it.)=m1_base_years" "M5 final (30k it.)=m5_long" "M8 (30k it. + offset)=m8_offset_30k")
$PY -m moviedelphi.figures --main DELPHI --curves "${CURVES[@]}"
$PY -m moviedelphi.figures --slides --main DELPHI --curves "${CURVES[@]}"
$PY -m moviedelphi.report_numbers
$PY -m moviedelphi.readme_results

# 4. the demo notebook (examples picked from the test results by fixed rules), executed, with an HTML copy
$PY scripts/build_demo_notebook.py test DELPHI notebooks/demo.ipynb
(cd notebooks && jupyter nbconvert --to notebook --execute --inplace demo.ipynb && jupyter nbconvert --to html demo.ipynb)
