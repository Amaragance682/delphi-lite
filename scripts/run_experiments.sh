#!/usr/bin/env bash
# Every training run in the report, in the order of the validation path (one change per step).
# Each run writes runs/<name>/{config.json,log.csv,ckpt.pt,summary.json}.
# Times are for one RTX 5080 Laptop GPU with two runs sharing it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
train() { $PY -m moviedelphi.train --config configs/base.json "$@"; }

# baselines: choose the RECENT-POP window and counting rule on validation (seconds)
$PY -m moviedelphi.baselines --split val

# M1: Delphi, small (4 layers, 128 dims), Delphi's year-unit age encoding, 10k iterations (~8 min)
train --name m1_base_years
# M2: M1 with the age encoding in days instead of years
train --name m2_age_days --age_scale 1.0
# M3: M1 with the every-position next-day objective
train --name m3_every_position --objective every_position
# M4: M1 with each day-to-next-day transition weighted equally
train --name m4_basket_weighting --basket_weighting true
# M5: M1 trained 3x longer (30k iterations, ~29 min)
train --name m5_long --max_iters 30000 --eval_interval 2000
# M6: M1 with a 3.5x larger model (6 layers, 8 heads, 256 dims, ~20 min)
train --name m6_big --n_layer 6 --n_head 8 --n_embd 256
# M7: M1 with the popularity offset beta * log(1 + 30-day training count) on every log-rate
train --name m7_popularity_offset --popularity_offset true

# extra: data efficiency, M1's configuration on 10% and 30% of the eligible training users
train --name data_10pct --user_fraction 0.1
train --name data_30pct --user_fraction 0.3

# M8: M5 with the popularity offset (configs/m8_offset_30k.json, the candidate final configuration)
$PY -m moviedelphi.train --config configs/m8_offset_30k.json --name m8_offset_30k --seed 1
# M8 did not beat M5 on validation, so the final model is M5's configuration: two more seeds
train --name m5_long_s2 --seed 2 --max_iters 30000 --eval_interval 2000
train --name m5_long_s3 --seed 3 --max_iters 30000 --eval_interval 2000
