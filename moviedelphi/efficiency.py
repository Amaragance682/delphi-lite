"""Model size, training time and inference cost.

    python -m moviedelphi.efficiency --model NAME=runs/NAME/ckpt.pt [...]

Inference is timed on validation cases with a full 256-event history: one case at a time
(latency of a single request) and in batches of 256 (throughput), on CPU and, if present, GPU.
Writes results/tables/efficiency.csv.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .batches import history_inputs
from .common import load_eval_cases, load_split
from .train import load_model


def timed(fn, sync):
    sync()
    t = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t


def inference_times(model, tok, age, device, n_single=200, batch=256):
    model = model.to(device)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    t = torch.from_numpy(tok).to(device)
    a = torch.from_numpy(age).to(device)
    model.predict_last(t[:batch], a[:batch])  # warm-up
    single = [timed(lambda i=i: model.predict_last(t[i:i + 1], a[i:i + 1]), sync) for i in range(n_single)]
    batched = timed(lambda: [model.predict_last(t[i:i + batch], a[i:i + batch]) for i in range(0, len(t), batch)], sync)
    return {"latency_ms_median": 1000 * float(np.median(single)), "throughput_cases_per_s": len(t) / batched}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="NAME=path/to/ckpt.pt")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="results/tables/efficiency.csv")
    ap.add_argument("--n_cases", type=int, default=2048)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("highest")

    split, cases = load_split("val", args.processed), load_eval_cases("val", args.processed)
    cases = cases[cases.n_hist >= 256].head(args.n_cases).reset_index(drop=True)
    rows = []
    for spec in args.model:
        name, path = spec.split("=", 1)
        model, ckpt = load_model(path, "cpu")
        tok, age = history_inputs(split, cases, None, model.config.block_size)
        with open(os.path.join("runs", ckpt["config"]["name"], "summary.json")) as f:  # the run that made it
            summary = json.load(f)
        row = {"model": name, "params": model.get_num_params(), "checkpoint_mb": os.path.getsize(path) / 2**20,
               "train_minutes": summary["train_seconds"] / 60, "train_iters": ckpt["config"]["max_iters"],
               "cpu_threads": torch.get_num_threads(), "cases": len(cases)}
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for k, v in inference_times(model, tok, age, device).items():
                row[f"{device}_{k}"] = v
        rows.append(row)
        print(row)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
