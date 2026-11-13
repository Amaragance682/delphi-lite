"""Train Delphi on movie histories (adapted from Delphi's train.py / nanoGPT).

    python -m moviedelphi.train --config configs/base.json [--name NAME] [--key value ...]

Training uses eligible training users only. Every eval_interval iterations the run logs the
validation loss and the validation ranking metrics, and keeps the checkpoint with the best
validation recall@20. Everything is written to runs/<name>/:
  config.json   the full configuration
  log.csv       one row per evaluation: iteration, learning rate, train/val losses, val metrics
  ckpt.pt       best checkpoint (model weights, model config, training config, val metrics)
  summary.json  best iteration, val metrics, parameter count, wall-clock time
"""

import argparse
import csv
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch

from .baselines import recent_log_counts
from .batches import TrainBatches
from .common import load_eval_cases, load_split, vocab_size
from .evaluation import PRIMARY, evaluate_topk, model_topk
from .model import Delphi, DelphiConfig


@dataclass
class TrainConfig:
    name: str = "delphi"
    out_root: str = "runs"
    processed: str = "data/processed"
    seed: int = 1
    # data
    block_size: int = 256
    batch_size: int = 64
    user_fraction: float = 1.0      # fraction of eligible training users (data-efficiency runs)
    # model (DelphiConfig)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    token_dropout: float = 0.0
    bias: bool = False
    t_min: float = 0.1
    age_scale: float = 365.25
    basket_weighting: bool = False
    objective: str = "last_position"
    popularity_offset: bool = False
    popularity_window: int = 30
    # AdamW with linear warm-up and cosine decay, as in Delphi
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    warmup_iters: int = 500
    max_iters: int = 10000
    # evaluation and system
    eval_interval: int = 1000
    val_loss_batches: int = 20
    device: str = "cuda"
    dtype: str = "bfloat16"


def get_lr(it, c):
    if it < c.warmup_iters:
        return c.learning_rate * it / c.warmup_iters
    if it > c.max_iters:
        return c.min_lr
    ratio = (it - c.warmup_iters) / (c.max_iters - c.warmup_iters)
    return c.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (c.learning_rate - c.min_lr)


def model_config(c, n_vocab):
    return DelphiConfig(block_size=c.block_size, vocab_size=n_vocab, n_layer=c.n_layer, n_head=c.n_head,
                        n_embd=c.n_embd, dropout=c.dropout, token_dropout=c.token_dropout, t_min=c.t_min,
                        bias=c.bias, age_scale=c.age_scale, basket_weighting=c.basket_weighting,
                        objective=c.objective, popularity_offset=c.popularity_offset,
                        popularity_window=c.popularity_window)


def load_model(path, device="cpu", processed=None):
    """Model in eval mode. A popularity-offset model gets its table rebuilt from the training split."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Delphi(DelphiConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    if model.config.popularity_offset:
        processed = processed or ckpt["config"]["processed"]
        model.set_popularity(recent_log_counts(load_split("train", processed), model.config.vocab_size,
                                               model.config.popularity_window))
    return model.eval(), ckpt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="JSON file with TrainConfig fields")
    for f in fields(TrainConfig):
        ap.add_argument(f"--{f.name}", type=(lambda s: s.lower() in ("1", "true", "yes")) if f.type is bool else f.type)
    args = ap.parse_args()
    values = {}
    if args.config:
        with open(args.config) as fh:
            values.update(json.load(fh))
    values.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    return TrainConfig(**values)


def main():
    c = parse_args()
    out_dir = os.path.join(c.out_root, c.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(c), f, indent=2)

    torch.manual_seed(c.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device_type = "cuda" if "cuda" in c.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[c.dtype]
    ctx = nullcontext() if device_type == "cpu" or c.dtype == "float32" else torch.autocast(device_type, dtype=ptdtype)

    n_vocab = vocab_size(c.processed)
    train_split = load_split("train", c.processed)
    train_batches = TrainBatches(train_split, c.block_size, c.user_fraction, seed=c.seed)
    val_split = load_split("val", c.processed)
    val_cases = load_eval_cases("val", c.processed)
    # fixed validation windows so that the validation loss is comparable across evaluations
    val_windows = TrainBatches(val_split, c.block_size, seed=12345)
    val_loss_set = [val_windows.get(c.batch_size, "cpu") for _ in range(c.val_loss_batches)]

    model = Delphi(model_config(c, n_vocab)).to(c.device)
    if c.popularity_offset:  # counts from training users only, in the window up to each day
        model.set_popularity(recent_log_counts(train_split, n_vocab, c.popularity_window))
    optimizer = model.configure_optimizers(c.weight_decay, c.learning_rate, (c.beta1, c.beta2), device_type)
    n_params = model.get_num_params()
    print(f"{c.name}: {n_params / 1e6:.2f}M parameters, {train_batches.n_users} training histories")

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for x, a, y, b in val_loss_set:
            with ctx:
                loss = model(x.to(c.device), a.to(c.device), y.to(c.device), b.to(c.device))
            losses.append([loss["loss_ce"].item(), loss["loss_dt"].item()])
        topk = model_topk(model, val_split, val_cases, device=c.device)  # float32 (TF32 matmuls) ranking
        _, metrics = evaluate_topk(topk, val_split, val_cases)
        model.train()
        ce, dt = np.mean(losses, 0)
        return {"val_ce": ce, "val_dt": dt, **{k: metrics[k] for k in ("recall@10", "recall@20", "recall@50", "ndcg@20")}}

    log_path = os.path.join(out_dir, "log.csv")
    log_cols = ["iter", "lr", "train_ce", "train_dt", "val_ce", "val_dt",
                "recall@10", "recall@20", "recall@50", "ndcg@20", "seconds"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(log_cols)

    best, best_iter, t_start = -1.0, 0, time.time()
    run_ce, run_dt, run_n = 0.0, 0.0, 0
    model.train()
    for it in range(c.max_iters + 1):
        lr = get_lr(it, c)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % c.eval_interval == 0 and it > 0:
            m = evaluate()
            row = {"iter": it, "lr": lr, "train_ce": float(run_ce) / max(run_n, 1),
                   "train_dt": float(run_dt) / max(run_n, 1), **m, "seconds": time.time() - t_start}
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([row[k] for k in log_cols])
            print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
            run_ce, run_dt, run_n = 0.0, 0.0, 0
            if m[PRIMARY] > best:
                best, best_iter = m[PRIMARY], it
                torch.save({"model": model.state_dict(), "model_args": asdict(model.config), "config": asdict(c),
                            "iter": it, "val": m}, os.path.join(out_dir, "ckpt.pt"))
        if it == c.max_iters:
            break

        x, a, y, b = train_batches.get(c.batch_size, c.device)
        with ctx:
            loss = model(x, a, y, b)
        total = loss["loss_ce"] + loss["loss_dt"]
        total.backward()
        if c.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        run_ce += loss["loss_ce"].detach()
        run_dt += loss["loss_dt"].detach()
        run_n += 1

    summary = {"name": c.name, "best_iter": best_iter, "best_val_" + PRIMARY: best, "params": n_params,
               "train_seconds": time.time() - t_start, "training_histories": train_batches.n_users,
               "best_val": torch.load(os.path.join(out_dir, "ckpt.pt"), weights_only=False)["val"]}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
