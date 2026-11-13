"""Constants and loaders shared by every step of the pipeline."""

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Time unit: whole UTC calendar days, counted from 1995-01-01 (MovieLens-25M starts 1995-01-09).
DAY0_UNIX_DAYS = 9131  # 1995-01-01 as days since 1970-01-01
SECONDS_PER_DAY = 86400

# Delphi's token convention: the .bin files store movies as raw tokens 1..N and the loader adds 1,
# so model token 0 is padding and model token 1 is Delphi's "no event" token.
PAD = 0
NO_EVENT = 1
RESERVED_TOKENS = [PAD, NO_EVENT]  # never predicted, never targets

SPLITS = ("train", "val", "test")
SPLIT_SEED = 42        # user split
ORDER_SEED = 1234      # random order of ratings that share a day
TARGET_SEED = 2026     # which active day is the evaluation target for each val/test user

DEFAULT_PROCESSED = "data/processed"


def day_to_date(day):
    """Model day index -> numpy datetime64[D]."""
    return np.datetime64("1995-01-01") + np.asarray(day).astype("timedelta64[D]")


@dataclass
class Split:
    """Event rows of one split plus per-user row ranges.

    data[:, 0] user index within the split, data[:, 1] day, data[:, 2] raw token (1..N).
    Rows are sorted by user and day; rows that share a day are in a fixed random order.
    """
    name: str
    data: np.ndarray
    users: pd.DataFrame

    @property
    def tokens(self):
        """Model tokens (raw token + 1) as int64."""
        return self.data[:, 2].astype(np.int64) + 1

    @property
    def days(self):
        return self.data[:, 1].astype(np.int64)


def load_split(name, processed=DEFAULT_PROCESSED):
    data = np.fromfile(os.path.join(processed, f"{name}.bin"), dtype=np.uint32).reshape(-1, 3)
    users = pd.read_csv(os.path.join(processed, f"users_{name}.csv"))
    return Split(name, data, users)


def load_eval_cases(name, processed=DEFAULT_PROCESSED):
    return pd.read_csv(os.path.join(processed, f"eval_{name}.csv"))


def load_vocab(processed=DEFAULT_PROCESSED):
    return pd.read_csv(os.path.join(processed, "vocab.csv"))


def vocab_size(processed=DEFAULT_PROCESSED):
    """Number of model tokens: the movies plus the two reserved tokens."""
    with open(os.path.join(processed, "summary.json")) as f:
        return json.load(f)["vocab"]["model_vocab_size"]
