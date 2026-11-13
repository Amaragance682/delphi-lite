"""Package the hand-in as one zip (run from anywhere): python scripts/make_handin.py [out.zip]

Included: code, README, report, slides, demo notebook, results, the logs of every run, the final model
and the checkpoints of its two extra seeds. Excluded: MovieLens files (data/, not redistributable), the
upstream Delphi clone (external/), checkpoints of runs that are not the final model (rebuild them with
scripts/run_experiments.sh) and the per-case top-100 arrays (rebuilt by scripts/final_evaluation.sh).
"""

import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["README.md", "THIRD_PARTY_NOTICES.md", "requirements.txt", ".gitignore", "models/delphi_final.pt"]
FOLDERS = ["configs", "moviedelphi", "scripts", "tests", "notebooks", "report", "slides", "results"]
RUN_FILES = ["config.json", "log.csv", "summary.json"]
SEED_CHECKPOINTS = {"m5_long_s2", "m5_long_s3"}  # seed 1 is models/delphi_final.pt


def skip(path):
    return ("__pycache__" in path or path.endswith(".pyc") or ".ipynb_checkpoints" in path
            or path.endswith("_top100.npy"))


def main(out=os.path.join(os.path.dirname(ROOT), "delphi_lite_handin.zip")):
    paths = list(FILES)
    for folder in FOLDERS:
        for dirpath, _, files in os.walk(os.path.join(ROOT, folder)):
            paths += [os.path.relpath(os.path.join(dirpath, f), ROOT) for f in files]
    for run in sorted(os.listdir(os.path.join(ROOT, "runs"))):
        if os.path.isdir(os.path.join(ROOT, "runs", run)):
            paths += [f"runs/{run}/{f}" for f in RUN_FILES + (["ckpt.pt"] if run in SEED_CHECKPOINTS else [])]
    paths = sorted(p for p in set(paths) if not skip(p))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(os.path.join(ROOT, p), os.path.join("delphi_lite", p))
    print(f"wrote {out}: {len(paths)} files, {os.path.getsize(out) / 2**20:.1f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
