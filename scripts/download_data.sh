#!/usr/bin/env bash
# Download MovieLens-25M (GroupLens) into data/raw and check it against the published MD5.
# The data may not be redistributed, so it is not part of the hand-in.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
cd data/raw
curl -sS -o ml-25m.zip https://files.grouplens.org/datasets/movielens/ml-25m.zip
curl -sS -o ml-25m.zip.md5 https://files.grouplens.org/datasets/movielens/ml-25m.zip.md5
expected=$(cut -d' ' -f1 ml-25m.zip.md5)
actual=$(md5sum ml-25m.zip | cut -d' ' -f1)
if [ "$expected" != "$actual" ]; then
  echo "MD5 mismatch: expected $expected, got $actual" >&2
  exit 1
fi
unzip -q -o ml-25m.zip
echo "MovieLens-25M ready in data/raw/ml-25m (md5 $actual)"
