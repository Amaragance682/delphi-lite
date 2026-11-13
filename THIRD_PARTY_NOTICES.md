# Third-party code and data

## Delphi (code adapted in `moviedelphi/model.py`, `moviedelphi/train.py`, `moviedelphi/batches.py`)

Source: https://github.com/gerstung-lab/Delphi (commit fb72166be6b29d8db819227a59487e51c1df1454).
Shmatko, Jung, Gaurav et al., *Learning the natural history of human disease with generative
transformers*, Nature (2025), https://doi.org/10.1038/s41586-025-09529-3.
No Delphi weights are used: every model in this project is trained from scratch on MovieLens.

```
MIT License

Copyright (c) 2024 Gerstung Lab

MIT license applies only to code; the model weights use CC BY-NC-ND 4.0 instead.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Delphi's implementation is based on nanoGPT by Andrej Karpathy (https://github.com/karpathy/nanoGPT),
also MIT-licensed.

## MovieLens-25M (data, not included)

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
https://doi.org/10.1145/2827872

The GroupLens usage licence permits research use, requires this citation, and forbids
redistribution and commercial use without permission. The hand-in therefore contains no
MovieLens files: `scripts/download_data.sh` fetches the archive and checks its MD5.
