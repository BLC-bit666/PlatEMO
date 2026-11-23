# Native evaluator kernels

Local target: macOS arm64, pinned CPython 3.12 from `outputs/real_world_exact_reproduction_20261119/.venv`.

Git retains source, build scripts, patches and licenses only; compiled libraries and Python caches are ignored. The pinned evaluator environment, input data and decoder rebuild assets under `outputs/` are local prerequisites, not included in a clone of this repository.

- `realapps_kernels.cpp`: one-pass exact route costs, conservative A* search, exact RGB blending and integer-sum image MSE. Build with `python3 build_kernels.py`; no fast-math, no FMA/reassociation.
- `librealapps_qr.dylib`: local reader-only build of the upstream PyPI `zxing_cpp-3.1.1.tar.gz`, SHA-256 `1051a521b21a9fe206702ad4186aeb195154e3e1badcd99576d030723f36382b`. Keeps decoder options identical to the pinned Python `read_barcode` defaults and explicitly identifies RGB pixels. Original installed package remains unchanged.
- `qr_stability.patch`: skip undefined/zero direction tracing and use the reader's existing geometric fallback. This is not a timeout-to-infeasible rule.
- `ZXing-LICENSE`: upstream Apache-2.0 license.

Reproducible decoder source, upstream tarball, build logs and regression inputs are retained in `outputs/realapps_performance_diagnosis_20261120/`. Its `code/rebuild_decoder.py` recreates this local bundle from the pinned source.

The arithmetic library contains no problem/result caches. Route heuristics are fixed lower bounds prepared once per problem instance; their cost is reported separately from per-evaluation timing. All result-cache hits still consume logical FE.
