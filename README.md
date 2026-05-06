# yet-another-onnxruntime-extensions

[![core](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/ci_core.yml/badge.svg)](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/ci_core.yml)
[![ortops](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/ci_ortops.yml/badge.svg)](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/ci_ortops.yml)
[![coverage](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/ci_coverage.yml/badge.svg)](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/ci_coverage.yml)
[![build](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/build.yml/badge.svg)](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/build.yml)
[![mypy](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/mypy.yml/badge.svg)](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/mypy.yml)
[![Documentation](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/docs.yml/badge.svg)](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/docs.yml)
[![Style](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/style.yml/badge.svg)](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/style.yml)
[![pyrefly](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/pyrefly.yml/badge.svg)](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/pyrefly.yml)
[![Spelling](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/spelling.yml/badge.svg)](https://github.com/xadupre/yet-another-onnxruntime-extensions/actions/workflows/spelling.yml)
[![codecov](https://codecov.io/gh/xadupre/yet-another-onnxruntime-extensions/branch/main/graph/badge.svg)](https://codecov.io/gh/xadupre/yet-another-onnxruntime-extensions)
[![GitHub repo size](https://img.shields.io/github/repo-size/xadupre/yet-another-onnxruntime-extensions)](https://github.com/xadupre/yet-another-onnxruntime-extensions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img src="docs/_static/logo.svg" alt="yaourt logo" width="116"/>
</p>

**yet-another-onnxruntime-extensions** (`yaourt`) is an experimental library of
[ONNX Runtime](https://onnxruntime.ai/) extensions: custom C++ operators,
profiling utilities, and plotting helpers.

## Features

- **Custom C++ operators** (`yaourt.ortops`) — sparse CPU operators ship as
  pre-built binaries inside the wheel; fused-kernel CUDA operators require a
  CUDA-enabled CMake build (see [docs/getting_started.rst](docs/getting_started.rst)).
  Both are registered directly with ONNX Runtime.
- **Profiling tools** (`yaourt.tools`) — parse ONNX Runtime JSON profiling files
  into pandas DataFrames and visualize execution timelines and per-operator
  breakdowns with matplotlib.
- **Plot helpers** (`yaourt.plot`) — benchmark plotting and histogram utilities
  for model analysis.
- **Reference evaluator** (`yaourt.reference`) — a pure-Python ONNX evaluator
  useful for testing and debugging custom operators without a full ONNX Runtime
  build.

## Installation

```bash
pip install yet-another-onnxruntime-extensions
```

> **Note:** The pre-built wheel includes sparse CPU operators only.
> Fused-kernel CUDA operators must be compiled from source.
> With the CUDA toolkit installed, run:
>
> ```bash
> cmake -S cmake -B build -DCMAKE_BUILD_TYPE=Release
> cmake --build build --config Release
> ```
>
> See [docs/getting_started.rst](docs/getting_started.rst) for full build instructions.

Verify the installation:

```python
import yaourt
print(yaourt.__version__)
```

## Quick Start

### Run inference with ONNX Runtime

```python
import numpy as np
import onnxruntime
from yaourt.doc import demo_mlp_model

# Build a small demo MLP model (filename argument is unused)
model = demo_mlp_model("")

# Run inference
sess = onnxruntime.InferenceSession(
    model.SerializeToString(), providers=["CPUExecutionProvider"]
)
x = np.random.randn(3, 10).astype(np.float32)
(output,) = sess.run(None, {"x": x})
print("Output shape:", output.shape)
```

### Load the custom C++ operators

```python
import onnxruntime as ort
from yaourt.ortops import SPARSE_CPU_LIB_PATH

opts = ort.SessionOptions()
opts.register_custom_ops_library(str(SPARSE_CPU_LIB_PATH))
```

### Profile an ONNX Runtime session

```python
from onnxruntime import InferenceSession, SessionOptions
from yaourt.tools.js_profile import js_profile_to_dataframe, plot_ort_profile
import matplotlib.pyplot as plt

opts = SessionOptions()
opts.enable_profiling = True
opts.profile_file_prefix = "/tmp/ort_profile"

sess = InferenceSession(model.SerializeToString(), sess_options=opts,
                        providers=["CPUExecutionProvider"])
# ... run inference ...
profile_file = sess.end_profiling()

df = js_profile_to_dataframe(profile_file, first_it_out=True)
fig, ax = plt.subplots(figsize=(8, 4))
plot_ort_profile(df, ax0=ax, title="Time per operator (µs)")
plt.tight_layout()
plt.show()
```

## Documentation

Full documentation (API reference, examples, getting started guide) is available at:
[https://xadupre.github.io/yet-another-onnxruntime-extensions/](https://xadupre.github.io/yet-another-onnxruntime-extensions/)

## Contributing

Contributions are welcome! Please read the
[Getting Started for Developers](https://xadupre.github.io/yet-another-onnxruntime-extensions/getting_started.html)
guide for instructions on how to clone, build, test, and submit changes.

The project uses [black](https://black.readthedocs.io/) for formatting and
[ruff](https://docs.astral.sh/ruff/) for linting. Run both before committing:

```bash
black . && ruff check .
```

## License

[MIT](LICENSE)
