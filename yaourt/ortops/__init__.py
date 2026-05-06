"""Python descriptions of the ORT custom ops shipped in this package."""

import platform
from pathlib import Path

from .doc import CPU_OPS, OrtOpDesc, OrtOpInput, OrtOpOutput, print_cpu_ops

__all__ = [
    "CPU_OPS",
    "FUSED_KERNEL_CUDA_LIB_PATH",
    "SPARSE_CPU_LIB_PATH",
    "OrtOpDesc",
    "OrtOpInput",
    "OrtOpOutput",
    "print_cpu_ops",
]

_HERE = Path(__file__).parent
_SYSTEM = platform.system()

if _SYSTEM == "Windows":
    _PREFIX = ""
    _EXT = ".dll"
elif _SYSTEM == "Darwin":
    _PREFIX = "lib"
    _EXT = ".dylib"
else:
    _PREFIX = "lib"
    _EXT = ".so"

FUSED_KERNEL_CUDA_LIB_PATH: Path = (
    _HERE / "fused_kernel" / "cuda" / f"{_PREFIX}ortops_fused_kernel_cuda{_EXT}"
)

SPARSE_CPU_LIB_PATH: Path = _HERE / "sparse" / "cpu_v1" / f"{_PREFIX}ortops_sparse_cpu{_EXT}"
