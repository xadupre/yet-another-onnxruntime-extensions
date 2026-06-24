"""Exposes CPU feature detection from the fused-kernel CPU shared library."""

from __future__ import annotations

import ctypes
import platform
from functools import lru_cache
from pathlib import Path

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

_FUSED_KERNEL_CPU_LIB_PATH = (
    _HERE / "fused_kernel" / "cpu" / f"{_PREFIX}ortops_fused_kernel_cpu{_EXT}"
)


@lru_cache(maxsize=1)
def _load_cpu_features_lib() -> ctypes.CDLL:
    """Loads the fused-kernel CPU shared library once."""
    if not _FUSED_KERNEL_CPU_LIB_PATH.exists():
        raise FileNotFoundError(f"CPU extension library not found: {_FUSED_KERNEL_CPU_LIB_PATH}")
    return ctypes.CDLL(str(_FUSED_KERNEL_CPU_LIB_PATH))


def cpu_supports_avx2() -> bool:
    """Returns whether the running CPU supports AVX2 instructions."""
    lib = _load_cpu_features_lib()
    func = lib.CpuSupportsAvx2
    func.restype = ctypes.c_bool
    return bool(func())


def cpu_supports_avx512f() -> bool:
    """Returns whether the running CPU supports AVX512F instructions."""
    lib = _load_cpu_features_lib()
    func = lib.CpuSupportsAvx512f
    func.restype = ctypes.c_bool
    return bool(func())


def cpu_supports_f16c() -> bool:
    """Returns whether the running CPU supports F16C instructions."""
    lib = _load_cpu_features_lib()
    func = lib.CpuSupportsF16c
    func.restype = ctypes.c_bool
    return bool(func())
