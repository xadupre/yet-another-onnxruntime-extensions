"""Python descriptions of the ORT custom ops shipped in this package."""

from __future__ import annotations

import platform
from pathlib import Path

from .doc import CPU_OPS, OrtOpDesc, OrtOpInput, OrtOpOutput, print_cpu_ops

__all__ = [
    "CPU_OPS",
    "FUSED_KERNEL_CUDA_LIB_PATH",
    "SPARSE_CPU2_LIB_PATH",
    "SPARSE_CPU_LIB_PATH",
    "OrtOpDesc",
    "OrtOpInput",
    "OrtOpOutput",
    "get_ort_ext_libs",
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

SPARSE_CPU2_LIB_PATH: Path = _HERE / "sparse" / "cpu" / f"{_PREFIX}ortops_sparse_cpu2{_EXT}"

_DOMAIN_PREFIX = "yaourt.ortops."
_LIB_SUFFIXES = {".so", ".dll", ".dylib"}


def get_ort_ext_libs(
    ep: str, subfolder: str | None = None, domain: str | None = None
) -> list[Path]:
    """Returns paths to ORT extension library files for the given execution provider.

    Searches ``yaourt/ortops/{subfolder}`` for compiled shared libraries
    (``.so``, ``.dll``, ``.dylib``).  Exactly one of *subfolder* or *domain*
    must be supplied.

    When *domain* is provided it is converted to a subfolder path by stripping
    the ``"yaourt.ortops."`` prefix and replacing the remaining dots with path
    separators (e.g. ``"yaourt.ortops.sparse.cpu"`` → ``"sparse/cpu"``).

    :param ep: execution provider string (e.g. ``"CPUExecutionProvider"`` or
        ``"CUDAExecutionProvider"``).
    :param subfolder: path to the subdirectory under ``yaourt/ortops/`` that
        contains the library, e.g. ``"sparse/cpu"``.
    :param domain: ONNX domain string whose prefix encodes the subfolder,
        e.g. ``"yaourt.ortops.sparse.cpu"``.
    :returns: sorted list of :class:`pathlib.Path` objects pointing to the
        shared-library files found in the directory.
    :raises ValueError: when neither or both of *subfolder* and *domain* are
        supplied, or when *domain* does not start with ``"yaourt.ortops."``.
    :raises FileNotFoundError: when the derived directory does not exist.
    """
    if subfolder is None and domain is None:
        raise ValueError("Supply either 'subfolder' or 'domain'.")
    if subfolder is not None and domain is not None:
        raise ValueError("Supply either 'subfolder' or 'domain', not both.")
    if domain is not None:
        if not domain.startswith(_DOMAIN_PREFIX):
            raise ValueError(f"Domain {domain!r} does not start with {_DOMAIN_PREFIX!r}.")
        subfolder = domain[len(_DOMAIN_PREFIX) :].replace(".", "/")
    search_dir = _HERE / subfolder
    if not search_dir.is_dir():
        raise FileNotFoundError(f"Library directory not found: {search_dir}")
    return sorted(p for p in search_dir.iterdir() if p.suffix in _LIB_SUFFIXES)
