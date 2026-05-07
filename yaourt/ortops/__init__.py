"""Python descriptions of the ORT custom ops shipped in this package."""

from __future__ import annotations

import platform
from pathlib import Path

from .doc import (
    CPU_OPS,
    CUDA_OPS,
    OrtOpDesc,
    OrtOpInput,
    OrtOpOutput,
    print_cpu_ops,
    print_cpu_ops_rst,
    print_cuda_ops,
    print_cuda_ops_rst,
)

__all__ = [
    "CPU_OPS",
    "CUDA_OPS",
    "FUSED_KERNEL_CUDA_LIB_PATH",
    "SPARSE_CPU2_LIB_PATH",
    "SPARSE_CPU_LIB_PATH",
    "OrtOpDesc",
    "OrtOpInput",
    "OrtOpOutput",
    "get_ort_ext_libs",
    "print_cpu_ops",
    "print_cpu_ops_rst",
    "print_cuda_ops",
    "print_cuda_ops_rst",
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

# Registry mapping subfolder (relative to yaourt/ortops/) to its known library
# Path constants.  Using named constants keeps the mapping explicit and avoids
# scanning arbitrary directories.
_KNOWN_LIBS: dict[str, list[Path]] = {
    "sparse/cpu_v1": [SPARSE_CPU_LIB_PATH],
    "sparse/cpu": [SPARSE_CPU2_LIB_PATH],
    "fused_kernel/cuda": [FUSED_KERNEL_CUDA_LIB_PATH],
}


def get_ort_ext_libs(
    ep: str, subfolder: str | None = None, domain: str | None = None
) -> list[Path]:
    """Returns paths to ORT extension library files for the given execution provider.

    Looks up the known library paths registered for *subfolder* and returns
    those that exist on disk.  Exactly one of *subfolder* or *domain* must be
    supplied.

    When *domain* is provided it is converted to a subfolder path by stripping
    the ``"yaourt.ortops."`` prefix and replacing the remaining dots with path
    separators (e.g. ``"yaourt.ortops.sparse.cpu"`` → ``"sparse/cpu"``).

    :param ep: execution provider string (e.g. ``"CPUExecutionProvider"`` or
        ``"CUDAExecutionProvider"``).  Accepted as part of the public API for
        forward-compatibility; callers should always pass the intended provider
        so that future versions can use it to validate or filter results.
    :param subfolder: path to the subdirectory under ``yaourt/ortops/`` that
        contains the library, e.g. ``"sparse/cpu"``.
    :param domain: ONNX domain string whose prefix encodes the subfolder,
        e.g. ``"yaourt.ortops.sparse.cpu"``.
    :returns: sorted list of :class:`pathlib.Path` objects for the known
        libraries registered under *subfolder* that exist on disk.
    :raises ValueError: when neither or both of *subfolder* and *domain* are
        supplied, or when *domain* does not start with ``"yaourt.ortops."``.
    :raises KeyError: when *subfolder* has no registered libraries.
    """
    if subfolder is None and domain is None:
        raise ValueError("Supply either 'subfolder' or 'domain'.")
    if subfolder is not None and domain is not None:
        raise ValueError("Supply either 'subfolder' or 'domain', not both.")
    if domain is not None:
        if not domain.startswith(_DOMAIN_PREFIX):
            raise ValueError(f"Domain {domain!r} does not start with {_DOMAIN_PREFIX!r}.")
        subfolder = domain[len(_DOMAIN_PREFIX) :].replace(".", "/")
    if subfolder not in _KNOWN_LIBS:
        raise KeyError(
            f"No libraries registered for subfolder {subfolder!r}. "
            f"Known subfolders: {sorted(_KNOWN_LIBS)}"
        )
    return sorted(p for p in _KNOWN_LIBS[subfolder] if p.exists())
