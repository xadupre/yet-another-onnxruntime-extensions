"""Custom operators for ONNX Runtime, organised by execution provider.

This package documents all custom operators shipped with
*yet-another-onnxruntime-extensions*.  Metadata is defined in Python so that
Sphinx can generate the API reference automatically.

Sub-modules
-----------

* :mod:`yaourt.ortops.cpu` — operators for the CPU execution provider
"""

from yaourt.ortops.cpu import CPU_OPS, DENSE_TO_SPARSE, SPARSE_TO_DENSE, OpDocumentation

__all__ = ["CPU_OPS", "DENSE_TO_SPARSE", "SPARSE_TO_DENSE", "OpDocumentation"]
