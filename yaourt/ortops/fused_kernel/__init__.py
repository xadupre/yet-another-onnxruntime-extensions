"""Python package for fused-kernel custom ops.

Provides :mod:`~yaourt.ortops.fused_kernel.reference_ops` with CPU-only
:class:`~onnx.reference.op_run.OpRun` reference implementations for every
operator in the ``yaourt.ortops.fused_kernel.cuda`` ONNX domain.  These
kernels are registered automatically in
:attr:`~yaourt.reference.ExtendedReferenceEvaluator.default_ops` so that
models using the fused-kernel CUDA ops can be evaluated on CPU without a GPU
or the compiled CUDA shared library.
"""

from .reference_ops import ALL_OPS

__all__ = ["ALL_OPS"]
