"""
CPU custom operators shipped with *yet-another-onnxruntime-extensions*.

Every operator listed in :data:`CPU_OPS` is backed by a C++ kernel compiled
into a shared library (``libortops_optim_cpu.so`` on Linux,
``ortops_optim_cpu.dll`` on Windows, ``libortops_optim_cpu.dylib`` on macOS)
that must be built from the ``ortops/optim/cpu`` sources before use.

All CPU custom operators belong to the ``yaourt.ortops.optim.cpu`` ONNX domain.

Loading the library
-------------------

Register the shared library with :class:`onnxruntime.SessionOptions` before
creating an inference session::

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.register_custom_ops_library("/path/to/libortops_optim_cpu.so")
    sess = ort.InferenceSession(
        model_bytes,
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )

Operator catalogue
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Name
     - Domain
     - Inputs
     - Outputs
     - Description
   * - ``DenseToSparse``
     - ``yaourt.ortops.optim.cpu``
     - float32 (N×M)
     - float32 (K,)
     - Encodes a 2-D dense matrix as a compact 1-D sparse buffer.
   * - ``SparseToDense``
     - ``yaourt.ortops.optim.cpu``
     - float32 (K,)
     - float32 (N×M)
     - Decodes a sparse buffer produced by ``DenseToSparse``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class OpDocumentation:
    """Defines the interface of a single custom ONNX operator.

    :param name: Operator name as registered with ONNX Runtime.
    :param domain: ONNX domain string.
    :param execution_provider: Execution-provider type string
        (e.g. ``"CPUExecutionProvider"``).
    :param n_inputs: Number of input tensors.
    :param input_types: Human-readable description for each input tensor.
    :param n_outputs: Number of output tensors.
    :param output_types: Human-readable description for each output tensor.
    :param description: Concise description of what the operator does.
    """

    name: str
    domain: str
    execution_provider: str
    n_inputs: int
    input_types: List[str]
    n_outputs: int
    output_types: List[str]
    description: str


#: Sparse encoding of a 2-D float32 matrix.
#:
#: Converts a 2-D dense float32 tensor (shape ``(N, M)``) into a flat 1-D
#: float32 buffer that stores a compact sparse representation:
#: a header with shape metadata, followed by the non-zero indices and their
#: values.  Only 2-D inputs are accepted; a runtime error is raised otherwise.
#:
#: **Inputs**
#:
#: * ``X`` — ``float32`` tensor of shape ``(N, M)``
#:
#: **Outputs**
#:
#: * ``Y`` — ``float32`` 1-D tensor of shape ``(K,)`` containing the sparse
#:   encoding (``K`` depends on the number of non-zero elements)
#:
#: Pair with :data:`SPARSE_TO_DENSE` to round-trip a dense matrix through
#: sparse form.
DENSE_TO_SPARSE: OpDocumentation = OpDocumentation(
    name="DenseToSparse",
    domain="yaourt.ortops.optim.cpu",
    execution_provider="CPUExecutionProvider",
    n_inputs=1,
    input_types=["float32, shape (N, M) — 2-D dense matrix"],
    n_outputs=1,
    output_types=["float32, shape (K,) — flat sparse encoding"],
    description=(
        "Encodes a 2-D dense float32 matrix as a compact 1-D float32 buffer "
        "containing a sparse representation (header, non-zero indices, and values). "
        "Only 2-D inputs are supported."
    ),
)

#: Dense reconstruction from a 1-D sparse buffer.
#:
#: Decodes the flat sparse buffer produced by :data:`DENSE_TO_SPARSE` and
#: reconstructs the original 2-D dense float32 tensor.
#:
#: **Inputs**
#:
#: * ``X`` — ``float32`` 1-D tensor of shape ``(K,)`` as produced by
#:   :data:`DENSE_TO_SPARSE`
#:
#: **Outputs**
#:
#: * ``Y`` — ``float32`` tensor of shape ``(N, M)`` — the reconstructed
#:   dense matrix
SPARSE_TO_DENSE: OpDocumentation = OpDocumentation(
    name="SparseToDense",
    domain="yaourt.ortops.optim.cpu",
    execution_provider="CPUExecutionProvider",
    n_inputs=1,
    input_types=["float32, shape (K,) — flat sparse encoding produced by DenseToSparse"],
    n_outputs=1,
    output_types=["float32, shape (N, M) — reconstructed 2-D dense matrix"],
    description=(
        "Decodes the flat sparse buffer produced by DenseToSparse and "
        "reconstructs the original 2-D dense float32 tensor."
    ),
)

#: Complete list of custom operators compiled for the CPU execution provider,
#: in registration order.
CPU_OPS: List[OpDocumentation] = [DENSE_TO_SPARSE, SPARSE_TO_DENSE]

__all__ = ["CPU_OPS", "DENSE_TO_SPARSE", "SPARSE_TO_DENSE", "OpDocumentation"]
