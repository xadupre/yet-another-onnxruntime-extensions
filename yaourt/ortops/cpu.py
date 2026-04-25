"""Catalogue of CPU custom ORT ops, derived from C++ source files.

Structural metadata (op name, domain, execution provider, input/output names
and element types) is parsed directly from the C++ lite-API source files at
import time so that the Python catalogue always stays in sync with the C++
implementation without any manual maintenance.

Human-readable documentation strings are kept in :data:`_OP_DOCS` so that
prose descriptions can be edited independently of the C++ code.

Supported C++ sources
---------------------
- ``ortops/sparse/cpu/ort_optim_cpu2_lib.cc`` — provides the op domain and the
  ``CreateLiteCustomOp`` registrations (op name → kernel class + exec provider).
- ``ortops/sparse/cpu/ort_sparse_lite.h`` — provides the ``Compute`` method
  signatures used to extract input/output argument names and element types.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

_CPP_DTYPE_MAP: dict[str, str] = {
    "float": "float32",
    "double": "float64",
    "int8_t": "int8",
    "int16_t": "int16",
    "int32_t": "int32",
    "int64_t": "int64",
    "uint8_t": "uint8",
    "uint16_t": "uint16",
    "uint32_t": "uint32",
    "uint64_t": "uint64",
    "bool": "bool",
}


@dataclass
class OrtOpInput:
    """Describes one input of a custom ORT op.

    :param name: argument name used in the op signature
    :param dtype: ONNX element type (e.g. ``"float32"``)
    :param description: human-readable description of what the input represents
    """

    name: str
    dtype: str
    description: str


@dataclass
class OrtOpOutput:
    """Describes one output of a custom ORT op.

    :param name: argument name used in the op signature
    :param dtype: ONNX element type (e.g. ``"float32"``)
    :param description: human-readable description of what the output represents
    """

    name: str
    dtype: str
    description: str


@dataclass
class OrtOpDesc:
    """Describes a single custom ORT op.

    :param name: op name as registered with OrtRuntime
    :param domain: ONNX domain the op belongs to
    :param since_version: opset version in which the op was introduced
    :param execution_provider: execution provider (e.g. ``"CPUExecutionProvider"``)
    :param inputs: ordered list of input descriptors
    :param outputs: ordered list of output descriptors
    :param doc: longer plain-text description of the op's semantics
    """

    name: str
    domain: str
    since_version: int
    execution_provider: str
    inputs: List[OrtOpInput] = field(default_factory=list)
    outputs: List[OrtOpOutput] = field(default_factory=list)
    doc: str = ""


# ---------------------------------------------------------------------------
# Human-readable documentation strings (prose that does not belong in C++)
# ---------------------------------------------------------------------------

#: Per-op documentation augmenting the auto-parsed structural metadata.
#: Keys are op names; each value may contain ``"doc"``, ``"input_descs"``,
#: and ``"output_descs"`` entries.
_OP_DOCS: dict[str, dict[str, object]] = {
    "DenseToSparse": {
        "doc": (
            "Converts a 2-D dense ``float32`` tensor into a compact flat sparse"
            " encoding.  Only non-zero elements are stored together with their"
            " flat indices, yielding a 1-D ``float32`` output.\n\n"
            "The encoding format is::\n\n"
            "    [header | indices (uint32) | values (float32)]\n\n"
            "where the header records the original shape and the number of"
            " non-zero elements.  The output is suitable as input to"
            " *SparseToDense* for a lossless round-trip.\n\n"
            "**Constraints**\n\n"
            "- Input must be exactly 2-D.\n"
            "- Only ``float32`` element type is supported.\n"
        ),
        "input_descs": {
            "X": (
                "2-D dense float32 input tensor of shape ``[n_rows, n_cols]``."
                "  Zero elements are not stored in the sparse encoding."
            )
        },
        "output_descs": {
            "Y": (
                "1-D float32 output tensor that encodes the sparse representation"
                " of **X**.  The encoding layout is implementation-defined and"
                " intended to be consumed by the paired *SparseToDense* op."
            )
        },
    },
    "SparseToDense": {
        "doc": (
            "Converts the compact sparse encoding produced by *DenseToSparse*"
            " back into a 2-D dense ``float32`` tensor.  Positions that were"
            " zero in the original tensor are filled with ``0.0``.\n\n"
            "**Constraints**\n\n"
            "- Input must be exactly 1-D and contain a valid sparse header.\n"
            "- The encoded shape must be 2-D.\n"
            "- Only ``float32`` element type is supported.\n"
        ),
        "input_descs": {"X": "1-D float32 sparse encoding produced by *DenseToSparse*."},
        "output_descs": {
            "Y": (
                "Reconstructed 2-D dense float32 tensor.  The shape is"
                " recovered from the sparse header embedded in **X**."
            )
        },
    },
}

# ---------------------------------------------------------------------------
# C++ source parsers
# ---------------------------------------------------------------------------


def _repo_root() -> str:
    """Returns the repository root directory derived from this file's location."""
    # This module lives at yaourt/ortops/cpu.py; root is two levels up.
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_lite_lib_cc(path: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Parses a lite-API lib ``.cc`` file for domain and op registrations.

    :param path: absolute path to the ``.cc`` file
    :returns: ``(domain, [(kernel_class, op_name, exec_provider), ...])``
    """
    with open(path, encoding="utf-8") as fh:
        content = fh.read()

    m = re.search(r'c_OpDomain\s*=\s*"([^"]+)"', content)
    if not m:
        warnings.warn(
            f"Could not find 'c_OpDomain' in {path!r}; domain will be empty.", stacklevel=2
        )
    domain = m.group(1) if m else ""

    # CreateLiteCustomOp<ortops::KernelClass>("OpName", "ExecProvider")
    registrations: list[tuple[str, str, str]] = re.findall(
        r'CreateLiteCustomOp<[^>]*?(\w+)>\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)', content
    )
    return domain, registrations


def _parse_lite_header(path: str) -> dict[str, list[tuple[str, str, bool]]]:
    """Parses a lite-API ``.h`` header for ``Compute`` parameter lists.

    :param path: absolute path to the ``.h`` file
    :returns: ``{kernel_class: [(arg_name, dtype, is_input), ...]}``
        where each tuple is ``(arg_name: str, dtype: str, is_input: bool)``.
        *is_input* is ``True`` for ``const``-qualified parameters (inputs) and
        ``False`` for mutable reference parameters (outputs).

    .. note::
        The struct-body regex ``[^}]+`` does not support nested braces.  The
        C++ kernel structs targeted here contain only flat declarations so this
        is sufficient for the current source layout.
    """
    with open(path, encoding="utf-8") as fh:
        content = fh.read()

    result: dict[str, list[tuple[str, str, bool]]] = {}
    struct_re = re.compile(r"struct\s+(\w+)\s*\{([^}]+)\}", re.DOTALL)
    compute_re = re.compile(r"Compute\s*\(([^)]+)\)")
    tensor_re = re.compile(r"(const\s+)?Ort::Custom::Tensor<(\w+)>\s*&\s*(\w+)")

    for sm in struct_re.finditer(content):
        struct_name = sm.group(1)
        cm = compute_re.search(sm.group(2))
        if not cm:
            continue
        params: list[tuple[str, str, bool]] = [
            # (arg_name, numpy_dtype, is_input)
            # is_input is True when the param is const-qualified (read-only input),
            # False when it is a mutable reference (output to be written).
            (pm.group(3), _CPP_DTYPE_MAP.get(pm.group(2), pm.group(2)), bool(pm.group(1)))
            for pm in tensor_re.finditer(cm.group(1))
        ]
        if params:
            result[struct_name] = params

    return result


def _build_cpu_ops(
    lib_cc_path: str | None = None, header_path: str | None = None
) -> dict[str, OrtOpDesc]:
    """Builds the CPU_OPS catalogue by parsing C++ source files.

    Structural metadata (op name, domain, execution provider, input/output
    argument names and element types) is extracted from the C++ files.
    Human-readable descriptions are taken from :data:`_OP_DOCS`.

    :param lib_cc_path: path to the lite-API lib ``.cc`` file; defaults to
        ``ortops/sparse/cpu/ort_optim_cpu2_lib.cc`` inside the repo root.
    :param header_path: path to the lite-API ``.h`` header; defaults to
        ``ortops/sparse/cpu/ort_sparse_lite.h`` inside the repo root.
    :returns: dict mapping op name to :class:`OrtOpDesc`; returns an empty
        dict when the C++ source files are not present.
    """
    root = _repo_root()
    if lib_cc_path is None:
        lib_cc_path = os.path.join(root, "ortops", "sparse", "cpu", "ort_optim_cpu2_lib.cc")
    if header_path is None:
        header_path = os.path.join(root, "ortops", "sparse", "cpu", "ort_sparse_lite.h")

    if not (os.path.exists(lib_cc_path) and os.path.exists(header_path)):
        return {}

    domain, registrations = _parse_lite_lib_cc(lib_cc_path)
    kernel_params = _parse_lite_header(header_path)

    ops: dict[str, OrtOpDesc] = {}
    for kernel_class, op_name, exec_provider in registrations:
        params = kernel_params.get(kernel_class, [])
        extra = _OP_DOCS.get(op_name, {})
        input_descs: dict[str, str] = extra.get("input_descs", {})  # type: ignore[assignment]
        output_descs: dict[str, str] = extra.get("output_descs", {})  # type: ignore[assignment]
        ops[op_name] = OrtOpDesc(
            name=op_name,
            domain=domain,
            since_version=1,
            execution_provider=exec_provider,
            inputs=[
                OrtOpInput(name=n, dtype=t, description=input_descs.get(n, ""))
                for n, t, is_in in params
                if is_in
            ],
            outputs=[
                OrtOpOutput(name=n, dtype=t, description=output_descs.get(n, ""))
                for n, t, is_in in params
                if not is_in
            ],
            doc=extra.get("doc", ""),  # type: ignore[arg-type]
        )

    return ops


# ---------------------------------------------------------------------------
# Public catalogue
# ---------------------------------------------------------------------------

#: All CPU custom ops provided by *yet-another-onnxruntime-extensions*, keyed
#: by op name.  Populated at import time by parsing the C++ source files.
CPU_OPS: dict[str, OrtOpDesc] = _build_cpu_ops()
