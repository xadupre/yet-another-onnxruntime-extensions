"""Documentation catalogue of CPU custom ORT ops, derived from C++ source files.

Structural metadata (op name, domain, execution provider, input/output names
and element types) is parsed directly from the C++ lite-API source files at
import time so that the Python catalogue always stays in sync with the C++
implementation without any manual maintenance.

Human-readable documentation strings are parsed from ``///`` Doxygen-style
doc comments in the C++ header file, so prose descriptions live alongside
the kernel declarations and are never duplicated in Python.

Supported C++ sources
---------------------
- ``yaourt/ortops/sparse/cpu/ort_optim_cpu2_lib.cc`` — provides the op domain and the
  ``CreateLiteCustomOp`` registrations (op name → kernel class + exec provider).
- ``yaourt/ortops/sparse/cpu/ort_sparse_lite.h`` — provides the ``Compute`` method
  signatures and ``///`` doc comments used to extract input/output argument
  names, element types, and prose descriptions.

The :func:`print_cpu_ops` function renders the catalogue as plain text and is
intended to be called from a ``.. runpython::`` block in the Sphinx docs so
that the rendered output is always in sync with the C++ source.
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
# C++ source parsers
# ---------------------------------------------------------------------------


def _repo_root() -> str:
    """Returns the repository root directory derived from this file's location."""
    # This module lives at yaourt/ortops/doc.py; root is two levels up.
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


def _parse_lite_header_docs(path: str) -> dict[str, tuple[str, dict[str, str]]]:
    """Parses ``///`` Doxygen-style doc comments from a lite-API ``.h`` header.

    Locates blocks of consecutive ``///`` comment lines immediately preceding
    each ``struct`` definition, strips the ``///`` prefix, and extracts:

    - the op description (all text before the first ``@param`` tag), and
    - per-parameter descriptions (``@param[in] name desc`` and
      ``@param[out] name desc`` tags, with optional continuation lines).

    Continuation lines for a parameter are non-empty ``///`` lines that follow
    a ``@param`` tag and precede the next ``@param`` tag or an empty ``///``
    line.

    :param path: absolute path to the ``.h`` file
    :returns: ``{kernel_class: (doc, {param_name: description})}``
    """
    with open(path, encoding="utf-8") as fh:
        content = fh.read()

    doc_struct_re = re.compile(
        r"((?:[ \t]*///[^\n]*\n)+)"  # group 1: consecutive /// lines
        r"[ \t]*struct\s+(\w+)"  # group 2: struct name
    )

    result: dict[str, tuple[str, dict[str, str]]] = {}

    for m in doc_struct_re.finditer(content):
        raw_block = m.group(1)
        struct_name = m.group(2)

        # Strip the /// prefix (and one optional trailing space) from every line.
        lines = [re.sub(r"^[ \t]*///[ ]?", "", line) for line in raw_block.splitlines()]

        doc_lines: list[str] = []
        param_docs: dict[str, str] = {}
        current_param: str | None = None
        current_desc_lines: list[str] = []

        for line in lines:
            pm = re.match(r"@param\[(?:in|out)\]\s+(\w+)\s*(.*)", line)
            if pm:
                if current_param is not None:
                    param_docs[current_param] = " ".join(
                        part for part in current_desc_lines if part
                    ).strip()
                current_param = pm.group(1)
                first_desc = pm.group(2).strip()
                current_desc_lines = [first_desc] if first_desc else []
            elif current_param is not None:
                stripped = line.strip()
                if stripped:
                    current_desc_lines.append(stripped)
            else:
                doc_lines.append(line)

        if current_param is not None:
            param_docs[current_param] = " ".join(
                part for part in current_desc_lines if part
            ).strip()

        doc = "\n".join(doc_lines).strip()
        result[struct_name] = (doc, param_docs)

    return result


def _build_cpu_ops(
    lib_cc_path: str | None = None, header_path: str | None = None
) -> dict[str, OrtOpDesc]:
    """Builds the CPU_OPS catalogue by parsing C++ source files.

    Structural metadata (op name, domain, execution provider, input/output
    argument names and element types) is extracted from the C++ files.
    Human-readable descriptions are parsed from ``///`` Doxygen-style doc
    comments in the header file via :func:`_parse_lite_header_docs`.

    :param lib_cc_path: path to the lite-API lib ``.cc`` file; defaults to
        ``yaourt/ortops/sparse/cpu/ort_optim_cpu2_lib.cc`` inside the repo root.
    :param header_path: path to the lite-API ``.h`` header; defaults to
        ``yaourt/ortops/sparse/cpu/ort_sparse_lite.h`` inside the repo root.
    :returns: dict mapping op name to :class:`OrtOpDesc`; returns an empty
        dict when the C++ source files are not present.
    """
    root = _repo_root()
    if lib_cc_path is None:
        lib_cc_path = os.path.join(
            root, "yaourt", "ortops", "sparse", "cpu", "ort_optim_cpu2_lib.cc"
        )
    if header_path is None:
        header_path = os.path.join(root, "yaourt", "ortops", "sparse", "cpu", "ort_sparse_lite.h")

    if not (os.path.exists(lib_cc_path) and os.path.exists(header_path)):
        return {}

    domain, registrations = _parse_lite_lib_cc(lib_cc_path)
    kernel_params = _parse_lite_header(header_path)
    kernel_docs = _parse_lite_header_docs(header_path)

    ops: dict[str, OrtOpDesc] = {}
    for kernel_class, op_name, exec_provider in registrations:
        params = kernel_params.get(kernel_class, [])
        doc, param_docs = kernel_docs.get(kernel_class, ("", {}))
        ops[op_name] = OrtOpDesc(
            name=op_name,
            domain=domain,
            since_version=1,
            execution_provider=exec_provider,
            inputs=[
                OrtOpInput(name=n, dtype=t, description=param_docs.get(n, ""))
                for n, t, is_in in params
                if is_in
            ],
            outputs=[
                OrtOpOutput(name=n, dtype=t, description=param_docs.get(n, ""))
                for n, t, is_in in params
                if not is_in
            ],
            doc=doc,
        )

    return ops


# ---------------------------------------------------------------------------
# Public catalogue
# ---------------------------------------------------------------------------

#: All CPU custom ops provided by *yet-another-onnxruntime-extensions*, keyed
#: by op name.  Populated at import time by parsing the C++ source files.
CPU_OPS: dict[str, OrtOpDesc] = _build_cpu_ops()


def print_cpu_ops() -> None:
    """Prints the CPU custom-op catalogue to stdout.

    Renders :data:`CPU_OPS` as plain text suitable for a ``.. runpython::``
    block in the Sphinx documentation, ensuring the rendered output is always
    derived from the C++ source files.

    .. runpython::
        :showcode:

        from yaourt.ortops.doc import print_cpu_ops
        print_cpu_ops()
    """
    if not CPU_OPS:
        print("No CPU ops found (C++ source tree not present).")
        return
    for op_name, op in sorted(CPU_OPS.items()):
        print(f"{op_name}")
        print(f"  domain   : {op.domain}")
        print(f"  provider : {op.execution_provider}")
        print(f"  version  : {op.since_version}")
        if op.doc:
            for line in op.doc.splitlines():
                print(f"  {line}")
        if op.inputs:
            print("  inputs:")
            for inp in op.inputs:
                desc = f" — {inp.description}" if inp.description else ""
                print(f"    {inp.name} ({inp.dtype}){desc}")
        if op.outputs:
            print("  outputs:")
            for out in op.outputs:
                desc = f" — {out.description}" if out.description else ""
                print(f"    {out.name} ({out.dtype}){desc}")
        print()
