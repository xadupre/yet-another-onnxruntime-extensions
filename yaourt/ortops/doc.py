"""Documentation catalogue of custom ORT ops, derived from C++ source files.

Structural metadata (op name, domain, execution provider, input/output names
and element types) is parsed directly from the C++ source files at import time
so that the Python catalogue always stays in sync with the C++ implementation
without any manual maintenance.

Human-readable documentation strings are parsed from Doxygen-style doc
comments in the C++ header files, so prose descriptions live alongside the
kernel declarations and are never duplicated in Python.

Supported C++ sources — sparse CPU (lite API)
----------------------------------------------
- ``yaourt/ortops/sparse/cpu/ort_sparse_cpu2_lib.cc`` — provides the op domain and the
  ``CreateLiteCustomOp`` registrations (op name → kernel class + exec provider).
- ``yaourt/ortops/sparse/cpu/ort_sparse_lite.h`` — provides the ``Compute`` method
  signatures and ``///`` doc comments used to extract input/output argument
  names, element types, and prose descriptions.

Supported C++ sources — fused kernel CUDA (custom-op-base API)
---------------------------------------------------------------
- ``yaourt/ortops/fused_kernel/cuda/ort_fused_kernel_cuda_lib.cu`` — provides the op
  domain name.
- ``yaourt/ortops/fused_kernel/cuda/*.cu`` (individual kernel files) — provide
  ``GetName()``, ``GetInputTypeCount()``, ``GetOutputTypeCount()``, and
  ``GetExecutionProviderType()`` implementations.
- ``yaourt/ortops/fused_kernel/cuda/*.h`` (individual header files) — provide
  ``/** @file … @brief … */`` Doxygen doc blocks used as op descriptions.

The :func:`print_cpu_ops` / :func:`print_cpu_ops_rst` and
:func:`print_cuda_ops` / :func:`print_cuda_ops_rst` functions render the
catalogues as plain text or RST and are intended to be called from
``.. runpython::`` blocks in the Sphinx docs.
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
        ``yaourt/ortops/sparse/cpu/ort_sparse_cpu2_lib.cc`` inside the repo root.
    :param header_path: path to the lite-API ``.h`` header; defaults to
        ``yaourt/ortops/sparse/cpu/ort_sparse_lite.h`` inside the repo root.
    :returns: dict mapping op name to :class:`OrtOpDesc`; returns an empty
        dict when the C++ source files are not present.
    """
    root = _repo_root()
    if lib_cc_path is None:
        lib_cc_path = os.path.join(
            root, "yaourt", "ortops", "sparse", "cpu", "ort_sparse_cpu2_lib.cc"
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
# CUDA fused-kernel parsers
# ---------------------------------------------------------------------------


def _parse_cuda_lib_cu(path: str) -> str:
    """Parses a CUDA lib ``.cu`` file for the op domain name.

    :param path: absolute path to the ``.cu`` lib file
    :returns: domain string, or empty string when not found.
    """
    with open(path, encoding="utf-8") as fh:
        content = fh.read()

    # c_OpDomain is the C++ static string constant that holds the ONNX domain
    # for all operators registered in this library (e.g. "yaourt.ortops.fused_kernel.cuda").
    m = re.search(r'c_OpDomain\s*=\s*"([^"]+)"', content)
    if not m:
        warnings.warn(
            f"Could not find 'c_OpDomain' in {path!r}; domain will be empty.", stacklevel=2
        )
    return m.group(1) if m else ""


def _parse_cuda_kernel_cu(path: str) -> tuple[list[str], int, int, str]:
    """Parses a single CUDA kernel ``.cu`` file for op metadata.

    Extracts all op names returned by ``GetName()``, the input/output counts
    from ``GetInputTypeCount()`` / ``GetOutputTypeCount()``, and the execution
    provider from ``GetExecutionProviderType()``.

    :param path: absolute path to the ``.cu`` kernel file.
    :returns: ``(op_names, n_inputs, n_outputs, exec_provider)`` where
        *op_names* is the list of distinct string literals returned by all
        ``GetName()`` implementations in the file.  Returns an empty list of
        op names when none are found.
    """
    with open(path, encoding="utf-8") as fh:
        content = fh.read()

    # Locate GetName() method bodies using a regex that handles one level of
    # nested braces (needed for switch-statement bodies in GetName).
    # Pattern explanation:
    #   GetName\s*\(\s*\)\s*const\s*\{  — matches 'GetName() const {'
    #   (                                — capture group: method body
    #     [^}]+                          — any non-'}' chars (base case, no nesting)
    #     (?:\{[^}]*\}[^}]*)*            — optionally followed by {...} blocks (one level deep)
    #   )                                — end capture
    #   \}                               — closing brace of the method
    get_name_re = re.compile(
        r"GetName\s*\(\s*\)\s*const\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}", re.DOTALL
    )
    # Match string literals only from lines that contain a 'return' keyword,
    # which avoids picking up error-message strings in EXT_THROW() macros on
    # non-return lines.
    string_in_return_re = re.compile(r'"([A-Za-z][A-Za-z0-9_]*)"')

    op_names: list[str] = []
    for m in get_name_re.finditer(content):
        body = m.group(1)
        for line in body.splitlines():
            if "return" in line:
                for sm in string_in_return_re.finditer(line):
                    name = sm.group(1)
                    if name not in op_names:
                        op_names.append(name)

    # Extract input and output counts (take the first match in each case).
    count_re = re.compile(
        r"GetInputTypeCount\s*\(\s*\)\s*const\s*\{[^}]*return\s+(\d+)", re.DOTALL
    )
    cm = count_re.search(content)
    n_inputs = int(cm.group(1)) if cm else 0

    count_re = re.compile(
        r"GetOutputTypeCount\s*\(\s*\)\s*const\s*\{[^}]*return\s+(\d+)", re.DOTALL
    )
    cm = count_re.search(content)
    n_outputs = int(cm.group(1)) if cm else 0

    # Extract execution provider.
    ep_re = re.compile(
        r"GetExecutionProviderType\s*\(\s*\)\s*const\s*\{[^}]*return\s*\"([^\"]+)\"", re.DOTALL
    )
    em = ep_re.search(content)
    exec_provider = em.group(1) if em else "CUDAExecutionProvider"

    return op_names, n_inputs, n_outputs, exec_provider


def _parse_cuda_header_file_doc(path: str) -> str:
    """Parses the ``/** @file … @brief … */`` Doxygen doc block from a CUDA ``.h`` header.

    Extracts the content of the block comment immediately following the
    ``#pragma once`` line, strips the ``*`` prefixes, removes Doxygen
    ``@file``, ``@brief``, ``@c``, ``@f$``/``@f[``/``@f]``, ``@code``/
    ``@endcode``, and ``@tparam`` / ``@param`` tags to produce readable plain
    text suitable for the documentation catalogue.

    :param path: absolute path to the ``.h`` header file.
    :returns: plain-text description string, or empty string when no block
        comment is found.
    """
    with open(path, encoding="utf-8") as fh:
        content = fh.read()

    # Match the first /* ... */ block comment in the file.
    m = re.search(r"/\*\*(.*?)\*/", content, re.DOTALL)
    if not m:
        return ""

    raw = m.group(1)
    # Replace @f$ ... @f$ and @f[ ... @f] LaTeX math spans (including
    # multi-line spans) with a plain-text placeholder before splitting.
    raw = re.sub(r"@f\$.*?@f\$", "<math>", raw, flags=re.DOTALL)
    raw = re.sub(r"@f\[.*?@f\]", "<math>", raw, flags=re.DOTALL)
    # Strip leading ' * ' or ' *' from each line.
    lines = [re.sub(r"^\s*\*\s?", "", line) for line in raw.splitlines()]

    cleaned: list[str] = []
    for line in lines:
        # Remove @file tag + filename.
        line = re.sub(r"@file\s+\S+", "", line).strip()
        # Remove @brief tag (keep its text).
        line = re.sub(r"@brief\s*", "", line).strip()
        # Remove @c <word> inline code tags (keep the word).
        line = re.sub(r"@c\s+(\S+)", r"\1", line)
        # Drop @code and @endcode lines (fenced pseudo-code blocks).
        if re.match(r"\s*@(code|endcode)\b", line):
            continue
        # Drop @tparam and @param lines (implementation details).
        if re.match(r"\s*@(tparam|param)\b", line):
            continue
        cleaned.append(line)

    # Post-process: ensure continuation lines of bullet-list items are
    # indented so that docutils does not emit "bullet list ends without a
    # blank line; unexpected unindent." warnings.  Any non-blank, unindented
    # line that follows a bullet item (without an intervening blank line) is
    # indented by two spaces to make it a valid RST continuation.  A blank
    # line is also inserted before bullet items that directly follow non-bullet
    # content (e.g. "Inputs:" label lines), so that docutils recognises them
    # as a proper list rather than a continuation of the preceding paragraph.
    result_lines: list[str] = []
    in_bullet = False
    for line in cleaned:
        if not line:
            in_bullet = False
            result_lines.append(line)
            continue
        if line.startswith(("- ", "* ")):
            if not in_bullet and result_lines and result_lines[-1]:
                result_lines.append("")  # blank line before the bullet list
            in_bullet = True
            result_lines.append(line)
        elif in_bullet and not line.startswith(" "):
            result_lines.append("  " + line)
        else:
            result_lines.append(line)
            if not line.startswith(" "):
                in_bullet = False

    # Remove leading/trailing blank lines and join.
    doc = "\n".join(result_lines).strip()
    return doc


def _build_cuda_ops(cuda_dir: str | None = None) -> dict[str, OrtOpDesc]:
    """Builds the CUDA_OPS catalogue by parsing the fused-kernel CUDA source files.

    Scans the ``yaourt/ortops/fused_kernel/cuda/`` directory for individual
    kernel ``.cu`` files, extracts op names, input/output counts, and the
    execution provider, and pairs each op with the description from its
    corresponding ``.h`` header file.

    :param cuda_dir: path to the fused-kernel CUDA source directory; defaults
        to ``yaourt/ortops/fused_kernel/cuda/`` inside the repo root.
    :returns: dict mapping op name to :class:`OrtOpDesc`; returns an empty
        dict when the CUDA source directory is not present.
    """
    root = _repo_root()
    if cuda_dir is None:
        cuda_dir = os.path.join(root, "yaourt", "ortops", "fused_kernel", "cuda")

    lib_cu = os.path.join(cuda_dir, "ort_fused_kernel_cuda_lib.cu")
    if not os.path.isdir(cuda_dir) or not os.path.exists(lib_cu):
        return {}

    domain = _parse_cuda_lib_cu(lib_cu)

    ops: dict[str, OrtOpDesc] = {}
    for fname in sorted(os.listdir(cuda_dir)):
        if not fname.endswith(".cu") or fname == "ort_fused_kernel_cuda_lib.cu":
            continue
        cu_path = os.path.join(cuda_dir, fname)
        op_names, n_inputs, n_outputs, exec_provider = _parse_cuda_kernel_cu(cu_path)
        if not op_names:
            continue

        # Try to load the doc from the matching .h file.
        h_path = os.path.join(cuda_dir, fname.replace(".cu", ".h"))
        doc = _parse_cuda_header_file_doc(h_path) if os.path.exists(h_path) else ""

        for op_name in op_names:
            ops[op_name] = OrtOpDesc(
                name=op_name,
                domain=domain,
                since_version=1,
                execution_provider=exec_provider,
                inputs=[
                    OrtOpInput(name=f"input_{i}", dtype="T", description="")
                    for i in range(n_inputs)
                ],
                outputs=[
                    OrtOpOutput(name=f"output_{i}", dtype="T", description="")
                    for i in range(n_outputs)
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

#: All fused-kernel CUDA custom ops provided by
#: *yet-another-onnxruntime-extensions*, keyed by op name.  Populated at
#: import time by parsing the C++ source files.
CUDA_OPS: dict[str, OrtOpDesc] = _build_cuda_ops()


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


def print_cpu_ops_rst() -> None:
    """Renders the CPU custom-op catalogue as RST and writes it to stdout.

    Renders :data:`CPU_OPS` as valid reStructuredText suitable for a
    ``.. runpython:: :rst:`` block in the Sphinx documentation.  Each op is
    rendered as a sub-section with a ``list-table`` for its metadata, and
    bulleted lists for its inputs and outputs, ensuring the rendered page is
    always derived from the C++ source files without manual maintenance.

    .. runpython::
        :showcode:
        :rst:

        from yaourt.ortops.doc import print_cpu_ops_rst
        print_cpu_ops_rst()
    """
    if not CPU_OPS:
        print("*No CPU ops found (C++ source tree not present).*")
        return
    for op_name, op in sorted(CPU_OPS.items()):
        print(op_name)
        print("~" * len(op_name))
        print()
        print(".. list-table::")
        print("   :widths: 20 80")
        print("   :header-rows: 0")
        print()
        print("   * - **Domain**")
        print(f"     - ``{op.domain}``")
        print("   * - **Execution provider**")
        print(f"     - ``{op.execution_provider}``")
        print("   * - **Since version**")
        print(f"     - {op.since_version}")
        print()
        if op.doc:
            for line in op.doc.splitlines():
                print(line)
            print()
        if op.inputs:
            print("**Inputs**")
            print()
            for inp in op.inputs:
                desc = f" — {inp.description}" if inp.description else ""
                print(f"* ``{inp.name}`` (*{inp.dtype}*){desc}")
            print()
        if op.outputs:
            print("**Outputs**")
            print()
            for out in op.outputs:
                desc = f" — {out.description}" if out.description else ""
                print(f"* ``{out.name}`` (*{out.dtype}*){desc}")
            print()


def _print_ops_catalogue(
    ops: dict[str, OrtOpDesc], empty_message: str, plain: bool = False
) -> None:
    """Renders an op catalogue to stdout in plain text or RST format.

    Shared implementation used by :func:`print_cuda_ops` and
    :func:`print_cuda_ops_rst`.

    :param ops: op catalogue to render.
    :param empty_message: message to print when *ops* is empty.
    :param plain: when ``True`` renders plain text; when ``False`` renders RST.
    """
    if not ops:
        print(empty_message)
        return
    for op_name, op in sorted(ops.items()):
        if plain:
            print(f"{op_name}")
            print(f"  domain   : {op.domain}")
            print(f"  provider : {op.execution_provider}")
            print(f"  version  : {op.since_version}")
            if op.doc:
                for line in op.doc.splitlines():
                    print(f"  {line}")
            if op.inputs:
                print(f"  inputs   : {op.inputs[0].dtype} x{len(op.inputs)}")
            if op.outputs:
                print(f"  outputs  : {op.outputs[0].dtype} x{len(op.outputs)}")
        else:
            print(op_name)
            print("~" * len(op_name))
            print()
            print(".. list-table::")
            print("   :widths: 20 80")
            print("   :header-rows: 0")
            print()
            print("   * - **Domain**")
            print(f"     - ``{op.domain}``")
            print("   * - **Execution provider**")
            print(f"     - ``{op.execution_provider}``")
            print("   * - **Inputs**")
            print(f"     - {len(op.inputs)}")
            print("   * - **Outputs**")
            print(f"     - {len(op.outputs)}")
            print()
            if op.doc:
                for line in op.doc.splitlines():
                    print(line)
                print()
        print()


def print_cuda_ops() -> None:
    """Prints the fused-kernel CUDA custom-op catalogue to stdout.

    Renders :data:`CUDA_OPS` as plain text suitable for a ``.. runpython::``
    block in the Sphinx documentation, ensuring the rendered output is always
    derived from the C++ source files.

    .. runpython::
        :showcode:

        from yaourt.ortops.doc import print_cuda_ops
        print_cuda_ops()
    """
    _print_ops_catalogue(
        CUDA_OPS, empty_message="No CUDA ops found (C++ source tree not present).", plain=True
    )


def print_cuda_ops_rst() -> None:
    """Renders the fused-kernel CUDA custom-op catalogue as RST and writes it to stdout.

    Renders :data:`CUDA_OPS` as valid reStructuredText suitable for a
    ``.. runpython:: :rst:`` block in the Sphinx documentation.  Each op is
    rendered as a sub-section with a ``list-table`` for its metadata followed
    by its description, ensuring the rendered page is always derived from the
    C++ source files without manual maintenance.

    .. runpython::
        :showcode:
        :rst:

        from yaourt.ortops.doc import print_cuda_ops_rst
        print_cuda_ops_rst()
    """
    _print_ops_catalogue(
        CUDA_OPS, empty_message="*No CUDA ops found (C++ source tree not present).*", plain=False
    )
