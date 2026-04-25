"""Catalogue of CPU custom ORT ops shipped in this package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


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


#: All CPU custom ops provided by *yet-another-onnxruntime-extensions*, keyed by op name.
CPU_OPS: dict[str, OrtOpDesc] = {
    op.name: op
    for op in [
        OrtOpDesc(
            name="DenseToSparse",
            domain="yaourt.ortops.optim.cpu",
            since_version=1,
            execution_provider="CPUExecutionProvider",
            inputs=[
                OrtOpInput(
                    name="X",
                    dtype="float32",
                    description=(
                        "2-D dense float32 input tensor of shape ``[n_rows, n_cols]``."
                        "  Zero elements are not stored in the sparse encoding."
                    ),
                )
            ],
            outputs=[
                OrtOpOutput(
                    name="Y",
                    dtype="float32",
                    description=(
                        "1-D float32 output tensor that encodes the sparse representation"
                        " of **X**.  The encoding layout is implementation-defined and"
                        " intended to be consumed by the paired *SparseToDense* op."
                    ),
                )
            ],
            doc=(
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
        ),
        OrtOpDesc(
            name="SparseToDense",
            domain="yaourt.ortops.optim.cpu",
            since_version=1,
            execution_provider="CPUExecutionProvider",
            inputs=[
                OrtOpInput(
                    name="X",
                    dtype="float32",
                    description="1-D float32 sparse encoding produced by *DenseToSparse*.",
                )
            ],
            outputs=[
                OrtOpOutput(
                    name="Y",
                    dtype="float32",
                    description=(
                        "Reconstructed 2-D dense float32 tensor.  The shape is"
                        " recovered from the sparse header embedded in **X**."
                    ),
                )
            ],
            doc=(
                "Converts the compact sparse encoding produced by *DenseToSparse*"
                " back into a 2-D dense ``float32`` tensor.  Positions that were"
                " zero in the original tensor are filled with ``0.0``.\n\n"
                "**Constraints**\n\n"
                "- Input must be exactly 1-D and contain a valid sparse header.\n"
                "- The encoded shape must be 2-D.\n"
                "- Only ``float32`` element type is supported.\n"
            ),
        ),
    ]
}
