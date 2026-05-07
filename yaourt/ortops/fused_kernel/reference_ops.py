"""Python reference implementations of the fused-kernel CUDA custom ops.

These :class:`~onnx.reference.op_run.OpRun` subclasses provide CPU-only
reference kernels for every operator registered in the
``yaourt.ortops.fused_kernel.cuda`` domain. They are useful for unit-testing
model topology and numeric correctness without a CUDA device, and can be
passed directly to :class:`~yaourt.reference.ExtendedReferenceEvaluator` via
``new_ops``.

All classes set ``op_schema = None`` so that the ONNX reference runtime does
not attempt to validate attributes against a schema that does not exist for
custom-domain operators.
"""

from __future__ import annotations

import numpy
from onnx.reference.op_run import OpRun

_DOMAIN = "yaourt.ortops.fused_kernel.cuda"


# ---------------------------------------------------------------------------
# Unary operators
# ---------------------------------------------------------------------------


class NegXplus1(OpRun):
    """Computes ``1 - x`` element-wise."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, X):  # noqa: N803
        return (numpy.ones_like(X) - X,)


class ReplaceZero(OpRun):
    """Replaces zero elements with the scalar attribute ``by``."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, X, by=0.0):  # noqa: N803
        result = numpy.where(X == 0, numpy.array(by, dtype=X.dtype), X)
        return (result,)


class MulSigmoid(OpRun):
    """Computes ``x * sigmoid(x)`` element-wise (Swish activation)."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, X):  # noqa: N803
        x64 = X.astype(numpy.float64)
        sigmoid_x = 1.0 / (1.0 + numpy.exp(-x64))
        return ((X * sigmoid_x).astype(X.dtype),)


class Transpose2DCastFP16(OpRun):
    """Transposes a 2-D float32 matrix and casts the result to float16."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, X):  # noqa: N803
        return (X.T.astype(numpy.float16),)


class Transpose2DCastFP32(OpRun):
    """Transposes a 2-D float16 matrix and casts the result to float32."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, X):  # noqa: N803
        return (X.T.astype(numpy.float32),)


# ---------------------------------------------------------------------------
# Binary operators
# ---------------------------------------------------------------------------


class MulMulSigmoid(OpRun):
    """Computes ``x * y * sigmoid(y)`` element-wise."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, X, Y):  # noqa: N803
        y64 = Y.astype(numpy.float64)
        sigmoid_y = 1.0 / (1.0 + numpy.exp(-y64))
        return ((X * Y * sigmoid_y).astype(X.dtype),)


# ---------------------------------------------------------------------------
# Ternary operators — 3 inputs, 1 output
# ---------------------------------------------------------------------------


class AddMul(OpRun):
    """Computes ``(A + B) * C`` element-wise."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return ((A + B) * C,)


class MulAdd(OpRun):
    """Computes ``A * B + C`` element-wise."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return (A * B + C,)


class SubMul(OpRun):
    """Computes ``(A - B) * C`` element-wise."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return ((A - B) * C,)


class MulSub(OpRun):
    """Computes ``A * B - C`` element-wise."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return (A * B - C,)


class AddAdd(OpRun):
    """Computes ``A + B + C`` element-wise."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return (A + B + C,)


class MulMul(OpRun):
    """Computes ``A * B * C`` element-wise."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return (A * B * C,)


# ---------------------------------------------------------------------------
# Ternary operators — 3 inputs, 2 outputs
# ---------------------------------------------------------------------------


class AddSharedInput(OpRun):
    """Computes ``(A + B, A + C)`` element-wise, producing two outputs."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return (A + B, A + C)


class MulSharedInput(OpRun):
    """Computes ``(A * B, A * C)`` element-wise, producing two outputs."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return (A * B, A * C)


# ---------------------------------------------------------------------------
# Quaternary operators — 4 inputs, 1 output
# ---------------------------------------------------------------------------


class AddAddAdd(OpRun):
    """Computes ``A + B + C + D`` element-wise."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C, D):  # noqa: N803
        return (A + B + C + D,)


class MulMulMul(OpRun):
    """Computes ``A * B * C * D`` element-wise."""

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C, D):  # noqa: N803
        return (A * B * C * D,)


# ---------------------------------------------------------------------------
# Rotary positional embedding
# ---------------------------------------------------------------------------


class Rotary(OpRun):
    """Applies a rotary positional transformation to the last dimension of X.

    The ``side`` attribute (string ``"left"`` or ``"right"``) controls which
    half of the rotation is computed:

    * **left**: ``out[..., :half] = x[..., half:]``,
      ``out[..., half:] = -x[..., :half]``
    * **right**: ``out[..., :half] = -x[..., half:]``,
      ``out[..., half:] = x[..., :half]``

    The second input ``splits`` is a 1-D int64 tensor ``[half, half]`` whose
    first element provides the half-size of the last dimension.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, X, splits, side="left"):  # noqa: N803
        half = int(splits[0])
        out = numpy.empty_like(X)
        if side == "left":
            out[..., :half] = X[..., half:]
            out[..., half:] = -X[..., :half]
        else:
            out[..., :half] = -X[..., half:]
            out[..., half:] = X[..., :half]
        return (out,)


# ---------------------------------------------------------------------------
# ScatterNDOfShape
# ---------------------------------------------------------------------------

_SCATTER_REDUCTIONS = {
    "add": numpy.add.at,
    "none": None,
    "mul": numpy.multiply.at,
    "min": numpy.minimum.at,
    "max": numpy.maximum.at,
}


def _scatter_nd_of_shape(shape, indices, updates, reduction, masked_value=None):
    """Scatters *updates* into a zero tensor of *shape*.

    :param shape: 1-D int64 array giving the output shape.
    :param indices: integer indices tensor (last dimension = index depth).
    :param updates: data tensor to scatter.
    :param reduction: one of ``"add"``, ``"none"``, ``"mul"``, ``"min"``,
        ``"max"``.
    :param masked_value: when not ``None``, index entries equal to this value
        are skipped.
    :returns: the output tensor after scattering.
    """
    output_shape = tuple(int(d) for d in shape)
    output = numpy.zeros(output_shape, dtype=updates.dtype)

    index_depth = indices.shape[-1]
    flat_indices = indices.reshape(-1, index_depth)
    update_shape = updates.shape[len(indices.shape) - 1 :]
    flat_updates = updates.reshape(-1, *update_shape)

    ufunc_at = _SCATTER_REDUCTIONS.get(reduction)

    for i in range(flat_indices.shape[0]):
        idx = flat_indices[i]
        if masked_value is not None and numpy.any(idx == masked_value):
            continue
        target_idx = tuple(idx)
        if ufunc_at is not None:
            ufunc_at(output, target_idx, flat_updates[i])
        else:
            output[target_idx] = flat_updates[i]

    return output


class ScatterNDOfShape(OpRun):
    """Scatters ``updates`` into a zero tensor of shape ``shape``.

    Inputs:
     - ``shape`` — 1-D int64 tensor defining the output shape.
     - ``indices`` — integer indices tensor.
     - ``updates`` — data tensor to scatter.

    The ``reduction`` attribute (string, default ``"add"``) controls how
    conflicts are resolved.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, shape, indices, updates, reduction="add"):
        return (_scatter_nd_of_shape(shape, indices, updates, reduction),)


class MaskedScatterNDOfShape(OpRun):
    """Scatters ``updates`` into a zero tensor of shape ``shape``, skipping
    index entries equal to ``maskedValue``.

    Inputs:
     - ``shape`` — 1-D int64 tensor defining the output shape.
     - ``indices`` — integer indices tensor.
     - ``updates`` — data tensor to scatter.

    Attributes:
     - ``reduction`` (string, default ``"add"``): reduction mode.
     - ``maskedValue`` (int, default ``-1``): index value to skip.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, shape, indices, updates, reduction="add", maskedValue=-1):
        return (_scatter_nd_of_shape(shape, indices, updates, reduction, maskedValue),)


# ---------------------------------------------------------------------------
# TriMatrix
# ---------------------------------------------------------------------------


class TriMatrix(OpRun):
    """Fills a 2-D matrix whose elements depend on their position relative to
    the main diagonal.

    Inputs:
     - ``shape`` — 1-D int64 tensor ``[n_rows, n_cols]``.
     - ``csts``  — 1-D tensor with three values ``[lower, diag, upper]``.

    The output element at ``(r, c)`` equals:

    * ``csts[0]`` when ``r > c`` (lower triangle),
    * ``csts[1]`` when ``r == c`` (diagonal),
    * ``csts[2]`` when ``r < c`` (upper triangle).
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, shape, csts):
        n_rows, n_cols = int(shape[0]), int(shape[1])
        lower, diag, upper = csts[0], csts[1], csts[2]
        rows = numpy.arange(n_rows).reshape(-1, 1)
        cols = numpy.arange(n_cols).reshape(1, -1)
        result = numpy.where(rows > cols, lower, numpy.where(rows == cols, diag, upper))
        return (result.astype(csts.dtype),)


# ---------------------------------------------------------------------------
# Convenience list of all reference ops in this module
# ---------------------------------------------------------------------------

ALL_OPS = [
    NegXplus1,
    ReplaceZero,
    MulSigmoid,
    Transpose2DCastFP16,
    Transpose2DCastFP32,
    MulMulSigmoid,
    AddMul,
    MulAdd,
    SubMul,
    MulSub,
    AddAdd,
    MulMul,
    AddSharedInput,
    MulSharedInput,
    AddAddAdd,
    MulMulMul,
    Rotary,
    ScatterNDOfShape,
    MaskedScatterNDOfShape,
    TriMatrix,
]

__all__ = [cls.__name__ for cls in ALL_OPS] + ["ALL_OPS"]
