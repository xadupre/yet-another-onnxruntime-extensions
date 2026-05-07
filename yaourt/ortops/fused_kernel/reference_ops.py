"""Python reference implementations of the fused-kernel CUDA custom ops.

These :class:`~onnx.reference.op_run.OpRun` subclasses provide CPU-only
reference kernels for every operator registered in the
``yaourt.ortops.fused_kernel.cuda`` domain.

All kernels are listed in :data:`ALL_OPS` and are pre-registered in
:attr:`~yaourt.reference.ExtendedReferenceEvaluator.default_ops`, so models
using fused-kernel CUDA ops can be evaluated on CPU without a GPU or the
compiled shared library — no ``new_ops`` argument is required:

.. runpython::
    :showcode:

    import numpy as np
    import onnx.helper as oh
    import onnx
    from yaourt.reference import ExtendedReferenceEvaluator

    TFLOAT = onnx.TensorProto.FLOAT
    DOMAIN = "yaourt.ortops.fused_kernel.cuda"
    model = oh.make_model(
        oh.make_graph(
            [oh.make_node("MulMul", ["A", "B", "C"], ["Z"], domain=DOMAIN)],
            "mulmul_graph",
            [oh.make_tensor_value_info(n, TFLOAT, [None]) for n in "ABC"],
            [oh.make_tensor_value_info("Z", TFLOAT, [None])],
        ),
        opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(DOMAIN, 1)],
        ir_version=10,
    )
    ref = ExtendedReferenceEvaluator(model)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    c = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    (result,) = ref.run(None, {"A": a, "B": b, "C": c})
    print(result)

Individual kernels can also be passed explicitly via ``new_ops`` when only a
subset of operators is needed.

All classes set ``op_schema = None`` so that the ONNX reference runtime does
not attempt to validate attributes against a schema that does not exist for
custom-domain operators.

Operators provided
------------------

Unary (1 input → 1 output):

* :class:`NegXplus1` — ``1 - x``
* :class:`ReplaceZero` — replace zero elements with scalar attribute ``by``
* :class:`MulSigmoid` — ``x * sigmoid(x)`` (Swish)
* :class:`Transpose2DCastFP16` — transpose 2-D float32 → float16
* :class:`Transpose2DCastFP32` — transpose 2-D float16 → float32

Binary (2 inputs → 1 output):

* :class:`MulMulSigmoid` — ``x * y * sigmoid(y)``

Ternary (3 inputs → 1 output):

* :class:`AddMul` — ``(A + B) * C``
* :class:`MulAdd` — ``A * B + C``
* :class:`SubMul` — ``(A - B) * C``
* :class:`MulSub` — ``A * B - C``
* :class:`AddAdd` — ``A + B + C``
* :class:`MulMul` — ``A * B * C``

Ternary (3 inputs → 2 outputs):

* :class:`AddSharedInput` — ``(A + B, A + C)``
* :class:`MulSharedInput` — ``(A * B, A * C)``

Quaternary (4 inputs → 1 output):

* :class:`AddAddAdd` — ``A + B + C + D``
* :class:`MulMulMul` — ``A * B * C * D``

Other:

* :class:`Rotary` — rotary positional embedding (RoPE)
* :class:`ScatterNDOfShape` — scatter into a zero tensor
* :class:`MaskedScatterNDOfShape` — scatter with index masking
* :class:`TriMatrix` — triangular matrix from scalar constants
"""

from __future__ import annotations

import numpy
from onnx.reference.op_run import OpRun

_DOMAIN = "yaourt.ortops.fused_kernel.cuda"


# ---------------------------------------------------------------------------
# Unary operators
# ---------------------------------------------------------------------------


class NegXplus1(OpRun):
    """Computes ``1 - x`` element-wise.

    :param X: input tensor (any numeric dtype).
    :returns: output tensor of the same shape and dtype as *X*.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, X):  # noqa: N803
        return (numpy.ones_like(X) - X,)


class ReplaceZero(OpRun):
    """Replaces every zero element with the scalar attribute ``by``.

    :param X: input tensor (any numeric dtype).
    :param by: scalar replacement value for zero elements (default ``0.0``).
    :returns: output tensor of the same shape and dtype as *X* with zeros
        replaced by *by*.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, X, by=0.0):  # noqa: N803
        result = numpy.where(X == 0, numpy.array(by, dtype=X.dtype), X)
        return (result,)


class MulSigmoid(OpRun):
    """Computes ``x * sigmoid(x)`` element-wise (Swish / SiLU activation).

    :param X: input tensor (float32 or float64).
    :returns: output tensor of the same shape and dtype as *X*.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, X):  # noqa: N803
        x64 = X.astype(numpy.float64)
        sigmoid_x = 1.0 / (1.0 + numpy.exp(-x64))
        return ((X * sigmoid_x).astype(X.dtype),)


class Transpose2DCastFP16(OpRun):
    """Transposes a 2-D float32 matrix and casts the result to float16.

    :param X: 2-D input tensor of dtype float32 with shape ``(M, N)``.
    :returns: 2-D output tensor of dtype float16 with shape ``(N, M)``.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, X):  # noqa: N803
        return (X.T.astype(numpy.float16),)


class Transpose2DCastFP32(OpRun):
    """Transposes a 2-D float16 matrix and casts the result to float32.

    :param X: 2-D input tensor of dtype float16 with shape ``(M, N)``.
    :returns: 2-D output tensor of dtype float32 with shape ``(N, M)``.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, X):  # noqa: N803
        return (X.T.astype(numpy.float32),)


# ---------------------------------------------------------------------------
# Binary operators
# ---------------------------------------------------------------------------


class MulMulSigmoid(OpRun):
    """Computes ``x * y * sigmoid(y)`` element-wise.

    :param X: first input tensor.
    :param Y: second input tensor (used for both the multiplication and the
        sigmoid gate); must be broadcastable with *X*.
    :returns: output tensor of the same shape and dtype as *X*.
    """

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
    """Computes ``(A + B) * C`` element-wise.

    :param A: first input tensor.
    :param B: second input tensor; must be broadcastable with *A*.
    :param C: third input tensor (scale); must be broadcastable with *A* + *B*.
    :returns: output tensor of the same shape and dtype as *A*.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return ((A + B) * C,)


class MulAdd(OpRun):
    """Computes ``A * B + C`` element-wise.

    :param A: first input tensor.
    :param B: second input tensor; must be broadcastable with *A*.
    :param C: bias tensor; must be broadcastable with *A* * *B*.
    :returns: output tensor of the same shape and dtype as *A*.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return (A * B + C,)


class SubMul(OpRun):
    """Computes ``(A - B) * C`` element-wise.

    :param A: first input tensor.
    :param B: second input tensor; must be broadcastable with *A*.
    :param C: scale tensor; must be broadcastable with *A* - *B*.
    :returns: output tensor of the same shape and dtype as *A*.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return ((A - B) * C,)


class MulSub(OpRun):
    """Computes ``A * B - C`` element-wise.

    :param A: first input tensor.
    :param B: second input tensor; must be broadcastable with *A*.
    :param C: bias tensor to subtract; must be broadcastable with *A* * *B*.
    :returns: output tensor of the same shape and dtype as *A*.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return (A * B - C,)


class AddAdd(OpRun):
    """Computes ``A + B + C`` element-wise.

    :param A: first input tensor.
    :param B: second input tensor; must be broadcastable with *A*.
    :param C: third input tensor; must be broadcastable with *A* + *B*.
    :returns: output tensor of the same shape and dtype as *A*.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return (A + B + C,)


class MulMul(OpRun):
    """Computes ``A * B * C`` element-wise.

    :param A: first input tensor.
    :param B: second input tensor; must be broadcastable with *A*.
    :param C: third input tensor; must be broadcastable with *A* * *B*.
    :returns: output tensor of the same shape and dtype as *A*.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return (A * B * C,)


# ---------------------------------------------------------------------------
# Ternary operators — 3 inputs, 2 outputs
# ---------------------------------------------------------------------------


class AddSharedInput(OpRun):
    """Computes ``(A + B, A + C)`` element-wise, producing two outputs.

    :param A: shared input tensor.
    :param B: second input tensor; must be broadcastable with *A*.
    :param C: third input tensor; must be broadcastable with *A*.
    :returns: tuple ``(A + B, A + C)``, each with the same shape and dtype
        as *A*.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return (A + B, A + C)


class MulSharedInput(OpRun):
    """Computes ``(A * B, A * C)`` element-wise, producing two outputs.

    :param A: shared input tensor.
    :param B: second input tensor; must be broadcastable with *A*.
    :param C: third input tensor; must be broadcastable with *A*.
    :returns: tuple ``(A * B, A * C)``, each with the same shape and dtype
        as *A*.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C):  # noqa: N803
        return (A * B, A * C)


# ---------------------------------------------------------------------------
# Quaternary operators — 4 inputs, 1 output
# ---------------------------------------------------------------------------


class AddAddAdd(OpRun):
    """Computes ``A + B + C + D`` element-wise.

    :param A: first input tensor.
    :param B: second input tensor; must be broadcastable with *A*.
    :param C: third input tensor; must be broadcastable with *A* + *B*.
    :param D: fourth input tensor; must be broadcastable with *A* + *B* + *C*.
    :returns: output tensor of the same shape and dtype as *A*.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C, D):  # noqa: N803
        return (A + B + C + D,)


class MulMulMul(OpRun):
    """Computes ``A * B * C * D`` element-wise.

    :param A: first input tensor.
    :param B: second input tensor; must be broadcastable with *A*.
    :param C: third input tensor; must be broadcastable with *A* * *B*.
    :param D: fourth input tensor; must be broadcastable with *A* * *B* * *C*.
    :returns: output tensor of the same shape and dtype as *A*.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, A, B, C, D):  # noqa: N803
        return (A * B * C * D,)


# ---------------------------------------------------------------------------
# Rotary positional embedding
# ---------------------------------------------------------------------------


class Rotary(OpRun):
    """Applies a rotary positional transformation to the last dimension of X.

    The ``side`` attribute controls which half of the rotation is computed:

    * **left**: ``out[..., :half] = x[..., half:]``,
      ``out[..., half:] = -x[..., :half]``
    * **right**: ``out[..., :half] = -x[..., half:]``,
      ``out[..., half:] = x[..., :half]``

    :param X: input tensor; the last dimension must be ``2 * half``.
    :param splits: 1-D int64 tensor ``[half, half]``; the first element
        provides the half-size of the last dimension.
    :param side: string attribute ``"left"`` (default) or ``"right"``
        selecting the rotation direction.
    :returns: output tensor of the same shape and dtype as *X*.
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

    :param shape: 1-D int64 tensor defining the output shape.
    :param indices: integer indices tensor; the last dimension gives the index
        depth into the output tensor.
    :param updates: data tensor to scatter into the output.
    :param reduction: string attribute controlling conflict resolution; one of
        ``"add"`` (default), ``"none"``, ``"mul"``, ``"min"``, ``"max"``.
    :returns: output tensor of dtype matching *updates* and shape given by
        *shape*, filled with scattered values.
    """

    op_domain = _DOMAIN
    op_schema = None

    def _run(self, shape, indices, updates, reduction="add"):
        return (_scatter_nd_of_shape(shape, indices, updates, reduction),)


class MaskedScatterNDOfShape(OpRun):
    """Scatters ``updates`` into a zero tensor, skipping masked index entries.

    Index entries equal to ``maskedValue`` are ignored, leaving the
    corresponding output positions at zero.

    :param shape: 1-D int64 tensor defining the output shape.
    :param indices: integer indices tensor; the last dimension gives the index
        depth into the output tensor.
    :param updates: data tensor to scatter into the output.
    :param reduction: string attribute controlling conflict resolution; one of
        ``"add"`` (default), ``"none"``, ``"mul"``, ``"min"``, ``"max"``.
    :param maskedValue: integer attribute; index entries equal to this value
        are skipped (default ``-1``).
    :returns: output tensor of dtype matching *updates* and shape given by
        *shape*, filled with scattered values.
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

    The output element at ``(r, c)`` equals:

    * ``csts[0]`` when ``r > c`` (lower triangle),
    * ``csts[1]`` when ``r == c`` (diagonal),
    * ``csts[2]`` when ``r < c`` (upper triangle).

    :param shape: 1-D int64 tensor ``[n_rows, n_cols]`` giving the output
        dimensions.
    :param csts: 1-D tensor with exactly three values
        ``[lower_value, diag_value, upper_value]``.
    :returns: 2-D output tensor with shape ``(n_rows, n_cols)`` and dtype
        matching *csts*.
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
