"""Tests for the Python reference implementations of fused-kernel CUDA custom ops.

These tests exercise every op in
``yaourt.ortops.fused_kernel.reference_ops`` against the same expected values
that the CUDA tests use, verifying that the reference kernels are correct.
"""

import unittest

import numpy
import numpy.testing
import onnx
import onnx.helper as oh

from yaourt.ext_test_case import ExtTestCase
from yaourt.ortops.fused_kernel.reference_ops import (
    ALL_OPS,
    AddAdd,
    AddAddAdd,
    AddMul,
    AddSharedInput,
    MaskedScatterNDOfShape,
    MulAdd,
    MulMul,
    MulMulMul,
    MulMulSigmoid,
    MulSharedInput,
    MulSigmoid,
    MulSub,
    NegXplus1,
    ReplaceZero,
    Rotary,
    ScatterNDOfShape,
    SubMul,
    Transpose2DCastFP16,
    Transpose2DCastFP32,
    TriMatrix,
)
from yaourt.reference import ExtendedReferenceEvaluator

_OP_DOMAIN = "yaourt.ortops.fused_kernel.cuda"
TFLOAT = onnx.TensorProto.FLOAT
TFLOAT16 = onnx.TensorProto.FLOAT16
TINT64 = onnx.TensorProto.INT64


def _make_unary_model(op_name, dtype=TFLOAT, shape=(4,), **attrs):
    X = oh.make_tensor_value_info("X", dtype, list(shape))
    Y = oh.make_tensor_value_info("Y", dtype, None)
    node = oh.make_node(op_name, ["X"], ["Y"], domain=_OP_DOMAIN, **attrs)
    graph = oh.make_graph([node], op_name + "Graph", [X], [Y])
    return oh.make_model(
        graph,
        opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)],
        ir_version=10,
    )


def _make_binary_model(
    op_name, dtype=TFLOAT, in_names=("X", "Y"), shape_a=(4,), shape_b=(4,), **attrs
):
    inputs = [
        oh.make_tensor_value_info(n, dtype, list(s)) for n, s in zip(in_names, [shape_a, shape_b])
    ]
    Z = oh.make_tensor_value_info("Z", dtype, None)
    node = oh.make_node(op_name, list(in_names), ["Z"], domain=_OP_DOMAIN, **attrs)
    graph = oh.make_graph([node], op_name + "Graph", inputs, [Z])
    return oh.make_model(
        graph,
        opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)],
        ir_version=10,
    )


def _make_ternary_model(
    op_name, dtype=TFLOAT, in_names=("A", "B", "C"), out_names=("Z",), shape=(4,), **attrs
):
    inputs = [oh.make_tensor_value_info(n, dtype, list(shape)) for n in in_names]
    outputs = [oh.make_tensor_value_info(n, dtype, None) for n in out_names]
    node = oh.make_node(op_name, list(in_names), list(out_names), domain=_OP_DOMAIN, **attrs)
    graph = oh.make_graph([node], op_name + "Graph", inputs, outputs)
    return oh.make_model(
        graph,
        opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)],
        ir_version=10,
    )


def _make_quaternary_model(
    op_name, dtype=TFLOAT, in_names=("A", "B", "C", "D"), shape=(4,), **attrs
):
    inputs = [oh.make_tensor_value_info(n, dtype, list(shape)) for n in in_names]
    Z = oh.make_tensor_value_info("Z", dtype, None)
    node = oh.make_node(op_name, list(in_names), ["Z"], domain=_OP_DOMAIN, **attrs)
    graph = oh.make_graph([node], op_name + "Graph", inputs, [Z])
    return oh.make_model(
        graph,
        opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)],
        ir_version=10,
    )


class TestFusedKernelReferenceOps(ExtTestCase):
    """Tests for Python reference kernels of the fused-kernel CUDA custom ops."""

    # ------------------------------------------------------------------
    # Unary ops
    # ------------------------------------------------------------------

    def test_negxplus1(self):
        """NegXplus1 computes 1 - x correctly."""
        model = _make_unary_model("NegXplus1")
        ref = ExtendedReferenceEvaluator(model, new_ops=[NegXplus1])
        x = numpy.random.rand(4, 8).astype(numpy.float32)
        (y,) = ref.run(None, {"X": x})
        numpy.testing.assert_allclose(y, 1.0 - x, rtol=1e-5)

    def test_replace_zero(self):
        """ReplaceZero replaces zero entries with the given scalar."""
        model = _make_unary_model("ReplaceZero", shape=(2, 3), by=7.0)
        ref = ExtendedReferenceEvaluator(model, new_ops=[ReplaceZero])
        x = numpy.array([[1.0, 0.0, 2.0], [0.0, 5.0, 0.0]], dtype=numpy.float32)
        (y,) = ref.run(None, {"X": x})
        expected = numpy.where(x == 0.0, 7.0, x)
        numpy.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_mul_sigmoid(self):
        """MulSigmoid computes x * sigmoid(x)."""
        model = _make_unary_model("MulSigmoid", shape=(4, 4))
        ref = ExtendedReferenceEvaluator(model, new_ops=[MulSigmoid])
        x = numpy.random.randn(4, 4).astype(numpy.float32)
        (y,) = ref.run(None, {"X": x})
        x64 = x.astype(numpy.float64)
        expected = (x * (1.0 / (1.0 + numpy.exp(-x64)))).astype(numpy.float32)
        numpy.testing.assert_allclose(y, expected, rtol=1e-4)

    def test_transpose2d_cast_fp16(self):
        """Transpose2DCastFP16 transposes a float32 2-D matrix and casts to float16."""
        model = _make_unary_model("Transpose2DCastFP16", shape=(8, 16))
        ref = ExtendedReferenceEvaluator(model, new_ops=[Transpose2DCastFP16])
        x = numpy.arange(128, dtype=numpy.float32).reshape(8, 16)
        (y,) = ref.run(None, {"X": x})
        expected = x.T.astype(numpy.float16)
        self.assertEqual(y.dtype, numpy.float16)
        numpy.testing.assert_allclose(
            y.astype(numpy.float32), expected.astype(numpy.float32), rtol=1e-3
        )

    def test_transpose2d_cast_fp32(self):
        """Transpose2DCastFP32 transposes a float16 2-D matrix and casts to float32."""
        model = _make_unary_model("Transpose2DCastFP32", dtype=TFLOAT16, shape=(8, 16))
        ref = ExtendedReferenceEvaluator(model, new_ops=[Transpose2DCastFP32])
        x = numpy.arange(128, dtype=numpy.float32).reshape(8, 16).astype(numpy.float16)
        (y,) = ref.run(None, {"X": x})
        expected = x.T.astype(numpy.float32)
        self.assertEqual(y.dtype, numpy.float32)
        numpy.testing.assert_allclose(y, expected, rtol=1e-3)

    # ------------------------------------------------------------------
    # Binary ops
    # ------------------------------------------------------------------

    def test_mul_mul_sigmoid(self):
        """MulMulSigmoid computes x * y * sigmoid(y)."""
        model = _make_binary_model("MulMulSigmoid", shape_a=(4, 4), shape_b=(4, 4))
        ref = ExtendedReferenceEvaluator(model, new_ops=[MulMulSigmoid])
        rng = numpy.random.default_rng(4)
        x = rng.standard_normal((4, 4)).astype(numpy.float32)
        y = rng.standard_normal((4, 4)).astype(numpy.float32)
        (z,) = ref.run(None, {"X": x, "Y": y})
        y64 = y.astype(numpy.float64)
        sigmoid_y = 1.0 / (1.0 + numpy.exp(-y64))
        expected = (x * y * sigmoid_y).astype(numpy.float32)
        numpy.testing.assert_allclose(z, expected, rtol=1e-4)

    # ------------------------------------------------------------------
    # Ternary ops — 1 output
    # ------------------------------------------------------------------

    def test_add_mul(self):
        """AddMul computes (A + B) * C element-wise."""
        model = _make_ternary_model("AddMul")
        ref = ExtendedReferenceEvaluator(model, new_ops=[AddMul])
        rng = numpy.random.default_rng(0)
        a, b, c = [rng.standard_normal((4,)).astype(numpy.float32) for _ in range(3)]
        (z,) = ref.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, (a + b) * c, rtol=1e-5)

    def test_mul_add(self):
        """MulAdd computes A * B + C element-wise."""
        model = _make_ternary_model("MulAdd")
        ref = ExtendedReferenceEvaluator(model, new_ops=[MulAdd])
        rng = numpy.random.default_rng(1)
        a, b, c = [rng.standard_normal((4,)).astype(numpy.float32) for _ in range(3)]
        (z,) = ref.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, a * b + c, rtol=1e-5)

    def test_sub_mul(self):
        """SubMul computes (A - B) * C element-wise."""
        model = _make_ternary_model("SubMul")
        ref = ExtendedReferenceEvaluator(model, new_ops=[SubMul])
        rng = numpy.random.default_rng(2)
        a, b, c = [rng.standard_normal((4,)).astype(numpy.float32) for _ in range(3)]
        (z,) = ref.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, (a - b) * c, rtol=1e-5)

    def test_mul_sub(self):
        """MulSub computes A * B - C element-wise."""
        model = _make_ternary_model("MulSub")
        ref = ExtendedReferenceEvaluator(model, new_ops=[MulSub])
        rng = numpy.random.default_rng(3)
        a, b, c = [rng.standard_normal((4,)).astype(numpy.float32) for _ in range(3)]
        (z,) = ref.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, a * b - c, rtol=1e-5)

    def test_add_add(self):
        """AddAdd computes A + B + C element-wise."""
        model = _make_ternary_model("AddAdd")
        ref = ExtendedReferenceEvaluator(model, new_ops=[AddAdd])
        rng = numpy.random.default_rng(5)
        a, b, c = [rng.standard_normal((4,)).astype(numpy.float32) for _ in range(3)]
        (z,) = ref.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, a + b + c, rtol=1e-5)

    def test_mul_mul(self):
        """MulMul computes A * B * C element-wise."""
        model = _make_ternary_model("MulMul")
        ref = ExtendedReferenceEvaluator(model, new_ops=[MulMul])
        rng = numpy.random.default_rng(6)
        a, b, c = [rng.standard_normal((4,)).astype(numpy.float32) for _ in range(3)]
        (z,) = ref.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, a * b * c, rtol=1e-5)

    # ------------------------------------------------------------------
    # Ternary ops — 2 outputs
    # ------------------------------------------------------------------

    def test_add_shared_input(self):
        """AddSharedInput computes (A+B, A+C) element-wise."""
        model = _make_ternary_model("AddSharedInput", out_names=("Z0", "Z1"))
        ref = ExtendedReferenceEvaluator(model, new_ops=[AddSharedInput])
        rng = numpy.random.default_rng(9)
        a, b, c = [rng.standard_normal((4,)).astype(numpy.float32) for _ in range(3)]
        z0, z1 = ref.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z0, a + b, rtol=1e-5)
        numpy.testing.assert_allclose(z1, a + c, rtol=1e-5)

    def test_mul_shared_input(self):
        """MulSharedInput computes (A*B, A*C) element-wise."""
        model = _make_ternary_model("MulSharedInput", out_names=("Z0", "Z1"))
        ref = ExtendedReferenceEvaluator(model, new_ops=[MulSharedInput])
        rng = numpy.random.default_rng(10)
        a, b, c = [rng.standard_normal((4,)).astype(numpy.float32) for _ in range(3)]
        z0, z1 = ref.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z0, a * b, rtol=1e-5)
        numpy.testing.assert_allclose(z1, a * c, rtol=1e-5)

    # ------------------------------------------------------------------
    # Quaternary ops
    # ------------------------------------------------------------------

    def test_add_add_add(self):
        """AddAddAdd computes A + B + C + D element-wise."""
        model = _make_quaternary_model("AddAddAdd")
        ref = ExtendedReferenceEvaluator(model, new_ops=[AddAddAdd])
        rng = numpy.random.default_rng(7)
        a, b, c, d = [rng.standard_normal((4,)).astype(numpy.float32) for _ in range(4)]
        (z,) = ref.run(None, {"A": a, "B": b, "C": c, "D": d})
        numpy.testing.assert_allclose(z, a + b + c + d, rtol=1e-5)

    def test_mul_mul_mul(self):
        """MulMulMul computes A * B * C * D element-wise."""
        model = _make_quaternary_model("MulMulMul")
        ref = ExtendedReferenceEvaluator(model, new_ops=[MulMulMul])
        rng = numpy.random.default_rng(8)
        a, b, c, d = [rng.standard_normal((4,)).astype(numpy.float32) for _ in range(4)]
        (z,) = ref.run(None, {"A": a, "B": b, "C": c, "D": d})
        numpy.testing.assert_allclose(z, a * b * c * d, rtol=1e-5)

    # ------------------------------------------------------------------
    # Rotary
    # ------------------------------------------------------------------

    def test_rotary_left(self):
        """Rotary left swaps halves: left_out=right_in, right_out=-left_in."""
        shape = (3, 2, 3, 4)
        X = oh.make_tensor_value_info("X", TFLOAT, list(shape))
        splits_in = oh.make_tensor_value_info("splits", TINT64, [2])
        Y = oh.make_tensor_value_info("Y", TFLOAT, list(shape))
        node = oh.make_node("Rotary", ["X", "splits"], ["Y"], domain=_OP_DOMAIN, side="left")
        graph = oh.make_graph([node], "RotaryGraph", [X, splits_in], [Y])
        model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)],
            ir_version=10,
        )
        ref = ExtendedReferenceEvaluator(model, new_ops=[Rotary])
        x = numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape) + 1.0
        half = shape[-1] // 2
        splits = numpy.array([half, half], dtype=numpy.int64)
        (y,) = ref.run(None, {"X": x, "splits": splits})
        expected = x.copy()
        expected[..., :half] = x[..., half:]
        expected[..., half:] = -x[..., :half]
        numpy.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_rotary_right(self):
        """Rotary right swaps halves: left_out=-right_in, right_out=left_in."""
        shape = (3, 2, 3, 4)
        X = oh.make_tensor_value_info("X", TFLOAT, list(shape))
        splits_in = oh.make_tensor_value_info("splits", TINT64, [2])
        Y = oh.make_tensor_value_info("Y", TFLOAT, list(shape))
        node = oh.make_node("Rotary", ["X", "splits"], ["Y"], domain=_OP_DOMAIN, side="right")
        graph = oh.make_graph([node], "RotaryGraph", [X, splits_in], [Y])
        model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)],
            ir_version=10,
        )
        ref = ExtendedReferenceEvaluator(model, new_ops=[Rotary])
        x = numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape) + 1.0
        half = shape[-1] // 2
        splits = numpy.array([half, half], dtype=numpy.int64)
        (y,) = ref.run(None, {"X": x, "splits": splits})
        expected = x.copy()
        expected[..., :half] = -x[..., half:]
        expected[..., half:] = x[..., :half]
        numpy.testing.assert_allclose(y, expected, rtol=1e-5)

    # ------------------------------------------------------------------
    # ScatterNDOfShape
    # ------------------------------------------------------------------

    def test_scatter_nd_of_shape(self):
        """ScatterNDOfShape performs scatter-add into a zero tensor of given shape."""
        output_shape_val = numpy.array([4, 6], dtype=numpy.int64)
        indices_val = numpy.array([[0], [1], [2], [0]], dtype=numpy.int64)
        updates_val = numpy.ones((4, 6), dtype=numpy.float32)

        shape_in = oh.make_tensor_value_info("shape", TINT64, [None])
        indices_in = oh.make_tensor_value_info("indices", TINT64, list(indices_val.shape))
        updates_in = oh.make_tensor_value_info("updates", TFLOAT, list(updates_val.shape))
        Y = oh.make_tensor_value_info("Y", TFLOAT, None)
        node = oh.make_node(
            "ScatterNDOfShape",
            ["shape", "indices", "updates"],
            ["Y"],
            domain=_OP_DOMAIN,
            reduction="add",
        )
        graph = oh.make_graph(
            [node], "ScatterNDOfShapeGraph", [shape_in, indices_in, updates_in], [Y]
        )
        model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)],
            ir_version=10,
        )
        ref = ExtendedReferenceEvaluator(model, new_ops=[ScatterNDOfShape])
        (y,) = ref.run(
            None, {"shape": output_shape_val, "indices": indices_val, "updates": updates_val}
        )

        expected = numpy.zeros((4, 6), dtype=numpy.float32)
        for i, idx in enumerate(indices_val[:, 0]):
            expected[idx] += updates_val[i]
        numpy.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_masked_scatter_nd_of_shape(self):
        """MaskedScatterNDOfShape skips scatter-add for masked index value (-1)."""
        output_shape_val = numpy.array([8, 4], dtype=numpy.int64)
        indices_val = numpy.array([[[0]], [[1]], [[-1]], [[2]], [[-1]], [[3]]], dtype=numpy.int64)
        updates_val = numpy.ones((6, 1, 4), dtype=numpy.float32)

        shape_in = oh.make_tensor_value_info("shape", TINT64, [None])
        indices_in = oh.make_tensor_value_info("indices", TINT64, list(indices_val.shape))
        updates_in = oh.make_tensor_value_info("updates", TFLOAT, list(updates_val.shape))
        Y = oh.make_tensor_value_info("Y", TFLOAT, None)
        node = oh.make_node(
            "MaskedScatterNDOfShape",
            ["shape", "indices", "updates"],
            ["Y"],
            domain=_OP_DOMAIN,
            reduction="add",
            maskedValue=-1,
        )
        graph = oh.make_graph(
            [node], "MaskedScatterNDOfShapeGraph", [shape_in, indices_in, updates_in], [Y]
        )
        model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)],
            ir_version=10,
        )
        ref = ExtendedReferenceEvaluator(model, new_ops=[MaskedScatterNDOfShape])
        (y,) = ref.run(
            None, {"shape": output_shape_val, "indices": indices_val, "updates": updates_val}
        )

        expected = numpy.zeros((8, 4), dtype=numpy.float32)
        for i in range(indices_val.shape[0]):
            idx = indices_val[i, 0, 0]
            if idx != -1:
                expected[idx] += updates_val[i, 0]
        numpy.testing.assert_allclose(y, expected, rtol=1e-5)

    # ------------------------------------------------------------------
    # TriMatrix
    # ------------------------------------------------------------------

    def test_tri_matrix(self):
        """TriMatrix fills a 2-D matrix with lower/diag/upper scalar constants."""
        shape_val = numpy.array([6, 6], dtype=numpy.int64)
        csts_val = numpy.array([2.0, 3.0, 4.0], dtype=numpy.float32)

        shape_in = oh.make_tensor_value_info("shape", TINT64, [2])
        csts_in = oh.make_tensor_value_info("csts", TFLOAT, [3])
        Y = oh.make_tensor_value_info("Y", TFLOAT, None)
        node = oh.make_node("TriMatrix", ["shape", "csts"], ["Y"], domain=_OP_DOMAIN)
        graph = oh.make_graph([node], "TriMatrixGraph", [shape_in, csts_in], [Y])
        model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)],
            ir_version=10,
        )
        ref = ExtendedReferenceEvaluator(model, new_ops=[TriMatrix])
        (y,) = ref.run(None, {"shape": shape_val, "csts": csts_val})

        n = 6
        i1 = numpy.arange(n).reshape((-1, 1))
        i2 = numpy.arange(n).reshape((1, -1))
        expected = numpy.empty((n, n), dtype=numpy.float32)
        expected[i1 > i2] = 2.0
        expected[i1 == i2] = 3.0
        expected[i1 < i2] = 4.0
        numpy.testing.assert_array_equal(y, expected)

    # ------------------------------------------------------------------
    # ALL_OPS list
    # ------------------------------------------------------------------

    def test_all_ops_list_completeness(self):
        """ALL_OPS contains exactly the expected operator classes."""
        expected_names = {
            "NegXplus1",
            "ReplaceZero",
            "MulSigmoid",
            "Transpose2DCastFP16",
            "Transpose2DCastFP32",
            "MulMulSigmoid",
            "AddMul",
            "MulAdd",
            "SubMul",
            "MulSub",
            "AddAdd",
            "MulMul",
            "AddSharedInput",
            "MulSharedInput",
            "AddAddAdd",
            "MulMulMul",
            "Rotary",
            "ScatterNDOfShape",
            "MaskedScatterNDOfShape",
            "TriMatrix",
        }
        actual_names = {cls.__name__ for cls in ALL_OPS}
        self.assertEqual(expected_names, actual_names)

    def test_all_ops_have_correct_domain(self):
        """Every op in ALL_OPS declares the fused-kernel CUDA domain."""
        for cls in ALL_OPS:
            with self.subTest(op=cls.__name__):
                self.assertEqual(cls.op_domain, _OP_DOMAIN)

    def test_all_ops_have_null_schema(self):
        """Every op in ALL_OPS sets op_schema = None."""
        for cls in ALL_OPS:
            with self.subTest(op=cls.__name__):
                self.assertIsNone(cls.op_schema)


if __name__ == "__main__":
    unittest.main(verbosity=2)
