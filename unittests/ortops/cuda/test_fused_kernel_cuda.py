"""
Tests for the CUDA custom ORT ops built by cmake.

The shared library ``yaourt/ortops/fused_kernel/cuda/libortops_fused_kernel_cuda.so`` must be
built with ``cmake --build cmake`` before running this test.  Tests are
skipped when the library is absent or when no CUDA device is available.
"""

import os
import platform
import unittest

import numpy
import onnx
import onnx.helper as oh

from yaourt.ext_test_case import ExtTestCase, requires_cuda_onnxruntime, requires_onnxruntime

# Path to the shared library produced by the cmake build.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SYSTEM = platform.system()
if _SYSTEM == "Windows":
    _LIB_NAME = "ortops_fused_kernel_cuda.dll"
elif _SYSTEM == "Darwin":
    _LIB_NAME = "libortops_fused_kernel_cuda.dylib"
else:
    _LIB_NAME = "libortops_fused_kernel_cuda.so"
_LIB_PATH = os.path.join(_REPO_ROOT, "yaourt", "ortops", "fused_kernel", "cuda", _LIB_NAME)
_OP_DOMAIN = "yaourt.ortops.fused_kernel.cuda"


def _lib_available() -> bool:
    """Returns True if the CUDA custom op shared library is present."""
    return os.path.exists(_LIB_PATH)


def _get_bfloat16_dtype():
    """Returns the available numpy-compatible bfloat16 dtype, if any."""
    if hasattr(numpy, "bfloat16"):
        return numpy.bfloat16
    try:
        import ml_dtypes
    except ImportError:
        return None
    return ml_dtypes.bfloat16


_BFLOAT16_DTYPE = _get_bfloat16_dtype()


@unittest.skipUnless(_lib_available(), f"CUDA custom op library not found at {_LIB_PATH!r}")
@requires_cuda_onnxruntime()
@requires_onnxruntime("1.18")
class TestFusedKernelCudaCustomOps(ExtTestCase):
    """Tests for CUDA custom ops (NegXplus1, ReplaceZero, MulSigmoid, etc.)."""

    def _make_inference_session(self, model_bytes: bytes):
        """Creates an OrtInferenceSession with the custom op library loaded (CUDA EP)."""
        self.assertIn("cuda", _LIB_PATH)
        return self.make_inference_session(
            model_bytes,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            custom_ops_library=_LIB_PATH,
        )

    def _make_unary_model(self, op_name: str, dtype_onnx: int, shape, **kwargs) -> bytes:
        """Builds an ONNX model with a single custom unary op."""
        X = oh.make_tensor_value_info("X", dtype_onnx, list(shape))
        Y = oh.make_tensor_value_info("Y", dtype_onnx, list(shape))
        node = oh.make_node(op_name, inputs=["X"], outputs=["Y"], domain=_OP_DOMAIN, **kwargs)
        graph = oh.make_graph([node], op_name + "Graph", [X], [Y])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def _make_binary_model(
        self, op_name: str, dtype_onnx: int, shape_a, shape_b, **kwargs
    ) -> bytes:
        """Builds an ONNX model with a single custom binary op."""
        X = oh.make_tensor_value_info("X", dtype_onnx, list(shape_a))
        Y_in = oh.make_tensor_value_info("Y", dtype_onnx, list(shape_b))
        Z = oh.make_tensor_value_info("Z", dtype_onnx, None)
        node = oh.make_node(
            op_name, inputs=["X", "Y"], outputs=["Z"], domain=_OP_DOMAIN, **kwargs
        )
        graph = oh.make_graph([node], op_name + "Graph", [X, Y_in], [Z])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def _make_ternary_model(
        self, op_name: str, dtype_onnx: int, shape_a, shape_b, shape_c, **kwargs
    ) -> bytes:
        """Builds an ONNX model with a single custom ternary op."""
        A = oh.make_tensor_value_info("A", dtype_onnx, list(shape_a))
        B = oh.make_tensor_value_info("B", dtype_onnx, list(shape_b))
        C = oh.make_tensor_value_info("C", dtype_onnx, list(shape_c))
        Z = oh.make_tensor_value_info("Z", dtype_onnx, None)
        node = oh.make_node(
            op_name, inputs=["A", "B", "C"], outputs=["Z"], domain=_OP_DOMAIN, **kwargs
        )
        graph = oh.make_graph([node], op_name + "Graph", [A, B, C], [Z])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def _make_quaternary_model(
        self, op_name: str, dtype_onnx: int, shape_a, shape_b, shape_c, shape_d, **kwargs
    ) -> bytes:
        """Builds an ONNX model with a single custom 4-input op."""
        A = oh.make_tensor_value_info("A", dtype_onnx, list(shape_a))
        B = oh.make_tensor_value_info("B", dtype_onnx, list(shape_b))
        C = oh.make_tensor_value_info("C", dtype_onnx, list(shape_c))
        D = oh.make_tensor_value_info("D", dtype_onnx, list(shape_d))
        Z = oh.make_tensor_value_info("Z", dtype_onnx, None)
        node = oh.make_node(
            op_name, inputs=["A", "B", "C", "D"], outputs=["Z"], domain=_OP_DOMAIN, **kwargs
        )
        graph = oh.make_graph([node], op_name + "Graph", [A, B, C, D], [Z])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def _make_shared_input_model(
        self, op_name: str, dtype_onnx: int, shape_a, shape_b, shape_c, **kwargs
    ) -> bytes:
        """Builds an ONNX model for AddSharedInput/MulSharedInput (3 in, 2 out)."""
        A = oh.make_tensor_value_info("A", dtype_onnx, list(shape_a))
        B = oh.make_tensor_value_info("B", dtype_onnx, list(shape_b))
        C = oh.make_tensor_value_info("C", dtype_onnx, list(shape_c))
        Z0 = oh.make_tensor_value_info("Z0", dtype_onnx, None)
        Z1 = oh.make_tensor_value_info("Z1", dtype_onnx, None)
        node = oh.make_node(
            op_name,
            inputs=["A", "B", "C"],
            outputs=["Z0", "Z1"],
            domain=_OP_DOMAIN,
            name=f"type_{op_name}",
            **kwargs,
        )
        graph = oh.make_graph([node], op_name + "Graph", [A, B, C], [Z0, Z1])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 22), oh.make_opsetid(_OP_DOMAIN, 1)]
        )
        model.ir_version = 10
        return model.SerializeToString()

    def _make_rotary_model(self, dtype_onnx: int, shape, side: str) -> bytes:
        """Builds an ONNX model for the Rotary op."""
        X = oh.make_tensor_value_info("X", dtype_onnx, list(shape))
        splits = oh.make_tensor_value_info("splits", onnx.TensorProto.INT64, [2])
        Y = oh.make_tensor_value_info("Y", dtype_onnx, list(shape))
        node = oh.make_node(
            "Rotary", inputs=["X", "splits"], outputs=["Y"], domain=_OP_DOMAIN, side=side
        )
        graph = oh.make_graph([node], "RotaryGraph", [X, splits], [Y])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def _make_scatter_nd_of_shape_model(
        self, dtype_onnx: int, indices_shape, updates_shape, **kwargs
    ) -> bytes:
        """Builds an ONNX model for the ScatterNDOfShape op."""
        shape_in = oh.make_tensor_value_info("shape", onnx.TensorProto.INT64, [None])
        indices = oh.make_tensor_value_info(
            "indices", onnx.TensorProto.INT64, list(indices_shape)
        )
        updates = oh.make_tensor_value_info("updates", dtype_onnx, list(updates_shape))
        Y = oh.make_tensor_value_info("Y", dtype_onnx, None)
        node = oh.make_node(
            "ScatterNDOfShape",
            inputs=["shape", "indices", "updates"],
            outputs=["Y"],
            domain=_OP_DOMAIN,
            reduction="add",
            **kwargs,
        )
        graph = oh.make_graph([node], "ScatterNDOfShapeGraph", [shape_in, indices, updates], [Y])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def _make_masked_scatter_nd_model(
        self, dtype_onnx: int, indices_shape, updates_shape, masked_value: int = -1
    ) -> bytes:
        """Builds an ONNX model for the MaskedScatterNDOfShape op."""
        shape_in = oh.make_tensor_value_info("shape", onnx.TensorProto.INT64, [None])
        indices = oh.make_tensor_value_info(
            "indices", onnx.TensorProto.INT64, list(indices_shape)
        )
        updates = oh.make_tensor_value_info("updates", dtype_onnx, list(updates_shape))
        Y = oh.make_tensor_value_info("Y", dtype_onnx, None)
        node = oh.make_node(
            "MaskedScatterNDOfShape",
            inputs=["shape", "indices", "updates"],
            outputs=["Y"],
            domain=_OP_DOMAIN,
            reduction="add",
            maskedValue=masked_value,
        )
        graph = oh.make_graph(
            [node], "MaskedScatterNDOfShapeGraph", [shape_in, indices, updates], [Y]
        )
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def _make_transpose_cast_model(
        self, op_name: str, input_dtype_onnx: int, output_dtype_onnx: int, shape
    ) -> bytes:
        """Builds an ONNX model for Transpose2DCastFP16/Transpose2DCastFP32."""
        X = oh.make_tensor_value_info("X", input_dtype_onnx, list(shape))
        Y = oh.make_tensor_value_info("Y", output_dtype_onnx, None)
        node = oh.make_node(op_name, inputs=["X"], outputs=["Y"], domain=_OP_DOMAIN)
        graph = oh.make_graph([node], op_name + "Graph", [X], [Y])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def _make_tri_matrix_model(self, dtype_onnx: int) -> bytes:
        """Builds an ONNX model for the TriMatrix op."""
        shape_in = oh.make_tensor_value_info("shape", onnx.TensorProto.INT64, [2])
        csts = oh.make_tensor_value_info("csts", dtype_onnx, [3])
        Y = oh.make_tensor_value_info("Y", dtype_onnx, None)
        node = oh.make_node(
            "TriMatrix", inputs=["shape", "csts"], outputs=["Y"], domain=_OP_DOMAIN
        )
        graph = oh.make_graph([node], "TriMatrixGraph", [shape_in, csts], [Y])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def _to_bfloat16(self, value):
        """Converts an array to bfloat16 through float32.

        Returns:
            The converted bfloat16 array.
        """
        return value.astype(numpy.float32).astype(_BFLOAT16_DTYPE)

    def _assert_bfloat16_allclose(
        self, got, expected, rtol: float = 1e-2, atol: float = 1e-2
    ) -> None:
        """Compares two arrays using float32 views with bfloat16-friendly tolerances."""
        self.assertEqual(got.dtype, _BFLOAT16_DTYPE)
        self.assertEqual(expected.dtype, _BFLOAT16_DTYPE)
        numpy.testing.assert_allclose(
            got.astype(numpy.float32), expected.astype(numpy.float32), rtol=rtol, atol=atol
        )

    def test_lib_path_exists(self):
        """Sanity check: the library file is present on disk."""
        self.assertTrue(os.path.exists(_LIB_PATH), f"Library not found: {_LIB_PATH}")

    def test_negxplus1_float32(self):
        """NegXplus1 computes 1 - x correctly for float32."""
        shape = (4, 8)
        model = self._make_unary_model("NegXplus1", onnx.TensorProto.FLOAT, shape)
        sess = self._make_inference_session(model)

        x = numpy.random.rand(*shape).astype(numpy.float32)
        (y,) = sess.run(None, {"X": x})
        numpy.testing.assert_allclose(y, 1.0 - x, rtol=1e-5)

    @unittest.skipUnless(_BFLOAT16_DTYPE is not None, "No bfloat16 dtype available")
    def test_negxplus1_bfloat16(self):
        """NegXplus1 computes 1 - x correctly for bfloat16."""
        import onnx

        shape = (4, 8)
        model = self._make_unary_model("NegXplus1", onnx.TensorProto.BFLOAT16, shape)
        sess = self._make_inference_session(model)

        x = numpy.random.rand(*shape).astype(numpy.float32).astype(_BFLOAT16_DTYPE)
        (y,) = sess.run(None, {"X": x})
        expected = (1.0 - x.astype(numpy.float32)).astype(_BFLOAT16_DTYPE)
        numpy.testing.assert_allclose(
            y.astype(numpy.float32), expected.astype(numpy.float32), rtol=5e-3
        )

    def test_replace_zero_float32(self):
        """ReplaceZero replaces zero entries with the given scalar."""
        shape = (2, 3)
        model = self._make_unary_model("ReplaceZero", onnx.TensorProto.FLOAT, shape, by=7.0)
        sess = self._make_inference_session(model)

        x = numpy.array([[1.0, 0.0, 2.0], [0.0, 5.0, 0.0]], dtype=numpy.float32)
        (y,) = sess.run(None, {"X": x})
        expected = numpy.where(x == 0.0, 7.0, x)
        numpy.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_mul_sigmoid_float32(self):
        """MulSigmoid computes x * sigmoid(x) (Swish activation)."""
        shape = (4, 4)
        model = self._make_unary_model("MulSigmoid", onnx.TensorProto.FLOAT, shape)
        sess = self._make_inference_session(model)

        x = numpy.random.randn(*shape).astype(numpy.float32)
        (y,) = sess.run(None, {"X": x})

        sigmoid_x = 1.0 / (1.0 + numpy.exp(-x.astype(numpy.float64)))
        expected = (x * sigmoid_x).astype(numpy.float32)
        numpy.testing.assert_allclose(y, expected, rtol=1e-4)

    def test_addmul_float32(self):
        """AddMul computes (A + B) * C element-wise."""
        shape = (4, 4)
        model = self._make_ternary_model("AddMul", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(0)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, (a + b) * c, rtol=1e-5)

    @unittest.skipUnless(_BFLOAT16_DTYPE is not None, "No bfloat16 dtype available")
    def test_addmul_bfloat16(self):
        """AddMul computes (A + B) * C element-wise for bfloat16."""
        import onnx

        shape = (4, 4)
        model = self._make_ternary_model("AddMul", onnx.TensorProto.BFLOAT16, shape, shape, shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(12)
        a = rng.standard_normal(shape).astype(numpy.float32).astype(_BFLOAT16_DTYPE)
        b = rng.standard_normal(shape).astype(numpy.float32).astype(_BFLOAT16_DTYPE)
        c = rng.standard_normal(shape).astype(numpy.float32).astype(_BFLOAT16_DTYPE)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        expected = (a.astype(numpy.float32) + b.astype(numpy.float32)) * c.astype(numpy.float32)
        numpy.testing.assert_allclose(z.astype(numpy.float32), expected, rtol=1e-2, atol=1e-2)

    @unittest.skipUnless(_BFLOAT16_DTYPE is not None, "No bfloat16 dtype available")
    def test_bfloat16_fused_kernel_suite(self):
        """Tests bfloat16 execution for the fused-kernel bf16 coverage suite."""
        import onnx

        shape = (4, 4)
        rng = numpy.random.default_rng(42)

        a = self._to_bfloat16(rng.standard_normal(shape))
        b = self._to_bfloat16(rng.standard_normal(shape))
        c = self._to_bfloat16(rng.standard_normal(shape))
        d = self._to_bfloat16(rng.standard_normal(shape))

        # Unary kernels.
        model = self._make_unary_model("ReplaceZero", onnx.TensorProto.BFLOAT16, shape, by=7.0)
        sess = self._make_inference_session(model)
        x = self._to_bfloat16(
            numpy.array(
                [
                    [1.0, 0.0, 2.0, 0.0],
                    [0.0, 5.0, 0.0, 8.0],
                    [0.0, 0.0, 3.0, 4.0],
                    [9.0, 0.0, 0.0, 6.0],
                ]
            )
        )
        (y,) = sess.run(None, {"X": x})
        expected = self._to_bfloat16(
            numpy.where(x.astype(numpy.float32) == 0.0, 7.0, x.astype(numpy.float32))
        )
        self._assert_bfloat16_allclose(y, expected)

        model = self._make_unary_model("MulSigmoid", onnx.TensorProto.BFLOAT16, shape)
        sess = self._make_inference_session(model)
        x = self._to_bfloat16(rng.standard_normal(shape))
        (y,) = sess.run(None, {"X": x})
        x32 = x.astype(numpy.float32)
        sigmoid_x = 1.0 / (1.0 + numpy.exp(-x32))
        expected = self._to_bfloat16(x32 * sigmoid_x)
        self._assert_bfloat16_allclose(y, expected, rtol=2e-2, atol=2e-2)

        # Ternary kernels.
        ternary_cases = [
            ("MulAdd", lambda a32, b32, c32: a32 * b32 + c32),
            ("SubMul", lambda a32, b32, c32: (a32 - b32) * c32),
            ("MulSub", lambda a32, b32, c32: a32 * b32 - c32),
            ("AddAdd", lambda a32, b32, c32: a32 + b32 + c32),
            ("MulMul", lambda a32, b32, c32: a32 * b32 * c32),
        ]
        for op_name, expected_fn in ternary_cases:
            model = self._make_ternary_model(
                op_name, onnx.TensorProto.BFLOAT16, shape, shape, shape
            )
            sess = self._make_inference_session(model)
            (z,) = sess.run(None, {"A": a, "B": b, "C": c})
            expected = self._to_bfloat16(
                expected_fn(
                    a.astype(numpy.float32), b.astype(numpy.float32), c.astype(numpy.float32)
                )
            )
            self._assert_bfloat16_allclose(z, expected)

        # Quaternary kernels.
        quaternary_cases = [
            ("AddAddAdd", lambda a32, b32, c32, d32: a32 + b32 + c32 + d32),
            ("MulMulMul", lambda a32, b32, c32, d32: a32 * b32 * c32 * d32),
        ]
        for op_name, expected_fn in quaternary_cases:
            model = self._make_quaternary_model(
                op_name, onnx.TensorProto.BFLOAT16, shape, shape, shape, shape
            )
            sess = self._make_inference_session(model)
            (z,) = sess.run(None, {"A": a, "B": b, "C": c, "D": d})
            expected = self._to_bfloat16(
                expected_fn(
                    a.astype(numpy.float32),
                    b.astype(numpy.float32),
                    c.astype(numpy.float32),
                    d.astype(numpy.float32),
                )
            )
            self._assert_bfloat16_allclose(z, expected)

        # Shared-input kernels.
        shared_cases = [
            ("AddSharedInput", lambda a32, b32, c32: (a32 + b32, a32 + c32)),
            ("MulSharedInput", lambda a32, b32, c32: (a32 * b32, a32 * c32)),
        ]
        for op_name, expected_fn in shared_cases:
            model = self._make_shared_input_model(
                op_name, onnx.TensorProto.BFLOAT16, shape, shape, shape
            )
            sess = self._make_inference_session(model)
            z0, z1 = sess.run(None, {"A": a, "B": b, "C": c})
            e0, e1 = expected_fn(
                a.astype(numpy.float32), b.astype(numpy.float32), c.astype(numpy.float32)
            )
            self._assert_bfloat16_allclose(z0, self._to_bfloat16(e0))
            self._assert_bfloat16_allclose(z1, self._to_bfloat16(e1))

        # Binary kernel.
        model = self._make_binary_model("MulMulSigmoid", onnx.TensorProto.BFLOAT16, shape, shape)
        sess = self._make_inference_session(model)
        (z,) = sess.run(None, {"X": a, "Y": b})
        b32 = b.astype(numpy.float32)
        sigmoid_b = 1.0 / (1.0 + numpy.exp(-b32))
        expected = self._to_bfloat16(a.astype(numpy.float32) * b32 * sigmoid_b)
        self._assert_bfloat16_allclose(z, expected, rtol=2e-2, atol=2e-2)

        # Rotary kernel.
        rotary_shape = (3, 2, 3, 4)
        x = self._to_bfloat16(numpy.arange(numpy.prod(rotary_shape)).reshape(rotary_shape) + 1.0)
        half = rotary_shape[-1] // 2
        splits = numpy.array([half, half], dtype=numpy.int64)
        expected_left = x.astype(numpy.float32).copy()
        expected_left[..., :half] = x.astype(numpy.float32)[..., half:]
        expected_left[..., half:] = -x.astype(numpy.float32)[..., :half]
        model = self._make_rotary_model(onnx.TensorProto.BFLOAT16, rotary_shape, "left")
        sess = self._make_inference_session(model)
        (y_left,) = sess.run(None, {"X": x, "splits": splits})
        self._assert_bfloat16_allclose(y_left, self._to_bfloat16(expected_left))

        expected_right = x.astype(numpy.float32).copy()
        expected_right[..., :half] = -x.astype(numpy.float32)[..., half:]
        expected_right[..., half:] = x.astype(numpy.float32)[..., :half]
        model = self._make_rotary_model(onnx.TensorProto.BFLOAT16, rotary_shape, "right")
        sess = self._make_inference_session(model)
        (y_right,) = sess.run(None, {"X": x, "splits": splits})
        self._assert_bfloat16_allclose(y_right, self._to_bfloat16(expected_right))

        # TriMatrix kernel.
        shape_i64 = numpy.array([6, 6], dtype=numpy.int64)
        csts = self._to_bfloat16(numpy.array([2.0, 3.0, 4.0], dtype=numpy.float32))
        model = self._make_tri_matrix_model(onnx.TensorProto.BFLOAT16)
        sess = self._make_inference_session(model)
        (y,) = sess.run(None, {"shape": shape_i64, "csts": csts})
        n = int(shape_i64[0])
        i1 = numpy.arange(n).reshape((-1, 1))
        i2 = numpy.arange(n).reshape((1, -1))
        expected = numpy.empty((n, n), dtype=numpy.float32)
        expected[i1 > i2] = 2.0
        expected[i1 == i2] = 3.0
        expected[i1 < i2] = 4.0
        self._assert_bfloat16_allclose(y, self._to_bfloat16(expected))

    def test_muladd_float32(self):
        """MulAdd computes A * B + C element-wise."""
        shape = (4, 4)
        model = self._make_ternary_model("MulAdd", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(1)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, a * b + c, rtol=1e-5)

    def test_submul_float32(self):
        """SubMul computes (A - B) * C element-wise."""
        shape = (4, 4)
        model = self._make_ternary_model("SubMul", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(2)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, (a - b) * c, rtol=1e-5)

    def test_mulsub_float32(self):
        """MulSub computes A * B - C element-wise."""
        shape = (4, 4)
        model = self._make_ternary_model("MulSub", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(3)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, a * b - c, rtol=1e-5)

    def test_mul_mul_sigmoid_float32(self):
        """MulMulSigmoid computes x * y * sigmoid(y) element-wise."""
        shape = (4, 4)
        model = self._make_binary_model("MulMulSigmoid", onnx.TensorProto.FLOAT, shape, shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(4)
        x = rng.standard_normal(shape).astype(numpy.float32)
        y = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"X": x, "Y": y})
        sigmoid_y = 1.0 / (1.0 + numpy.exp(-y.astype(numpy.float64)))
        expected = (x * y * sigmoid_y).astype(numpy.float32)
        numpy.testing.assert_allclose(z, expected, rtol=1e-4)

    def test_addadd_float32(self):
        """AddAdd computes A + B + C element-wise."""
        shape = (4, 4)
        model = self._make_ternary_model("AddAdd", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(5)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, a + b + c, rtol=1e-5)

    def test_mulmul_float32(self):
        """MulMul computes A * B * C element-wise."""
        shape = (4, 4)
        model = self._make_ternary_model("MulMul", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(6)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, a * b * c, rtol=1e-5)

    def test_addaddadd_float32(self):
        """AddAddAdd computes A + B + C + D element-wise."""
        shape = (4, 4)
        model = self._make_quaternary_model(
            "AddAddAdd", onnx.TensorProto.FLOAT, shape, shape, shape, shape
        )
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(7)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        d = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c, "D": d})
        numpy.testing.assert_allclose(z, a + b + c + d, rtol=1e-5)

    def test_mulmulmul_float32(self):
        """MulMulMul computes A * B * C * D element-wise."""
        shape = (4, 4)
        model = self._make_quaternary_model(
            "MulMulMul", onnx.TensorProto.FLOAT, shape, shape, shape, shape
        )
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(8)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        d = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c, "D": d})
        numpy.testing.assert_allclose(z, a * b * c * d, rtol=1e-5)

    def test_add_shared_input_float32(self):
        """AddSharedInput computes (A+B, A+C) as two outputs element-wise."""
        shape = (4, 4)
        model = self._make_shared_input_model(
            "AddSharedInput", onnx.TensorProto.FLOAT, shape, shape, shape
        )
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(9)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        z0, z1 = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z0, a + b, rtol=1e-5)
        numpy.testing.assert_allclose(z1, a + c, rtol=1e-5)

    def test_mul_shared_input_float32(self):
        """MulSharedInput computes (A*B, A*C) as two outputs element-wise."""
        shape = (4, 4)
        model = self._make_shared_input_model(
            "MulSharedInput", onnx.TensorProto.FLOAT, shape, shape, shape
        )
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(10)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        z0, z1 = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z0, a * b, rtol=1e-5)
        numpy.testing.assert_allclose(z1, a * c, rtol=1e-5)

    def test_rotary_left_float32(self):
        """Rotary left swaps halves: left_out=right_in, right_out=-left_in."""
        shape = (3, 2, 3, 4)
        model = self._make_rotary_model(onnx.TensorProto.FLOAT, shape, "left")
        sess = self._make_inference_session(model)

        x = numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape) + 1.0
        half = shape[-1] // 2
        splits = numpy.array([half, half], dtype=numpy.int64)

        expected = x.copy()
        expected[..., :half] = x[..., half:]
        expected[..., half:] = -x[..., :half]

        (y,) = sess.run(None, {"X": x, "splits": splits})
        numpy.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_rotary_right_float32(self):
        """Rotary right swaps halves: left_out=-right_in, right_out=left_in."""
        shape = (3, 2, 3, 4)
        model = self._make_rotary_model(onnx.TensorProto.FLOAT, shape, "right")
        sess = self._make_inference_session(model)

        x = numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape) + 1.0
        half = shape[-1] // 2
        splits = numpy.array([half, half], dtype=numpy.int64)

        expected = x.copy()
        expected[..., :half] = -x[..., half:]
        expected[..., half:] = x[..., :half]

        (y,) = sess.run(None, {"X": x, "splits": splits})
        numpy.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_scatter_nd_of_shape_float32(self):
        """ScatterNDOfShape performs scatter-add into a zero tensor of given shape."""
        output_shape = numpy.array([4, 6], dtype=numpy.int64)
        indices = numpy.array([[0], [1], [2], [0]], dtype=numpy.int64)
        updates = numpy.ones((4, 6), dtype=numpy.float32)
        model = self._make_scatter_nd_of_shape_model(
            onnx.TensorProto.FLOAT, indices.shape, updates.shape
        )
        sess = self._make_inference_session(model)

        (y,) = sess.run(None, {"shape": output_shape, "indices": indices, "updates": updates})

        expected = numpy.zeros((4, 6), dtype=numpy.float32)
        for i, idx in enumerate(indices[:, 0]):
            expected[idx] += updates[i]
        numpy.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_masked_scatter_nd_of_shape_float32(self):
        """MaskedScatterNDOfShape skips scatter-add for masked index value (-1)."""
        output_shape = numpy.array([8, 4], dtype=numpy.int64)
        indices = numpy.array([[[0]], [[1]], [[-1]], [[2]], [[-1]], [[3]]], dtype=numpy.int64)
        updates = numpy.ones((6, 1, 4), dtype=numpy.float32)
        model = self._make_masked_scatter_nd_model(
            onnx.TensorProto.FLOAT, indices.shape, updates.shape, masked_value=-1
        )
        sess = self._make_inference_session(model)

        (y,) = sess.run(None, {"shape": output_shape, "indices": indices, "updates": updates})

        expected = numpy.zeros((8, 4), dtype=numpy.float32)
        for i in range(indices.shape[0]):
            idx = indices[i, 0, 0]
            if idx != -1:
                expected[idx] += updates[i, 0]
        numpy.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_transpose2d_cast_fp16(self):
        """Transpose2DCastFP16 transposes a float32 2D matrix and casts to float16."""
        shape = (32, 96)
        model = self._make_transpose_cast_model(
            "Transpose2DCastFP16", onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, shape
        )
        sess = self._make_inference_session(model)

        x = numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape) + 1.0
        (y,) = sess.run(None, {"X": x})

        expected = x.T.astype(numpy.float16)
        numpy.testing.assert_allclose(
            y.astype(numpy.float32), expected.astype(numpy.float32), rtol=1e-3
        )

    def test_transpose2d_cast_fp32(self):
        """Transpose2DCastFP32 transposes a float16 2D matrix and casts to float32."""
        shape = (32, 96)
        model = self._make_transpose_cast_model(
            "Transpose2DCastFP32", onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT, shape
        )
        sess = self._make_inference_session(model)

        x = (numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape) + 1.0).astype(
            numpy.float16
        )
        (y,) = sess.run(None, {"X": x})

        expected = x.T.astype(numpy.float32)
        numpy.testing.assert_allclose(y, expected, rtol=1e-3)

    def test_tri_matrix_float32(self):
        """TriMatrix fills a 2D matrix with lower/diag/upper scalar constants."""
        shape = numpy.array([6, 6], dtype=numpy.int64)
        csts = numpy.array([2.0, 3.0, 4.0], dtype=numpy.float32)
        model = self._make_tri_matrix_model(onnx.TensorProto.FLOAT)
        sess = self._make_inference_session(model)

        (y,) = sess.run(None, {"shape": shape, "csts": csts})

        n = int(shape[0])
        i1 = numpy.arange(n).reshape((-1, 1))
        i2 = numpy.arange(n).reshape((1, -1))
        expected = numpy.empty((n, n), dtype=numpy.float32)
        expected[i1 > i2] = 2.0
        expected[i1 == i2] = 3.0
        expected[i1 < i2] = 4.0
        numpy.testing.assert_array_equal(y, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
