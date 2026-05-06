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


@unittest.skipUnless(_lib_available(), f"CUDA custom op library not found at {_LIB_PATH!r}")
@requires_cuda_onnxruntime()
@requires_onnxruntime("1.18")
class TestFusedKernelCudaCustomOps(ExtTestCase):
    """Tests for CUDA custom ops (NegXplus1, ReplaceZero, MulSigmoid, etc.)."""

    def _make_inference_session(self, model_bytes: bytes):
        """Creates an OrtInferenceSession with the custom op library loaded (CUDA EP)."""
        import onnxruntime as ort

        self.assertExists(_LIB_PATH)
        self.assertIn("cuda", _LIB_PATH)
        so = ort.SessionOptions()
        so.register_custom_ops_library(_LIB_PATH)
        return ort.InferenceSession(
            model_bytes,
            sess_options=so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def _make_unary_model(self, op_name: str, dtype_onnx: int, shape, **kwargs) -> bytes:
        """Builds an ONNX model with a single custom unary op."""
        import onnx.helper as oh

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
        import onnx.helper as oh

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
        import onnx.helper as oh

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
        import onnx.helper as oh

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
        import onnx.helper as oh

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
        import onnx
        import onnx.helper as oh

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
        import onnx
        import onnx.helper as oh

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
        import onnx
        import onnx.helper as oh

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
        import onnx.helper as oh

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
        import onnx
        import onnx.helper as oh

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

    def test_lib_path_exists(self):
        """Sanity check: the library file is present on disk."""
        self.assertTrue(os.path.exists(_LIB_PATH), f"Library not found: {_LIB_PATH}")

    def test_negxplus1_float32(self):
        """NegXplus1 computes 1 - x correctly for float32."""
        import onnx

        shape = (4, 8)
        model = self._make_unary_model("NegXplus1", onnx.TensorProto.FLOAT, shape)
        sess = self._make_inference_session(model)

        x = numpy.random.rand(*shape).astype(numpy.float32)
        (y,) = sess.run(None, {"X": x})
        numpy.testing.assert_allclose(y, 1.0 - x, rtol=1e-5)

    def test_replace_zero_float32(self):
        """ReplaceZero replaces zero entries with the given scalar."""
        import onnx

        shape = (2, 3)
        model = self._make_unary_model("ReplaceZero", onnx.TensorProto.FLOAT, shape, by=7.0)
        sess = self._make_inference_session(model)

        x = numpy.array([[1.0, 0.0, 2.0], [0.0, 5.0, 0.0]], dtype=numpy.float32)
        (y,) = sess.run(None, {"X": x})
        expected = numpy.where(x == 0.0, 7.0, x)
        numpy.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_mul_sigmoid_float32(self):
        """MulSigmoid computes x * sigmoid(x) (Swish activation)."""
        import onnx

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
        import onnx

        shape = (4, 4)
        model = self._make_ternary_model("AddMul", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(0)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, (a + b) * c, rtol=1e-5)

    def test_muladd_float32(self):
        """MulAdd computes A * B + C element-wise."""
        import onnx

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
        import onnx

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
        import onnx

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
        import onnx

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
        import onnx

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
        import onnx

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
        import onnx

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
        import onnx

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
        import onnx

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
        import onnx

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
        import onnx

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
        import onnx

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
        import onnx

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
        import onnx

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
        import onnx

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
        import onnx

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
        import onnx

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


@unittest.skipUnless(_lib_available(), f"CUDA custom op library not found at {_LIB_PATH!r}")
@requires_cuda_onnxruntime()
@requires_onnxruntime("1.18")
class TestMixedOnnxAndContribOpsCuda(ExtTestCase):
    """Tests that combine standard ONNX ops with custom CUDA contrib ops in one graph."""

    def _make_inference_session(self, model_bytes: bytes):
        """Creates an OrtInferenceSession with the custom op library loaded (CUDA EP)."""
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.register_custom_ops_library(_LIB_PATH)
        return ort.InferenceSession(
            model_bytes,
            sess_options=so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def _make_relu_then_neg_xplus1_model(self, shape) -> bytes:
        """Builds a model: X -> Relu -> Z -> NegXplus1 -> Y.

        Combines a standard ONNX ``Relu`` op with the custom ``NegXplus1``
        contrib op so that both execute together under the CUDA EP.
        """
        import onnx
        import onnx.helper as oh

        X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, list(shape))
        Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, list(shape))
        relu_node = oh.make_node("Relu", inputs=["X"], outputs=["Z"])
        neg_node = oh.make_node("NegXplus1", inputs=["Z"], outputs=["Y"], domain=_OP_DOMAIN)
        graph = oh.make_graph([relu_node, neg_node], "ReluNegXplus1Graph", [X], [Y])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def _make_add_then_mul_sigmoid_model(self, shape) -> bytes:
        """Builds a model: (X, Y) -> Add -> Z -> MulSigmoid -> Out.

        Combines the standard ONNX ``Add`` op with the custom ``MulSigmoid``
        contrib op (Swish-like activation) in a single CUDA graph.
        """
        import onnx
        import onnx.helper as oh

        X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, list(shape))
        Y_in = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, list(shape))
        Out = oh.make_tensor_value_info("Out", onnx.TensorProto.FLOAT, list(shape))
        add_node = oh.make_node("Add", inputs=["X", "Y"], outputs=["Z"])
        swish_node = oh.make_node("MulSigmoid", inputs=["Z"], outputs=["Out"], domain=_OP_DOMAIN)
        graph = oh.make_graph([add_node, swish_node], "AddMulSigmoidGraph", [X, Y_in], [Out])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def _make_neg_xplus1_then_add_model(self, shape) -> bytes:
        """Builds a model: X -> NegXplus1 -> Z, then (Z, X) -> Add -> Y.

        Combines the custom ``NegXplus1`` contrib op with the standard ONNX
        ``Add`` op.  The result should be (1 - X) + X = 1 everywhere.
        """
        import onnx
        import onnx.helper as oh

        X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, list(shape))
        Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, list(shape))
        neg_node = oh.make_node("NegXplus1", inputs=["X"], outputs=["Z"], domain=_OP_DOMAIN)
        add_node = oh.make_node("Add", inputs=["Z", "X"], outputs=["Y"])
        graph = oh.make_graph([neg_node, add_node], "NegXplus1AddGraph", [X], [Y])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def _make_mul_then_add_mul_model(self, shape) -> bytes:
        """Builds a model: (A, B) -> Mul -> P, then (P, B, C) -> AddMul -> Y.

        Combines the standard ONNX ``Mul`` op with the custom three-input
        ``AddMul`` contrib op, verifying that both execute on the CUDA EP.
        ``AddMul`` computes ``(P + B) * C``, so the full expression is
        ``(A * B + B) * C = B * (A + 1) * C``.
        """
        import onnx
        import onnx.helper as oh

        A = oh.make_tensor_value_info("A", onnx.TensorProto.FLOAT, list(shape))
        B = oh.make_tensor_value_info("B", onnx.TensorProto.FLOAT, list(shape))
        C = oh.make_tensor_value_info("C", onnx.TensorProto.FLOAT, list(shape))
        Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)
        mul_node = oh.make_node("Mul", inputs=["A", "B"], outputs=["P"])
        add_mul_node = oh.make_node(
            "AddMul", inputs=["P", "B", "C"], outputs=["Y"], domain=_OP_DOMAIN
        )
        graph = oh.make_graph([mul_node, add_mul_node], "MulAddMulGraph", [A, B, C], [Y])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def test_relu_then_neg_xplus1(self):
        """Standard Relu followed by custom NegXplus1: result is 1 - max(0, x)."""
        shape = (4, 8)
        model = self._make_relu_then_neg_xplus1_model(shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(100)
        x = rng.standard_normal(shape).astype(numpy.float32)
        (y,) = sess.run(None, {"X": x})

        expected = 1.0 - numpy.maximum(0.0, x)
        numpy.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_add_then_mul_sigmoid(self):
        """Standard Add followed by custom MulSigmoid: result is (x+y)*sigmoid(x+y)."""
        shape = (4, 4)
        model = self._make_add_then_mul_sigmoid_model(shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(101)
        x = rng.standard_normal(shape).astype(numpy.float32)
        y = rng.standard_normal(shape).astype(numpy.float32)
        (out,) = sess.run(None, {"X": x, "Y": y})

        s = (x + y).astype(numpy.float64)
        expected = (s * (1.0 / (1.0 + numpy.exp(-s)))).astype(numpy.float32)
        numpy.testing.assert_allclose(out, expected, rtol=1e-4)

    def test_neg_xplus1_then_add_equals_one(self):
        """Custom NegXplus1 then standard Add: (1 - x) + x = 1 for all elements."""
        shape = (3, 5)
        model = self._make_neg_xplus1_then_add_model(shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(102)
        x = rng.standard_normal(shape).astype(numpy.float32)
        (y,) = sess.run(None, {"X": x})

        numpy.testing.assert_allclose(y, numpy.ones(shape, dtype=numpy.float32), rtol=1e-5)

    def test_mul_then_add_mul(self):
        """Standard Mul then custom AddMul: (A*B + B) * C = B*(A+1)*C."""
        shape = (4, 4)
        model = self._make_mul_then_add_mul_model(shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(103)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        (y,) = sess.run(None, {"A": a, "B": b, "C": c})

        # p = a * b; y = (p + b) * c = b * (a + 1) * c
        p = a * b
        expected = (p + b) * c
        numpy.testing.assert_allclose(y, expected, rtol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
