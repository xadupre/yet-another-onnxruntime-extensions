"""
Tests for the CUDA custom ORT ops built by cmake.

The shared library ``ortops/optim/cuda/libortops_optim_cuda.so`` must be
built with ``cmake --build cmake`` before running this test.  Tests are
skipped when the library is absent or when no CUDA device is available.
"""

import os
import platform
import unittest

import numpy

from yaourt.ext_test_case import ExtTestCase, has_cuda_onnxruntime, requires_onnxruntime

# Path to the shared library produced by the cmake build.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SYSTEM = platform.system()
if _SYSTEM == "Windows":
    _LIB_NAME = "ortops_optim_cuda.dll"
elif _SYSTEM == "Darwin":
    _LIB_NAME = "libortops_optim_cuda.dylib"
else:
    _LIB_NAME = "libortops_optim_cuda.so"
_LIB_PATH = os.path.join(_REPO_ROOT, "ortops", "optim", "cuda", _LIB_NAME)
_OP_DOMAIN = "yaourt.ortops.optim.cuda"


def _lib_available() -> bool:
    """Returns True if the CUDA custom op shared library is present."""
    return os.path.exists(_LIB_PATH)


def _make_inference_session(model_bytes: bytes):
    """Creates an OrtInferenceSession with the custom op library loaded (CUDA EP)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.register_custom_ops_library(_LIB_PATH)
    return ort.InferenceSession(
        model_bytes, sess_options=so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )


def _make_unary_model(op_name: str, dtype_onnx: int, shape, **kwargs) -> bytes:
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


def _make_binary_model(op_name: str, dtype_onnx: int, shape_a, shape_b, **kwargs) -> bytes:
    """Builds an ONNX model with a single custom binary op."""
    import onnx.helper as oh

    X = oh.make_tensor_value_info("X", dtype_onnx, list(shape_a))
    Y_in = oh.make_tensor_value_info("Y", dtype_onnx, list(shape_b))
    Z = oh.make_tensor_value_info("Z", dtype_onnx, None)
    node = oh.make_node(op_name, inputs=["X", "Y"], outputs=["Z"], domain=_OP_DOMAIN, **kwargs)
    graph = oh.make_graph([node], op_name + "Graph", [X, Y_in], [Z])
    model = oh.make_model(
        graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
    )
    model.ir_version = 8
    return model.SerializeToString()


def _make_ternary_model(
    op_name: str, dtype_onnx: int, shape_a, shape_b, shape_c, **kwargs
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


_skip_reason = f"CUDA custom op library not found at {_LIB_PATH!r} or CUDA unavailable"


@unittest.skipUnless(_lib_available() and has_cuda_onnxruntime(), _skip_reason)
@requires_onnxruntime("1.18")
class TestCudaCustomOps(ExtTestCase):
    """Tests for CUDA custom ops (NegXplus1, ReplaceZero, MulSigmoid, etc.)."""

    def test_lib_path_exists(self):
        """Sanity check: the library file is present on disk."""
        self.assertTrue(os.path.exists(_LIB_PATH), f"Library not found: {_LIB_PATH}")

    def test_negxplus1_float32(self):
        """NegXplus1 computes 1 - x correctly for float32."""
        import onnx

        shape = (4, 8)
        model = _make_unary_model("NegXplus1", onnx.TensorProto.FLOAT, shape)
        sess = _make_inference_session(model)

        x = numpy.random.rand(*shape).astype(numpy.float32)
        (y,) = sess.run(None, {"X": x})
        numpy.testing.assert_allclose(y, 1.0 - x, rtol=1e-5)

    def test_replace_zero_float32(self):
        """ReplaceZero replaces zero entries with the given scalar."""
        import onnx

        shape = (2, 3)
        model = _make_unary_model("ReplaceZero", onnx.TensorProto.FLOAT, shape, by=7.0)
        sess = _make_inference_session(model)

        x = numpy.array([[1.0, 0.0, 2.0], [0.0, 5.0, 0.0]], dtype=numpy.float32)
        (y,) = sess.run(None, {"X": x})
        expected = numpy.where(x == 0.0, 7.0, x)
        numpy.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_mul_sigmoid_float32(self):
        """MulSigmoid computes x * sigmoid(x) (Swish activation)."""
        import onnx

        shape = (4, 4)
        model = _make_unary_model("MulSigmoid", onnx.TensorProto.FLOAT, shape)
        sess = _make_inference_session(model)

        x = numpy.random.randn(*shape).astype(numpy.float32)
        (y,) = sess.run(None, {"X": x})

        sigmoid_x = 1.0 / (1.0 + numpy.exp(-x.astype(numpy.float64)))
        expected = (x * sigmoid_x).astype(numpy.float32)
        numpy.testing.assert_allclose(y, expected, rtol=1e-4)

    def test_addmul_float32(self):
        """AddMul computes (A + B) * C element-wise."""
        import onnx

        shape = (4, 4)
        model = _make_ternary_model("AddMul", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = _make_inference_session(model)

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
        model = _make_ternary_model("MulAdd", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = _make_inference_session(model)

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
        model = _make_ternary_model("SubMul", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = _make_inference_session(model)

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
        model = _make_ternary_model("MulSub", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = _make_inference_session(model)

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
        model = _make_binary_model("MulMulSigmoid", onnx.TensorProto.FLOAT, shape, shape)
        sess = _make_inference_session(model)

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
        model = _make_ternary_model("AddAdd", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = _make_inference_session(model)

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
        model = _make_ternary_model("MulMul", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = _make_inference_session(model)

        rng = numpy.random.default_rng(6)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, a * b * c, rtol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
