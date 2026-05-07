"""
Tests for the MulMul CPU custom ORT op built by cmake.

The shared library
``yaourt/ortops/fused_kernel/cpu/libortops_fused_kernel_cpu.so`` must be
built with ``cmake --build cmake`` before running this test.  Tests are
skipped when the library is absent.
"""

import os
import platform
import unittest

import numpy

from yaourt.ext_test_case import ExtTestCase, requires_onnxruntime

# Path to the shared library produced by the cmake build.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SYSTEM = platform.system()
if _SYSTEM == "Windows":
    _LIB_NAME = "ortops_fused_kernel_cpu.dll"
elif _SYSTEM == "Darwin":
    _LIB_NAME = "libortops_fused_kernel_cpu.dylib"
else:
    _LIB_NAME = "libortops_fused_kernel_cpu.so"
_LIB_PATH = os.path.join(_REPO_ROOT, "yaourt", "ortops", "fused_kernel", "cpu", _LIB_NAME)
_OP_DOMAIN = "yaourt.ortops.fused_kernel.cpu"


def _lib_available() -> bool:
    """Returns True if the CPU custom op shared library is present."""
    return os.path.exists(_LIB_PATH)


@unittest.skipUnless(_lib_available(), f"CPU custom op library not found at {_LIB_PATH!r}")
@requires_onnxruntime("1.18")
class TestMulMulCpuOp(ExtTestCase):
    """Tests for the MulMul CPU custom op."""

    def _make_inference_session(self, model_bytes: bytes):
        """Creates an OrtInferenceSession with the custom op library loaded (CPU EP)."""
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.register_custom_ops_library(_LIB_PATH)
        return ort.InferenceSession(
            model_bytes, sess_options=so, providers=["CPUExecutionProvider"]
        )

    def _make_ternary_model(
        self, op_name: str, dtype_onnx: int, shape_a, shape_b, shape_c
    ) -> bytes:
        """Builds an ONNX model with a single custom ternary op."""
        import onnx.helper as oh

        A = oh.make_tensor_value_info("A", dtype_onnx, list(shape_a))
        B = oh.make_tensor_value_info("B", dtype_onnx, list(shape_b))
        C = oh.make_tensor_value_info("C", dtype_onnx, list(shape_c))
        Z = oh.make_tensor_value_info("Z", dtype_onnx, None)
        node = oh.make_node(op_name, inputs=["A", "B", "C"], outputs=["Z"], domain=_OP_DOMAIN)
        graph = oh.make_graph([node], op_name + "Graph", [A, B, C], [Z])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    # ------------------------------------------------------------------
    # Basic correctness tests
    # ------------------------------------------------------------------

    def test_mulmul_float32_same_shape(self):
        """MulMul computes A * B * C element-wise for equal-shape inputs."""
        import onnx

        shape = (4, 4)
        model = self._make_ternary_model("MulMul", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(0)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, a * b * c, rtol=1e-5)

    def test_mulmul_float32_ones(self):
        """MulMul with all-ones inputs returns all-ones."""
        import onnx

        shape = (8, 8)
        model = self._make_ternary_model("MulMul", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = self._make_inference_session(model)

        a = numpy.ones(shape, dtype=numpy.float32)
        b = numpy.ones(shape, dtype=numpy.float32)
        c = numpy.ones(shape, dtype=numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, numpy.ones(shape, dtype=numpy.float32))

    def test_mulmul_float32_zeros(self):
        """MulMul with any zero input returns all-zeros."""
        import onnx

        shape = (3, 5)
        model = self._make_ternary_model("MulMul", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(1)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = numpy.zeros(shape, dtype=numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, numpy.zeros(shape, dtype=numpy.float32))

    def test_mulmul_float32_negative_values(self):
        """MulMul handles negative values correctly."""
        import onnx

        shape = (2, 4)
        model = self._make_ternary_model("MulMul", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = self._make_inference_session(model)

        a = numpy.array([[-1.0, 2.0, -3.0, 4.0], [1.0, -1.0, 1.0, -1.0]], dtype=numpy.float32)
        b = numpy.array([[2.0, -1.0, 1.0, -2.0], [-1.0, 1.0, -1.0, 1.0]], dtype=numpy.float32)
        c = numpy.array([[3.0, 3.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0]], dtype=numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, a * b * c, rtol=1e-5)

    # ------------------------------------------------------------------
    # Broadcasting tests
    # ------------------------------------------------------------------

    def test_mulmul_float32_broadcast_scalar_c(self):
        """MulMul broadcasts a scalar C over all elements."""
        import onnx

        shape = (4, 4)
        model = self._make_ternary_model("MulMul", onnx.TensorProto.FLOAT, shape, shape, (1,))
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(2)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = numpy.array([2.0], dtype=numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, a * b * 2.0, rtol=1e-5)

    def test_mulmul_float32_broadcast_b_scalar(self):
        """MulMul broadcasts a scalar B over all elements."""
        import onnx

        shape = (5, 6)
        model = self._make_ternary_model("MulMul", onnx.TensorProto.FLOAT, shape, (1,), shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(3)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = numpy.array([3.0], dtype=numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, a * 3.0 * c, rtol=1e-5)

    # ------------------------------------------------------------------
    # Size-related edge cases
    # ------------------------------------------------------------------

    def test_mulmul_float32_single_element(self):
        """MulMul works correctly for single-element tensors."""
        import onnx

        shape = (1,)
        model = self._make_ternary_model("MulMul", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = self._make_inference_session(model)

        a = numpy.array([2.0], dtype=numpy.float32)
        b = numpy.array([3.0], dtype=numpy.float32)
        c = numpy.array([4.0], dtype=numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, numpy.array([24.0], dtype=numpy.float32))

    def test_mulmul_float32_large_vector(self):
        """MulMul is numerically correct on a large 1-D vector (AVX path)."""
        import onnx

        shape = (1024,)
        model = self._make_ternary_model("MulMul", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(4)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, a * b * c, rtol=1e-5)

    def test_mulmul_float32_non_multiple_of_8(self):
        """MulMul is correct when the element count is not a multiple of 8 (scalar tail)."""
        import onnx

        shape = (13,)  # 13 is not a multiple of 8
        model = self._make_ternary_model("MulMul", onnx.TensorProto.FLOAT, shape, shape, shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(5)
        a = rng.standard_normal(shape).astype(numpy.float32)
        b = rng.standard_normal(shape).astype(numpy.float32)
        c = rng.standard_normal(shape).astype(numpy.float32)
        (z,) = sess.run(None, {"A": a, "B": b, "C": c})
        numpy.testing.assert_allclose(z, a * b * c, rtol=1e-5)

    # ------------------------------------------------------------------
    # Library path sanity check
    # ------------------------------------------------------------------

    def test_lib_path_exists(self):
        """Sanity check that the library file is present."""
        self.assertTrue(os.path.exists(_LIB_PATH), f"Library not found: {_LIB_PATH}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
