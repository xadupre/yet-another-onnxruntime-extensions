"""
Tests for the DenseToSparse and SparseToDense custom ORT ops using the
second (lite) custom op API, built by cmake.

The shared library ``ortops/sparse/cpu/libortops_optim_cpu2.so`` must be
built with ``cmake --build cmake`` before running this test.  Tests are
skipped when the library is absent.
"""

import os
import platform
import unittest

import numpy

from yaourt.ext_test_case import ExtTestCase, requires_onnxruntime

# Path to the shared library produced by the cmake build.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SYSTEM = platform.system()
if _SYSTEM == "Windows":
    _LIB_NAME = "ortops_optim_cpu2.dll"
elif _SYSTEM == "Darwin":
    _LIB_NAME = "libortops_optim_cpu2.dylib"
else:
    _LIB_NAME = "libortops_optim_cpu2.so"
_LIB_PATH = os.path.join(_REPO_ROOT, "ortops", "sparse", "cpu", _LIB_NAME)
_OP_DOMAIN = "yaourt.ortops.optim.cpu"


def _lib_available() -> bool:
    return os.path.exists(_LIB_PATH)


def _make_dense_to_sparse_model(shape: tuple) -> bytes:
    """Builds an ONNX model that calls the DenseToSparse custom op."""
    import onnx
    import onnx.helper as oh

    X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, list(shape))
    Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)
    node = oh.make_node("DenseToSparse", inputs=["X"], outputs=["Y"], domain=_OP_DOMAIN)
    graph = oh.make_graph([node], "DenseToSparseGraph", [X], [Y])
    model = oh.make_model(
        graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
    )
    model.ir_version = 8
    return model.SerializeToString()


def _make_sparse_to_dense_model() -> bytes:
    """Builds an ONNX model that calls the SparseToDense custom op."""
    import onnx
    import onnx.helper as oh

    X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, None)
    Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)
    node = oh.make_node("SparseToDense", inputs=["X"], outputs=["Y"], domain=_OP_DOMAIN)
    graph = oh.make_graph([node], "SparseToDenseGraph", [X], [Y])
    model = oh.make_model(
        graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)]
    )
    model.ir_version = 8
    return model.SerializeToString()


def _make_inference_session(model_bytes: bytes):
    """Creates an OrtInferenceSession with the custom op library loaded."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.register_custom_ops_library(_LIB_PATH)
    return ort.InferenceSession(model_bytes, sess_options=so, providers=["CPUExecutionProvider"])


@unittest.skipUnless(_lib_available(), f"Custom op library not found at {_LIB_PATH!r}")
@requires_onnxruntime("1.18")
class TestSparseCustomOpsLite(ExtTestCase):
    """Tests for DenseToSparse and SparseToDense custom ops using the lite API."""

    def test_dense_to_sparse_basic(self):
        """DenseToSparse produces a 1-D float output (sparse encoding)."""
        model = _make_dense_to_sparse_model((3, 4))
        sess = _make_inference_session(model)

        x = numpy.array([[1, 0, 2, 0], [0, 3, 0, 0], [0, 0, 0, 4]], dtype=numpy.float32)
        (y,) = sess.run(None, {"X": x})

        # The output is a flat float32 vector encoding the sparse structure.
        self.assertEqual(y.ndim, 1)
        self.assertEqual(y.dtype, numpy.float32)
        # Length must be positive (sparse header + indices + values).
        self.assertGreater(y.shape[0], 0)

    def test_sparse_to_dense_roundtrip(self):
        """Dense → Sparse → Dense round-trip preserves values."""
        shape = (4, 5)
        model_d2s = _make_dense_to_sparse_model(shape)
        model_s2d = _make_sparse_to_dense_model()
        sess_d2s = _make_inference_session(model_d2s)
        sess_s2d = _make_inference_session(model_s2d)

        rng = numpy.random.default_rng(0)
        x = rng.random(shape).astype(numpy.float32)
        # Introduce sparsity: zero out ~50 % of elements.
        x[rng.random(shape) < 0.5] = 0.0

        (sparse,) = sess_d2s.run(None, {"X": x})
        (recovered,) = sess_s2d.run(None, {"X": sparse})

        numpy.testing.assert_array_equal(recovered, x)

    def test_all_zeros(self):
        """DenseToSparse on an all-zero matrix followed by SparseToDense gives zeros."""
        shape = (3, 3)
        model_d2s = _make_dense_to_sparse_model(shape)
        model_s2d = _make_sparse_to_dense_model()
        sess_d2s = _make_inference_session(model_d2s)
        sess_s2d = _make_inference_session(model_s2d)

        x = numpy.zeros(shape, dtype=numpy.float32)
        (sparse,) = sess_d2s.run(None, {"X": x})
        (recovered,) = sess_s2d.run(None, {"X": sparse})

        numpy.testing.assert_array_equal(recovered, x)

    def test_all_nonzero(self):
        """DenseToSparse on a fully dense matrix round-trips correctly."""
        shape = (2, 6)
        model_d2s = _make_dense_to_sparse_model(shape)
        model_s2d = _make_sparse_to_dense_model()
        sess_d2s = _make_inference_session(model_d2s)
        sess_s2d = _make_inference_session(model_s2d)

        x = numpy.ones(shape, dtype=numpy.float32) * 3.14
        (sparse,) = sess_d2s.run(None, {"X": x})
        (recovered,) = sess_s2d.run(None, {"X": sparse})

        numpy.testing.assert_array_almost_equal(recovered, x)

    def test_negative_values(self):
        """Negative values survive the sparse round-trip."""
        x = numpy.array([[-1.0, 0.0, 2.5], [0.0, -0.5, 0.0]], dtype=numpy.float32)
        shape = x.shape
        model_d2s = _make_dense_to_sparse_model(shape)
        model_s2d = _make_sparse_to_dense_model()
        sess_d2s = _make_inference_session(model_d2s)
        sess_s2d = _make_inference_session(model_s2d)

        (sparse,) = sess_d2s.run(None, {"X": x})
        (recovered,) = sess_s2d.run(None, {"X": sparse})

        numpy.testing.assert_array_almost_equal(recovered, x)

    def test_large_matrix(self):
        """Round-trip correctness on a larger sparse matrix."""
        shape = (50, 100)
        rng = numpy.random.default_rng(42)
        x = rng.standard_normal(shape).astype(numpy.float32)
        x[rng.random(shape) < 0.9] = 0.0  # ~90 % sparse

        model_d2s = _make_dense_to_sparse_model(shape)
        model_s2d = _make_sparse_to_dense_model()
        sess_d2s = _make_inference_session(model_d2s)
        sess_s2d = _make_inference_session(model_s2d)

        (sparse,) = sess_d2s.run(None, {"X": x})
        (recovered,) = sess_s2d.run(None, {"X": sparse})

        numpy.testing.assert_array_equal(recovered, x)

    def test_lib_path_exists(self):
        """Sanity check that the library file is present."""
        self.assertTrue(os.path.exists(_LIB_PATH), f"Library not found: {_LIB_PATH}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
