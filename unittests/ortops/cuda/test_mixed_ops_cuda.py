"""
Tests that mix standard ONNX ops with ORT built-in ``com.microsoft`` contrib ops on CUDA.

No custom shared library is required; the tests rely solely on the contrib ops
that ship with ``onnxruntime-gpu``.  Each test is skipped automatically when
``CUDAExecutionProvider`` is unavailable.
"""

import unittest

import numpy

from yaourt.ext_test_case import ExtTestCase, requires_cuda_onnxruntime, requires_onnxruntime


@requires_cuda_onnxruntime()
@requires_onnxruntime("1.18")
class TestMixedOnnxAndContribOpsCuda(ExtTestCase):
    """Tests that combine standard ONNX ops with ORT built-in contrib ops on CUDA.

    These tests do **not** require any custom shared library; they rely solely
    on the `com.microsoft` contrib ops that ship with `onnxruntime-gpu`.
    """

    def _make_inference_session(self, model_bytes: bytes):
        """Creates an InferenceSession targeting the CUDA execution provider.

        Raises:
            AssertionError: If ORT falls back to CPU and does not use
                ``CUDAExecutionProvider``.
        """
        import onnxruntime as ort

        sess = ort.InferenceSession(
            model_bytes, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.assertIn(
            "CUDAExecutionProvider",
            sess.get_providers(),
            "ORT fell back to CPU-only execution; CUDAExecutionProvider is not active.",
        )
        return sess

    @staticmethod
    def _gelu_ref(x: numpy.ndarray) -> numpy.ndarray:
        """Computes the reference GELU: x * 0.5 * (1 + erf(x / sqrt(2))).

        Returns:
            float32 array of the same shape as *x*.
        """
        import math

        x64 = x.astype(numpy.float64)
        erf_vec = numpy.frompyfunc(math.erf, 1, 1)
        return (x64 * 0.5 * (1.0 + erf_vec(x64 / numpy.sqrt(2.0)).astype(numpy.float64))).astype(
            numpy.float32
        )

    @staticmethod
    def _fast_gelu_ref(x: numpy.ndarray) -> numpy.ndarray:
        """Computes the tanh approximation used by the FastGelu contrib op.

        FastGelu(x) = x * 0.5 * (1 + tanh(x * (c + x^2 * d)))
        where c = sqrt(2/pi) and d = 0.044715.

        Returns:
            float32 array of the same shape as *x*.
        """
        x64 = x.astype(numpy.float64)
        c = numpy.sqrt(2.0 / numpy.pi)
        d = 0.044715
        return (x64 * 0.5 * (1.0 + numpy.tanh(x64 * (c + x64 * x64 * d)))).astype(numpy.float32)

    def _make_relu_then_gelu_model(self, shape) -> bytes:
        """Builds a model: X -> Relu (ONNX) -> Z -> Gelu (com.microsoft) -> Y."""
        import onnx
        import onnx.helper as oh

        X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, list(shape))
        Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, list(shape))
        relu_node = oh.make_node("Relu", inputs=["X"], outputs=["Z"])
        gelu_node = oh.make_node("Gelu", inputs=["Z"], outputs=["Y"], domain="com.microsoft")
        graph = oh.make_graph([relu_node, gelu_node], "ReluGeluGraph", [X], [Y])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def _make_add_then_gelu_model(self, shape) -> bytes:
        """Builds a model: (X, Y) -> Add (ONNX) -> Z -> Gelu (com.microsoft) -> Out."""
        import onnx
        import onnx.helper as oh

        X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, list(shape))
        Y_in = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, list(shape))
        Out = oh.make_tensor_value_info("Out", onnx.TensorProto.FLOAT, list(shape))
        add_node = oh.make_node("Add", inputs=["X", "Y"], outputs=["Z"])
        gelu_node = oh.make_node("Gelu", inputs=["Z"], outputs=["Out"], domain="com.microsoft")
        graph = oh.make_graph([add_node, gelu_node], "AddGeluGraph", [X, Y_in], [Out])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def _make_gelu_then_mul_model(self, shape) -> bytes:
        """Builds a model: X -> Gelu (com.microsoft) -> Z -> Mul (ONNX, Z * X) -> Y."""
        import onnx
        import onnx.helper as oh

        X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, list(shape))
        Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, list(shape))
        gelu_node = oh.make_node("Gelu", inputs=["X"], outputs=["Z"], domain="com.microsoft")
        mul_node = oh.make_node("Mul", inputs=["Z", "X"], outputs=["Y"])
        graph = oh.make_graph([gelu_node, mul_node], "GeluMulGraph", [X], [Y])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def _make_add_then_fast_gelu_model(self, shape) -> bytes:
        """Builds a model: (X, bias) -> Add (ONNX) -> Z -> FastGelu (com.microsoft) -> Out."""
        import onnx
        import onnx.helper as oh

        X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, list(shape))
        bias = oh.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, list(shape))
        Out = oh.make_tensor_value_info("Out", onnx.TensorProto.FLOAT, list(shape))
        add_node = oh.make_node("Add", inputs=["X", "bias"], outputs=["Z"])
        fast_gelu_node = oh.make_node(
            "FastGelu", inputs=["Z"], outputs=["Out"], domain="com.microsoft"
        )
        graph = oh.make_graph([add_node, fast_gelu_node], "AddFastGeluGraph", [X, bias], [Out])
        model = oh.make_model(
            graph, opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)]
        )
        model.ir_version = 8
        return model.SerializeToString()

    def test_relu_then_gelu(self):
        """Standard Relu followed by ORT contrib Gelu: Gelu(max(0, x))."""
        shape = (4, 8)
        model = self._make_relu_then_gelu_model(shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(100)
        x = rng.standard_normal(shape).astype(numpy.float32)
        (y,) = sess.run(None, {"X": x})

        expected = self._gelu_ref(numpy.maximum(0.0, x))
        numpy.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-5)

    def test_add_then_gelu(self):
        """Standard Add followed by ORT contrib Gelu: Gelu(x + y)."""
        shape = (4, 4)
        model = self._make_add_then_gelu_model(shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(101)
        x = rng.standard_normal(shape).astype(numpy.float32)
        y = rng.standard_normal(shape).astype(numpy.float32)
        (out,) = sess.run(None, {"X": x, "Y": y})

        expected = self._gelu_ref(x + y)
        numpy.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    def test_gelu_then_mul(self):
        """ORT contrib Gelu followed by standard Mul: Gelu(x) * x."""
        shape = (3, 6)
        model = self._make_gelu_then_mul_model(shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(102)
        x = rng.standard_normal(shape).astype(numpy.float32)
        (y,) = sess.run(None, {"X": x})

        expected = (self._gelu_ref(x).astype(numpy.float64) * x.astype(numpy.float64)).astype(
            numpy.float32
        )
        numpy.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-5)

    def test_add_then_fast_gelu(self):
        shape = (4, 4)
        model = self._make_add_then_fast_gelu_model(shape)
        sess = self._make_inference_session(model)

        rng = numpy.random.default_rng(103)
        x = rng.standard_normal(shape).astype(numpy.float32)
        bias = rng.standard_normal(shape).astype(numpy.float32)
        (out,) = sess.run(None, {"X": x, "bias": bias})

        expected = self._fast_gelu_ref(x + bias)
        numpy.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
