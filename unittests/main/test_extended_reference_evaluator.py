import unittest

import numpy as np
import onnx
import onnx.helper as oh
from onnx.reference.op_run import OpRun

from yaourt.ext_test_case import ExtTestCase
from yaourt.reference import ExtendedReferenceEvaluator

TFLOAT = onnx.TensorProto.FLOAT


def _make_add_function() -> onnx.FunctionProto:
    return oh.make_function(
        domain="test.fn",
        fname="MyAdd",
        inputs=["X", "Y"],
        outputs=["Z"],
        nodes=[oh.make_node("Add", ["X", "Y"], ["Z"])],
        opset_imports=[oh.make_opsetid("", 18)],
    )


def _make_add_model() -> onnx.ModelProto:
    return oh.make_model(
        oh.make_graph(
            [oh.make_node("Add", ["X", "Y"], ["Z"])],
            "add_graph",
            [
                oh.make_tensor_value_info("X", TFLOAT, [None, None]),
                oh.make_tensor_value_info("Y", TFLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, [None, None])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )


class TestExtendedReferenceEvaluator(ExtTestCase):
    def test_run_standard_ops(self):
        model = _make_add_model()
        ref = ExtendedReferenceEvaluator(model)
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        (result,) = ref.run(None, {"X": x, "Y": x})
        self.assertEqualArray(x + x, result)

    def test_run_list_inputs_shortcut(self):
        model = _make_add_model()
        ref = ExtendedReferenceEvaluator(model)
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        (result,) = ref.run([x, x])
        self.assertEqualArray(x + x, result)

    def test_default_ops_is_nonempty(self):
        """default_ops is pre-populated with the fused-kernel CUDA reference kernels."""
        from yaourt.ortops.fused_kernel.reference_ops import ALL_OPS

        self.assertGreater(len(ExtendedReferenceEvaluator.default_ops), 0)
        for cls in ALL_OPS:
            self.assertIn(cls, ExtendedReferenceEvaluator.default_ops)

    def test_fused_kernel_ops_work_without_new_ops(self):
        """Fused-kernel CUDA ops run via default_ops without passing new_ops."""
        _OP_DOMAIN = "yaourt.ortops.fused_kernel.cuda"
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("MulMul", ["A", "B", "C"], ["Z"], domain=_OP_DOMAIN)],
                "mulmul_graph",
                [oh.make_tensor_value_info(n, TFLOAT, [None]) for n in ("A", "B", "C")],
                [oh.make_tensor_value_info("Z", TFLOAT, [None])],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(_OP_DOMAIN, 1)],
            ir_version=10,
        )
        ref = ExtendedReferenceEvaluator(model)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        c = np.array([7.0, 8.0, 9.0], dtype=np.float32)
        (result,) = ref.run(None, {"A": a, "B": b, "C": c})
        self.assertEqualArray(a * b * c, result)

    def test_custom_op_via_new_ops(self):
        class DoubleOp(OpRun):
            op_domain = "test.domain"

            def _run(self, X):
                return (X * 2,)

        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("DoubleOp", ["X"], ["Z"], domain="test.domain")],
                "custom_graph",
                [oh.make_tensor_value_info("X", TFLOAT, [None])],
                [oh.make_tensor_value_info("Z", TFLOAT, [None])],
            ),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("test.domain", 1)],
            ir_version=10,
        )
        ref = ExtendedReferenceEvaluator(model, new_ops=[DoubleOp])
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (result,) = ref.run(None, {"X": x})
        self.assertEqualArray(x * 2, result)

    def test_filter_ops_selects_best_version(self):
        class MyOp_1(OpRun):
            op_domain = "test"
            op_schema = None

            def _run(self, X):
                return (X,)

        class MyOp_3(OpRun):
            op_domain = "test"
            op_schema = None

            def _run(self, X):
                return (X * 3,)

        model = oh.make_model(
            oh.make_graph([], "g", [], []),
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("test", 2)],
        )
        filtered = ExtendedReferenceEvaluator.filter_ops(model, [MyOp_1, MyOp_3], None)
        # MyOp_3 requires version 3 but model declares opset 2; only MyOp_1 qualifies.
        names = [cl.__name__ for cl in filtered]
        self.assertIn("MyOp", names)
        self.assertNotIn("MyOp_1", names)
        self.assertNotIn("MyOp_3", names)

    def test_filter_ops_unversioned_kept_as_is(self):
        class PlainOp(OpRun):
            op_domain = "test"

            def _run(self, X):
                return (X,)

        filtered = ExtendedReferenceEvaluator.filter_ops(None, [PlainOp], None)
        self.assertEqual(len(filtered), 1)
        self.assertIs(filtered[0], PlainOp)

    def test_input_names_accessible(self):
        model = _make_add_model()
        ref = ExtendedReferenceEvaluator(model)
        self.assertEqual(list(ref.input_names), ["X", "Y"])

    def test_verbose_parameter(self):
        model = _make_add_model()
        ref = ExtendedReferenceEvaluator(model, verbose=0)
        x = np.ones((2, 2), dtype=np.float32)
        (result,) = ref.run(None, {"X": x, "Y": x})
        self.assertEqualArray(x + x, result)

    def test_run_function_proto(self):
        fn = _make_add_function()
        ref = ExtendedReferenceEvaluator(fn)
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        (result,) = ref.run(None, {"X": x, "Y": x})
        self.assertEqualArray(x + x, result)

    def test_run_function_proto_list_inputs(self):
        fn = _make_add_function()
        ref = ExtendedReferenceEvaluator(fn)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (result,) = ref.run([x, x])
        self.assertEqualArray(x + x, result)

    def test_run_function_proto_intermediate(self):
        fn = oh.make_function(
            domain="test.fn",
            fname="AddMul",
            inputs=["X", "Y"],
            outputs=["Z"],
            nodes=[
                oh.make_node("Add", ["X", "Y"], ["tmp"]),
                oh.make_node("Mul", ["tmp", "X"], ["Z"]),
            ],
            opset_imports=[oh.make_opsetid("", 18)],
        )
        ref = ExtendedReferenceEvaluator(fn)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        inter = ref._run_function(None, {"X": x, "Y": y}, intermediate=True)
        self.assertIsInstance(inter, dict)
        self.assertIn("tmp", inter)
        self.assertIn("Z", inter)
        self.assertEqualArray(x + y, inter["tmp"])
        self.assertEqualArray((x + y) * x, inter["Z"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
