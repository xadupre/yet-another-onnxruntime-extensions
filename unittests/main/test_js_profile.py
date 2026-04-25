import json
import os
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yaourt.ext_test_case import (
    ExtTestCase,
    requires_matplotlib,
    skipif_ci_apple,
    skipif_ci_windows,
)


def _make_simple_model() -> onnx.ModelProto:
    """Returns a small MLP ONNX model for testing."""
    return oh.make_model(
        oh.make_graph(
            [
                oh.make_node("MatMul", ["x", "W1"], ["h"]),
                oh.make_node("Add", ["h", "b1"], ["relu_in"]),
                oh.make_node("Relu", ["relu_in"], ["output"]),
            ],
            "test_graph",
            [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, (2, 4))],
            [oh.make_tensor_value_info("output", onnx.TensorProto.FLOAT, (2, 8))],
            [
                onh.from_array(np.random.randn(4, 8).astype(np.float32), name="W1"),
                onh.from_array(np.zeros(8, dtype=np.float32), name="b1"),
            ],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=8,
    )


def _run_ort_with_profiling(model: onnx.ModelProto) -> str:
    """
    Runs onnxruntime with profiling enabled and returns the profile filename.
    """
    from onnxruntime import InferenceSession, SessionOptions

    with tempfile.TemporaryDirectory() as tmpdir:
        opts = SessionOptions()
        opts.enable_profiling = True
        opts.profile_file_prefix = os.path.join(tmpdir, "ort_profile")
        sess = InferenceSession(
            model.SerializeToString(), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        x = np.random.randn(2, 4).astype(np.float32)
        for _ in range(3):
            sess.run(None, {"x": x})
        profile_file = sess.end_profiling()
        # Copy to a permanent location so we can read it after tmpdir is gone
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as dest:
            dest_name = dest.name
        with open(profile_file) as src_f:
            content = src_f.read()
        with open(dest_name, "w") as dst_f:
            dst_f.write(content)
    return dest_name


class TestJsProfile(ExtTestCase):
    def test_js_profile_to_dataframe_returns_dataframe(self):
        from pandas import DataFrame
        from yaourt.tools.js_profile import js_profile_to_dataframe

        model = _make_simple_model()
        profile_file = _run_ort_with_profiling(model)
        try:
            df = js_profile_to_dataframe(profile_file)
            self.assertIsInstance(df, DataFrame)
            self.assertGreater(df.shape[0], 0)
        finally:
            os.unlink(profile_file)

    def test_js_profile_to_dataframe_as_list(self):
        from yaourt.tools.js_profile import js_profile_to_dataframe

        model = _make_simple_model()
        profile_file = _run_ort_with_profiling(model)
        try:
            rows = js_profile_to_dataframe(profile_file, as_df=False)
            self.assertIsInstance(rows, list)
            self.assertGreater(len(rows), 0)
            self.assertIsInstance(rows[0], dict)
        finally:
            os.unlink(profile_file)

    def test_js_profile_to_dataframe_has_event_name_column(self):
        from yaourt.tools.js_profile import js_profile_to_dataframe

        model = _make_simple_model()
        profile_file = _run_ort_with_profiling(model)
        try:
            df = js_profile_to_dataframe(profile_file)
            self.assertIn("event_name", df.columns)
            self.assertIn("iteration", df.columns)
        finally:
            os.unlink(profile_file)

    def test_js_profile_to_dataframe_agg(self):
        from pandas import DataFrame
        from yaourt.tools.js_profile import js_profile_to_dataframe

        model = _make_simple_model()
        profile_file = _run_ort_with_profiling(model)
        try:
            df = js_profile_to_dataframe(profile_file, agg=True, first_it_out=True)
            self.assertIsInstance(df, DataFrame)
            self.assertGreater(df.shape[0], 0)
        finally:
            os.unlink(profile_file)

    def test_post_process_df_profile_iteration_column(self):
        from pandas import DataFrame
        from yaourt.tools.js_profile import post_process_df_profile

        # Build a minimal synthetic profiling dataframe
        rows = [
            {
                "name": "SequentialExecutor::Execute",
                "cat": "Session",
                "dur": 100,
                "ts": 0,
                "args_node_index": None,
                "args_op_name": None,
                "args_provider": None,
                "args_input_type_shape": [],
                "args_output_type_shape": [],
            },
            {
                "name": "Relu_kernel_time",
                "cat": "Node",
                "dur": 10,
                "ts": 5,
                "args_node_index": 0,
                "args_op_name": "Relu",
                "args_provider": "CPUExecutionProvider",
                "args_input_type_shape": [],
                "args_output_type_shape": [],
            },
        ]
        df_raw = DataFrame(rows)
        df = post_process_df_profile(df_raw)
        self.assertIn("iteration", df.columns)
        self.assertIn("event_name", df.columns)
        # The Relu row is in iteration 0 (after the first Execute marker)
        self.assertEqual(df.loc[1, "iteration"], 0)

    def test_js_profile_to_dataframe_from_synthetic_file(self):
        """Tests js_profile_to_dataframe directly from a synthetic JSON file."""
        from pandas import DataFrame
        from yaourt.tools.js_profile import js_profile_to_dataframe

        profile_data = [
            {"name": "model_run", "cat": "Session", "dur": 500, "ts": 0, "args": {}},
            {
                "name": "SequentialExecutor::Execute",
                "cat": "Session",
                "dur": 400,
                "ts": 10,
                "args": {},
            },
            {
                "name": "Relu_kernel_time",
                "cat": "Node",
                "dur": 20,
                "ts": 50,
                "args": {
                    "node_index": 0,
                    "op_name": "Relu",
                    "provider": "CPUExecutionProvider",
                    "input_type_shape": [],
                    "output_type_shape": [],
                },
            },
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(profile_data, f)
            fname = f.name
        try:
            df = js_profile_to_dataframe(fname)
            self.assertIsInstance(df, DataFrame)
            self.assertIn("event_name", df.columns)
            self.assertGreater(df.shape[0], 0)
        finally:
            os.unlink(fname)


@skipif_ci_windows("too long")
@skipif_ci_apple("too long")
@requires_matplotlib()
class TestJsProfilePlot(ExtTestCase):
    @classmethod
    def setUp(cls):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cls.plt = plt

    def _get_profile_df(self):
        from yaourt.tools.js_profile import js_profile_to_dataframe

        model = _make_simple_model()
        profile_file = _run_ort_with_profiling(model)
        try:
            df = js_profile_to_dataframe(profile_file, first_it_out=True)
        finally:
            os.unlink(profile_file)
        return df

    def test_plot_ort_profile_returns_axes(self):
        import matplotlib.axes
        from yaourt.tools.js_profile import plot_ort_profile

        df = self._get_profile_df()
        ax = plot_ort_profile(df)
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.plt.close("all")

    def test_plot_ort_profile_with_title(self):
        import matplotlib.axes
        from yaourt.tools.js_profile import plot_ort_profile

        df = self._get_profile_df()
        _fig, ax0 = self.plt.subplots()
        result = plot_ort_profile(df, ax0=ax0, title="Profile")
        self.assertIsInstance(result, matplotlib.axes.Axes)
        self.assertEqual(result.get_title(), "Profile")
        self.plt.close("all")

    def test_plot_ort_profile_timeline_returns_axes(self):
        import matplotlib.axes
        from yaourt.tools.js_profile import plot_ort_profile_timeline

        df = self._get_profile_df()
        ax = plot_ort_profile_timeline(df)
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.plt.close("all")

    def test_plot_ort_profile_timeline_with_title(self):
        import matplotlib.axes
        from yaourt.tools.js_profile import plot_ort_profile_timeline

        df = self._get_profile_df()
        _fig, ax = self.plt.subplots()
        result = plot_ort_profile_timeline(df, ax=ax, title="Timeline")
        self.assertIsInstance(result, matplotlib.axes.Axes)
        self.assertEqual(result.get_title(), "Timeline")
        self.plt.close("all")


if __name__ == "__main__":
    unittest.main(verbosity=2)
