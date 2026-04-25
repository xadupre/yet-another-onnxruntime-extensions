"""Tests for the yaourt.ortops.cpu catalogue module."""

import unittest

from yaourt.ext_test_case import ExtTestCase


class TestOrtOpsCpuCatalogue(ExtTestCase):
    """Verifies the CPU custom-op catalogue exposed by yaourt.ortops.cpu."""

    def test_cpu_ops_is_dict(self):
        from yaourt.ortops.cpu import CPU_OPS

        self.assertIsInstance(CPU_OPS, dict)

    def test_cpu_ops_contains_expected_ops(self):
        from yaourt.ortops.cpu import CPU_OPS

        self.assertIn("DenseToSparse", CPU_OPS)
        self.assertIn("SparseToDense", CPU_OPS)

    def test_dense_to_sparse_domain(self):
        from yaourt.ortops.cpu import CPU_OPS

        op = CPU_OPS["DenseToSparse"]
        self.assertEqual(op.domain, "yaourt.ortops.optim.cpu")

    def test_sparse_to_dense_domain(self):
        from yaourt.ortops.cpu import CPU_OPS

        op = CPU_OPS["SparseToDense"]
        self.assertEqual(op.domain, "yaourt.ortops.optim.cpu")

    def test_dense_to_sparse_execution_provider(self):
        from yaourt.ortops.cpu import CPU_OPS

        op = CPU_OPS["DenseToSparse"]
        self.assertEqual(op.execution_provider, "CPUExecutionProvider")

    def test_sparse_to_dense_execution_provider(self):
        from yaourt.ortops.cpu import CPU_OPS

        op = CPU_OPS["SparseToDense"]
        self.assertEqual(op.execution_provider, "CPUExecutionProvider")

    def test_dense_to_sparse_has_one_input_and_one_output(self):
        from yaourt.ortops.cpu import CPU_OPS

        op = CPU_OPS["DenseToSparse"]
        self.assertEqual(len(op.inputs), 1)
        self.assertEqual(len(op.outputs), 1)

    def test_sparse_to_dense_has_one_input_and_one_output(self):
        from yaourt.ortops.cpu import CPU_OPS

        op = CPU_OPS["SparseToDense"]
        self.assertEqual(len(op.inputs), 1)
        self.assertEqual(len(op.outputs), 1)

    def test_input_output_dtype_is_float32(self):
        from yaourt.ortops.cpu import CPU_OPS

        for name, op in CPU_OPS.items():
            for inp in op.inputs:
                self.assertEqual(inp.dtype, "float32", msg=f"{name} input dtype")
            for out in op.outputs:
                self.assertEqual(out.dtype, "float32", msg=f"{name} output dtype")

    def test_all_ops_have_non_empty_doc(self):
        from yaourt.ortops.cpu import CPU_OPS

        for name, op in CPU_OPS.items():
            self.assertGreater(len(op.doc), 0, msg=f"{name} doc is empty")

    def test_since_version_is_positive(self):
        from yaourt.ortops.cpu import CPU_OPS

        for name, op in CPU_OPS.items():
            self.assertGreater(op.since_version, 0, msg=f"{name} since_version")

    def test_op_name_matches_dict_key(self):
        from yaourt.ortops.cpu import CPU_OPS

        for key, op in CPU_OPS.items():
            self.assertEqual(key, op.name)

    def test_ort_op_desc_repr_contains_name(self):
        from yaourt.ortops.cpu import CPU_OPS

        op = CPU_OPS["DenseToSparse"]
        self.assertIn("DenseToSparse", repr(op))

    def test_package_init_re_exports_symbols(self):
        from yaourt.ortops import CPU_OPS, OrtOpDesc, OrtOpInput, OrtOpOutput

        self.assertIsInstance(CPU_OPS, dict)
        self.assertTrue(callable(OrtOpDesc))
        self.assertTrue(callable(OrtOpInput))
        self.assertTrue(callable(OrtOpOutput))


if __name__ == "__main__":
    unittest.main(verbosity=2)
