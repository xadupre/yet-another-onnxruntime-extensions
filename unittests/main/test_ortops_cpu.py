"""Tests for the yaourt.ortops.doc catalogue module."""

import os
import tempfile
import unittest

from yaourt.ext_test_case import ExtTestCase

# Absolute path to the C++ source files used by the parser.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LIB_CC = os.path.join(_REPO_ROOT, "yaourt", "ortops", "sparse", "cpu", "ort_optim_cpu2_lib.cc")
_HEADER = os.path.join(_REPO_ROOT, "yaourt", "ortops", "sparse", "cpu", "ort_sparse_lite.h")


class TestOrtOpsCpuCatalogue(ExtTestCase):
    """Verifies the CPU custom-op catalogue exposed by yaourt.ortops.doc."""

    def test_cpu_ops_is_dict(self):
        from yaourt.ortops.doc import CPU_OPS

        self.assertIsInstance(CPU_OPS, dict)

    def test_cpu_ops_contains_expected_ops(self):
        from yaourt.ortops.doc import CPU_OPS

        self.assertIn("DenseToSparse", CPU_OPS)
        self.assertIn("SparseToDense", CPU_OPS)

    def test_dense_to_sparse_domain(self):
        from yaourt.ortops.doc import CPU_OPS

        op = CPU_OPS["DenseToSparse"]
        self.assertEqual(op.domain, "yaourt.ortops.optim.cpu")

    def test_sparse_to_dense_domain(self):
        from yaourt.ortops.doc import CPU_OPS

        op = CPU_OPS["SparseToDense"]
        self.assertEqual(op.domain, "yaourt.ortops.optim.cpu")

    def test_dense_to_sparse_execution_provider(self):
        from yaourt.ortops.doc import CPU_OPS

        op = CPU_OPS["DenseToSparse"]
        self.assertEqual(op.execution_provider, "CPUExecutionProvider")

    def test_sparse_to_dense_execution_provider(self):
        from yaourt.ortops.doc import CPU_OPS

        op = CPU_OPS["SparseToDense"]
        self.assertEqual(op.execution_provider, "CPUExecutionProvider")

    def test_dense_to_sparse_has_one_input_and_one_output(self):
        from yaourt.ortops.doc import CPU_OPS

        op = CPU_OPS["DenseToSparse"]
        self.assertEqual(len(op.inputs), 1)
        self.assertEqual(len(op.outputs), 1)

    def test_sparse_to_dense_has_one_input_and_one_output(self):
        from yaourt.ortops.doc import CPU_OPS

        op = CPU_OPS["SparseToDense"]
        self.assertEqual(len(op.inputs), 1)
        self.assertEqual(len(op.outputs), 1)

    def test_input_output_dtype_is_float32(self):
        from yaourt.ortops.doc import CPU_OPS

        for name, op in CPU_OPS.items():
            for inp in op.inputs:
                self.assertEqual(inp.dtype, "float32", msg=f"{name} input dtype")
            for out in op.outputs:
                self.assertEqual(out.dtype, "float32", msg=f"{name} output dtype")

    def test_all_ops_have_non_empty_doc(self):
        from yaourt.ortops.doc import CPU_OPS

        for name, op in CPU_OPS.items():
            self.assertGreater(len(op.doc), 0, msg=f"{name} doc is empty")

    def test_since_version_is_positive(self):
        from yaourt.ortops.doc import CPU_OPS

        for name, op in CPU_OPS.items():
            self.assertGreater(op.since_version, 0, msg=f"{name} since_version")

    def test_op_name_matches_dict_key(self):
        from yaourt.ortops.doc import CPU_OPS

        for key, op in CPU_OPS.items():
            self.assertEqual(key, op.name)

    def test_ort_op_desc_repr_contains_name(self):
        from yaourt.ortops.doc import CPU_OPS

        op = CPU_OPS["DenseToSparse"]
        self.assertIn("DenseToSparse", repr(op))

    def test_package_init_re_exports_symbols(self):
        from yaourt.ortops import CPU_OPS, OrtOpDesc, OrtOpInput, OrtOpOutput

        self.assertIsInstance(CPU_OPS, dict)
        self.assertTrue(callable(OrtOpDesc))
        self.assertTrue(callable(OrtOpInput))
        self.assertTrue(callable(OrtOpOutput))


@unittest.skipUnless(os.path.exists(_LIB_CC), f"C++ source not found: {_LIB_CC}")
class TestParseLiteLibCc(ExtTestCase):
    """Unit tests for the C++ lib .cc parser."""

    def test_returns_correct_domain(self):
        from yaourt.ortops.doc import _parse_lite_lib_cc

        domain, _ = _parse_lite_lib_cc(_LIB_CC)
        self.assertEqual(domain, "yaourt.ortops.optim.cpu")

    def test_returns_two_registrations(self):
        from yaourt.ortops.doc import _parse_lite_lib_cc

        _, registrations = _parse_lite_lib_cc(_LIB_CC)
        self.assertEqual(len(registrations), 2)

    def test_registration_op_names(self):
        from yaourt.ortops.doc import _parse_lite_lib_cc

        _, registrations = _parse_lite_lib_cc(_LIB_CC)
        op_names = {r[1] for r in registrations}
        self.assertIn("DenseToSparse", op_names)
        self.assertIn("SparseToDense", op_names)

    def test_registration_exec_providers(self):
        from yaourt.ortops.doc import _parse_lite_lib_cc

        _, registrations = _parse_lite_lib_cc(_LIB_CC)
        for _, _, provider in registrations:
            self.assertEqual(provider, "CPUExecutionProvider")

    def test_missing_file_raises(self):
        from yaourt.ortops.doc import _parse_lite_lib_cc

        with self.assertRaises(OSError):
            _parse_lite_lib_cc("/nonexistent/path.cc")

    def test_empty_file_returns_empty(self):
        from yaourt.ortops.doc import _parse_lite_lib_cc

        with tempfile.NamedTemporaryFile(suffix=".cc", mode="w", delete=True) as fh:
            fh.write("// empty\n")
            fh.flush()
            domain, regs = _parse_lite_lib_cc(fh.name)
        self.assertEqual(domain, "")
        self.assertEqual(regs, [])


@unittest.skipUnless(os.path.exists(_HEADER), f"C++ header not found: {_HEADER}")
class TestParseLiteHeader(ExtTestCase):
    """Unit tests for the C++ header parser."""

    def test_finds_both_kernels(self):
        from yaourt.ortops.doc import _parse_lite_header

        result = _parse_lite_header(_HEADER)
        self.assertIn("DenseToSparseKernelLite", result)
        self.assertIn("SparseToDenseKernelLite", result)

    def test_dense_to_sparse_has_one_input_one_output(self):
        from yaourt.ortops.doc import _parse_lite_header

        result = _parse_lite_header(_HEADER)
        params = result["DenseToSparseKernelLite"]
        inputs = [p for p in params if p[2]]
        outputs = [p for p in params if not p[2]]
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(outputs), 1)

    def test_dtypes_mapped_to_float32(self):
        from yaourt.ortops.doc import _parse_lite_header

        result = _parse_lite_header(_HEADER)
        for params in result.values():
            for _, dtype, _ in params:
                self.assertEqual(dtype, "float32")

    def test_empty_file_returns_empty(self):
        from yaourt.ortops.doc import _parse_lite_header

        with tempfile.NamedTemporaryFile(suffix=".h", mode="w", delete=True) as fh:
            fh.write("// empty\n")
            fh.flush()
            result = _parse_lite_header(fh.name)
        self.assertEqual(result, {})


@unittest.skipUnless(os.path.exists(_HEADER), f"C++ header not found: {_HEADER}")
class TestParseLiteHeaderDocs(ExtTestCase):
    """Unit tests for the C++ header doc-comment parser."""

    def test_finds_docs_for_both_kernels(self):
        from yaourt.ortops.doc import _parse_lite_header_docs

        result = _parse_lite_header_docs(_HEADER)
        self.assertIn("DenseToSparseKernelLite", result)
        self.assertIn("SparseToDenseKernelLite", result)

    def test_docs_are_non_empty(self):
        from yaourt.ortops.doc import _parse_lite_header_docs

        result = _parse_lite_header_docs(_HEADER)
        for name, (doc, _) in result.items():
            self.assertGreater(len(doc), 0, msg=f"{name} doc is empty")

    def test_param_docs_contain_x_and_y(self):
        from yaourt.ortops.doc import _parse_lite_header_docs

        result = _parse_lite_header_docs(_HEADER)
        for name, (_, param_docs) in result.items():
            self.assertIn("X", param_docs, msg=f"{name} missing @param X")
            self.assertIn("Y", param_docs, msg=f"{name} missing @param Y")

    def test_param_descriptions_are_non_empty(self):
        from yaourt.ortops.doc import _parse_lite_header_docs

        result = _parse_lite_header_docs(_HEADER)
        for kernel, (_, param_docs) in result.items():
            for param, desc in param_docs.items():
                self.assertGreater(len(desc), 0, msg=f"{kernel}.{param} description is empty")

    def test_empty_file_returns_empty(self):
        from yaourt.ortops.doc import _parse_lite_header_docs

        with tempfile.NamedTemporaryFile(suffix=".h", mode="w", delete=True) as fh:
            fh.write("// empty\n")
            fh.flush()
            result = _parse_lite_header_docs(fh.name)
        self.assertEqual(result, {})


class TestBuildCpuOps(ExtTestCase):
    """Tests for _build_cpu_ops() with explicit file paths."""

    def test_build_with_explicit_paths(self):
        from yaourt.ortops.doc import _build_cpu_ops

        ops = _build_cpu_ops(lib_cc_path=_LIB_CC, header_path=_HEADER)
        self.assertIn("DenseToSparse", ops)
        self.assertIn("SparseToDense", ops)

    def test_returns_empty_when_files_missing(self):
        from yaourt.ortops.doc import _build_cpu_ops

        ops = _build_cpu_ops(lib_cc_path="/nonexistent.cc", header_path="/nonexistent.h")
        self.assertEqual(ops, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
