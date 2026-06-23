"""Tests for the yaourt.ortops.doc catalogue module."""

import os
import tempfile
import unittest

from yaourt.ext_test_case import ExtTestCase

# Absolute path to the C++ source files used by the parser.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LIB_CC = os.path.join(_REPO_ROOT, "yaourt", "ortops", "sparse", "cpu", "ort_sparse_cpu2_lib.cc")
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
        self.assertEqual(op.domain, "yaourt.ortops.sparse.cpu")

    def test_sparse_to_dense_domain(self):
        from yaourt.ortops.doc import CPU_OPS

        op = CPU_OPS["SparseToDense"]
        self.assertEqual(op.domain, "yaourt.ortops.sparse.cpu")

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
        self.assertEqual(domain, "yaourt.ortops.sparse.cpu")

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


class TestPrintCpuOpsRst(ExtTestCase):
    """Tests for print_cpu_ops_rst()."""

    def _capture(self) -> str:
        """Captures stdout from print_cpu_ops_rst() and returns it as a string."""
        import io
        from contextlib import redirect_stdout

        from yaourt.ortops.doc import print_cpu_ops_rst

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_cpu_ops_rst()
        return buf.getvalue()

    def test_returns_non_empty_output(self):
        output = self._capture()
        self.assertGreater(len(output), 0)

    def test_output_contains_op_names(self):
        from yaourt.ortops.doc import CPU_OPS

        output = self._capture()
        for op_name in CPU_OPS:
            self.assertIn(op_name, output)

    def test_output_contains_list_table_directive(self):
        output = self._capture()
        self.assertIn(".. list-table::", output)

    def test_output_contains_domain(self):
        from yaourt.ortops.doc import CPU_OPS

        output = self._capture()
        for op in CPU_OPS.values():
            self.assertIn(op.domain, output)

    def test_output_contains_inputs_and_outputs_sections(self):
        from yaourt.ortops.doc import CPU_OPS

        output = self._capture()
        if any(op.inputs for op in CPU_OPS.values()):
            self.assertIn("**Inputs**", output)
        if any(op.outputs for op in CPU_OPS.values()):
            self.assertIn("**Outputs**", output)

    def test_package_init_re_exports_print_cpu_ops_rst(self):
        from yaourt.ortops import print_cpu_ops_rst

        self.assertTrue(callable(print_cpu_ops_rst))

    def test_empty_catalogue_prints_fallback_message(self):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch

        from yaourt.ortops import doc as doc_module

        buf = io.StringIO()
        with patch.object(doc_module, "CPU_OPS", {}), redirect_stdout(buf):
            doc_module.print_cpu_ops_rst()
        self.assertIn("No CPU ops found", buf.getvalue())


# ---------------------------------------------------------------------------
# CUDA ops catalogue tests
# ---------------------------------------------------------------------------

_CUDA_DIR = os.path.join(_REPO_ROOT, "yaourt", "ortops", "fused_kernel", "cuda")
_CUDA_LIB_CU = os.path.join(_CUDA_DIR, "ort_fused_kernel_cuda_lib.cu")

# ---------------------------------------------------------------------------
# Fused-kernel CPU ops catalogue tests
# ---------------------------------------------------------------------------

_FUSED_KERNEL_CPU_DIR = os.path.join(_REPO_ROOT, "yaourt", "ortops", "fused_kernel", "cpu")
_FUSED_KERNEL_CPU_LIB_CC = os.path.join(_FUSED_KERNEL_CPU_DIR, "ort_fused_kernel_cpu_lib.cc")


class TestFusedKernelCpuOpsCatalogue(ExtTestCase):
    """Verifies the fused-kernel CPU custom-op catalogue exposed by yaourt.ortops.doc."""

    def test_fused_kernel_cpu_ops_is_dict(self):
        from yaourt.ortops.doc import FUSED_KERNEL_CPU_OPS

        self.assertIsInstance(FUSED_KERNEL_CPU_OPS, dict)

    def test_fused_kernel_cpu_ops_contains_mulmul(self):
        from yaourt.ortops.doc import FUSED_KERNEL_CPU_OPS

        self.assertIn("MulMul", FUSED_KERNEL_CPU_OPS)

    def test_fused_kernel_cpu_ops_domain(self):
        from yaourt.ortops.doc import FUSED_KERNEL_CPU_OPS

        for op in FUSED_KERNEL_CPU_OPS.values():
            self.assertEqual(op.domain, "yaourt.ortops.fused_kernel.cpu")

    def test_fused_kernel_cpu_ops_execution_provider(self):
        from yaourt.ortops.doc import FUSED_KERNEL_CPU_OPS

        for op in FUSED_KERNEL_CPU_OPS.values():
            self.assertEqual(op.execution_provider, "CPUExecutionProvider")

    def test_mulmul_has_three_inputs_one_output(self):
        from yaourt.ortops.doc import FUSED_KERNEL_CPU_OPS

        op = FUSED_KERNEL_CPU_OPS["MulMul"]
        self.assertEqual(len(op.inputs), 3)
        self.assertEqual(len(op.outputs), 1)

    def test_mulmul_inputs_are_float32(self):
        from yaourt.ortops.doc import FUSED_KERNEL_CPU_OPS

        op = FUSED_KERNEL_CPU_OPS["MulMul"]
        for inp in op.inputs:
            self.assertEqual(inp.dtype, "float32")
        for out in op.outputs:
            self.assertEqual(out.dtype, "float32")

    def test_mulmul_has_non_empty_doc(self):
        from yaourt.ortops.doc import FUSED_KERNEL_CPU_OPS

        op = FUSED_KERNEL_CPU_OPS["MulMul"]
        self.assertGreater(len(op.doc), 0)

    def test_package_init_re_exports_fused_kernel_cpu_ops(self):
        from yaourt.ortops import FUSED_KERNEL_CPU_OPS

        self.assertIsInstance(FUSED_KERNEL_CPU_OPS, dict)

    def test_package_init_re_exports_fused_kernel_cpu_lib_path(self):
        from yaourt.ortops import FUSED_KERNEL_CPU_LIB_PATH

        self.assertIsNotNone(FUSED_KERNEL_CPU_LIB_PATH)

    def test_package_init_re_exports_print_fused_kernel_cpu_ops_rst(self):
        from yaourt.ortops import print_fused_kernel_cpu_ops_rst

        self.assertTrue(callable(print_fused_kernel_cpu_ops_rst))


@unittest.skipUnless(
    os.path.isdir(_FUSED_KERNEL_CPU_DIR),
    f"Fused-kernel CPU source dir not found: {_FUSED_KERNEL_CPU_DIR}",
)
class TestBuildFusedKernelCpuOps(ExtTestCase):
    """Tests for _build_fused_kernel_cpu_ops()."""

    def test_build_finds_mulmul(self):
        from yaourt.ortops.doc import _build_fused_kernel_cpu_ops

        ops = _build_fused_kernel_cpu_ops(cpu_dir=_FUSED_KERNEL_CPU_DIR)
        self.assertIn("MulMul", ops)

    def test_returns_empty_when_dir_missing(self):
        from yaourt.ortops.doc import _build_fused_kernel_cpu_ops

        ops = _build_fused_kernel_cpu_ops(cpu_dir="/nonexistent/cpu")
        self.assertEqual(ops, {})

    def test_get_ort_ext_libs_fused_kernel_cpu_domain(self):
        from yaourt.ortops import get_ort_ext_libs

        # Should not raise even when the .so is absent.
        libs = get_ort_ext_libs("CPUExecutionProvider", domain="yaourt.ortops.fused_kernel.cpu")
        self.assertIsInstance(libs, list)


@unittest.skipUnless(
    os.path.isdir(_FUSED_KERNEL_CPU_DIR),
    f"Fused-kernel CPU source dir not found: {_FUSED_KERNEL_CPU_DIR}",
)
class TestPrintFusedKernelCpuOpsRst(ExtTestCase):
    """Tests for print_fused_kernel_cpu_ops_rst()."""

    def _capture(self) -> str:
        """Captures stdout from print_fused_kernel_cpu_ops_rst() and returns it."""
        import io
        from contextlib import redirect_stdout

        from yaourt.ortops.doc import print_fused_kernel_cpu_ops_rst

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_fused_kernel_cpu_ops_rst()
        return buf.getvalue()

    def test_returns_non_empty_output(self):
        output = self._capture()
        self.assertGreater(len(output), 0)

    def test_output_contains_mulmul(self):
        output = self._capture()
        self.assertIn("MulMul", output)

    def test_output_contains_list_table_directive(self):
        output = self._capture()
        self.assertIn(".. list-table::", output)

    def test_output_contains_domain(self):
        output = self._capture()
        self.assertIn("yaourt.ortops.fused_kernel.cpu", output)

    def test_empty_catalogue_prints_fallback_message(self):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch

        from yaourt.ortops import doc as doc_module

        buf = io.StringIO()
        with patch.object(doc_module, "FUSED_KERNEL_CPU_OPS", {}), redirect_stdout(buf):
            doc_module.print_fused_kernel_cpu_ops_rst()
        self.assertIn("No fused-kernel CPU ops found", buf.getvalue())


class TestCudaOpsCatalogue(ExtTestCase):
    """Verifies the CUDA custom-op catalogue exposed by yaourt.ortops.doc."""

    def test_cuda_ops_is_dict(self):
        from yaourt.ortops.doc import CUDA_OPS

        self.assertIsInstance(CUDA_OPS, dict)

    def test_cuda_ops_contains_expected_ops(self):
        from yaourt.ortops.doc import CUDA_OPS

        expected = {"AddMul", "MulAdd", "NegXplus1", "Rotary", "ScatterNDOfShape"}
        for name in expected:
            self.assertIn(name, CUDA_OPS)

    def test_cuda_ops_domain(self):
        from yaourt.ortops.doc import CUDA_OPS

        for op in CUDA_OPS.values():
            self.assertEqual(op.domain, "yaourt.ortops.fused_kernel.cuda")

    def test_cuda_ops_execution_provider(self):
        from yaourt.ortops.doc import CUDA_OPS

        for op in CUDA_OPS.values():
            self.assertEqual(op.execution_provider, "CUDAExecutionProvider")

    def test_package_init_re_exports_cuda_ops(self):
        from yaourt.ortops import CUDA_OPS

        self.assertIsInstance(CUDA_OPS, dict)

    def test_package_init_re_exports_print_cuda_ops_rst(self):
        from yaourt.ortops import print_cuda_ops_rst

        self.assertTrue(callable(print_cuda_ops_rst))


@unittest.skipUnless(os.path.exists(_CUDA_LIB_CU), f"CUDA lib source not found: {_CUDA_LIB_CU}")
class TestParseCudaLibCu(ExtTestCase):
    """Unit tests for the CUDA lib .cu domain parser."""

    def test_returns_correct_domain(self):
        from yaourt.ortops.doc import _parse_cuda_lib_cu

        domain = _parse_cuda_lib_cu(_CUDA_LIB_CU)
        self.assertEqual(domain, "yaourt.ortops.fused_kernel.cuda")

    def test_missing_file_raises(self):
        from yaourt.ortops.doc import _parse_cuda_lib_cu

        with self.assertRaises(OSError):
            _parse_cuda_lib_cu("/nonexistent/path.cu")


@unittest.skipUnless(os.path.isdir(_CUDA_DIR), f"CUDA source dir not found: {_CUDA_DIR}")
class TestParseCudaKernelCu(ExtTestCase):
    """Unit tests for the CUDA kernel .cu parser."""

    def test_addmul_cu_returns_two_op_names(self):
        from yaourt.ortops.doc import _parse_cuda_kernel_cu

        cu_path = os.path.join(_CUDA_DIR, "addmul.cu")
        op_names, n_in, n_out, ep = _parse_cuda_kernel_cu(cu_path)
        self.assertIn("AddMul", op_names)
        self.assertIn("MulAdd", op_names)
        self.assertEqual(n_in, 3)
        self.assertEqual(n_out, 1)
        self.assertEqual(ep, "CUDAExecutionProvider")

    def test_negxplus1_cu_returns_one_op_name(self):
        from yaourt.ortops.doc import _parse_cuda_kernel_cu

        cu_path = os.path.join(_CUDA_DIR, "negxplus1.cu")
        op_names, n_in, n_out, _ep = _parse_cuda_kernel_cu(cu_path)
        self.assertEqual(op_names, ["NegXplus1"])
        self.assertEqual(n_in, 1)
        self.assertEqual(n_out, 1)

    def test_transpose_cast_2d_cu_returns_two_op_names(self):
        from yaourt.ortops.doc import _parse_cuda_kernel_cu

        cu_path = os.path.join(_CUDA_DIR, "transpose_cast_2d.cu")
        op_names, _, _, _ = _parse_cuda_kernel_cu(cu_path)
        self.assertIn("Transpose2DCastFP16", op_names)
        self.assertIn("Transpose2DCastFP32", op_names)


@unittest.skipUnless(os.path.isdir(_CUDA_DIR), f"CUDA source dir not found: {_CUDA_DIR}")
class TestBuildCudaOps(ExtTestCase):
    """Tests for _build_cuda_ops()."""

    def test_build_finds_expected_ops(self):
        from yaourt.ortops.doc import _build_cuda_ops

        ops = _build_cuda_ops(cuda_dir=_CUDA_DIR)
        self.assertIn("AddMul", ops)
        self.assertIn("NegXplus1", ops)
        self.assertIn("Rotary", ops)

    def test_returns_empty_when_dir_missing(self):
        from yaourt.ortops.doc import _build_cuda_ops

        ops = _build_cuda_ops(cuda_dir="/nonexistent/cuda")
        self.assertEqual(ops, {})


@unittest.skipUnless(os.path.isdir(_CUDA_DIR), f"CUDA source dir not found: {_CUDA_DIR}")
class TestPrintCudaOpsRst(ExtTestCase):
    """Tests for print_cuda_ops_rst()."""

    def _capture(self) -> str:
        """Captures stdout from print_cuda_ops_rst() and returns it as a string."""
        import io
        from contextlib import redirect_stdout

        from yaourt.ortops.doc import print_cuda_ops_rst

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_cuda_ops_rst()
        return buf.getvalue()

    def test_returns_non_empty_output(self):
        output = self._capture()
        self.assertGreater(len(output), 0)

    def test_output_contains_op_names(self):
        from yaourt.ortops.doc import CUDA_OPS

        output = self._capture()
        for op_name in CUDA_OPS:
            self.assertIn(op_name, output)

    def test_output_contains_list_table_directive(self):
        output = self._capture()
        self.assertIn(".. list-table::", output)

    def test_output_contains_domain(self):
        from yaourt.ortops.doc import CUDA_OPS

        output = self._capture()
        for op in CUDA_OPS.values():
            self.assertIn(op.domain, output)

    def test_empty_catalogue_prints_fallback_message(self):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch

        from yaourt.ortops import doc as doc_module

        buf = io.StringIO()
        with patch.object(doc_module, "CUDA_OPS", {}), redirect_stdout(buf):
            doc_module.print_cuda_ops_rst()
        self.assertIn("No CUDA ops found", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
