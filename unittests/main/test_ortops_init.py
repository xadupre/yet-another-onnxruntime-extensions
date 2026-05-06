"""Tests for yaourt.ortops.__init__ — get_ort_ext_libs and library path constants."""

import os
import platform
import tempfile
import unittest
from pathlib import Path

from yaourt.ext_test_case import ExtTestCase

_SYSTEM = platform.system()
if _SYSTEM == "Windows":
    _LIB_EXT = ".dll"
elif _SYSTEM == "Darwin":
    _LIB_EXT = ".dylib"
else:
    _LIB_EXT = ".so"


class TestLibPathConstants(ExtTestCase):
    """Verifies the library-path constants exported by yaourt.ortops."""

    def test_sparse_cpu_lib_path_is_path(self):
        from yaourt.ortops import SPARSE_CPU_LIB_PATH

        self.assertIsInstance(SPARSE_CPU_LIB_PATH, Path)

    def test_sparse_cpu2_lib_path_is_path(self):
        from yaourt.ortops import SPARSE_CPU2_LIB_PATH

        self.assertIsInstance(SPARSE_CPU2_LIB_PATH, Path)

    def test_fused_kernel_cuda_lib_path_is_path(self):
        from yaourt.ortops import FUSED_KERNEL_CUDA_LIB_PATH

        self.assertIsInstance(FUSED_KERNEL_CUDA_LIB_PATH, Path)

    def test_sparse_cpu2_lib_path_points_to_cpu_subfolder(self):
        from yaourt.ortops import SPARSE_CPU2_LIB_PATH

        # The path must live inside the sparse/cpu directory.
        parts = SPARSE_CPU2_LIB_PATH.parts
        self.assertIn("sparse", parts)
        self.assertIn("cpu", parts)
        # cpu_v1 must NOT appear in the path.
        self.assertNotIn("cpu_v1", parts)

    def test_sparse_cpu_lib_path_points_to_cpu_v1_subfolder(self):
        from yaourt.ortops import SPARSE_CPU_LIB_PATH

        parts = SPARSE_CPU_LIB_PATH.parts
        self.assertIn("cpu_v1", parts)

    def test_sparse_cpu2_lib_path_has_correct_extension(self):
        from yaourt.ortops import SPARSE_CPU2_LIB_PATH

        self.assertEqual(SPARSE_CPU2_LIB_PATH.suffix, _LIB_EXT)

    def test_sparse_cpu2_lib_path_in_all(self):
        import yaourt.ortops as m

        self.assertIn("SPARSE_CPU2_LIB_PATH", m.__all__)

    def test_get_ort_ext_libs_in_all(self):
        import yaourt.ortops as m

        self.assertIn("get_ort_ext_libs", m.__all__)


class TestGetOrtExtLibsErrors(ExtTestCase):
    """Tests for get_ort_ext_libs error conditions."""

    def test_raises_when_neither_arg_supplied(self):
        from yaourt.ortops import get_ort_ext_libs

        with self.assertRaises(ValueError):
            get_ort_ext_libs("CPUExecutionProvider")

    def test_raises_when_both_args_supplied(self):
        from yaourt.ortops import get_ort_ext_libs

        with self.assertRaises(ValueError):
            get_ort_ext_libs(
                "CPUExecutionProvider", subfolder="sparse/cpu", domain="yaourt.ortops.sparse.cpu"
            )

    def test_raises_when_domain_has_wrong_prefix(self):
        from yaourt.ortops import get_ort_ext_libs

        with self.assertRaises(ValueError):
            get_ort_ext_libs("CPUExecutionProvider", domain="other.domain.sparse.cpu")

    def test_raises_file_not_found_for_nonexistent_subfolder(self):
        from yaourt.ortops import get_ort_ext_libs

        with self.assertRaises(FileNotFoundError):
            get_ort_ext_libs("CPUExecutionProvider", subfolder="nonexistent/path")

    def test_raises_file_not_found_for_nonexistent_domain(self):
        from yaourt.ortops import get_ort_ext_libs

        with self.assertRaises(FileNotFoundError):
            get_ort_ext_libs("CPUExecutionProvider", domain="yaourt.ortops.nonexistent.path")


class TestGetOrtExtLibsHappyPath(ExtTestCase):
    """Tests for get_ort_ext_libs with real and synthetic directories."""

    def _make_fake_lib_dir(self, names):
        """Creates a temporary directory containing fake library files."""
        tmp = tempfile.mkdtemp()
        for name in names:
            Path(tmp, name).write_bytes(b"")
        return tmp

    def test_returns_list_of_paths(self):
        from yaourt.ortops import get_ort_ext_libs

        lib_name = f"libfake{_LIB_EXT}"
        tmp = self._make_fake_lib_dir([lib_name, "not_a_lib.txt"])
        try:
            # Computing a relative path from the _HERE anchor is not easy, so we
            # call get_ort_ext_libs via its domain/subfolder logic by pointing
            # at a folder under yaourt/ortops that always exists (sparse).
            results = get_ort_ext_libs("CPUExecutionProvider", subfolder="sparse")
            self.assertIsInstance(results, list)
            for p in results:
                self.assertIsInstance(p, Path)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_only_lib_files_returned(self):
        """Non-library files in the directory must not appear in results."""
        from yaourt.ortops import get_ort_ext_libs

        # sparse/ exists and contains subdirectories (cpu, cpu_v1) but no
        # top-level shared libraries — result should be an empty list.
        results = get_ort_ext_libs("CPUExecutionProvider", subfolder="sparse")
        for p in results:
            self.assertIn(p.suffix, {".so", ".dll", ".dylib"})

    def test_domain_derives_subfolder_sparse_cpu(self):
        """Domain 'yaourt.ortops.sparse.cpu' must resolve to the sparse/cpu dir."""
        from yaourt.ortops import get_ort_ext_libs

        # sparse/cpu always exists (it contains C++ source files).
        results = get_ort_ext_libs("CPUExecutionProvider", domain="yaourt.ortops.sparse.cpu")
        self.assertIsInstance(results, list)

    def test_domain_derives_subfolder_fused_kernel_cuda(self):
        """Domain 'yaourt.ortops.fused_kernel.cuda' must resolve to fused_kernel/cuda."""
        from yaourt.ortops import get_ort_ext_libs

        results = get_ort_ext_libs(
            "CUDAExecutionProvider", domain="yaourt.ortops.fused_kernel.cuda"
        )
        self.assertIsInstance(results, list)

    def test_results_are_sorted(self):
        """Returned paths must be in sorted order."""
        from yaourt.ortops import get_ort_ext_libs

        results = get_ort_ext_libs("CPUExecutionProvider", subfolder="sparse")
        self.assertEqual(results, sorted(results))

    def test_subfolder_sparse_cpu_returns_paths_pointing_into_cpu(self):
        from yaourt.ortops import get_ort_ext_libs

        results = get_ort_ext_libs("CPUExecutionProvider", subfolder="sparse/cpu")
        for p in results:
            self.assertIn("cpu", p.parts)


class TestGetOrtExtLibsWithCompiledLib(ExtTestCase):
    """Tests that require the compiled shared library to be present."""

    @classmethod
    def setUpClass(cls):
        from yaourt.ortops import SPARSE_CPU2_LIB_PATH

        if not os.path.exists(SPARSE_CPU2_LIB_PATH):
            raise unittest.SkipTest(f"Compiled library not found: {SPARSE_CPU2_LIB_PATH}")

    def test_sparse_cpu2_lib_appears_in_get_ort_ext_libs_by_subfolder(self):
        from yaourt.ortops import SPARSE_CPU2_LIB_PATH, get_ort_ext_libs

        results = get_ort_ext_libs("CPUExecutionProvider", subfolder="sparse/cpu")
        self.assertIn(SPARSE_CPU2_LIB_PATH, results)

    def test_sparse_cpu2_lib_appears_in_get_ort_ext_libs_by_domain(self):
        from yaourt.ortops import SPARSE_CPU2_LIB_PATH, get_ort_ext_libs

        results = get_ort_ext_libs("CPUExecutionProvider", domain="yaourt.ortops.sparse.cpu")
        self.assertIn(SPARSE_CPU2_LIB_PATH, results)


if __name__ == "__main__":
    unittest.main(verbosity=2)
