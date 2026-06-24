"""Tests for Python CPU feature helpers exposed by yaourt.ortops."""

import os
import unittest

from yaourt.ext_test_case import ExtTestCase


class TestCpuFeaturesExports(ExtTestCase):
    """Verifies CPU feature helpers are exported in yaourt.ortops."""

    def test_cpu_feature_functions_in_all(self):
        import yaourt.ortops as m

        self.assertIn("cpu_supports_avx2", m.__all__)
        self.assertIn("cpu_supports_avx512f", m.__all__)
        self.assertIn("cpu_supports_f16c", m.__all__)


class TestCpuFeaturesRuntime(ExtTestCase):
    """Verifies CPU feature helpers call into the compiled shared library."""

    @classmethod
    def setUpClass(cls):
        from yaourt.ortops import FUSED_KERNEL_CPU_LIB_PATH

        if not os.path.exists(FUSED_KERNEL_CPU_LIB_PATH):
            raise unittest.SkipTest(f"Compiled library not found: {FUSED_KERNEL_CPU_LIB_PATH}")

    def test_cpu_supports_avx2_returns_bool(self):
        from yaourt.ortops import cpu_supports_avx2

        self.assertIsInstance(cpu_supports_avx2(), bool)

    def test_cpu_supports_f16c_returns_bool(self):
        from yaourt.ortops import cpu_supports_f16c

        self.assertIsInstance(cpu_supports_f16c(), bool)

    def test_cpu_supports_avx512f_returns_bool(self):
        from yaourt.ortops import cpu_supports_avx512f

        self.assertIsInstance(cpu_supports_avx512f(), bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
