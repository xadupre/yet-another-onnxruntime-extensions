"""
Tests that every ``plot_*.py`` script in ``docs/examples/`` runs without
error.  The approach mirrors the one used in *yet-another-onnx-builder*.

Each example is imported as a module via :func:`importlib.util`; when that
fails (e.g. the file is not importable on its own) the script is executed
in a subprocess instead.  A test method is generated dynamically for every
script that is discovered so that failures are reported individually.
"""

import importlib.util
import os
import subprocess
import sys
import time
import unittest

from yaourt import __file__ as _yaourt_file
from yaourt.ext_test_case import ExtTestCase, ignore_errors, is_windows

VERBOSE = 0
ROOT = os.path.realpath(os.path.abspath(os.path.join(_yaourt_file, "..", "..")))


def _import_source(module_file_path: str, module_name: str):
    """Executes the module at *module_file_path* as a Python module and returns it.

    :param module_file_path: absolute path to the Python file
    :param module_name: logical name to assign to the module
    :returns: the executed module object
    """
    if not os.path.exists(module_file_path):
        raise FileNotFoundError(module_file_path)
    spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    if spec is None:
        raise FileNotFoundError(f"Unable to find {module_name!r} in {module_file_path!r}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDocumentationExamples(ExtTestCase):
    """Runs every ``plot_*.py`` script found under ``docs/examples/``."""

    def run_test(self, fold: str, name: str, verbose: int = 0) -> int:
        """Executes the example at *fold/name* in-process; falls back to subprocess on failure.

        :param fold: directory that contains the example script
        :param name: filename of the example script
        :param verbose: when non-zero, prints the elapsed time
        :returns: 1 on success
        """
        ppath = os.environ.get("PYTHONPATH", "")
        if not ppath:
            os.environ["PYTHONPATH"] = ROOT
        elif ROOT not in ppath:
            sep = ";" if is_windows() else ":"
            os.environ["PYTHONPATH"] = ppath + sep + ROOT
        perf = time.perf_counter()
        try:
            mod = _import_source(os.path.join(fold, name), os.path.splitext(name)[0])
            assert mod is not None
        except FileNotFoundError:
            # Fall back to running the script in a subprocess.
            cmds = [sys.executable, "-u", os.path.join(fold, name)]
            p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _out, err = p.communicate()
            st = err.decode("ascii", errors="ignore")
            if st and "Traceback" in st:
                if '"dot" not found in path.' in st:
                    raise unittest.SkipTest(f"failed: {name!r} due to missing dot.")
                if (
                    "We couldn't connect to 'https://huggingface.co'" in st
                    or "Cannot access content at: https://huggingface.co/" in st
                ):
                    raise unittest.SkipTest(f"Connectivity issues due to\n{err}")
                raise AssertionError(
                    f"Example {name!r} (cmd: {cmds!r}, exec_prefix={sys.exec_prefix!r}) "
                    f"failed due to\n{st}"
                )
        dt = time.perf_counter() - perf
        if verbose:
            print(f"{dt:.3f}: run {name!r}")
        return 1

    @classmethod
    def add_test_methods(cls):
        """Scans ``docs/examples/`` for ``plot_*.py`` files and attaches one test method per script."""  # noqa: E501
        this = os.path.abspath(os.path.dirname(__file__))
        root_fold = os.path.normpath(os.path.join(this, "..", "..", "docs", "examples"))

        found = []
        if os.path.isdir(root_fold):
            for entry in sorted(os.listdir(root_fold)):
                fold = os.path.join(root_fold, entry)
                if os.path.isdir(fold):
                    for name in sorted(os.listdir(fold)):
                        if name.endswith(".py") and name.startswith("plot_"):
                            found.append((fold, name))

        for fold, name in found:
            reason = None

            if not reason and is_windows():
                reason = "CI complains on Windows"

            short_name = os.path.splitext(name)[0]

            if reason:

                @unittest.skip(reason)
                def _test_(self, fold=fold, name=name):
                    res = self.run_test(fold, name, verbose=VERBOSE)
                    self.assertTrue(res)

            else:

                @ignore_errors(OSError)
                def _test_(self, fold=fold, name=name):
                    res = self.run_test(fold, name, verbose=VERBOSE)
                    self.assertTrue(res)

            setattr(cls, f"test_{short_name}", _test_)


TestDocumentationExamples.add_test_methods()

if __name__ == "__main__":
    unittest.main(verbosity=2)
