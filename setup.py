"""Integrates the cmake build step into pip install for the C++ custom-op libraries.

When ``pip install .`` or ``pip install -e .`` is run, this module triggers
cmake to configure and build the shared-library custom ops so they are
available alongside the Python sources without a separate build step.

If a CUDA compiler (``nvcc``) is detected on the PATH at build time the
package is distributed as ``yet-another-onnxruntime-extensions-cuda``; without
CUDA it falls back to the plain ``yet-another-onnxruntime-extensions`` name.
"""

from __future__ import annotations

import warnings

# Suppress deprecation warnings emitted by pytest-runner (the `ptr` package)
# when setuptools >= 77 loads all installed distutils command entry-points
# during setup().  These originate in a third-party package and cannot be
# fixed here; they will disappear once pytest-runner is updated or uninstalled.
warnings.filterwarnings("ignore", module=r"ptr(\..+)?$")

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop
from setuptools.dist import Distribution as _Distribution

_HERE = Path(__file__).parent.resolve()

_BASE_PACKAGE_NAME = "yet-another-onnxruntime-extensions"

# Extra cmake options exposed as command-line arguments on all build commands.
_CMAKE_OPTIONS: list[tuple[str, str | None, str]] = [
    (
        "cuda-architectures=",
        None,
        'CUDA architectures to compile for, semi-colon separated (e.g. "86;89;90")',
    ),
    (
        "cmake-args=",
        None,
        'additional cmake configure arguments, space-separated (e.g. "-DFOO=1 -DBAR=2")',
    ),
    ("cpp-tests", None, "build the C++ unit tests (passes -DBUILD_CPP_TESTS=ON to cmake)"),
]

# Names of boolean (flag) options in ``_CMAKE_OPTIONS`` that take no value.
_CMAKE_BOOLEAN_OPTIONS = ("cpp-tests",)


def _cuda_available() -> bool:
    """Checks whether an ``nvcc`` CUDA compiler is found on the PATH."""
    return shutil.which("nvcc") is not None


def _package_name() -> str:
    """Computes the distribution name, appending '-cuda' when CUDA is available."""
    if _cuda_available():
        return f"{_BASE_PACKAGE_NAME}-cuda"
    return _BASE_PACKAGE_NAME


def _run_cmake(
    cuda_architectures: str | None = None, cmake_args: str | None = None, cpp_tests: bool = False
) -> None:
    """Configures and builds the C++ custom-op shared libraries via cmake.

    Prints a warning and returns without error when cmake is absent or when
    the build fails (e.g. because CUDA is not available), so that the Python
    package can still be installed in reduced-functionality mode.

    :param cuda_architectures: optional semi-colon-separated list of CUDA
        compute architectures passed as ``-DCMAKE_CUDA_ARCHITECTURES``
        (e.g. ``"86;89;90"``).
    :param cmake_args: optional space-separated extra cmake configure
        arguments (e.g. ``"-DFOO=bar -DBAZ=1"``).
    :param cpp_tests: when ``True``, builds the C++ unit tests by passing
        ``-DBUILD_CPP_TESTS=ON`` to the cmake configure step.
    """
    cmake = shutil.which("cmake")
    if cmake is None:
        print(
            "yaourt: cmake executable not found on PATH; "
            "C++ custom-op libraries will not be built.",
            flush=True,
        )
        return

    cmake_src = _HERE / "cmake"
    if not cmake_src.is_dir():
        print(
            f"yaourt: cmake source directory not found at {cmake_src}; "
            "C++ custom-op libraries will not be built.",
            flush=True,
        )
        return

    build_dir = _HERE / "build"
    configure_cmd = [cmake, f"-S{cmake_src}", f"-B{build_dir}", "-DCMAKE_BUILD_TYPE=Release"]

    ort_version = os.environ.get("ORT_VERSION")
    if ort_version:
        configure_cmd.append(f"-DORT_VERSION={ort_version}")
        print(f"yaourt: using local ONNX Runtime from {ort_version}", flush=True)

    if cuda_architectures:
        configure_cmd.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}")
        print(f"yaourt: CUDA architectures: {cuda_architectures}", flush=True)

    if cmake_args:
        configure_cmd.extend(cmake_args.split())
        print(f"yaourt: extra cmake args: {cmake_args}", flush=True)

    if cpp_tests:
        configure_cmd.append("-DBUILD_CPP_TESTS=ON")
        print("yaourt: building C++ unit tests", flush=True)

    build_cmd = [cmake, "--build", str(build_dir), "--config", "Release"]

    print("yaourt: cmake configure ...", flush=True)
    result = subprocess.run(configure_cmd, cwd=str(_HERE))
    if result.returncode != 0:
        print(
            f"yaourt: cmake configure step failed (exit code {result.returncode}); "
            "C++ custom-op libraries will not be built.",
            flush=True,
        )
        return

    print("yaourt: cmake build ...", flush=True)
    result = subprocess.run(build_cmd, cwd=str(_HERE))
    if result.returncode != 0:
        print(
            f"yaourt: cmake build step failed (exit code {result.returncode}); "
            "C++ custom-op libraries may be incomplete.",
            flush=True,
        )


class _CMakeMixin:
    """Mixin that adds cmake-specific command-line options to a setuptools command.

    Provides ``--cuda-architectures``, ``--cmake-args`` and ``--cpp-tests``
    options that are forwarded to the cmake configure step.  Subclasses must
    combine ``_CMAKE_OPTIONS`` into their own ``user_options`` (and
    ``_CMAKE_BOOLEAN_OPTIONS`` into ``boolean_options``) so that setuptools
    includes them in the ``--help`` output.
    """

    def initialize_options(self) -> None:
        """Initializes cmake-related options to their defaults."""
        super().initialize_options()
        self.cuda_architectures: str | None = None
        self.cmake_args: str | None = None
        self.cpp_tests: bool = False

    def finalize_options(self) -> None:
        """Finalizes cmake-related options."""
        super().finalize_options()


class BuildPy(_CMakeMixin, _build_py):
    """Runs the cmake build before installing the Python sources."""

    user_options = _build_py.user_options + _CMAKE_OPTIONS
    boolean_options = _build_py.boolean_options + list(_CMAKE_BOOLEAN_OPTIONS)

    def run(self) -> None:
        _run_cmake(
            cuda_architectures=self.cuda_architectures,
            cmake_args=self.cmake_args,
            cpp_tests=self.cpp_tests,
        )
        super().run()


class Develop(_CMakeMixin, _develop):
    """Runs the cmake build before setting up the editable install."""

    user_options = _develop.user_options + _CMAKE_OPTIONS
    boolean_options = _develop.boolean_options + list(_CMAKE_BOOLEAN_OPTIONS)

    def run(self) -> None:
        _run_cmake(
            cuda_architectures=self.cuda_architectures,
            cmake_args=self.cmake_args,
            cpp_tests=self.cpp_tests,
        )
        super().run()


class BuildExt(_CMakeMixin, _build_ext):
    """Runs the CMake build before the standard build_ext step.

    This makes ``python setup.py build_ext --inplace`` trigger CMake so
    that the C++ shared-library custom ops are compiled and copied into
    the source tree before any extension processing occurs.
    """

    user_options = _build_ext.user_options + _CMAKE_OPTIONS
    boolean_options = _build_ext.boolean_options + list(_CMAKE_BOOLEAN_OPTIONS)

    def run(self) -> None:
        _run_cmake(
            cuda_architectures=self.cuda_architectures,
            cmake_args=self.cmake_args,
            cpp_tests=self.cpp_tests,
        )
        super().run()


class BinaryDistribution(_Distribution):
    """Forces setuptools to tag the wheel as platform-specific.

    The custom-op shared libraries are compiled by cmake and copied into
    the source tree rather than through the normal Python C-extension
    mechanism.  Without this override, setuptools would classify the
    wheel as pure Python (``py3-none-any``), which is incorrect for a
    package containing native binaries.
    """

    def has_ext_modules(self) -> bool:
        """Signals that this distribution contains binary extensions."""
        return True


setup(
    name=_package_name(),
    distclass=BinaryDistribution,
    cmdclass={"build_py": BuildPy, "develop": Develop, "build_ext": BuildExt},
)
