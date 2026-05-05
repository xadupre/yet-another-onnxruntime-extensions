"""Integrates the cmake build step into pip install for the C++ custom-op libraries.

When ``pip install .`` or ``pip install -e .`` is run, this module triggers
cmake to configure and build the shared-library custom ops so they are
available alongside the Python sources without a separate build step.

If a CUDA compiler (``nvcc``) is detected on the PATH at build time the
package is distributed as ``yet-another-onnxruntime-extensions-cuda``; without
CUDA it falls back to the plain ``yet-another-onnxruntime-extensions`` name.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop

_HERE = Path(__file__).parent.resolve()

_BASE_PACKAGE_NAME = "yet-another-onnxruntime-extensions"


def _cuda_available() -> bool:
    """Checks whether an ``nvcc`` CUDA compiler is found on the PATH."""
    return shutil.which("nvcc") is not None


def _package_name() -> str:
    """Computes the distribution name, appending '-cuda' when CUDA is available."""
    if _cuda_available():
        return f"{_BASE_PACKAGE_NAME}-cuda"
    return _BASE_PACKAGE_NAME


def _run_cmake() -> None:
    """Configures and builds the C++ custom-op shared libraries via cmake.

    Prints a warning and returns without error when cmake is absent or when
    the build fails (e.g. because CUDA is not available), so that the Python
    package can still be installed in reduced-functionality mode.
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


class BuildPy(_build_py):
    """Runs the cmake build before installing the Python sources."""

    def run(self) -> None:
        _run_cmake()
        super().run()


class Develop(_develop):
    """Runs the cmake build before setting up the editable install."""

    def run(self) -> None:
        _run_cmake()
        super().run()


class BuildExt(_build_ext):
    """Runs the CMake build before the standard build_ext step.

    This makes ``python setup.py build_ext --inplace`` trigger CMake so
    that the C++ shared-library custom ops are compiled and copied into
    the source tree before any extension processing occurs.
    """

    def run(self) -> None:
        _run_cmake()
        super().run()


setup(
    name=_package_name(),
    cmdclass={"build_py": BuildPy, "develop": Develop, "build_ext": BuildExt},
)
