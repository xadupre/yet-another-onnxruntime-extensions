"""Integrates the cmake build step into pip install for the C++ custom-op libraries.

When ``pip install .`` or ``pip install -e .`` is run, this module triggers
cmake to configure and build the shared-library custom ops so they are
available alongside the Python sources without a separate build step.
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


def _run_cmake() -> None:
    """Configures and builds the C++ custom-op shared libraries via cmake."""
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError(
            "cmake executable not found on PATH; install cmake and re-run the build."
        )

    cmake_src = _HERE / "cmake"
    if not cmake_src.is_dir():
        raise RuntimeError(
            f"cmake source directory not found at {cmake_src}; "
            "the repository may be incomplete."
        )

    build_dir = _HERE / "build"
    configure_cmd = [cmake, f"-S{cmake_src}", f"-B{build_dir}", "-DCMAKE_BUILD_TYPE=Release"]
    build_cmd = [cmake, "--build", str(build_dir), "--config", "Release"]

    print("yaourt: cmake configure ...", flush=True)
    try:
        subprocess.run(configure_cmd, check=True, cwd=str(_HERE))
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"cmake configure step failed with exit code {exc.returncode}."
        ) from exc
    print("yaourt: cmake build ...", flush=True)
    try:
        subprocess.run(build_cmd, check=True, cwd=str(_HERE))
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"cmake build step failed with exit code {exc.returncode}.") from exc


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


setup(cmdclass={"build_py": BuildPy, "develop": Develop, "build_ext": BuildExt})
