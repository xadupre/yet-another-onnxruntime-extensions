"""Integrates the cmake build step into pip install for the C++ custom-op libraries.

When ``pip install .`` or ``pip install -e .`` is run, this module triggers
cmake to configure and build the shared-library custom ops so they are
available alongside the Python sources without a separate build step.

Set the environment variable ``SKIP_CMAKE=1`` to skip the cmake build step
(useful in CI where cmake is run explicitly before pip install).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop

_HERE = Path(__file__).parent.resolve()


def _run_cmake() -> None:
    """Configures and builds the C++ custom-op shared libraries via cmake."""
    if os.environ.get("SKIP_CMAKE"):
        print("yaourt: cmake build skipped (SKIP_CMAKE is set).", flush=True)
        return

    cmake = shutil.which("cmake")
    if cmake is None:
        print(
            "WARNING: cmake executable not found on PATH; "
            "the C++ custom-op libraries will not be built.",
            file=sys.stderr,
        )
        return

    cmake_src = _HERE / "cmake"
    if not cmake_src.is_dir():
        print(
            "WARNING: cmake source directory not found; "
            "the C++ custom-op libraries will not be built.",
            file=sys.stderr,
        )
        return

    build_dir = _HERE / "_build"
    configure_cmd = [cmake, f"-S{cmake_src}", f"-B{build_dir}", "-DCMAKE_BUILD_TYPE=Release"]
    build_cmd = [cmake, "--build", str(build_dir), "--config", "Release"]

    try:
        print("yaourt: cmake configure ...", flush=True)
        subprocess.run(configure_cmd, check=True, cwd=str(_HERE))
        print("yaourt: cmake build ...", flush=True)
        subprocess.run(build_cmd, check=True, cwd=str(_HERE))
    except subprocess.CalledProcessError as exc:
        print(
            f"WARNING: cmake step failed ({exc}); "
            "the C++ custom-op libraries will not be built.",
            file=sys.stderr,
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


setup(cmdclass={"build_py": BuildPy, "develop": Develop})
