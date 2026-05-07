import logging
import os
import subprocess
import sys
from pathlib import Path

import yaourt

project = "yet-another-onnxruntime-extensions"
author = "yet-another-onnxruntime-extensions contributors"
release = yaourt.__version__

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Run Doxygen to generate the XML consumed by Breathe.
# The Doxyfile lives next to this conf.py.  A missing doxygen executable is
# treated as a soft failure so that the rest of the Sphinx build can still
# proceed.
# ---------------------------------------------------------------------------
_docs_dir = Path(__file__).parent
_doxygen_result = subprocess.run(
    ["doxygen", "Doxyfile"], cwd=str(_docs_dir), check=False, capture_output=True, text=True
)
if _doxygen_result.returncode != 0:
    _logger.warning(
        "Doxygen exited with code %d; C++ API docs may be incomplete.\n%s",
        _doxygen_result.returncode,
        _doxygen_result.stderr or _doxygen_result.stdout,
    )

extensions = [
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.duration",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
    "sphinx_runpython.runpython",
    "matplotlib.sphinxext.plot_directive",
]

breathe_projects = {"yaourt": str(_docs_dir / "_doxygen" / "xml")}
breathe_default_project = "yaourt"

sphinx_gallery_conf = {"examples_dirs": ["examples"], "gallery_dirs": ["auto_examples"]}

# templates_path = ["_templates"]
# Exclude the CI durations page when building in CI (e.g. GitHub Actions sets CI=true)
# because the page queries the GitHub REST API and makes network requests unsuitable for CI.
_exclude_patterns = ["build"]
if os.environ.get("CI"):
    _exclude_patterns.append("ci_durations.rst")

exclude_patterns = _exclude_patterns

# Tag used by the ``.. only::`` directive in docs to conditionally include content.
# Set when running under CI so that the ci_durations page is skipped.
if os.environ.get("CI"):
    tags.add("ci_build")  # noqa: F821  (``tags`` is injected by Sphinx)
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_theme_options = {
    "github_url": "https://github.com/xadupre/yet-another-onnxruntime-extensions",
    "logo": {"image_light": "_static/logo.svg", "image_dark": "_static/logo.svg"},
}

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "onnx": ("https://onnx.ai/onnx/", None),
    "onnxruntime": ("https://onnxruntime.ai/docs/api/python/", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
}

suppress_warnings = ["intersphinx.external"]
if os.environ.get("CI"):
    suppress_warnings.append("toc.excluded")
