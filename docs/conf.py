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
exclude_patterns = ["build"]
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

# ---------------------------------------------------------------------------
# On CI (GitHub Actions sets CI=true) exclude the CUDA custom-ops page since
# no CUDA device or driver is available in the runner environment.
# Also exclude the CI durations page because it queries the GitHub REST API
# and makes network requests unsuitable for CI.
# ---------------------------------------------------------------------------
if os.environ.get("CI"):
    tags.add("ci")  # noqa: F821 — ``tags`` is injected by Sphinx
    tags.add("ci_build")  # noqa: F821  (``tags`` is injected by Sphinx)
    exclude_patterns.append("custom_ops/fused_cuda.rst")
    exclude_patterns.append("ci_durations.rst")
    # fused_cuda.rst is excluded above, so suppress the "unknown document"
    # warning that Sphinx emits for :doc:`fused_cuda` references even inside
    # ``.. only:: not ci`` blocks (Sphinx resolves refs before evaluating tags).
    suppress_warnings.append("ref.doc")
    # Suppress the warning Sphinx emits when a toctree references an excluded
    # document (even when the toctree is inside a ``.. only::`` block, Sphinx
    # still processes toctree entries before evaluating ``.. only::`` conditions).
    suppress_warnings.append("toc.excluded")
