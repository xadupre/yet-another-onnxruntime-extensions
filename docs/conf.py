import sys
import yaourt

project = "yet-another-onnxruntime-extensions"
author = "yet-another-onnxruntime-extensions contributors"
release = yaourt.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.duration",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_issues",
    "matplotlib.sphinxext.plot_directive",
]

# templates_path = ["_templates"]
exclude_patterns = ["_build"]
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
