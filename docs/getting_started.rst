Getting Started
===============

This page explains how to install and build the project, both for regular users
and for developers who want to contribute or extend it.

.. contents:: Table of Contents
    :local:
    :depth: 2

For Users
---------

Prerequisites
~~~~~~~~~~~~~

- Python 3.10 or later
- ``pip``

Installation
~~~~~~~~~~~~

Install the package directly from `PyPI <https://pypi.org/project/yet-another-onnxruntime-extensions/>`_:

.. code-block:: bash

    pip install yet-another-onnxruntime-extensions

Verify the installation:

.. code-block:: python

    import yaourt
    print(yaourt.__version__)

Using Custom Operators (ortops)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The custom ONNX Runtime operators (``ortops``) require a compiled shared
library.  The library is shipped as a pre-built binary inside the wheel on
supported platforms, so no extra build step is needed when installing from
PyPI.

To load the library and register the operators with ONNX Runtime:

.. code-block:: python

    import onnxruntime as ort
    from yaourt.ortops import get_ort_ext_libs

    opts = ort.SessionOptions()
    opts.register_custom_ops_library(get_ort_ext_libs()[0])

For Developers
--------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.10 or later
- ``git``
- A C++ compiler supported by CMake (GCC, Clang, or MSVC)
- `CMake <https://cmake.org/>`_ 3.25 or later

Clone the Repository
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/xadupre/yet-another-onnxruntime-extensions.git
    cd yet-another-onnxruntime-extensions

Create a Virtual Environment (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate        # Linux / macOS
    # .venv\Scripts\activate         # Windows

Install the Package in Editable Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the Python package together with all development dependencies:

.. code-block:: bash

    pip install -e .[dev]

Build the C++ Custom Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ortops`` module contains custom ONNX Runtime operators implemented in
C++.  Configure and build them with CMake:

.. code-block:: bash

    cmake -S cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release

The compiled shared library is placed under ``build/`` and is automatically
discovered by :func:`yaourt.ortops.get_ort_ext_libs`.

Run the Tests
~~~~~~~~~~~~~

Run the pure-Python test suite:

.. code-block:: bash

    pytest unittests

Run the ``ortops`` tests (requires the C++ build above):

.. code-block:: bash

    pytest unittests/ortops

Run all tests with coverage:

.. code-block:: bash

    pytest --cov=yaourt --cov-report=term-missing unittests

Code Style and Linting
~~~~~~~~~~~~~~~~~~~~~~

The project uses `black <https://black.readthedocs.io/>`_ for formatting and
`ruff <https://docs.astral.sh/ruff/>`_ for linting.  Run both before
committing:

.. code-block:: bash

    black .
    ruff check .

Build the Documentation
~~~~~~~~~~~~~~~~~~~~~~~

Install the documentation dependencies first:

.. code-block:: bash

    pip install -e .[docs]

Then build the HTML documentation:

.. code-block:: bash

    python -m sphinx docs dist/html -j auto

Or use the convenience script at the root of the repository:

.. code-block:: bash

    bash make_doc.sh

The generated documentation is written to ``dist/html/``.

Build the Python Wheel
~~~~~~~~~~~~~~~~~~~~~~

To produce a distribution wheel (e.g. for local testing):

.. code-block:: bash

    pip install build
    python -m build

The wheel and source distribution are placed in the ``dist/`` directory.
