Available Custom Ops
====================

This page lists all custom ONNX Runtime operators provided by
*yet-another-onnxruntime-extensions*.  The catalogue is generated dynamically
at documentation-build time by parsing the C++ source files, so it always
reflects the actual implementation without any manual maintenance.

CPU Operators
-------------

The following operators are registered under the
``yaourt.ortops.sparse.cpu`` domain and run on the
``CPUExecutionProvider``.

.. runpython::
    :rst:

    from yaourt.ortops.doc import print_cpu_ops_rst
    print_cpu_ops_rst()
