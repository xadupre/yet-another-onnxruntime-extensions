Fused Kernel CUDA Custom Ops
============================

This page lists all custom ONNX Runtime operators in the fused-kernel CUDA
family provided by *yet-another-onnxruntime-extensions*.  The catalogue is
generated dynamically at documentation-build time by parsing the C++ source
files, so it always reflects the actual implementation without any manual
maintenance.

These operators are registered under the
``yaourt.ortops.fused_kernel.cuda`` domain and run on the
``CUDAExecutionProvider``.  The shared library must be compiled from source
with a CUDA-enabled CMake build — see :doc:`../getting_started` for instructions.
Once built, it can be loaded via
:data:`~yaourt.ortops.FUSED_KERNEL_CUDA_LIB_PATH`.

.. note::

    CUDA operators require a GPU and a CUDA-capable build of ONNX Runtime.
    They are not included in the pre-built wheel.

Operators
---------

.. runpython::
    :rst:

    from yaourt.ortops.doc import print_cuda_ops_rst
    print_cuda_ops_rst()
