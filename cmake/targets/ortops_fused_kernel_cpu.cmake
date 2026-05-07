#
# module: yaourt.ortops.fused_kernel.cpu (MulMul - optimised CPU kernel)
#
message(STATUS "+ KERNEL yaourt.ortops.fused_kernel.cpu")

ort_add_custom_op(
  ortops_fused_kernel_cpu
  "CPU"
  yaourt/ortops/fused_kernel/cpu
  ../yaourt/ortops/fused_kernel/cpu/ort_fused_kernel_cpu_lib.cc)

target_include_directories(
  ortops_fused_kernel_cpu
  PRIVATE
  "${ROOT_INCLUDE_PATH}"
  "${ORTOPS_INCLUDE_DIR}"
  "${ONNXRUNTIME_INCLUDE_DIR}"
  "${ROOT_PROJECT_PATH}/yaourt/ortops/fused_kernel/cpu")

# Parallelism is implemented via std::thread (safer than OpenMP inside a
# dlopen-able shared library).  Link against the platform threading library.
find_package(Threads REQUIRED)
target_link_libraries(ortops_fused_kernel_cpu PRIVATE Threads::Threads)

# AVX2 is already activated globally via -march=native (constants.cmake) on
# x86_64/AMD64 hosts; no extra per-target flag is required.

