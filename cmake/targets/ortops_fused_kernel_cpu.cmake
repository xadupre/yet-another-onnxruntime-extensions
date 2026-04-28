#
# module: yaourt.ortops.fused_kernel.cpu (DenseToSparse, SparseToDense)
#
message(STATUS "+ KERNEL yaourt.ortops.fused_kernel.cpu")

ort_add_custom_op(
  ortops_fused_kernel_cpu
  "CPU"
  yaourt/ortops/sparse/cpu_v1
  ../yaourt/ortops/sparse/cpu_v1/ort_fused_kernel_cpu_lib.cc)

target_include_directories(
  ortops_fused_kernel_cpu
  PRIVATE
  "${ROOT_INCLUDE_PATH}"
  "${ORTOPS_INCLUDE_DIR}"
  "${ONNXRUNTIME_INCLUDE_DIR}")

target_link_libraries(
  ortops_fused_kernel_cpu
  PRIVATE
  common)
