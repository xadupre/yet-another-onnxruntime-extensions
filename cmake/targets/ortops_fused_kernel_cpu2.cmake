#
# module: yaourt.ortops.fused_kernel.cpu (DenseToSparse, SparseToDense) - lite custom op API
#
message(STATUS "+ KERNEL yaourt.ortops.fused_kernel.cpu (lite)")

ort_add_custom_op(
  ortops_fused_kernel_cpu2
  "CPU"
  yaourt/ortops/sparse/cpu
  ../yaourt/ortops/sparse/cpu/ort_fused_kernel_cpu2_lib.cc)

target_include_directories(
  ortops_fused_kernel_cpu2
  PRIVATE
  "${ROOT_INCLUDE_PATH}"
  "${ORTOPS_INCLUDE_DIR}"
  "${ONNXRUNTIME_INCLUDE_DIR}")

target_link_libraries(
  ortops_fused_kernel_cpu2
  PRIVATE
  common)
