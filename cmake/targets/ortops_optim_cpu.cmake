#
# module: yaourt.ortops.optim.cpu (DenseToSparse, SparseToDense)
#
message(STATUS "+ KERNEL yaourt.ortops.optim.cpu")

ort_add_custom_op(
  ortops_optim_cpu
  "CPU"
  ortops/optim/cpu
  ../ortops/optim/cpu/ort_optim_cpu_lib.cc)

target_include_directories(
  ortops_optim_cpu
  PRIVATE
  "${ROOT_INCLUDE_PATH}"
  "${ORTOPS_INCLUDE_DIR}"
  "${ONNXRUNTIME_INCLUDE_DIR}")

target_link_libraries(
  ortops_optim_cpu
  PRIVATE
  common)
