#
# module: yaourt.ortops.optim.cpu (DenseToSparse, SparseToDense) - lite custom op API
#
message(STATUS "+ KERNEL yaourt.ortops.optim.cpu (lite)")

ort_add_custom_op(
  ortops_optim_cpu2
  "CPU"
  ortops/sparse/cpu
  ../ortops/sparse/cpu/ort_optim_cpu2_lib.cc)

target_include_directories(
  ortops_optim_cpu2
  PRIVATE
  "${ROOT_INCLUDE_PATH}"
  "${ORTOPS_INCLUDE_DIR}"
  "${ONNXRUNTIME_INCLUDE_DIR}")

target_link_libraries(
  ortops_optim_cpu2
  PRIVATE
  common)
