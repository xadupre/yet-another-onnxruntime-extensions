#
# module: yaourt.ortops.sparse.cpu (DenseToSparse, SparseToDense) - lite custom op API
#
message(STATUS "+ KERNEL yaourt.ortops.sparse.cpu (lite)")

ort_add_custom_op(
  ortops_sparse_cpu2
  "CPU"
  yaourt/ortops/sparse/cpu
  ../yaourt/ortops/sparse/cpu/ort_sparse_cpu2_lib.cc)

target_include_directories(
  ortops_sparse_cpu2
  PRIVATE
  "${ROOT_INCLUDE_PATH}"
  "${ORTOPS_INCLUDE_DIR}"
  "${ONNXRUNTIME_INCLUDE_DIR}")

target_link_libraries(
  ortops_sparse_cpu2
  PRIVATE
  common)
