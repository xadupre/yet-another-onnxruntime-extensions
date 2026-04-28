#
# module: yaourt.ortops.sparse.cpu (DenseToSparse, SparseToDense)
#
message(STATUS "+ KERNEL yaourt.ortops.sparse.cpu")

ort_add_custom_op(
  ortops_sparse_cpu
  "CPU"
  yaourt/ortops/sparse/cpu_v1
  ../yaourt/ortops/sparse/cpu_v1/ort_sparse_cpu_lib.cc)

target_include_directories(
  ortops_sparse_cpu
  PRIVATE
  "${ROOT_INCLUDE_PATH}"
  "${ORTOPS_INCLUDE_DIR}"
  "${ONNXRUNTIME_INCLUDE_DIR}")

target_link_libraries(
  ortops_sparse_cpu
  PRIVATE
  common)
