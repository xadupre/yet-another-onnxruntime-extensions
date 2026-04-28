#
# module: yaourt.ortops.fused_kernel.cuda
#
message(STATUS "+ KERNEL yaourt.ortops.fused_kernel.cuda")

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(STATUS "No CUDA compiler found - skipping yaourt.ortops.fused_kernel.cuda")
  return()
endif()

enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CUDA_SOURCES
  ../yaourt/ortops/fused_kernel/cuda/add_or_mul_shared_input.cu
  ../yaourt/ortops/fused_kernel/cuda/addaddaddmulmulmul.cu
  ../yaourt/ortops/fused_kernel/cuda/addaddmulmul.cu
  ../yaourt/ortops/fused_kernel/cuda/addmul.cu
  ../yaourt/ortops/fused_kernel/cuda/mul_mul_sigmoid.cu
  ../yaourt/ortops/fused_kernel/cuda/mul_sigmoid.cu
  ../yaourt/ortops/fused_kernel/cuda/negxplus1.cu
  ../yaourt/ortops/fused_kernel/cuda/replace_zero.cu
  ../yaourt/ortops/fused_kernel/cuda/rotary.cu
  ../yaourt/ortops/fused_kernel/cuda/scatter_nd_of_shape.cu
  ../yaourt/ortops/fused_kernel/cuda/scatter_nd_of_shape_masked.cu
  ../yaourt/ortops/fused_kernel/cuda/submul.cu
  ../yaourt/ortops/fused_kernel/cuda/transpose_cast_2d.cu
  ../yaourt/ortops/fused_kernel/cuda/tri_matrix.cu
  ../yaourt/ortops/fused_kernel/cuda/ort_fused_kernel_cuda_lib.cu)

if(WIN32)
  file(WRITE "yaourt/ortops/fused_kernel/cuda/ortops_fused_kernel_cuda.def"
       "LIBRARY \"ortops_fused_kernel_cuda.dll\"\nEXPORTS\n  RegisterCustomOps @1")
  list(APPEND CUDA_SOURCES "yaourt/ortops/fused_kernel/cuda/ortops_fused_kernel_cuda.def")
endif()

add_library(ortops_fused_kernel_cuda SHARED ${CUDA_SOURCES})

target_include_directories(
  ortops_fused_kernel_cuda
  PRIVATE
  "${ROOT_INCLUDE_PATH}"
  "${ORTOPS_INCLUDE_DIR}"
  "${ONNXRUNTIME_INCLUDE_DIR}"
  "${ROOT_PROJECT_PATH}/yaourt/ortops/fused_kernel/cuda")

target_compile_definitions(
  ortops_fused_kernel_cuda
  PRIVATE
  ORT_VERSION=${ORT_VERSION_INT})

target_link_libraries(
  ortops_fused_kernel_cuda
  PRIVATE
  common
  cublas
  cublasLt)

set_property(TARGET ortops_fused_kernel_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(ortops_fused_kernel_cuda PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(ortops_fused_kernel_cuda PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
set_target_properties(ortops_fused_kernel_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Copy the shared library to the yaourt/ortops/fused_kernel/cuda folder after build
add_custom_command(
  TARGET ortops_fused_kernel_cuda POST_BUILD
  COMMAND ${CMAKE_COMMAND} ARGS -E copy
          "$<TARGET_FILE:ortops_fused_kernel_cuda>"
          "${ROOT_PROJECT_PATH}/yaourt/ortops/fused_kernel/cuda")
