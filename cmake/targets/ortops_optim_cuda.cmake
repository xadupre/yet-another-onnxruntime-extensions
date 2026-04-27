#
# module: yaourt.ortops.optim.cuda
#
message(STATUS "+ KERNEL yaourt.ortops.optim.cuda")

option(USE_CUDA "Build CUDA custom-op extensions" OFF)
if(NOT USE_CUDA)
  message(STATUS "USE_CUDA=OFF - skipping yaourt.ortops.optim.cuda (set -DUSE_CUDA=ON to enable)")
  return()
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "USE_CUDA=ON but no CUDA compiler was found - install the CUDA toolkit and ensure nvcc is on PATH")
endif()

enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CUDA_SOURCES
  ../yaourt/ortops/optim/cuda/add_or_mul_shared_input.cu
  ../yaourt/ortops/optim/cuda/addaddaddmulmulmul.cu
  ../yaourt/ortops/optim/cuda/addaddmulmul.cu
  ../yaourt/ortops/optim/cuda/addmul.cu
  ../yaourt/ortops/optim/cuda/mul_mul_sigmoid.cu
  ../yaourt/ortops/optim/cuda/mul_sigmoid.cu
  ../yaourt/ortops/optim/cuda/negxplus1.cu
  ../yaourt/ortops/optim/cuda/replace_zero.cu
  ../yaourt/ortops/optim/cuda/rotary.cu
  ../yaourt/ortops/optim/cuda/scatter_nd_of_shape.cu
  ../yaourt/ortops/optim/cuda/scatter_nd_of_shape_masked.cu
  ../yaourt/ortops/optim/cuda/submul.cu
  ../yaourt/ortops/optim/cuda/transpose_cast_2d.cu
  ../yaourt/ortops/optim/cuda/tri_matrix.cu
  ../yaourt/ortops/optim/cuda/ort_optim_cuda_lib.cu)

if(WIN32)
  file(WRITE "yaourt/ortops/optim/cuda/ortops_optim_cuda.def"
       "LIBRARY \"ortops_optim_cuda.dll\"\nEXPORTS\n  RegisterCustomOps @1")
  list(APPEND CUDA_SOURCES "yaourt/ortops/optim/cuda/ortops_optim_cuda.def")
endif()

add_library(ortops_optim_cuda SHARED ${CUDA_SOURCES})

target_include_directories(
  ortops_optim_cuda
  PRIVATE
  "${ROOT_INCLUDE_PATH}"
  "${ORTOPS_INCLUDE_DIR}"
  "${ONNXRUNTIME_INCLUDE_DIR}"
  "${ROOT_PROJECT_PATH}/yaourt/ortops/optim/cuda")

target_compile_definitions(
  ortops_optim_cuda
  PRIVATE
  ORT_VERSION=${ORT_VERSION_INT})

target_link_libraries(
  ortops_optim_cuda
  PRIVATE
  common
  cublas
  cublasLt)

set_property(TARGET ortops_optim_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(ortops_optim_cuda PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(ortops_optim_cuda PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
set_target_properties(ortops_optim_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Copy the shared library to the yaourt/ortops/optim/cuda folder after build
add_custom_command(
  TARGET ortops_optim_cuda POST_BUILD
  COMMAND ${CMAKE_COMMAND} ARGS -E copy
          "$<TARGET_FILE:ortops_optim_cuda>"
          "${ROOT_PROJECT_PATH}/yaourt/ortops/optim/cuda")
