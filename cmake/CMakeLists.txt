cmake_minimum_required(VERSION 3.25.0)
project(yaourt_ortops VERSION 0.1.0)

list(APPEND CMAKE_MODULE_PATH
     "${CMAKE_CURRENT_SOURCE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/externals")

#
# Packages and constants
#

include("constants.cmake")
include("load_externals.cmake")

#
# modules
#

message(STATUS "--------------------------------------------")
set(ROOT_PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(ROOT_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../yaourt/cpp/include)
message(STATUS "ROOT_PROJECT_PATH=${ROOT_PROJECT_PATH}")
message(STATUS "ROOT_INCLUDE_PATH=${ROOT_INCLUDE_PATH}")
message(STATUS "--------------------------------------------")

#
# common library
#

include("targets/common.cmake")

#
# C++ unit tests
#

enable_testing()
include("targets/unittest_cpp.cmake")

#
# new kernels for onnxruntime
#

message(STATUS "---------------------------------------------------")
set(ORTOPS_INCLUDE_DIR "${ROOT_INCLUDE_PATH}")
message(STATUS "ROOT_INCLUDE_PATH=${ROOT_INCLUDE_PATH}")
message(STATUS "ORTOPS_INCLUDE_DIR=${ORTOPS_INCLUDE_DIR}")

include("targets/ortops_fused_kernel_cpu.cmake")
include("targets/ortops_fused_kernel_cpu2.cmake")
include("targets/ortops_fused_kernel_cuda.cmake")

#
# Final
#

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

get_property(
  all_targets DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY BUILDSYSTEM_TARGETS)
foreach(target ${all_targets})
  message(STATUS "Found target: ${target}")
endforeach()

message(STATUS "-------------------")
