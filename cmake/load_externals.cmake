
#
# Packages
#

message(STATUS "-------------------")

if(NOT ORT_VERSION)
  set(ORT_VERSION 1.25.0)
  set(ORT_VERSION_INT 1250)
endif()

include(FetchContent)

string(LENGTH "${ORT_VERSION}" ORT_VERSION_LENGTH)

if(ORT_VERSION_LENGTH LESS_EQUAL 12)
  message(STATUS "ORT - retrieve release version ${ORT_VERSION}")
  if(MSVC)
    set(ORT_NAME "onnxruntime-win-x64-${ORT_VERSION}.zip")
    set(ORT_FOLD "onnxruntime-win-x64-${ORT_VERSION}")
  elseif(APPLE)
    set(ORT_NAME "onnxruntime-osx-arm64-${ORT_VERSION}.tgz")
    set(ORT_FOLD "onnxruntime-osx-arm64-${ORT_VERSION}")
  else()
    set(ORT_NAME "onnxruntime-linux-x64-${ORT_VERSION}.tgz")
    set(ORT_FOLD "onnxruntime-linux-x64-${ORT_VERSION}")
  endif()
  set(ORT_ROOT "https://github.com/microsoft/onnxruntime/releases/download/")
  set(ORT_URL "${ORT_ROOT}v${ORT_VERSION}/${ORT_NAME}")

  string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" ORT_VERSION_MATCH ${ORT_VERSION})
  set(ORT_VERSION_MAJOR ${CMAKE_MATCH_1})
  set(ORT_VERSION_MINOR ${CMAKE_MATCH_2})
  set(ORT_VERSION_PATCH ${CMAKE_MATCH_3})
  math(
    EXPR
    ORT_VERSION_INT
    "${ORT_VERSION_MAJOR} * 1000 + ${ORT_VERSION_MINOR} * 10 + ${ORT_VERSION_PATCH}"
    OUTPUT_FORMAT DECIMAL)

  message(STATUS "ORT - ORT_URL=${ORT_URL}")
  FetchContent_Declare(onnxruntime URL ${ORT_URL})
  FetchContent_MakeAvailable(onnxruntime)
  set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
  set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)
else()
  message(STATUS "ORT - retrieve from directory '${ORT_VERSION}'")
  set(ORT_VERSION_INT 99999)
  set(ONNXRUNTIME_LIB_DIR "${ORT_VERSION}")
  if(MSVC)
    set(ONNXRUNTIME_INCLUDE_DIR
        "${ORT_VERSION}\\..\\..\\..\\..\\include\\onnxruntime\\core\\session")
  else()
    set(ONNXRUNTIME_INCLUDE_DIR
        "${ORT_VERSION}/../../../include/onnxruntime/core/session")
  endif()
  set(ORT_URL ${ORT_VERSION})
endif()

message(STATUS "ONNXRUNTIME_INCLUDE_DIR=${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNXRUNTIME_LIB_DIR=${ONNXRUNTIME_LIB_DIR}")
message(STATUS "ORT_VERSION_INT=${ORT_VERSION_INT}")

find_library(ONNXRUNTIME onnxruntime HINTS "${ONNXRUNTIME_LIB_DIR}")
if(NOT ONNXRUNTIME)
  message(FATAL_ERROR "onnxruntime cannot be found at '${ONNXRUNTIME_LIB_DIR}'")
endif()

message(STATUS "-------------------")

#
#! ort_add_custom_op : compiles a custom op shared library
#
# \arg:name     target name
# \arg:provider CPU or CUDA
# \arg:folder   output location (relative to ROOT_PROJECT_PATH)
# \argn:        C++ source files to compile
#
function(ort_add_custom_op name provider folder)
  if(WIN32)
    file(WRITE "${folder}/${name}.def"
         "LIBRARY \"${name}.dll\"\nEXPORTS\n  RegisterCustomOps @1")
    list(APPEND ARGN "${folder}/${name}.def")
  endif()
  message(STATUS "ort: custom op ${provider}: '${name}' in '${folder}'")
  add_library(${name} SHARED ${ARGN})
  target_include_directories(${name} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_compile_definitions(
    ${name}
    PRIVATE
    ORT_VERSION=${ORT_VERSION_INT})
  set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(${name} PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties(${name} PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
  add_custom_command(
    TARGET ${name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E copy
            "$<TARGET_FILE:${name}>"
            "${ROOT_PROJECT_PATH}/${folder}")
endfunction()
