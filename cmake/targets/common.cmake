#
# module: common C++ library (string utilities)
#
message(STATUS "+ KERNEL yaourt.common")
add_library(
    common
    STATIC
    ../yaourt/cpp/onnx_extended_helpers.cpp)
target_include_directories(common PRIVATE "${ROOT_INCLUDE_PATH}")
