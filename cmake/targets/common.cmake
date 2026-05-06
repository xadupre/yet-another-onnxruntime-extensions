#
# module: common C++ library (string utilities)
#
message(STATUS "+ KERNEL yaourt.common")
add_library(
    common
    STATIC
    ../yaourt/cpp/yaourt_helpers.cpp)
set_property(TARGET common PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(common PRIVATE "${ROOT_INCLUDE_PATH}")
