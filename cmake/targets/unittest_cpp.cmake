#
# module: C++ unit tests for common helper functions (MakeString, SplitString, etc.)
#
message(STATUS "+ unittest_cpp")

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
  URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(
  unittest_cpp
  ../cpp/unittests/test_make_string.cpp)

target_include_directories(
  unittest_cpp
  PRIVATE
  "${ROOT_INCLUDE_PATH}")

target_link_libraries(
  unittest_cpp
  PRIVATE
  common
  GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(unittest_cpp)
